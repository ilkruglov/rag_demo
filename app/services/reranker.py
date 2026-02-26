"""Lightweight reranker for keyword overlap and semantic boosting."""
from __future__ import annotations

import re
from typing import ClassVar, List, Optional

from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle

from app.services.semantic_enrichment import (
    detect_direction_from_query,
    get_boost_codes_for_query,
    OperationDirection,
    NRD_OPERATIONS,
)


# Pattern for operation codes like 16/2, 10/50
OPERATION_CODE_PATTERN = re.compile(r'\b(\d{1,2}/\d{1,2})\b')
# Pattern for simple operation codes like 35, 37 (standalone two-digit numbers, typically in context)
SIMPLE_OPERATION_CODE_PATTERN = re.compile(r'(?:код(?:\s+операции)?|операци[яию])\s*[-–]?\s*(\d{1,2}[А-Яа-яA-Za-z]?)\b', re.IGNORECASE)
# Pattern for form codes like MF170, AF005 (case-insensitive)
FORM_CODE_PATTERN = re.compile(r'\b([A-Za-z]{2}\d{3})\b')


class OperationCodeBooster(BaseNodePostprocessor):
    """Boost nodes that contain operation codes mentioned in the query.

    This is critical for queries like "условия исполнения операции 16/2"
    where the operation code must be matched even if semantic similarity is low.
    """

    def __init__(
        self,
        boost_factor: float = 0.5,
        *,
        callback_manager: CallbackManager | None = None
    ) -> None:
        super().__init__(callback_manager=callback_manager)
        self._boost_factor = boost_factor

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if not nodes or not query_bundle:
            return nodes

        query_text = query_bundle.query_str
        # Extract operation codes from query (e.g., "16/2", "10/50")
        query_op_codes = set(OPERATION_CODE_PATTERN.findall(query_text))
        # Extract simple operation codes from query (e.g., "код 35", "операция 37")
        query_simple_codes = set(SIMPLE_OPERATION_CODE_PATTERN.findall(query_text))
        # Extract form codes from query (e.g., "MF170", "AF005")
        query_form_codes = set(FORM_CODE_PATTERN.findall(query_text))

        if not query_op_codes and not query_form_codes and not query_simple_codes:
            return nodes

        for node in nodes:
            metadata = node.node.metadata or {}
            node_op_codes = set(metadata.get("operation_codes", []))
            node_form_codes = set(metadata.get("form_codes", []))

            # Also check text content for codes (in case metadata is missing)
            content = node.get_content()
            node_op_codes.update(OPERATION_CODE_PATTERN.findall(content))
            # Also extract simple codes from content
            node_simple_codes = set(SIMPLE_OPERATION_CODE_PATTERN.findall(content))
            node_form_codes.update(FORM_CODE_PATTERN.findall(content))

            # Calculate boost based on matching codes
            op_match = query_op_codes.intersection(node_op_codes)
            simple_match = query_simple_codes.intersection(node_simple_codes)
            form_match = query_form_codes.intersection(node_form_codes)

            if op_match or form_match or simple_match:
                # Strong boost for exact operation code match
                if op_match:
                    boost = self._boost_factor * len(op_match)
                    node.score = (node.score or 0.0) + boost
                # Strong boost for simple operation code match
                if simple_match:
                    boost = self._boost_factor * len(simple_match)
                    node.score = (node.score or 0.0) + boost
                # Moderate boost for form code match
                if form_match:
                    boost = (self._boost_factor / 2) * len(form_match)
                    node.score = (node.score or 0.0) + boost

        return nodes


class SectionTitleBooster(BaseNodePostprocessor):
    """Boost nodes where section_title matches query keywords."""

    _TOKEN_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"[\w-]+", re.IGNORECASE)

    def __init__(self, boost_factor: float = 0.3, *, callback_manager: CallbackManager | None = None) -> None:
        super().__init__(callback_manager=callback_manager)
        self._boost_factor = boost_factor

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        tokens = {token.lower() for token in SectionTitleBooster._TOKEN_PATTERN.findall(text)}
        return {token for token in tokens if len(token) > 2}

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if not nodes or not query_bundle:
            return nodes

        query_tokens = self._tokenize(query_bundle.query_str)
        if not query_tokens:
            return nodes

        for node in nodes:
            section_title = node.node.metadata.get("section_title", "")
            if not section_title:
                continue

            section_tokens = self._tokenize(section_title)
            if not section_tokens:
                continue

            # Calculate overlap between query and section title
            overlap = query_tokens.intersection(section_tokens)
            if overlap:
                coverage = len(overlap) / len(query_tokens)
                boost = coverage * self._boost_factor
                node.score = (node.score or 0.0) + boost

        return nodes


class KeywordOverlapReranker(BaseNodePostprocessor):
    """Rank nodes by keyword overlap with the query text."""

    _TOKEN_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"[\w-]+", re.IGNORECASE)

    def __init__(self, top_n: int = 4, *, callback_manager: CallbackManager | None = None) -> None:
        super().__init__(callback_manager=callback_manager)
        self._top_n = max(top_n, 0)

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        tokens = {token.lower() for token in KeywordOverlapReranker._TOKEN_PATTERN.findall(text)}
        return {token for token in tokens if len(token) > 2}

    @staticmethod
    def _score(node: NodeWithScore, query_tokens: set[str]) -> float:
        if not query_tokens:
            return node.score or 0.0
        node_tokens = KeywordOverlapReranker._tokenize(node.get_content())
        if not node_tokens:
            return 0.0
        overlap = query_tokens.intersection(node_tokens)
        if not overlap:
            return 0.0
        coverage = len(overlap) / len(query_tokens)
        density = len(overlap) / len(node_tokens)
        base = node.score or 0.0
        return base + coverage * 0.6 + density * 0.4

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if self._top_n == 0 or not nodes:
            return nodes

        query_text = query_bundle.query_str if query_bundle else None
        if not query_text:
            return nodes

        query_tokens = self._tokenize(query_text)
        if not query_tokens:
            return nodes

        scored = [
            (self._score(node, query_tokens), idx, node)
            for idx, node in enumerate(nodes)
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        reranked = [item[2] for item in scored[: self._top_n]]
        return reranked


class SemanticDirectionBooster(BaseNodePostprocessor):
    """Boost nodes based on semantic understanding of operation direction.

    This is the key component for solving the 35 vs 37 problem:
    - Detects if user is asking about incoming (прием) or outgoing (передача) operations
    - Boosts nodes that match the detected direction
    - Uses the NRD operations vocabulary for semantic understanding
    """

    def __init__(
        self,
        boost_factor: float = 0.6,
        *,
        callback_manager: CallbackManager | None = None
    ) -> None:
        super().__init__(callback_manager=callback_manager)
        self._boost_factor = boost_factor

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if not nodes or not query_bundle:
            return nodes

        query_text = query_bundle.query_str

        # Detect direction from query semantics
        direction = detect_direction_from_query(query_text)
        if not direction:
            return nodes

        # Get codes that should be boosted for this direction
        boost_codes = get_boost_codes_for_query(query_text)

        for node in nodes:
            metadata = node.node.metadata or {}
            content = node.get_content()

            # Get operation codes from metadata and content
            node_op_codes = set(metadata.get("operation_codes", []))
            node_op_codes.update(OPERATION_CODE_PATTERN.findall(content))
            node_simple_codes = set(SIMPLE_OPERATION_CODE_PATTERN.findall(content))
            node_op_codes.update(node_simple_codes)

            # Check direction match from metadata
            node_direction = metadata.get("operation_direction", "")

            # Boost if direction matches
            direction_match = False
            if direction == OperationDirection.INCOMING and node_direction == "incoming":
                direction_match = True
            elif direction == OperationDirection.OUTGOING and node_direction == "outgoing":
                direction_match = True

            # Boost if codes match
            code_match = bool(node_op_codes.intersection(boost_codes))

            if direction_match or code_match:
                boost = self._boost_factor
                if direction_match and code_match:
                    boost = self._boost_factor * 1.5  # Extra boost for both matches
                node.score = (node.score or 0.0) + boost

            # Penalize wrong direction (important for 35 vs 37 distinction)
            elif node_direction and node_direction != direction.value:
                # Only penalize if we're confident about direction
                penalty = self._boost_factor * 0.3
                node.score = max(0.0, (node.score or 0.0) - penalty)

        return nodes
