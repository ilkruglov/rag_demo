"""BM25 retriever with Russian lemmatization for precise keyword matching."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, List, Optional

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

try:
    import pymorphy3
    _MORPH = pymorphy3.MorphAnalyzer()
except ImportError:
    _MORPH = None

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

LOGGER = logging.getLogger(__name__)

# Стоп-слова для русского языка (расширенный список)
RUSSIAN_STOPWORDS = frozenset([
    'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все',
    'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по',
    'только', 'её', 'ее', 'мне', 'было', 'вот', 'от', 'меня', 'ещё', 'еще', 'нет',
    'о', 'из', 'ему', 'теперь', 'когда', 'уже', 'вам', 'ни', 'быть', 'был', 'него',
    'до', 'вас', 'нибудь', 'опять', 'уж', 'вам', 'ведь', 'там', 'потом', 'себя',
    'ничего', 'ей', 'может', 'они', 'тут', 'где', 'есть', 'надо', 'ней', 'для',
    'мы', 'тебя', 'их', 'чем', 'была', 'сам', 'чтоб', 'без', 'будто', 'чего',
    'раз', 'тоже', 'себе', 'под', 'будет', 'ж', 'тогда', 'кто', 'этот', 'того',
    'потому', 'этого', 'какой', 'совсем', 'ним', 'здесь', 'этом', 'один', 'почти',
    'мой', 'тем', 'чтобы', 'нее', 'нею', 'сейчас', 'были', 'куда', 'зачем', 'всех',
    'никогда', 'можно', 'при', 'наконец', 'два', 'об', 'другой', 'хоть', 'после',
    'над', 'больше', 'тот', 'через', 'эти', 'нас', 'про', 'всего', 'них', 'какая',
    'много', 'разве', 'три', 'эту', 'моя', 'впрочем', 'хорошо', 'свою', 'этой',
    'перед', 'иногда', 'лучше', 'чуть', 'том', 'нельзя', 'такой', 'им', 'более',
    'всегда', 'конечно', 'всю', 'между', 'которые', 'который', 'которая', 'которое',
    # Добавляем юридические стоп-слова
    'также', 'либо', 'или', 'если', 'иной', 'иным', 'иных', 'данный', 'данного',
    'настоящий', 'настоящего', 'настоящем', 'указанный', 'указанного', 'соответствии',
])


def lemmatize_text(text: str) -> List[str]:
    """Tokenize and lemmatize Russian text."""
    if _MORPH is None:
        # Fallback: простая токенизация без лемматизации
        tokens = re.findall(r'[а-яёa-z0-9]+', text.lower())
        return [t for t in tokens if t not in RUSSIAN_STOPWORDS and len(t) > 2]

    tokens = re.findall(r'[а-яёa-z0-9]+', text.lower())
    lemmas = []
    for token in tokens:
        if token in RUSSIAN_STOPWORDS or len(token) <= 2:
            continue
        # Лемматизируем только кириллицу
        if re.match(r'^[а-яё]+$', token):
            parsed = _MORPH.parse(token)
            if parsed:
                lemma = parsed[0].normal_form
                lemmas.append(lemma)
            else:
                lemmas.append(token)
        else:
            lemmas.append(token)
    return lemmas


class BM25Index:
    """BM25 index with Russian lemmatization support."""

    def __init__(self, nodes: Optional[List[TextNode]] = None):
        self.nodes: List[TextNode] = []
        self.node_ids: List[str] = []
        self.corpus: List[List[str]] = []
        self.bm25: Optional[Any] = None

        if nodes:
            self.build(nodes)

    def build(self, nodes: List[TextNode]) -> None:
        """Build BM25 index from nodes."""
        if BM25Okapi is None:
            LOGGER.warning("rank-bm25 not installed, BM25 index will not be built")
            return

        self.nodes = list(nodes)
        self.node_ids = [node.node_id for node in nodes]

        # Создаём корпус из лемматизированных текстов
        self.corpus = []
        for node in nodes:
            text = node.get_content()
            # Добавляем метаданные в текст для улучшения поиска
            metadata = node.metadata or {}
            doc_title = metadata.get('document_title', '')
            section = metadata.get('section_title', '')
            enriched_text = f"{doc_title} {section} {text}"
            lemmas = lemmatize_text(enriched_text)
            self.corpus.append(lemmas)

        # Строим BM25 индекс
        self.bm25 = BM25Okapi(self.corpus)
        LOGGER.info("Built BM25 index with %d documents", len(self.corpus))

    def search(self, query: str, top_k: int = 10) -> List[tuple[TextNode, float]]:
        """Search the index and return nodes with scores."""
        if self.bm25 is None or not self.nodes:
            return []

        query_lemmas = lemmatize_text(query)
        if not query_lemmas:
            return []

        scores = self.bm25.get_scores(query_lemmas)

        # Получаем top_k результатов
        scored_indices = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        results = []
        for idx, score in scored_indices:
            if score > 0:  # Только положительные scores
                results.append((self.nodes[idx], float(score)))

        return results

    def save(self, path: Path) -> None:
        """Save index to disk."""
        data = {
            'node_ids': self.node_ids,
            'corpus': self.corpus,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self, path: Path, nodes: List[TextNode]) -> bool:
        """Load index from disk. Returns True if successful."""
        if BM25Okapi is None:
            return False

        if not path.exists():
            return False

        try:
            with path.open('r', encoding='utf-8') as f:
                data = json.load(f)

            saved_ids = set(data.get('node_ids', []))
            current_ids = {node.node_id for node in nodes}

            # Проверяем, что nodes совпадают
            if saved_ids != current_ids:
                LOGGER.info("BM25 index outdated, will rebuild")
                return False

            self.corpus = data['corpus']
            self.node_ids = data['node_ids']

            # Восстанавливаем порядок nodes
            id_to_node = {n.node_id: n for n in nodes}
            self.nodes = [id_to_node[nid] for nid in self.node_ids]

            self.bm25 = BM25Okapi(self.corpus)
            LOGGER.info("Loaded BM25 index with %d documents", len(self.corpus))
            return True
        except Exception as e:
            LOGGER.warning("Failed to load BM25 index: %s", e)
            return False


class BM25Retriever(BaseRetriever):
    """LlamaIndex-compatible BM25 retriever with Russian lemmatization."""

    def __init__(
        self,
        index: BM25Index,
        similarity_top_k: int = 10,
        callback_manager: Optional[CallbackManager] = None,
    ):
        self._index = index
        self._similarity_top_k = similarity_top_k
        super().__init__(callback_manager=callback_manager)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes for a query."""
        query_str = query_bundle.query_str
        results = self._index.search(query_str, top_k=self._similarity_top_k)

        return [
            NodeWithScore(node=node, score=score)
            for node, score in results
        ]


# Глобальный кеш для BM25 индекса
_BM25_INDEX_CACHE: Optional[BM25Index] = None
_BM25_INDEX_PATH: Optional[str] = None


def get_bm25_index(storage_dir: str, nodes: List[TextNode]) -> BM25Index:
    """Get or create BM25 index with caching."""
    global _BM25_INDEX_CACHE, _BM25_INDEX_PATH

    index_path = Path(storage_dir) / "bm25_index.json"

    if _BM25_INDEX_CACHE is not None and _BM25_INDEX_PATH == str(index_path):
        return _BM25_INDEX_CACHE

    index = BM25Index()

    # Пробуем загрузить существующий индекс
    if not index.load(index_path, nodes):
        # Строим новый индекс
        index.build(nodes)
        index.save(index_path)

    _BM25_INDEX_CACHE = index
    _BM25_INDEX_PATH = str(index_path)

    return index


def clear_bm25_cache() -> None:
    """Clear the BM25 index cache."""
    global _BM25_INDEX_CACHE, _BM25_INDEX_PATH
    _BM25_INDEX_CACHE = None
    _BM25_INDEX_PATH = None
