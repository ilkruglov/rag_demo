"""Smart table parser for NRD documents with semantic enrichment."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TableType(Enum):
    """Types of tables found in NRD documents."""
    OPERATIONS_LIST = "operations_list"  # Перечень операций с кодами
    REQUIREMENTS = "requirements"  # Реквизиты/условия исполнения операций
    FORM_INSTRUCTIONS = "form_instructions"  # Порядок заполнения форм
    TERMS = "terms"  # Термины и определения
    REFERENCE = "reference"  # Справочники (типы счетов, разделов и т.д.)
    OTHER = "other"


@dataclass
class ParsedChunk:
    """A chunk of text extracted from a table with metadata."""
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    table_type: TableType = TableType.OTHER


# Regex patterns for extracting codes
OPERATION_CODE_PATTERN = re.compile(r'\b(\d{1,2}/\d{1,2})\b')
# Pattern for simple operation codes like 35, 37, 35А (in context)
SIMPLE_OPERATION_CODE_PATTERN = re.compile(r'(?:код(?:\s+операции)?|операци[яию])\s*[-–]?\s*(\d{1,2}[А-Яа-яA-Za-z]?)\b', re.IGNORECASE)
# Pattern for form codes like MF170, AF005 (case-insensitive)
FORM_CODE_PATTERN = re.compile(r'\b([A-Za-z]{2}\d{3})\b')


def extract_operation_codes(text: str) -> list[str]:
    """Extract operation codes like 16/2, 10/50 and simple codes like 35, 37 from text."""
    codes = set(OPERATION_CODE_PATTERN.findall(text))
    codes.update(SIMPLE_OPERATION_CODE_PATTERN.findall(text))
    return list(codes)


def extract_form_codes(text: str) -> list[str]:
    """Extract form codes like MF170, AF005 from text."""
    return list(set(FORM_CODE_PATTERN.findall(text)))


def classify_table(table) -> TableType:
    """Classify table type based on header content."""
    if not table.rows:
        return TableType.OTHER

    # Collect text from first 2 rows
    header_text = ""
    for row in table.rows[:2]:
        for cell in row.cells:
            header_text += " " + cell.text.lower()

    # Classification rules
    if "код операции" in header_text or "формы входящих" in header_text or "формы исходящих" in header_text:
        return TableType.OPERATIONS_LIST

    if "реквизит" in header_text or ("о/н" in header_text and "сверка" in header_text):
        return TableType.REQUIREMENTS

    if "наименование полей" in header_text and "пояснен" in header_text:
        return TableType.FORM_INSTRUCTIONS

    if "термин" in header_text and ("определен" in header_text or "значен" in header_text):
        return TableType.TERMS

    # Reference tables - types of accounts, sections etc.
    if "наименование типа" in header_text or "код типа" in header_text:
        return TableType.REFERENCE

    return TableType.OTHER


def _get_table_header(table) -> str:
    """Get first row as header text."""
    if not table.rows:
        return ""
    return " | ".join(cell.text.strip().replace("\n", " ")[:50] for cell in table.rows[0].cells)


def parse_operations_list_table(table, source_name: str) -> list[ParsedChunk]:
    """Parse operations list table - each row with operation code becomes a chunk."""
    chunks = []
    header = _get_table_header(table)

    for row_idx, row in enumerate(table.rows[1:], 1):  # Skip header row
        cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
        row_text = " | ".join(cells)

        # Check if row has operation code
        op_codes = extract_operation_codes(row_text)
        form_codes = extract_form_codes(row_text)

        if op_codes or form_codes:
            # Build enriched text
            enriched_parts = []
            if op_codes:
                enriched_parts.append(f"[Операции: {', '.join(sorted(op_codes))}]")
            if form_codes:
                enriched_parts.append(f"[Формы: {', '.join(sorted(form_codes))}]")

            text = "\n".join(enriched_parts) + f"\n{header}\n{row_text}"

            chunks.append(ParsedChunk(
                text=text,
                metadata={
                    "operation_codes": op_codes,
                    "form_codes": form_codes,
                    "content_type": "operations_list",
                    "table_row": row_idx,
                },
                table_type=TableType.OPERATIONS_LIST,
            ))

    return chunks


def parse_requirements_table(table, source_name: str) -> list[ParsedChunk]:
    """Parse requirements table - extract operation codes and create enriched chunk."""
    # Collect all text from table
    all_text = []
    all_op_codes = set()
    all_form_codes = set()

    for row in table.rows:
        cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
        row_text = " | ".join(cells)
        all_text.append(row_text)

        # Extract codes
        all_op_codes.update(extract_operation_codes(row_text))
        all_form_codes.update(extract_form_codes(row_text))

    if not all_op_codes:
        # No operation codes found, return as single chunk
        return [ParsedChunk(
            text="\n".join(all_text),
            metadata={"content_type": "requirements"},
            table_type=TableType.REQUIREMENTS,
        )]

    # Build enriched header
    op_codes_list = sorted(all_op_codes)
    form_codes_list = sorted(all_form_codes)

    enriched_header = []
    enriched_header.append(f"[Реквизиты/условия исполнения для операций: {', '.join(op_codes_list)}]")
    if form_codes_list:
        enriched_header.append(f"[Формы: {', '.join(form_codes_list)}]")
    enriched_header.append("[Тип: обязательные поля, условия исполнения сделок]")

    text = "\n".join(enriched_header) + "\n\n" + "\n".join(all_text)

    return [ParsedChunk(
        text=text,
        metadata={
            "operation_codes": op_codes_list,
            "form_codes": form_codes_list,
            "content_type": "operation_requirements",
        },
        table_type=TableType.REQUIREMENTS,
    )]


def parse_form_instructions_table(table, source_name: str) -> list[ParsedChunk]:
    """Parse form filling instructions table."""
    all_text = []
    form_codes = set()

    for row in table.rows:
        cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
        row_text = " | ".join(cells)
        all_text.append(row_text)
        form_codes.update(extract_form_codes(row_text))

    form_codes_list = sorted(form_codes)

    enriched_header = []
    if form_codes_list:
        enriched_header.append(f"[Порядок заполнения форм: {', '.join(form_codes_list)}]")
    enriched_header.append("[Тип: инструкция по заполнению полей]")

    text = "\n".join(enriched_header) + "\n\n" + "\n".join(all_text) if enriched_header else "\n".join(all_text)

    return [ParsedChunk(
        text=text,
        metadata={
            "form_codes": form_codes_list,
            "content_type": "form_instructions",
        },
        table_type=TableType.FORM_INSTRUCTIONS,
    )]


def parse_terms_table(table, source_name: str) -> list[ParsedChunk]:
    """Parse terms/definitions table - each term becomes a chunk."""
    chunks = []
    header = _get_table_header(table)

    for row_idx, row in enumerate(table.rows[1:], 1):  # Skip header
        cells = [cell.text.strip() for cell in row.cells]

        # Usually first cell is term, second is definition
        if len(cells) >= 2 and cells[0] and cells[1]:
            term = cells[0].replace("\n", " ")[:100]
            definition = cells[1].replace("\n", " ")

            text = f"[Термин: {term}]\n{term}: {definition}"

            chunks.append(ParsedChunk(
                text=text,
                metadata={
                    "term": term,
                    "content_type": "term_definition",
                },
                table_type=TableType.TERMS,
            ))

    # If no chunks created, return whole table as one chunk
    if not chunks:
        all_text = "\n".join(
            " | ".join(cell.text.strip().replace("\n", " ") for cell in row.cells)
            for row in table.rows
        )
        chunks.append(ParsedChunk(
            text=all_text,
            metadata={"content_type": "terms"},
            table_type=TableType.TERMS,
        ))

    return chunks


def parse_reference_table(table, source_name: str) -> list[ParsedChunk]:
    """Parse reference table (types of accounts, sections, etc.)."""
    chunks = []
    header = _get_table_header(table)

    # Determine what kind of reference this is from header
    header_lower = header.lower()
    ref_type = "reference"
    if "счет" in header_lower:
        ref_type = "account_types"
    elif "раздел" in header_lower:
        ref_type = "section_types"
    elif "субсчет" in header_lower:
        ref_type = "subaccount_types"

    for row_idx, row in enumerate(table.rows[1:], 1):  # Skip header
        cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
        if not any(cells):
            continue

        row_text = " | ".join(cells)

        # Extract any codes
        op_codes = extract_operation_codes(row_text)

        text = f"[{header}]\n{row_text}"

        metadata = {
            "content_type": ref_type,
            "table_row": row_idx,
        }
        if op_codes:
            metadata["operation_codes"] = op_codes

        chunks.append(ParsedChunk(
            text=text,
            metadata=metadata,
            table_type=TableType.REFERENCE,
        ))

    return chunks


def parse_table(table, source_name: str) -> list[ParsedChunk]:
    """Parse table based on its type and return enriched chunks."""
    table_type = classify_table(table)

    if table_type == TableType.OPERATIONS_LIST:
        return parse_operations_list_table(table, source_name)
    elif table_type == TableType.REQUIREMENTS:
        return parse_requirements_table(table, source_name)
    elif table_type == TableType.FORM_INSTRUCTIONS:
        return parse_form_instructions_table(table, source_name)
    elif table_type == TableType.TERMS:
        return parse_terms_table(table, source_name)
    elif table_type == TableType.REFERENCE:
        return parse_reference_table(table, source_name)
    else:
        # Default: return whole table as one chunk with extracted codes
        all_text = []
        all_op_codes = set()
        all_form_codes = set()

        for row in table.rows:
            cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
            row_text = " | ".join(cells)
            if any(cells):
                all_text.append(row_text)
                all_op_codes.update(extract_operation_codes(row_text))
                all_form_codes.update(extract_form_codes(row_text))

        if not all_text:
            return []

        # Add enriched header if codes found
        enriched_parts = []
        if all_op_codes:
            enriched_parts.append(f"[Операции: {', '.join(sorted(all_op_codes))}]")
        if all_form_codes:
            enriched_parts.append(f"[Формы: {', '.join(sorted(all_form_codes))}]")

        text = "\n".join(enriched_parts + all_text) if enriched_parts else "\n".join(all_text)

        metadata = {"content_type": "table"}
        if all_op_codes:
            metadata["operation_codes"] = sorted(all_op_codes)
        if all_form_codes:
            metadata["form_codes"] = sorted(all_form_codes)

        return [ParsedChunk(text=text, metadata=metadata, table_type=TableType.OTHER)]


def enrich_text_with_codes(text: str) -> tuple[str, dict[str, Any]]:
    """Extract codes from text and add enrichment header."""
    op_codes = extract_operation_codes(text)
    form_codes = extract_form_codes(text)

    metadata = {}
    enriched_parts = []

    if op_codes:
        metadata["operation_codes"] = sorted(op_codes)
        enriched_parts.append(f"[Операции: {', '.join(sorted(op_codes))}]")

    if form_codes:
        metadata["form_codes"] = sorted(form_codes)
        enriched_parts.append(f"[Формы: {', '.join(sorted(form_codes))}]")

    if enriched_parts:
        enriched_text = "\n".join(enriched_parts) + "\n\n" + text
        return enriched_text, metadata

    return text, metadata
