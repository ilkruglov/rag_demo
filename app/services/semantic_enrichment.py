"""Semantic enrichment for NRD operations vocabulary and query rewriting."""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Set


class OperationDirection(Enum):
    """Direction of operation (for depository operations)."""
    INCOMING = "incoming"  # Прием от депозитария/контрагента
    OUTGOING = "outgoing"  # Передача депозитарию/контрагенту
    INTERNAL = "internal"  # Внутренние операции (перевод между счетами)
    ADMIN = "admin"  # Административные операции
    BLOCK = "block"  # Блокировка/арест


@dataclass
class OperationInfo:
    """Information about a depository operation."""
    code: str
    name: str
    direction: OperationDirection
    keywords: List[str]
    forms: List[str]
    description: str


# Полный словарь операций НРД с семантическими описаниями
# Извлечено из документов НРД (Порядок, Приложения)
NRD_OPERATIONS = {
    # ===== АДМИНИСТРАТИВНЫЕ ОПЕРАЦИИ С АНКЕТАМИ (05-07) =====
    "05": OperationInfo(
        code="05",
        name="Регистрация/изменение анкеты юридического лица",
        direction=OperationDirection.ADMIN,
        keywords=["анкета", "юридическое лицо", "регистрация", "изменение анкеты"],
        forms=["AF005"],
        description="Регистрация или изменение анкеты юридического лица",
    ),
    "06": OperationInfo(
        code="06",
        name="Регистрация/изменение анкеты физического лица",
        direction=OperationDirection.ADMIN,
        keywords=["анкета", "физическое лицо", "регистрация", "изменение анкеты"],
        forms=["AF006"],
        description="Регистрация или изменение анкеты физического лица",
    ),
    "07": OperationInfo(
        code="07",
        name="Изменение банковских реквизитов",
        direction=OperationDirection.ADMIN,
        keywords=["банковские реквизиты", "реквизиты", "изменение реквизитов"],
        forms=["AF005"],
        description="Изменение банковских реквизитов в анкете",
    ),

    # ===== ОПЕРАЦИИ С РЕЕСТРОМ (21-22) =====
    "21": OperationInfo(
        code="21",
        name="Зачисление из реестра",
        direction=OperationDirection.INCOMING,
        keywords=["из реестра", "от регистратора", "зачисление из реестра", "реестр акционеров"],
        forms=["MF021", "AM021"],
        description="Зачисление ценных бумаг из реестра акционеров",
    ),
    "22": OperationInfo(
        code="22",
        name="Списание в реестр",
        direction=OperationDirection.OUTGOING,
        keywords=["в реестр", "регистратору", "списание в реестр", "реестр акционеров"],
        forms=["MF022"],
        description="Списание ценных бумаг в реестр акционеров",
    ),

    # ===== ОПЕРАЦИИ МЕЖДУ ДЕПОЗИТАРИЯМИ (26, 35-37) - КЛЮЧЕВЫЕ =====
    "26": OperationInfo(
        code="26",
        name="Перевод между депонентами",
        direction=OperationDirection.INTERNAL,
        keywords=["перевод", "между депонентами", "встречные поручения"],
        forms=["MF026", "MS026"],
        description="Перевод ценных бумаг между депонентами на основании встречных поручений",
    ),
    "26/1": OperationInfo(
        code="26/1",
        name="Перевод между депонентами (заблокировано Банком России)",
        direction=OperationDirection.INTERNAL,
        keywords=["перевод", "банк россии", "заблокировано"],
        forms=["MF026"],
        description="Перевод с раздела 'Заблокировано Банком России'",
    ),

    # ПРИЕМ ЦЕННЫХ БУМАГ (INCOMING) - код 35 и варианты
    "35": OperationInfo(
        code="35",
        name="Прием ценных бумаг от другого депозитария",
        direction=OperationDirection.INCOMING,
        keywords=[
            "принять", "прием", "приём", "получить", "получение", "зачислить", "зачисление",
            "от депозитария", "от другого депозитария", "из депозитария",
            "входящий", "входящее", "поступление", "поступить",
            "перевести к нам", "перевод к нам", "на наш счет", "на счет в нрд",
            "на хранение", "на учет", "mf035",
        ],
        forms=["MF035", "MS035", "GS035"],
        description="Операция зачисления ценных бумаг, поступающих из другого депозитария в НРД",
    ),
    "35А": OperationInfo(
        code="35А",
        name="Прием ценных бумаг от депозитария (альтернативная форма)",
        direction=OperationDirection.INCOMING,
        keywords=["принять", "прием", "от депозитария", "альтернативный"],
        forms=["MF035A"],
        description="Альтернативная форма операции приема ценных бумаг от другого депозитария",
    ),
    "35/2": OperationInfo(
        code="35/2",
        name="Прием ценных бумаг через ЦСУ ИП ПИФ",
        direction=OperationDirection.INCOMING,
        keywords=["прием", "цсу", "ип пиф", "пиф", "паевой фонд"],
        forms=["MF035"],
        description="Прием ценных бумаг через ЦСУ ИП ПИФ",
    ),
    "35/3": OperationInfo(
        code="35/3",
        name="Прием ценных бумаг через ЦСУ ИП ПИФ (альтернативный)",
        direction=OperationDirection.INCOMING,
        keywords=["прием", "цсу", "ип пиф"],
        forms=["MF035"],
        description="Альтернативный прием через ЦСУ ИП ПИФ",
    ),
    "10/35": OperationInfo(
        code="10/35",
        name="Прием ценных бумаг на хранение (служебная)",
        direction=OperationDirection.INCOMING,
        keywords=["прием", "хранение", "учет", "служебная"],
        forms=["MF035"],
        description="Прием ценных бумаг на хранение и/или учет (служебная операция)",
    ),

    # ПЕРЕДАЧА/СНЯТИЕ ЦЕННЫХ БУМАГ (OUTGOING) - коды 36, 37
    "36": OperationInfo(
        code="36",
        name="Снятие ценных бумаг с хранения и/или учета",
        direction=OperationDirection.OUTGOING,
        keywords=[
            "снятие", "снять", "с хранения", "с учета",
            "вывод", "списание",
        ],
        forms=["MF036", "MS036", "GS036"],
        description="Снятие ценных бумаг с хранения и/или учета в НРД",
    ),
    "36/2": OperationInfo(
        code="36/2",
        name="Снятие при погашении через ЦСУ ИП ПИФ",
        direction=OperationDirection.OUTGOING,
        keywords=["снятие", "погашение", "цсу", "ип пиф"],
        forms=["MF036"],
        description="Снятие при погашении через ЦСУ ИП ПИФ",
    ),
    "36/3": OperationInfo(
        code="36/3",
        name="Снятие при погашении через ЦСУ ИП ПИФ (альт.)",
        direction=OperationDirection.OUTGOING,
        keywords=["снятие", "погашение", "цсу", "ип пиф"],
        forms=["MF036"],
        description="Альтернативное снятие при погашении через ЦСУ ИП ПИФ",
    ),
    "36/35": OperationInfo(
        code="36/35",
        name="Обмен инвестиционных паев",
        direction=OperationDirection.OUTGOING,
        keywords=["обмен", "инвестиционные паи", "пиф"],
        forms=["MF036"],
        description="Снятие при обмене инвестиционных паев",
    ),
    "10/36": OperationInfo(
        code="10/36",
        name="Снятие ценных бумаг иностранного юридического лица",
        direction=OperationDirection.OUTGOING,
        keywords=["снятие", "иностранное", "юридическое лицо"],
        forms=["MF036"],
        description="Снятие ценных бумаг иностранного юридического лица",
    ),

    "37": OperationInfo(
        code="37",
        name="Передача ценных бумаг другому депозитарию",
        direction=OperationDirection.OUTGOING,
        keywords=[
            "передать", "передача", "отправить", "отправка", "списать", "списание",
            "другому депозитарию", "в депозитарий", "в другой депозитарий",
            "исходящий", "исходящее", "вывод", "вывести",
            "перевести из нрд", "перевод из нрд", "со счета в нрд",
            "mf037",
        ],
        forms=["MF037", "MS037", "GS037"],
        description="Операция списания ценных бумаг для передачи в другой депозитарий",
    ),

    # ===== ВНУТРЕННИЕ ПЕРЕВОДЫ (16/X, 10/50) =====
    "16": OperationInfo(
        code="16",
        name="Перевод ценных бумаг",
        direction=OperationDirection.INTERNAL,
        keywords=["перевод", "внутренний"],
        forms=["MF016"],
        description="Базовая операция перевода ценных бумаг",
    ),
    "16/1": OperationInfo(
        code="16/1",
        name="Перевод с раздела НИФИ EUROCLEAR BANK",
        direction=OperationDirection.INTERNAL,
        keywords=["перевод", "нифи", "euroclear", "индивидуальный счет"],
        forms=["MF016"],
        description="Перевод с раздела НИФИ на индивидуальном счете в EUROCLEAR BANK",
    ),
    "16/2": OperationInfo(
        code="16/2",
        name="Перевод между разделами счета депо",
        direction=OperationDirection.INTERNAL,
        keywords=[
            "перевод", "между разделами", "между счетами", "внутренний",
            "из раздела в раздел", "между субсчетами",
        ],
        forms=["MF162", "MS162"],
        description="Внутренний перевод ценных бумаг между разделами одного счета депо",
    ),
    "16/3": OperationInfo(
        code="16/3",
        name="Перевод с контролем расчетов по денежным средствам",
        direction=OperationDirection.INTERNAL,
        keywords=["перевод", "контроль расчетов", "денежные средства", "dvp"],
        forms=["MF016"],
        description="Перевод ценных бумаг с контролем расчетов по денежным средствам",
    ),
    "16/4": OperationInfo(
        code="16/4",
        name="Перевод на раздел Выкуплено Казначейского счета",
        direction=OperationDirection.INTERNAL,
        keywords=["перевод", "выкуп", "казначейский", "эмитент"],
        forms=["MF016"],
        description="Перевод на раздел 'Выкуплено' Казначейского счета депо эмитента",
    ),
    "16/5": OperationInfo(
        code="16/5",
        name="Перевод по Поручению оператора Финансовой платформы",
        direction=OperationDirection.INTERNAL,
        keywords=["перевод", "финансовая платформа", "оператор"],
        forms=["MF016"],
        description="Перевод по Поручению оператора Финансовой платформы",
    ),
    "10/50": OperationInfo(
        code="10/50",
        name="Перевод с изменением места хранения",
        direction=OperationDirection.INTERNAL,
        keywords=["перевод", "место хранения", "хранение", "смена хранения"],
        forms=["MF1050"],
        description="Перевод с изменением места хранения ценных бумаг",
    ),
    "10/51": OperationInfo(
        code="10/51",
        name="Снятие ценных бумаг с хранения при ликвидации",
        direction=OperationDirection.OUTGOING,
        keywords=["снятие", "ликвидация", "реестр", "депозитарий"],
        forms=["MF1051"],
        description="Снятие ценных бумаг с хранения и/или учета при ликвидации",
    ),
    "19/0": OperationInfo(
        code="19/0",
        name="Установка последовательности исполнения Поручений",
        direction=OperationDirection.ADMIN,
        keywords=["последовательность", "порядок исполнения", "поручения"],
        forms=["MF190"],
        description="Установка единой последовательности исполнения Поручений",
    ),
    "20/2": OperationInfo(
        code="20/2",
        name="Перевод на раздел Счета депо владельца",
        direction=OperationDirection.INTERNAL,
        keywords=["перевод", "раздел", "владелец", "доверительный управляющий"],
        forms=["MF020"],
        description="Перевод на раздел Счета депо владельца или доверительного управляющего",
    ),

    # ===== ИНФОРМАЦИОННЫЕ ОПЕРАЦИИ (40, 43, 4С) =====
    "40": OperationInfo(
        code="40",
        name="Информационный запрос об остатках",
        direction=OperationDirection.ADMIN,
        keywords=["информационный запрос", "остатки", "справка", "отчет"],
        forms=["IF040"],
        description="Информационный запрос об остатках ценных бумаг на Счетах депо",
    ),
    "43": OperationInfo(
        code="43",
        name="Информационный запрос о ценных бумагах на разделе",
        direction=OperationDirection.ADMIN,
        keywords=["информационный запрос", "раздел", "ценные бумаги"],
        forms=["IF043"],
        description="Информационный запрос о ценных бумагах на конкретном разделе",
    ),
    "4С": OperationInfo(
        code="4С",
        name="Повторное предоставление отчетных документов",
        direction=OperationDirection.ADMIN,
        keywords=["повторное", "отчетные документы", "дубликат"],
        forms=["IF04C"],
        description="Повторное предоставление отчетных документов",
    ),

    # ===== КОРПОРАТИВНЫЕ ДЕЙСТВИЯ (50, 53, 68, 70) =====
    "50": OperationInfo(
        code="50",
        name="Конвертация ценных бумаг",
        direction=OperationDirection.ADMIN,
        keywords=["конвертация", "конвертировать", "обмен", "преобразование"],
        forms=["MF050"],
        description="Операция конвертации ценных бумаг",
    ),
    "53": OperationInfo(
        code="53",
        name="Сверка поручения",
        direction=OperationDirection.ADMIN,
        keywords=["сверка", "для сверки", "подтверждение"],
        forms=["MF530"],
        description="Поручение со статусом 'Для сверки'",
    ),
    "68": OperationInfo(
        code="68",
        name="Участие в Корпоративном действии",
        direction=OperationDirection.ADMIN,
        keywords=["корпоративное действие", "ка", "участие", "инструкция"],
        forms=["GF068"],
        description="Поручение (инструкция) на участие в Корпоративном действии",
    ),
    "70": OperationInfo(
        code="70",
        name="Отмена ошибочной операции",
        direction=OperationDirection.ADMIN,
        keywords=["отмена", "ошибка", "ошибочная операция"],
        forms=["GF070"],
        description="Отмена ошибочной операции",
    ),

    # ===== БЛОКИРОВКИ И АРЕСТЫ (80/X, 81/X) =====
    "80/1": OperationInfo(
        code="80/1",
        name="Блокировка по аресту",
        direction=OperationDirection.BLOCK,
        keywords=["блокировка", "арест", "заблокировано по аресту"],
        forms=["MF801"],
        description="Операция ареста путем перевода на раздел 'Блокировано по аресту'",
    ),
    "80/3": OperationInfo(
        code="80/3",
        name="Наложение последующего ареста",
        direction=OperationDirection.BLOCK,
        keywords=["последующий арест", "наложение ареста"],
        forms=["MF803"],
        description="Наложение последующего ареста на ценные бумаги",
    ),
    "80/5": OperationInfo(
        code="80/5",
        name="Перевод арестованных ценных бумаг по итогам клиринга",
        direction=OperationDirection.BLOCK,
        keywords=["арест", "клиринг", "перевод арестованных"],
        forms=["MF805"],
        description="Перевод арестованных ценных бумаг в раздел 'Блокировано по аресту'",
    ),
    "81/1": OperationInfo(
        code="81/1",
        name="Снятие ареста с ценных бумаг",
        direction=OperationDirection.BLOCK,
        keywords=["снятие ареста", "разблокировка", "распределение"],
        forms=["MF811"],
        description="Снятие ареста или распределение ценных бумаг",
    ),
    "81/3": OperationInfo(
        code="81/3",
        name="Списание с раздела по Поручению (арест)",
        direction=OperationDirection.BLOCK,
        keywords=["списание", "арест", "поручение депонента"],
        forms=["MF813"],
        description="Списание ценных бумаг с раздела арестованных по Поручению Депонента",
    ),
    "10/82": OperationInfo(
        code="10/82",
        name="Ограничение по записи",
        direction=OperationDirection.BLOCK,
        keywords=["ограничение", "запись", "установление ограничения"],
        forms=["MF1082"],
        description="Установление ограничения путем предоставления отчета",
    ),

    # ===== ОПЕРАЦИИ СО СЧЕТАМИ И РАЗДЕЛАМИ (90-97) =====
    "90": OperationInfo(
        code="90",
        name="Открытие раздела счета депо",
        direction=OperationDirection.ADMIN,
        keywords=["открытие раздела", "открыть раздел", "новый раздел"],
        forms=["AF090", "AS090"],
        description="Открытие раздела различных типов на Счете депо",
    ),
    "91": OperationInfo(
        code="91",
        name="Закрытие раздела счета депо",
        direction=OperationDirection.ADMIN,
        keywords=["закрытие раздела", "закрыть раздел"],
        forms=["AF090"],
        description="Закрытие раздела счета депо",
    ),
    "92": OperationInfo(
        code="92",
        name="Внесение изменений в анкету ценной бумаги",
        direction=OperationDirection.ADMIN,
        keywords=["анкета ценной бумаги", "изменение анкеты", "эмитент"],
        forms=["AF092"],
        description="Внесение изменений в анкету ценной бумаги",
    ),
    "93": OperationInfo(
        code="93",
        name="Внесение изменений в анкету Счета депо",
        direction=OperationDirection.ADMIN,
        keywords=["анкета счета депо", "изменение анкеты", "доверительный управляющий"],
        forms=["AF093", "AS093"],
        description="Внесение изменений в анкету Счета депо доверительного управляющего",
    ),
    "94": OperationInfo(
        code="94",
        name="Поручение по форме AF094",
        direction=OperationDirection.ADMIN,
        keywords=["поручение", "af094"],
        forms=["AF094"],
        description="Специальное поручение по форме AF094",
    ),
    "97": OperationInfo(
        code="97",
        name="Изменение порядка направления отчетных документов",
        direction=OperationDirection.ADMIN,
        keywords=["отчетные документы", "порядок направления", "изменение порядка"],
        forms=["AF097"],
        description="Изменение стандартного порядка направления отчетных и информационных документов",
    ),

    # ===== ТОРГОВЫЕ ОПЕРАЦИИ (84) =====
    "84": OperationInfo(
        code="84",
        name="Распоряжение на Торговом счете депо",
        direction=OperationDirection.ADMIN,
        keywords=["торговый счет", "клиринг", "распоряжение"],
        forms=["MF084"],
        description="Распоряжение ценными бумагами на Торговом счете депо",
    ),
}


# Синонимы направлений для поиска в запросах
INCOMING_KEYWORDS = frozenset([
    "принять", "прием", "приём", "получить", "получение", "зачислить", "зачисление",
    "входящий", "входящее", "поступление", "поступить", "от депозитария",
    "из депозитария", "от другого", "перевод к нам", "на счет",
    "на хранение", "на учет", "mf035", "прием бумаг",
])

OUTGOING_KEYWORDS = frozenset([
    "передать", "передача", "отправить", "отправка", "списать", "списание",
    "исходящий", "исходящее", "вывод", "вывести", "в депозитарий",
    "другому депозитарию", "в другой", "перевод из нрд", "со счета",
    "снятие", "снять", "mf037", "отдать", "перевести другому",
])


def detect_direction_from_query(query: str) -> Optional[OperationDirection]:
    """Detect operation direction from user query."""
    query_lower = query.lower()

    incoming_count = sum(1 for kw in INCOMING_KEYWORDS if kw in query_lower)
    outgoing_count = sum(1 for kw in OUTGOING_KEYWORDS if kw in query_lower)

    if incoming_count > outgoing_count and incoming_count > 0:
        return OperationDirection.INCOMING
    elif outgoing_count > incoming_count and outgoing_count > 0:
        return OperationDirection.OUTGOING

    return None


def find_relevant_operations(query: str) -> List[OperationInfo]:
    """Find operations that match the query semantically."""
    query_lower = query.lower()
    # Tokenize query into words for whole-word matching
    query_words = set(re.findall(r'\b[\w-]+\b', query_lower))
    matches = []

    for code, info in NRD_OPERATIONS.items():
        keyword_matches = 0
        for kw in info.keywords:
            kw_lower = kw.lower()
            # For short keywords (<=3 chars), require whole word match
            if len(kw_lower) <= 3:
                if kw_lower in query_words:
                    keyword_matches += 1
            else:
                # For longer keywords, substring match is OK
                if kw_lower in query_lower:
                    keyword_matches += 1
        if keyword_matches > 0:
            matches.append((keyword_matches, info))

    # Sort by number of matches (most relevant first)
    matches.sort(key=lambda x: x[0], reverse=True)
    return [info for _, info in matches]


def get_operations_by_direction(direction: OperationDirection) -> List[OperationInfo]:
    """Get all operations with a given direction."""
    return [info for info in NRD_OPERATIONS.values() if info.direction == direction]


def expand_query_with_codes(query: str) -> str:
    """Expand query with relevant operation codes and forms.

    This is the key function for improving retrieval:
    - Detects user intent (incoming/outgoing)
    - Adds relevant operation codes to the query
    - This helps both BM25 (keyword match) and embedding similarity
    """
    expansions = []

    # Detect direction
    direction = detect_direction_from_query(query)
    if direction:
        ops = get_operations_by_direction(direction)
        for op in ops[:3]:  # Top 3 most relevant by direction
            expansions.append(f"код операции {op.code}")
            if op.forms:
                expansions.append(f"форма {op.forms[0]}")

    # Find operations by keywords
    relevant_ops = find_relevant_operations(query)
    for op in relevant_ops[:4]:  # Top 4 by keywords
        if f"код операции {op.code}" not in expansions:
            expansions.append(f"код операции {op.code}")

    if not expansions:
        return query

    # Add expansions to query (for hybrid retrieval)
    expansion_text = " ".join(expansions)
    return f"{query} [{expansion_text}]"


def get_boost_codes_for_query(query: str) -> Set[str]:
    """Get operation codes that should be boosted for this query."""
    codes = set()

    # By direction
    direction = detect_direction_from_query(query)
    if direction:
        for op in get_operations_by_direction(direction):
            codes.add(op.code)

    # By keywords
    for op in find_relevant_operations(query):
        codes.add(op.code)

    return codes


def get_form_codes_for_operations(operation_codes: Set[str]) -> Set[str]:
    """Get form codes associated with given operation codes."""
    forms = set()
    for code in operation_codes:
        if code in NRD_OPERATIONS:
            forms.update(NRD_OPERATIONS[code].forms)
    return forms


# Semantic enrichment for chunks during indexing
def enrich_chunk_with_semantic_info(text: str, metadata: dict) -> tuple[str, dict]:
    """Add semantic enrichment to a chunk based on detected operations.

    Used during indexing to add semantic tags that improve retrieval.
    """
    enriched_metadata = dict(metadata)
    enriched_parts = []

    # Check for operation codes in text
    operation_codes = metadata.get("operation_codes", [])

    # Detect direction based on codes
    directions = set()
    for code in operation_codes:
        if code in NRD_OPERATIONS:
            directions.add(NRD_OPERATIONS[code].direction)

    # Add direction tags
    if OperationDirection.INCOMING in directions:
        enriched_parts.append("[Направление: прием/зачисление]")
        enriched_metadata["operation_direction"] = "incoming"
    elif OperationDirection.OUTGOING in directions:
        enriched_parts.append("[Направление: передача/списание]")
        enriched_metadata["operation_direction"] = "outgoing"
    elif OperationDirection.INTERNAL in directions:
        enriched_parts.append("[Направление: внутренний перевод]")
        enriched_metadata["operation_direction"] = "internal"
    elif OperationDirection.BLOCK in directions:
        enriched_parts.append("[Направление: блокировка/арест]")
        enriched_metadata["operation_direction"] = "block"

    # Add operation descriptions
    for code in operation_codes:
        if code in NRD_OPERATIONS:
            op = NRD_OPERATIONS[code]
            enriched_parts.append(f"[{op.name}]")

    if enriched_parts:
        return "\n".join(enriched_parts) + "\n" + text, enriched_metadata

    return text, enriched_metadata
