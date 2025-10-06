from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton

# Fixed 6 survey questions (do not change wording), grouped by blocks
SURVEY = [
    {
        "key": "revenue_trend",
        "group": "Финансовые ситуации",
        "text": "Динамика выручки компании за последний отчетный период (год/квартал) по сравнению с аналогичным периодом годом ранее?",
        "reason": "Вопрос задан с целью зафиксировать тренд доходов компании для выбора финансовых гипотез.",
        "options": ["Растет", "Падает", "Остается стабильной"],
        "multi": False,
    },
    {
        "key": "profit_trend",
        "group": "Финансовые ситуации",
        "text": "Динамика чистой или операционной прибыли (EBITDA) компании за тот же период?",
        "reason": "Вопрос задан с целью понять динамику рентабельности и проверить согласованность с трендом выручки.",
        "options": [
            "Растет (быстрее выручки)",
            "Растет (медленнее выручки)",
            "Падает (при растущей/стабильной выручке)",
            "Остается стабильной",
            "Не знаю",
        ],
        "multi": False,
    },
    {
        "key": "market_state",
        "group": "Конкурентные ситуации",
        "text": "Как ведет себя рынок, на котором работает клиент, в целом?",
        "reason": "Вопрос задан с целью понять фазу рынка (рост/падение/стагнация) и корректно интерпретировать внутренние тренды.",
        "options": ["Растет", "Падает", "Стагнирует", "Не знаю"],
        "multi": False,
    },
    {
        "key": "ops_pains",
        "group": "Операционные ситуации",
        "text": "С какими из этих операционных вызовов сталкивается компания?",
        "reason": "Вопрос задан с целью выявить операционные ограничения (ликвидность, запасы, текучесть), влияющие на финрезультат.",
        "options": [
            "Падение маржинальности (рентабельности)",
            "Отрицательный денежный поток при наличии прибыли",
            "Проблемы с ликвидностью (нехватка денег на счетах)",
            "Рост дебиторской задолженности",
            "Рост складских запасов/неликвидов",
            "Снижение качества продукции или рост брака",
            "Низкая загрузка производственных мощностей",
            "Высокая текучесть персонала",
            "Длительный вывод новых продуктов на рынок",
            "Зависимость от одного-двух ключевых поставщиков",
            "Сложности с планированием (производства, закупок)",
            "Ничего из перечисленного / Не знаю",
        ],
        "multi": True,
    },
    {
        "key": "share_trend",
        "group": "Конкурентные ситуации",
        "text": "Как меняется доля рынка компании?",
        "reason": "Вопрос задан с целью оценить конкурентоспособность и силу позиционирования относительно рынка.",
        "options": [
            "Растет (за счет конкурентов)",
            "Падает (уступаем конкурентам)",
            "Остается стабильной",
            "Не знаю",
        ],
        "multi": False,
    },
    {
        "key": "competitor_actions",
        "group": "Конкурентные ситуации",
        "text": "Что из перечисленного характерно для действий конкурентов?",
        "reason": "Вопрос задан с целью зафиксировать давление конкурентов и факторы внешнего давления.",
        "options": [
            "Агрессивно снижают цены (демпинг)",
            "Выводят инновационные продукты/услуги",
            "Усиливают маркетинг и рекламу",
            "Уходят с рынка или испытывают трудности",
            "Активно консолидируются (поглощают других)",
            "Заметных изменений нет / Не знаю",
        ],
        "multi": True,
    },
]

SURVEY_PREFIX = "survey"
SURVEY_NEXT = "survey_next"


@dataclass
class SurveySession:
    step: int = 0
    answers: Dict[str, List[str] | str] = field(default_factory=dict)


def _marker_for_option(question_key: str, option: str) -> str:
    text = option.lower()
    if question_key in {"revenue_trend", "share_trend", "market_state"}:
        if "раст" in text:
            return "🟢 "
        if "падает" in text:
            return "🔴 "
        if "стабил" in text or "стагнирует" in text or "не знаю" in text:
            return "⚪️ "
    if question_key == "profit_trend":
        if "растет" in text or "растёт" in text:
            return "🟢 "
        if "падает" in text:
            return "🔴 "
        return "⚪️ "
    if question_key == "ops_pains":
        # Операционные боли — условно отрицательные, кроме варианта ничего/не знаю
        if "ничего" in text or "не знаю" in text:
            return "⚪️ "
        return "🔴 "
    if question_key == "competitor_actions":
        if "уходят с рынка" in text:
            return "🟢 "  # для клиента это скорее позитивно
        if "заметных изменений нет" in text or "не знаю" in text:
            return "⚪️ "
        return "🔴 "
    return ""


def make_keyboard(
    question_key: str,
    options: List[str],
    prefix: str,
    multi: bool,
    selected: List[str] | set[str] | None = None,
) -> InlineKeyboardMarkup:
    selected_set = set(selected or [])
    rows = []
    for i, o in enumerate(options):
        marker = _marker_for_option(question_key, o)
        label_core = ("✅ " + o) if o in selected_set else o
        label = (marker + label_core) if marker and not label_core.startswith("✅ ") else ("✅ " + marker + o) if marker and label_core.startswith("✅ ") else label_core
        rows.append([InlineKeyboardButton(text=label, callback_data=f"{prefix}:{i}")])
    if multi:
        rows.append([InlineKeyboardButton(text="Готово", callback_data=SURVEY_NEXT)])
    return InlineKeyboardMarkup(inline_keyboard=rows)
