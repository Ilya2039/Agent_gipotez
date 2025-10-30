from __future__ import annotations

"""
Назначение: UI-вспомогатели (клавиатуры, форматирование сообщений бота).
"""

from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton


def build_theme_keyboard() -> InlineKeyboardMarkup:
    """Клавиатура для шага выбора или пропуска темы."""
    return InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text="Пропустить", callback_data="theme:skip")]]
    )


def build_alt_keyboard(text: str) -> InlineKeyboardMarkup:
    """Клавиатура для кнопки генерации альтернативного раунда."""
    return InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text=text, callback_data="alt:more")]]
    )


def format_question(idx: int, qtext: str, example_prefix: str, example: str | None) -> str:
    """Форматирует текст вопроса с номером и опциональным примером ответа."""
    tail = f"\n\n<i>{example_prefix}</i> {example}" if example else ""
    return f"Вопрос {idx}:\n{qtext}{tail}"


def build_actions_keyboard(btn_correct: str, btn_more: str) -> InlineKeyboardMarkup:
    """Клавиатура с двумя кнопками: корректировка и новые гипотезы (в один ряд)."""
    row = [
        InlineKeyboardButton(text=btn_correct, callback_data="actions:correct"),
        InlineKeyboardButton(text=btn_more, callback_data="actions:more"),
    ]
    return InlineKeyboardMarkup(inline_keyboard=[row])

