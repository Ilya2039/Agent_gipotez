from __future__ import annotations

"""
Назначение: отдельный модуль состояний FSM для избежания циклических импортов.
"""

from aiogram.fsm.state import State, StatesGroup


class Flow(StatesGroup):
    """Состояния диалога: ожидание темы и ожидание ответа на динамический вопрос."""
    waiting_dynamic_answer = State()
    waiting_theme = State()
    preface1 = State()
    preface2 = State()
    preface3 = State()
    waiting_corrections = State()
    waiting_theme_for_corrections = State()


