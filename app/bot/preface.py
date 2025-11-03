from __future__ import annotations

"""
Назначение: шаги префейса (до динамических вопросов) и LLM‑утилиты извлечения
договорённостей/целей. Все функции принимают `app` (BotApp) и работают через его методы
и состояние.
"""

from typing import List
from aiogram.fsm.context import FSMContext
from aiogram.types import Message
from datetime import datetime
import random
import re

from app.bot.states import Flow
from app.bot.texts import (
    PREFACE_INTRO,
    PREFACE_STEP1,
    PREFACE_STEP2,
    PREFACE_STEP3,
    PREFACE_STEP1_FOCUS_INTRO,
    PREFACE_STEP1_CONFIRM_AND_GOALS,
    PREFACE_DIFFICULTIES_ASK,
    THEME_PROMPT,
    THEME_START_MSG,
)
from app.prompts.core import (
    build_extract_agreements_prompt,
    build_extract_overdue_agreements_prompt,
    build_filter_goals_prompt,
)
from app.bot.utils import compute_context_blob
from app.bot.utils import extract_json_array


async def show_preface(app, message: Message, state: FSMContext) -> None:
    """Показывает Шаг 1/3 в новом формате: интро → договорённости → 3 ключевые и вопросы."""
    chat_id = message.chat.id
    # Сообщение 1: короткое интро
    await message.answer(PREFACE_INTRO)

    # Извлекаем договорённости
    agreements = await llm_extract_agreements(app, chat_id)
    bullets = "\n".join([f"{i+1}. {a}" for i, a in enumerate(agreements)]) or "—"

    # Сообщение 2: список договорённостей под шапкой Шаг 1/3
    await message.answer(f"{PREFACE_STEP1}\n\n{bullets}")

    # Сообщение 3: выбрать ровно 3 ключевых (если меньше — дополним плейсхолдерами)
    top3 = list(agreements[:3])
    while len(top3) < 3:
        top3.append("—")
    top3_bullets = "\n".join([f"{i+1}. {a}" for i, a in enumerate(top3)])
    await message.answer(
        f"{PREFACE_STEP1_FOCUS_INTRO}\n\n{top3_bullets}\n\n{PREFACE_STEP1_CONFIRM_AND_GOALS}"
    )

    # Сохраняем в мету и ждём ответ на подтверждение/цели
    rec = app.meta.get(chat_id, {})
    rec["preface_agreements"] = agreements
    rec["preface_selected_top3"] = top3
    app.meta[chat_id] = rec
    await state.set_state(Flow.preface1)


async def llm_extract_agreements(app, chat_id: int) -> List[str]:
    """Извлекает текущие договорённости из контекста с помощью LLM (строгий JSON)."""
    dialog_blob = compute_context_blob(app, chat_id, limit=30000)
    prompt = build_extract_agreements_prompt(dialog_blob)
    try:
        # Полный контекст и промпт печатаем в консоль для отладки
        print("\n=== AGREEMENTS_CONTEXT START ===\n" + dialog_blob + "\n=== AGREEMENTS_CONTEXT END ===")
        print("\n=== AGREEMENTS_PROMPT START ===\n" + prompt + "\n=== AGREEMENTS_PROMPT END ===")
        app._log(chat_id, f"AGREEMENTS_PROMPT(len)={len(prompt)}")
        raw = await app._invoke_llm(prompt, system=app._system_strict_json)
        app._log(chat_id, f"AGREEMENTS_RAW: {raw}")
        arr = extract_json_array(raw)
        if isinstance(arr, list) and arr:
            return [str(x)[:140].strip() for x in arr][:8]
    except Exception:
        pass
    return []


def mark_preface_shown(app, chat_id: int) -> None:
    """Помечает, что префейс уже показан (чтобы не дублировать при последующих файлах)."""
    rec = app.meta.get(chat_id, {})
    rec["preface_shown"] = True
    app.meta[chat_id] = rec


async def llm_extract_overdue_agreements(app, chat_id: int) -> List[str]:
    """Ищет договорённости с дедлайнами в прошлом (через LLM)."""
    dialog_blob = compute_context_blob(app, chat_id, limit=30000)
    prompt = build_extract_overdue_agreements_prompt(dialog_blob, datetime.now().date().isoformat())
    try:
        # Печать полного контекста и промпта в терминал
        print("\n=== OVERDUE_CONTEXT START ===\n" + dialog_blob + "\n=== OVERDUE_CONTEXT END ===")
        print("\n=== OVERDUE_PROMPT START ===\n" + prompt + "\n=== OVERDUE_PROMPT END ===")
        app._log(chat_id, f"OVERDUE_PROMPT(len)={len(prompt)}")
        raw = await app._invoke_llm(prompt, system=app._system_strict_json)
        app._log(chat_id, f"OVERDUE_RAW: {raw}")
        arr = extract_json_array(raw)
        if isinstance(arr, list):
            return [str(x)[:140].strip() for x in arr][:8]
    except Exception:
        pass
    return []


async def on_preface_step1_answer(app, message: Message, state: FSMContext) -> None:
    """Получаем ответ на подтверждение/цели и задаём вопрос о сложностях (Сообщение 4)."""
    chat_id = message.chat.id
    rec = app.meta.get(chat_id, {})
    rec["preface_goals_input_text"] = message.text or ""
    app.meta[chat_id] = rec
    # Сообщение 4: сложности клиента
    await message.answer(PREFACE_DIFFICULTIES_ASK)
    await state.set_state(Flow.preface2)


async def ask_goals_prompt(app, message: Message, state: FSMContext) -> None:
    """Задаёт вопрос о целях/беспокойствах (Шаг 2/3)."""
    await message.answer(THEME_PROMPT)
    await state.set_state(Flow.preface2)


async def on_preface_step2_answer(app, message: Message, state: FSMContext) -> None:
    """Фиксируем сложности, затем фильтруем цели из предыдущего ответа и переходим к ранжированию."""
    chat_id = message.chat.id
    rec = app.meta.get(chat_id, {})
    rec["preface_difficulties_text"] = message.text or ""
    app.meta[chat_id] = rec

    # Фильтруем цели через LLM на основе ответа из предыдущего шага
    text = (rec.get("preface_goals_input_text") or "").strip()
    goals: List[str] = []
    try:
        prompt = build_filter_goals_prompt(text)
        raw = await app._invoke_llm(prompt, system=app._system_strict_json)
        import json as _json

        arr = _json.loads(raw)
        if isinstance(arr, list):
            goals = [str(s)[:140] for s in arr][:8]
    except Exception:
        raw_items = [p.strip() for p in re.split(r"[\n,;]+", text) if p.strip()]
        deny = {"пропустить", "skip", "хз", "не знаю", "да", "нет"}
        goals = [s[:140] for s in raw_items if s.lower() not in deny][:8]

    rec["preface_goals"] = goals
    agreements = list(rec.get("preface_agreements", []))
    
    combined: List[str] = []
    combined.extend(goals)
    combined.extend(agreements)
    if not combined:
        combined = ["—"]
    # Сразу переходим к динамическому анализу без ранжирования (пропускаем вопрос о приоритетах)
    app.meta[chat_id] = rec
    await message.answer(THEME_START_MSG)
    await app._ask_theme(message, state)


async def on_preface_step3_answer(app, message: Message, state: FSMContext) -> None:
    """Завершает префейс и запускает динамические вопросы."""
    await message.answer(THEME_START_MSG)
    await app._ask_theme(message, state)


