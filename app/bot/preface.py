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
    """Показывает вводную справку и список текущих договорённостей (Шаг 1/3)."""
    await message.answer(PREFACE_INTRO)
    agreements = await llm_extract_agreements(app, message.chat.id)
    bullets = "\n".join([f"{i+1}. {a}" for i, a in enumerate(agreements)]) or "—"
    await message.answer(f"{PREFACE_STEP1}\n\n{bullets}\n\nПодскажите, пожалуйста, актуален ли список выше?")
    rec = app.meta.get(message.chat.id, {})
    rec["preface_agreements"] = agreements
    app.meta[message.chat.id] = rec
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
    """Следующий шаг после подтверждения актуальности — показать просроченные пункты (если есть)."""
    overdue = await llm_extract_overdue_agreements(app, message.chat.id)
    if overdue:
        clean = [re.sub(r"^\s*\d+[\.)]\s*", "", str(t)).strip() for t in overdue]
        items = "\n".join([f"{i+1}. {t}" for i, t in enumerate(clean)])
        rec = app.meta.get(message.chat.id, {})
        rec["awaiting_overdue_answer"] = True
        rec["overdue_list"] = clean  # Сохраняем список просроченных договорённостей
        app.meta[message.chat.id] = rec
        await message.answer(
            f"{PREFACE_STEP2}\n\n{items}\n\nДействительно ли выполнены договорённости, срок по которым уже истёк?\nСталкивался ли клиент со сложностями в рамках их выполнения?"
        )
        await state.set_state(Flow.preface2)
    else:
        await ask_goals_prompt(app, message, state)


async def ask_goals_prompt(app, message: Message, state: FSMContext) -> None:
    """Задаёт вопрос о целях/беспокойствах (Шаг 2/3)."""
    await message.answer(THEME_PROMPT)
    await state.set_state(Flow.preface2)


async def on_preface_step2_answer(app, message: Message, state: FSMContext) -> None:
    """Обрабатывает ответ на шаге 2/3: сначала фиксация ответа о просроченных (если ждали), затем
    фильтрация целей LLM'ом и построение списка для ранжирования.
    """
    chat_id = message.chat.id
    rec = app.meta.get(chat_id, {})
    if rec.get("awaiting_overdue_answer"):
        rec["overdue_answer"] = message.text or ""
        rec["awaiting_overdue_answer"] = False
        app.meta[chat_id] = rec
        await ask_goals_prompt(app, message, state)
        return

    # фильтруем цели через LLM
    text = (message.text or "").strip()
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
    # Исключаем просроченные договорённости из списка
    overdue_list = rec.get("overdue_list", [])
    if overdue_list:
        overdue_set = set([o.lower().strip() for o in overdue_list])
        agreements = [a for a in agreements if a.lower().strip() not in overdue_set]
    
    combined: List[str] = []
    combined.extend(goals)
    combined.extend(agreements)
    if not combined:
        combined = ["—"]
    items = "\n".join([f"{i+1}. {t}" for i, t in enumerate(combined)])
    order = list(range(1, len(combined) + 1))
    random.shuffle(order)
    example = " ".join(str(x) for x in order) if order else "1"
    rank_tail = (
        "\n\nРасставьте их в порядке убывания приоритетов. Это поможет мне лучше понять клиента.\n"
        f"Пример: {example}"
    )
    await message.answer(f"{PREFACE_STEP3}{items}{rank_tail}")
    app.meta[chat_id] = rec
    await state.set_state(Flow.preface3)


async def on_preface_step3_answer(app, message: Message, state: FSMContext) -> None:
    """Завершает префейс и запускает динамические вопросы."""
    await message.answer(THEME_START_MSG)
    await app._ask_theme(message, state)


