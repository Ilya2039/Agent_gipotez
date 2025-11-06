from __future__ import annotations

"""
Назначение: бизнес-логика диалога (LLM-вопросы и финализация ответа).
Содержит функции, которые работают с состоянием бота через переданный объект app.
"""

import json
from pathlib import Path
from aiogram.fsm.context import FSMContext

from app.prompts.core import (
    build_generate_discovery_question_prompt,
    build_generate_free_hypothesis_prompt,
    build_generate_alternative_hypothesis_prompt,
    build_generate_meeting_questions_prompt,
    build_refine_hypotheses_prompt,
)
from app.bot.ui import build_alt_keyboard, format_question, build_actions_keyboard
from app.bot.texts import ACTIONS_PROMPT, BTN_CORRECT, BTN_MORE, BTN_AGREE, HELPFUL_TO_KNOW_TITLE
from app.bot.states import Flow
from app.services.examples import generate_short_example
from app.bot.states import Flow


async def ask_next_discovery(app, message, state: FSMContext) -> None:
    """Сгенерировать следующий уточняющий вопрос и отправить пользователю."""
    chat_id = message.chat.id
    asked = [it.get("question", "") for it in (app.dynamic_qa.get(chat_id) or []) if it.get("question")]
    cross_avoid = app.avoid_questions_all.get(chat_id) or []
    qa = app.dynamic_qa.get(chat_id) or []
    qa_json = json.dumps(qa, ensure_ascii=False, indent=2)
    dialog_blob = app._compute_context_blob(chat_id)
    examples_text = Path("data/hypotheses.txt").read_text(encoding="utf-8") if Path("data/hypotheses.txt").exists() else ""
    try:
        unknown_qs = list(app.meta.get(chat_id, {}).get("unknown_qs", []))
    except Exception:
        unknown_qs = []
    prompt = build_generate_discovery_question_prompt(
        qa_json=qa_json,
        dialog_json=dialog_blob,
        examples_text=examples_text,
        avoid=asked + cross_avoid,
        count=1,
        theme=app.dynamic_theme.get(chat_id, ""),
        avoid_unknown=unknown_qs,
    )
    try:
        raw = await app._invoke_llm(prompt, system=app._system_strict_json)
    except Exception:
        raw = ""
    try:
        arr = json.loads(raw)
    except Exception:
        arr = []
    if not isinstance(arr, list) or not arr:
        theme_txt = (app.dynamic_theme.get(chat_id, "") or "деятельности компании").strip()
        qtext = f"Какие ключевые задачи по теме {theme_txt} сейчас наиболее актуальны?"
    else:
        qtext = str(arr[0]).strip()
    if not qtext.endswith("?"):
        qtext = qtext.rstrip(". ") + "?"
    (app.dynamic_qa.setdefault(chat_id, [])).append({"question": qtext, "answer": ""})
    all_list = app.avoid_questions_all.setdefault(chat_id, [])
    if qtext not in all_list:
        all_list.append(qtext)
    try:
        example = await generate_short_example(app._invoke_llm, qtext, qa_json, dialog_blob)
    except Exception:
        example = ""
    idx = (app.dynamic_idx.get(chat_id) or 0) + 1
    app.dynamic_idx[chat_id] = idx
    EXAMPLE_PREFIX = "Пример ответа:"
    await message.answer(format_question(idx, qtext, EXAMPLE_PREFIX, example))
    app._log(chat_id, f"Q: {qtext}")
    await state.set_state(Flow.waiting_dynamic_answer)


async def finalize_dynamic_hypothesis(app, message, state: FSMContext) -> None:
    """Сформировать 3 гипотезы и вопросы к каждой + общий блок unknowns."""
    chat_id = message.chat.id
    qa = app.dynamic_qa.get(chat_id) or []
    qa_json = json.dumps(qa, ensure_ascii=False, indent=2)
    dialog_blob = app._compute_context_blob(chat_id)
    examples_text = Path("data/hypotheses.txt").read_text(encoding="utf-8") if Path("data/hypotheses.txt").exists() else ""
    hypos: list[dict] = []
    main_prompt = build_generate_free_hypothesis_prompt(
        app.dynamic_theme.get(chat_id, ""), qa_json, dialog_blob, examples_text, prefer_non_finance=True
    )
    main_raw = await app._invoke_llm(main_prompt, system=app._system_strict_json)
    try:
        main_obj = json.loads(main_raw)
    except Exception:
        main_obj = {"hypothesis": main_raw.strip(), "reason": ""}
    if (main_obj.get("hypothesis") or "").strip():
        hypos.append(main_obj)
    avoid_titles = (main_obj.get("hypothesis") or "").strip()
    for _ in range(2):
        alt_prompt = build_generate_alternative_hypothesis_prompt(
            qa_json, dialog_blob, examples_text, avoid_hypothesis=avoid_titles, prefer_non_finance=True
        )
        alt_raw = await app._invoke_llm(alt_prompt, system=app._system_strict_json)
        try:
            alt_obj = json.loads(alt_raw)
        except Exception:
            alt_obj = {"hypothesis": alt_raw.strip(), "reason": ""}
        title = (alt_obj.get("hypothesis") or "").strip()
        if title:
            hypos.append(alt_obj)
            avoid_titles = f"{avoid_titles}; {title}" if avoid_titles else title
    items = hypos[:3]
    # подготавливаем хранилище последних сообщений гипотез (для сохранения по кнопке согласия)
    app.last_hypotheses_messages[chat_id] = []
    # Формируем итоговую повестку с заголовком и нумерацией
    header = '<b>Гипотезы для фокусного обсуждения бизнеса клиента:</b>'
    blocks = []
    for idx, item in enumerate(items, start=1):
        title = (item.get("hypothesis") or "").strip()
        reason = (item.get("reason") or "").strip()
        q_prompt = build_generate_meeting_questions_prompt(title, qa_json, dialog_blob, count=3)
        q_raw = await app._invoke_llm(q_prompt, system=app._system_strict_json)
        try:
            q_arr = json.loads(q_raw)
        except Exception:
            q_arr = []
        qs_list = [str(q).strip() for q in (q_arr[:3] if isinstance(q_arr, list) else [])]
        bullets = "\n".join([f"- {q}" for q in qs_list]) if qs_list else "- —"
        reason_txt = f"Причина: {reason}" if reason else ""
        block = f"{idx}. {title}\n{reason_txt}\n{bullets}"
        blocks.append(block)
        app.last_hypotheses_messages[chat_id].append(block)
    await message.answer("Гипотезы для фокусного обсуждения бизнеса клиента:\n\n" + "\n\n".join(blocks))

    # Блок "Будет полезно узнать у клиента" + примеры ответов
    unknown_qs = list(app.meta.get(chat_id, {}).get("unknown_qs", [])) if app.meta.get(chat_id) else []
    if unknown_qs:
        lines = [f"<b>{HELPFUL_TO_KNOW_TITLE}</b>"]
        for q in unknown_qs:
            if not q:
                continue
            try:
                ex = await generate_short_example(app._invoke_llm, q, qa_json, dialog_blob)
            except Exception:
                ex = ""
            lines.append(q + (f"\nПример ответа: {ex}" if ex else ""))
        await message.answer("\n\n".join(lines))

    # Сообщение с действиями
    await message.answer(ACTIONS_PROMPT, reply_markup=build_actions_keyboard(BTN_CORRECT, BTN_MORE, BTN_AGREE))
    if hypos:
        first_title = (hypos[0].get("hypothesis") or "").strip()
        app._log(chat_id, f"FINAL HYPOTHESES: {[ (h.get('hypothesis') or '').strip() for h in hypos[:3] ]}")
        app.last_result[chat_id] = {"hypothesis": first_title}
        # Сохраняем последние 3 гипотезы для корректировки
        app.last_hypotheses[chat_id] = hypos[:3]
    await state.clear()


async def refine_hypotheses(app, message, state: FSMContext, correction_text: str) -> None:
    """Пересобирает 3 гипотезы с учётом замечаний КМ, оставляя без изменений те, что не упомянуты."""
    chat_id = message.chat.id
    last_hypos = app.last_hypotheses.get(chat_id) or []
    if not last_hypos:
        # Если нет предыдущих гипотез — вызываем обычную финализацию
        await finalize_dynamic_hypothesis(app, message, state)
        return
    
    qa = app.dynamic_qa.get(chat_id) or []
    qa_json = json.dumps(qa, ensure_ascii=False, indent=2)
    dialog_blob = app._compute_context_blob(chat_id)
    examples_text = Path("data/hypotheses.txt").read_text(encoding="utf-8") if Path("data/hypotheses.txt").exists() else ""
    
    hypos_json = json.dumps(last_hypos, ensure_ascii=False, indent=2)
    refine_prompt = build_refine_hypotheses_prompt(hypos_json, correction_text, qa_json, dialog_blob, examples_text)
    
    try:
        refine_raw = await app._invoke_llm(refine_prompt, system=app._system_strict_json)
        refined_arr = json.loads(refine_raw)
        if isinstance(refined_arr, list) and len(refined_arr) == 3:
            hypos = refined_arr
        else:
            hypos = last_hypos
    except Exception:
        hypos = last_hypos
    
    # Отправляем 3 гипотезы отдельными сообщениями
    app.last_hypotheses_messages[chat_id] = []
    for idx, item in enumerate(hypos[:3], start=1):
        title = (item.get("hypothesis") or "").strip()
        reason = (item.get("reason") or "").strip()
        # Для каждой гипотезы 2-3 вопроса
        q_prompt = build_generate_meeting_questions_prompt(title, qa_json, dialog_blob, count=3)
        q_raw = await app._invoke_llm(q_prompt, system=app._system_strict_json)
        try:
            q_arr = json.loads(q_raw)
        except Exception:
            q_arr = []
        qs_list = [str(q).strip() for q in (q_arr[:3] if isinstance(q_arr, list) else [])]
        bullets = "\n".join([f"- {q}" for q in qs_list]) if qs_list else "- —"
        reason_txt = f"Причина: {reason}" if reason else ""
        block = f"{idx}. {title}\n{reason_txt}\n{bullets}"
        await message.answer(block)
        app.last_hypotheses_messages[chat_id].append(block)
    
    # Блок "Будет полезно узнать у клиента" + примеры ответов
    unknown_qs = list(app.meta.get(chat_id, {}).get("unknown_qs", [])) if app.meta.get(chat_id) else []
    if unknown_qs:
        lines = [f"<b>{HELPFUL_TO_KNOW_TITLE}</b>"]
        for q in unknown_qs:
            if not q:
                continue
            try:
                ex = await generate_short_example(app._invoke_llm, q, qa_json, dialog_blob)
            except Exception:
                ex = ""
            lines.append(q + (f"\nПример ответа: {ex}" if ex else ""))
        await message.answer("\n\n".join(lines))
    
    # Сообщение с действиями
    await message.answer(ACTIONS_PROMPT, reply_markup=build_actions_keyboard(BTN_CORRECT, BTN_MORE, BTN_AGREE))
    
    # Обновляем last_hypotheses и last_result
    app.last_hypotheses[chat_id] = hypos[:3]
    if hypos:
        first_title = (hypos[0].get("hypothesis") or "").strip()
        app.last_result[chat_id] = {"hypothesis": first_title}
    
    await state.clear()


