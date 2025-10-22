from __future__ import annotations

import asyncio
import json
import re
from typing import Dict, List

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import CommandStart, Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import Message, CallbackQuery, FSInputFile, InlineKeyboardMarkup, InlineKeyboardButton

from app.llm.client import LLMClient
# legacy survey prompts removed
from app.prompts.core import (
    build_decide_next_action_prompt,
    build_generate_free_hypothesis_prompt,
    build_generate_discovery_question_prompt,
    build_generate_meeting_questions_prompt,
    build_generate_alternative_hypothesis_prompt,
)
from app.bot.services.unknowns import classify_unknown
from app.bot.services.examples import generate_short_example
from app.data.hypotheses_repo import (
    get_full_card_formatted_html,
    get_card_by_hypothesis_title_html,
    get_card_description_only_html,
    get_best_full_card_by_cosine,
    format_card_fields_to_html,
)
# survey legacy removed
from dotenv import load_dotenv
import logging
import os
from pathlib import Path
from datetime import datetime
import time
from app.parsers.docx_parser import parse_docx_to_json
from app.data.sub_repo import SubRepo


class Flow(StatesGroup):
    waiting_followup_answer = State()
    waiting_dynamic_answer = State()
    waiting_theme = State()


class BotApp:
    def __init__(self, token: str) -> None:
        self._ensure_logging()
        self.bot = Bot(token=token, default=DefaultBotProperties(parse_mode="HTML"))
        self.dp = Dispatcher()
        self.llm = LLMClient()
        self.surveys: Dict[int, dict] = {}
        self.followup_questions: Dict[int, List[str]] = {}
        self.followup_answers: Dict[int, List[str]] = {}
        self.followup_idx: Dict[int, int] = {}
        self.dialog_docs_json: Dict[int, dict] = {}
        self.dialog_docs_json_list: Dict[int, List[dict]] = {}
        self.hypotheses_pool: Dict[int, List[str]] = {}
        self.meta: Dict[int, Dict[str, str]] = {}
        self.dialog_log_path: Dict[int, Path] = {}
        self._system_strict_json = (
            "Ты отвечаешь строго в формате JSON без лишнего текста. "
            "Не добавляй пояснения, дисклеймеры, списки, markdown. Возвращай только валидный JSON."
        )
        self.subrepo = SubRepo()
        # store selected main hypothesis context per chat for sub-selection step
        self.selected_main: Dict[int, Dict[str, str]] = {}
        # dynamic flow storages
        self.dynamic_qa: Dict[int, List[Dict[str, str]]] = {}
        self.dynamic_idx: Dict[int, int] = {}
        self.waiting_file_first: Dict[int, bool] = {}
        self.last_result: Dict[int, Dict[str, object]] = {}
        # Avoid lists across alternative rounds
        self.avoid_questions_all: Dict[int, List[str]] = {}
        self.avoid_hypotheses: Dict[int, List[str]] = {}
        self.dynamic_theme: Dict[int, str] = {}
        # Track global avoids across alternative rounds
        self.avoid_questions_all: Dict[int, List[str]] = {}
        self.avoid_hypotheses: Dict[int, List[str]] = {}
        # throttle notice for multiple uploads sent in a burst
        self.last_upload_notice_at: Dict[int, float] = {}
        # debounce for file uploads (auto-start after last file)
        self.file_debounce_tasks: Dict[int, asyncio.Task] = {}

    async def start(self) -> None:
        self.dp.message.register(self.cmd_start, CommandStart())
        self.dp.message.register(self.cmd_analyze, Command(commands=["analyze"]))
        self.dp.message.register(self.cmd_skip, Command(commands=["skip"]))
        self.dp.message.register(self.on_document, F.document)
        self.dp.callback_query.register(self.on_alt_more, F.data == "alt:more")
        self.dp.callback_query.register(self.on_theme_skip, F.data == "theme:skip")
        self.dp.message.register(self.on_followup_answer, Flow.waiting_followup_answer)
        self.dp.message.register(self.on_dynamic_answer, Flow.waiting_dynamic_answer)
        self.dp.message.register(self.on_theme_answer, Flow.waiting_theme)
        await self.dp.start_polling(self.bot)

    def _render_q_text(self, q: Dict[str, object]) -> str:
        base = str(q["text"])  # type: ignore
        group = str(q.get("group") or "").strip()
        reason = str(q.get("reason") or "").strip()
        header = f"<b>{group}</b>\n" if group else ""
        reason_line = f"\n<i>Зачем спрашиваем:</i> {reason}" if reason else ""
        if bool(q.get("multi")):
            return header + base + reason_line + "\n<b>(Можно выбрать несколько вариантов; нажмите «Готово».)</b>"
        return header + base + reason_line

    async def cmd_start(self, message: Message, state: FSMContext) -> None:
        # init per-chat dialog log
        self._ensure_dialog_log(message.chat.id, None)
        self._log(message.chat.id, "== START ==")
        chat_id = message.chat.id
        self.waiting_file_first[chat_id] = True
        await message.answer(
            "Сначала пришлите ВСЕ доступные материалы по клиенту (.docx или .json).\n"
            "Я автоматически начну вопросы через пару секунд после последнего файла. Если файлов нет — напишите /skip."
        )
    async def cmd_analyze(self, message: Message, state: FSMContext) -> None:
        chat_id = message.chat.id
        # reset dynamic storages
        self.dynamic_qa.pop(chat_id, None)
        self.dynamic_idx.pop(chat_id, None)
        self.waiting_file_first[chat_id] = True
        await message.answer("Пришлите .docx/.json по клиенту. После загрузки начну до 5 вопросов. Если файлов нет — /skip.")

    async def cmd_skip(self, message: Message, state: FSMContext) -> None:
        chat_id = message.chat.id
        if not self.waiting_file_first.get(chat_id):
            await message.answer("Уже идём по вопросам.")
            return
        self.waiting_file_first[chat_id] = False
        await message.answer("Ок, начнём без файлов. Сначала коротко уточню тему.")
        await self._ask_theme(message, state)

    async def on_dynamic_answer(self, message: Message, state: FSMContext) -> None:
        chat_id = message.chat.id
        qa = self.dynamic_qa.get(chat_id) or []
        if qa and not qa[-1].get("answer"):
            qa[-1]["answer"] = message.text or ""
        else:
            qa.append({"question": "(free)", "answer": message.text or ""})
        self.dynamic_qa[chat_id] = qa
        # Track unknowns: if user says "не знаю" (or похожие), запомним вопрос
        try:
            # LLM-based classification for broader coverage of 'unknown' semantics
            unknown = await classify_unknown(self._invoke_llm, message.text or "", self._system_strict_json)
            if unknown and qa:
                unknown_list = self.meta.get(chat_id, {}).get("unknown_qs", [])
                unknown_list = list(unknown_list) + [qa[-1].get("question") or ""]
                rec = self.meta.get(chat_id, {})
                rec["unknown_qs"] = unknown_list
                self.meta[chat_id] = rec
        except Exception:
            pass
        self._log(chat_id, f"A: {message.text or ''}")
        idx = (self.dynamic_idx.get(chat_id) or 1)
        # Early stop decision: ask/request_file/hypothesis/stop
        try:
            import json as _json
            qa_json_dec = _json.dumps(self.dynamic_qa.get(chat_id) or [], ensure_ascii=False, indent=2)
            decision_raw = await self._invoke_llm(
                build_decide_next_action_prompt(self.dynamic_theme.get(chat_id, ""), qa_json_dec, self._compute_context_blob(chat_id), prefer_non_finance=True),
                system=self._system_strict_json,
            )
            decision = json.loads(decision_raw)
        except Exception:
            decision = {}
        act = (decision.get("action") or "").strip().lower() if isinstance(decision, dict) else ""
        if act == "request_file":
            # Не навязываем докуметы в динамическом раунде — продолжаем вопросы до 5
            logging.info("[FLOW] decision=request_file -> continue asking (chat %s)", chat_id)
            if idx < 5:
                await self._ask_next_discovery(message, state)
            else:
                await self._finalize_dynamic_hypothesis(message, state)
            return
        if act == "hypothesis" or act == "stop":
            logging.info("[FLOW] decision=%s -> finalize (chat %s)", act or "hypothesis", chat_id)
            await self._finalize_dynamic_hypothesis(message, state)
            return
        # default: continue asking until 5
        if idx < 5:
            await self._ask_next_discovery(message, state)
        else:
            await self._finalize_dynamic_hypothesis(message, state)

    async def _start_discovery(self, message: Message, state: FSMContext) -> None:
        chat_id = message.chat.id
        self.dynamic_qa[chat_id] = []
        self.dynamic_idx[chat_id] = 0
        # reset per-round unknown list so it stays within a single hypothesis
        rec = self.meta.get(chat_id, {})
        rec["unknown_qs"] = []
        self.meta[chat_id] = rec
        await self._ask_next_discovery(message, state)

    async def _ask_next_discovery(self, message: Message, state: FSMContext) -> None:
        chat_id = message.chat.id
        asked = [it.get("question", "") for it in (self.dynamic_qa.get(chat_id) or []) if it.get("question")]
        # also avoid cross-round previously asked
        cross_avoid = self.avoid_questions_all.get(chat_id) or []
        qa = self.dynamic_qa.get(chat_id) or []
        import json as _json
        qa_json = _json.dumps(qa, ensure_ascii=False, indent=2)
        dialog_blob = self._compute_context_blob(chat_id)
        examples_text = Path("data/hypotheses.txt").read_text(encoding="utf-8") if Path("data/hypotheses.txt").exists() else ""
        # build list of unknown topics to steer away
        unknown_qs = []
        try:
            unknown_qs = list(self.meta.get(chat_id, {}).get("unknown_qs", []))
        except Exception:
            unknown_qs = []
        prompt = build_generate_discovery_question_prompt(
            qa_json=qa_json,
            dialog_json=dialog_blob,
            examples_text=examples_text,
            avoid=asked + cross_avoid,
            count=1,
            theme=self.dynamic_theme.get(chat_id, ""),
            avoid_unknown=unknown_qs,
        )
        try:
            raw = await self._invoke_llm(prompt, system=self._system_strict_json)
        except Exception as e:
            logging.exception("[FLOW] LLM error generating discovery question: %s", e)
            raw = ""
        try:
            arr = json.loads(raw)
        except Exception:
            arr = []
        if not isinstance(arr, list) or not arr:
            # Fallback: задать базовый вопрос по теме или общим образом
            theme_txt = (self.dynamic_theme.get(chat_id, "") or "деятельности компании").strip()
            qtext = f"Какие ключевые задачи по теме {theme_txt} сейчас наиболее актуальны?"
            logging.info("[FLOW] Fallback discovery question for chat %s: %s", chat_id, qtext)
        else:
            qtext = str(arr[0]).strip()
        if not qtext.endswith("?"):
            qtext = qtext.rstrip(". ") + "?"
        (self.dynamic_qa.setdefault(chat_id, [])).append({"question": qtext, "answer": ""})
        # track for cross-round avoid
        all_list = self.avoid_questions_all.setdefault(chat_id, [])
        if qtext not in all_list:
            all_list.append(qtext)
        # example answer
        try:
            example = await generate_short_example(self._invoke_llm, qtext, qa_json, dialog_blob)
            tail = f"\n\n<i>Пример ответа:</i> {example}" if example else ""
        except Exception:
            tail = ""
        idx = (self.dynamic_idx.get(chat_id) or 0) + 1
        self.dynamic_idx[chat_id] = idx
        await message.answer(f"Вопрос {idx}:\n{qtext}{tail}")
        self._log(chat_id, f"Q: {qtext}")
        await state.set_state(Flow.waiting_dynamic_answer)

    # kept placeholder for fallback heuristics if needed in future

    async def _finalize_dynamic_hypothesis(self, message: Message, state: FSMContext) -> None:
        chat_id = message.chat.id
        qa = self.dynamic_qa.get(chat_id) or []
        import json as _json
        qa_json = _json.dumps(qa, ensure_ascii=False, indent=2)
        dialog_blob = self._compute_context_blob(chat_id)
        examples_text = Path("data/hypotheses.txt").read_text(encoding="utf-8") if Path("data/hypotheses.txt").exists() else ""
        # Build 3 creative hypotheses (1 main + 2 alternatives) and 5 meeting questions for each
        hypos: list[dict] = []
        # main
        main_prompt = build_generate_free_hypothesis_prompt(
            self.dynamic_theme.get(chat_id, ""), qa_json, dialog_blob, examples_text, prefer_non_finance=True
        )
        main_raw = await self._invoke_llm(main_prompt, system=self._system_strict_json)
        try:
            main_obj = json.loads(main_raw)
        except Exception:
            main_obj = {"hypothesis": main_raw.strip(), "reason": ""}
        if (main_obj.get("hypothesis") or "").strip():
            hypos.append(main_obj)
        # two alternatives
        avoid_titles = (main_obj.get("hypothesis") or "").strip()
        for _ in range(2):
            alt_prompt = build_generate_alternative_hypothesis_prompt(
                qa_json, dialog_blob, examples_text, avoid_hypothesis=avoid_titles, prefer_non_finance=True
            )
            alt_raw = await self._invoke_llm(alt_prompt, system=self._system_strict_json)
            try:
                alt_obj = json.loads(alt_raw)
            except Exception:
                alt_obj = {"hypothesis": alt_raw.strip(), "reason": ""}
            title = (alt_obj.get("hypothesis") or "").strip()
            if title:
                hypos.append(alt_obj)
                avoid_titles = f"{avoid_titles}; {title}" if avoid_titles else title

        # render 3 hypotheses list
        items = hypos[:3]
        hyp_lines: list[str] = []
        for idx, item in enumerate(items, start=1):
            title = (item.get("hypothesis") or "").strip()
            reason = (item.get("reason") or "").strip()
            hyp_lines.append(
                (f"<b>{idx}) Гипотеза</b>: <b>{title}</b>\n" + (f"Причина: {reason}" if reason else "")).strip()
            )
        # single shared set of 5 meeting questions across all 3 hypotheses
        combined_title = "Гипотезы:\n" + "\n".join([f"{i+1}) {(items[i].get('hypothesis') or '').strip()}" for i in range(len(items))])
        q_prompt = build_generate_meeting_questions_prompt(combined_title, qa_json, dialog_blob, count=5)
        q_raw = await self._invoke_llm(q_prompt, system=self._system_strict_json)
        try:
            q_arr = json.loads(q_raw)
        except Exception:
            q_arr = []
        qs = "\n".join([f"• {str(q).strip()}" for q in (q_arr[:5] if isinstance(q_arr, list) else [])])
        body = "\n\n".join(hyp_lines) + (f"\n\n<b>Вопросы к встрече (5):</b>\n{qs if qs else '—'}" if hyp_lines else "")

        # Build appendix with unknowns
        unknown_qs = list(self.meta.get(chat_id, {}).get("unknown_qs", [])) if self.meta.get(chat_id) else []
        appendix = ""
        if unknown_qs:
            appendix = "\n\n<b>Будет полезно узнать у клиента:</b>\n" + "\n".join([f"• {q}" for q in unknown_qs if q])
        text = f"Итог:\n\n{body}{appendix}"
        # inline button to request alternative hypothesis
        kb = InlineKeyboardMarkup(
            inline_keyboard=[[InlineKeyboardButton(text="Сгенерировать ещё", callback_data="alt:more")]]
        )
        await message.answer(text, reply_markup=kb)
        # store first hypothesis for avoidance in alt round
        if hypos:
            first_title = (hypos[0].get("hypothesis") or "").strip()
            self._log(chat_id, f"FINAL HYPOTHESES: {[ (h.get('hypothesis') or '').strip() for h in hypos[:3] ]}")
            # keep last_result with shared 5 questions for simplicity
            self.last_result[chat_id] = {"hypothesis": first_title, "questions": (q_arr[:5] if isinstance(q_arr, list) else [])}
        await state.clear()

    async def _ask_theme(self, message: Message, state: FSMContext) -> None:
        kb = InlineKeyboardMarkup(
            inline_keyboard=[[InlineKeyboardButton(text="Пропустить", callback_data="theme:skip")]]
        )
        await message.answer(
            "Какая тема вам интересна для анализа состояния клиента?\n\n"
            "Примеры:\n"
            "• Доля на рынке\n"
            "• Организационные ситуации\n"
            "• Операцонные ситуации\n"
            "• Финансовое положение\n"
            "• Состояние относительно конкурентов",
            reply_markup=kb,
        )
        await state.set_state(Flow.waiting_theme)

    async def on_theme_answer(self, message: Message, state: FSMContext) -> None:
        chat_id = message.chat.id
        theme = (message.text or "").strip()
        if not theme:
            await message.answer("Пожалуйста, укажите тему в одном-двух словах.")
            return
        self.dynamic_theme[chat_id] = theme
        await message.answer("Начинаю анализ клиента. Мне нужно задать вам несколько вопросов, чтобы узнать больше")
        await state.clear()
        await self._start_discovery(message, state)

    async def on_theme_skip(self, cq: CallbackQuery, state: FSMContext) -> None:
        chat_id = cq.message.chat.id
        self.dynamic_theme[chat_id] = ""
        try:
            await cq.answer("Тему пропустили")
        except Exception:
            pass
        await cq.message.answer("Начинаю анализ клиента. Мне нужно задать вам несколько вопросов, чтобы узнать больше")
        await state.clear()
        await self._start_discovery(cq.message, state)

    async def on_alt_more(self, cq: CallbackQuery, state: FSMContext) -> None:
        chat_id = cq.message.chat.id
        prev = self.last_result.get(chat_id) or {}
        prev_hypo = str(prev.get("hypothesis") or "")
        if prev_hypo:
            self.avoid_hypotheses.setdefault(chat_id, []).append(prev_hypo)
        # reset round and ask theme once per new hypothesis
        self.dynamic_qa[chat_id] = []
        self.dynamic_idx[chat_id] = 0
        try:
            await cq.answer("Новый раунд")
        except Exception:
            pass
        # без дополнительного текста — сразу спрашиваем тему (с кнопкой пропуска)
        logging.info("[FLOW] Start alternative round for chat %s; avoid hypo: %s", chat_id, prev_hypo)
        await self._ask_theme(cq.message, state)

    async def _decide_and_ask_next(self, message: Message, state: FSMContext) -> None:
        chat_id = message.chat.id
        theme = self.dynamic_theme.get(chat_id, "")
        qa = self.dynamic_qa.get(chat_id) or []
        import json as _json
        qa_json = _json.dumps(qa, ensure_ascii=False, indent=2)
        dialog_blob = self._compute_context_blob(chat_id)
        prompt = build_decide_next_action_prompt(theme, qa_json, dialog_blob, prefer_non_finance=True)
        raw = await self._invoke_llm(prompt, system=self._system_strict_json)
        try:
            obj = json.loads(raw)
        except Exception:
            obj = {}
        action = (obj.get("action") or "ask").strip()
        if action == "ask":
            qtext = (obj.get("question") or "Уточните, пожалуйста, ключевой аспект по теме?").strip()
            if not qtext.endswith("?"):
                qtext = qtext.rstrip(". ") + "?"
            (self.dynamic_qa.setdefault(chat_id, [])).append({"question": qtext, "answer": ""})
            await message.answer(qtext)
            self._log(chat_id, f"Q: {qtext}")
            await state.set_state(Flow.waiting_dynamic_answer)
            return
        if action == "request_file":
            await message.answer("Похоже, не хватает материалов. Пришлите .docx или .json по клиенту — учту в анализе.")
            # keep state depending on whether last item has answer
            await state.set_state(Flow.waiting_dynamic_answer)
            return
        if action == "hypothesis":
            # generate free-form, industry-focused hypothesis
            examples_text = Path("data/hypotheses.txt").read_text(encoding="utf-8") if Path("data/hypotheses.txt").exists() else ""
            hyp_prompt = build_generate_free_hypothesis_prompt(theme, qa_json, dialog_blob, examples_text, prefer_non_finance=True)
            hyp_raw = await self._invoke_llm(hyp_prompt, system=self._system_strict_json)
            try:
                hyp_obj = json.loads(hyp_raw)
            except Exception:
                hyp_obj = {"hypothesis": hyp_raw.strip(), "reason": ""}
            hypo = (hyp_obj.get("hypothesis") or "").strip()
            reason = (hyp_obj.get("reason") or "").strip()
            tags = hyp_obj.get("tags") or []
            # try find best matching card by cosine to add rich description if available
            desc_html = ""
            try:
                best = get_best_full_card_by_cosine(hypo)
                if best:
                    desc_html = format_card_fields_to_html(best.get("title", ""), best.get("text", ""))
            except Exception:
                desc_html = ""
            text = (
                f"<b>Гипотеза</b>: <b>{hypo}</b>\n"
                f"Причина: {reason}\n"
            )
            if tags:
                text += f"\nТеги: {', '.join(tags)}\n"
            if desc_html:
                text += f"\n<b>Карточка (по близости):</b>\n{desc_html}"
            await message.answer(text)
            self._log(chat_id, f"FINAL (FREE) HYPOTHESIS: {hypo}\nREASON: {reason}")
            await state.clear()
            return
        # stop or unknown
        await message.answer("Остановлюсь здесь. Можем продолжить в любой момент командой /analyze или пришлите материалы.")
        await state.clear()

    # survey legacy removed

    async def cmd_restart(self, message: Message, state: FSMContext) -> None:
        chat_id = message.chat.id
        # clear all per-chat structures
        self.surveys.pop(chat_id, None)
        self.followup_questions.pop(chat_id, None)
        self.followup_answers.pop(chat_id, None)
        self.followup_idx.pop(chat_id, None)
        self.dialog_docs_json.pop(chat_id, None)
        self.dialog_docs_json_list.pop(chat_id, None)
        self.hypotheses_pool.pop(chat_id, None)
        await state.clear()
        await message.answer("Сессия сброшена. Запустите заново командой /survey и при необходимости пришлите файлы.")

    async def on_document(self, message: Message, state: FSMContext) -> None:
        try:
            doc = message.document
            if not doc:
                return
            file = await self.bot.get_file(doc.file_id)
            os.makedirs("uploads", exist_ok=True)
            local_path = Path("uploads") / f"{message.chat.id}_{doc.file_unique_id}_{doc.file_name}"
            await self.bot.download_file(file.file_path, destination=local_path)
            logging.info("Received file %s -> %s", doc.file_name, local_path)
            # setup dialog log with file name
            self._ensure_dialog_log(message.chat.id, doc.file_name)
            self._log(message.chat.id, f"File: {doc.file_name}")
            parsed: dict
            if str(local_path).lower().endswith(".docx"):
                parsed = parse_docx_to_json(str(local_path))
            elif str(local_path).lower().endswith(".json"):
                import json as _json
                parsed = _json.loads(Path(local_path).read_text(encoding="utf-8"))
            else:
                await message.answer("Поддерживаются только .docx и .json")
                return
            self.dialog_docs_json[message.chat.id] = parsed
            self.dialog_docs_json_list.setdefault(message.chat.id, []).append(parsed)
            dump_path = Path("uploads") / f"{message.chat.id}_dialog.json"
            import json as _json
            dump_path.write_text(_json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
            self._log(message.chat.id, f"Parsed JSON saved: {dump_path}")
            # Не отправляем JSON в чат по требованию
            # show one consolidated notice per short burst of uploads (2s window)
            now = time.time()
            last = self.last_upload_notice_at.get(message.chat.id, 0.0)
            if now - last > 2.0:
                await message.answer(
                    "Помогу выявить гипотезы, описывающие возможные боли и точки роста клиента.\n"
                    "На их основе составлю полезные для встречи вопросы."
                )
                self.last_upload_notice_at[message.chat.id] = now
            # Auto-debounce: start theme after short idle (collect multiple files in one go)
            chat_id = message.chat.id
            if self.waiting_file_first.get(chat_id, True):
                # cancel previous pending task if any
                task = self.file_debounce_tasks.pop(chat_id, None)
                if task and not task.done():
                    task.cancel()
                async def _delayed_start():
                    try:
                        await asyncio.sleep(2.0)
                        if self.waiting_file_first.get(chat_id, False):
                            self.waiting_file_first[chat_id] = False
                            await self._ask_theme(message, state)
                    except asyncio.CancelledError:
                        return
                    except Exception:
                        logging.exception("debounce start failed")
                self.file_debounce_tasks[chat_id] = asyncio.create_task(_delayed_start())
        except Exception as e:
            logging.exception("Failed to handle document: %s", e)
            await message.answer("Не удалось обработать файл. Убедитесь, что это корректный DOCX/JSON.")

    # survey legacy removed

    # survey legacy removed

    # survey legacy removed

    async def on_followup_answer(self, message: Message, state: FSMContext) -> None:
        chat_id = message.chat.id
        idx = self.followup_idx.get(chat_id, 0)
        asked = self.followup_questions.get(chat_id, [])
        if asked is None:
            await message.answer("Сессия не найдена. Нажмите /survey")
            return
        self.followup_answers[chat_id].append(message.text or "")
        self._log(chat_id, f"A: {message.text or ''}")
        idx += 1
        self.followup_idx[chat_id] = idx
        # update hypothesis pool dynamically after each answer
        self._update_hypotheses_pool(chat_id)
        if idx < 4:
            await self._ask_next_followup(message, state)
            return
        # All answered → select best hypothesis
        sess = self.surveys.get(chat_id) or {}
        combined = {
            **sess.answers,
            "followups": [
                {"question": q, "answer": a}
                for q, a in zip(self.followup_questions.get(chat_id, []), self.followup_answers[chat_id])
            ],
        }
        answers_json = json.dumps(combined, ensure_ascii=False, indent=2)
        original_hypotheses_text = Path("data/hypotheses.txt").read_text(encoding="utf-8")
        filtered_hypotheses_text = self._filter_hypotheses_by_facts(original_hypotheses_text, sess.answers)
        try:
            self._log(chat_id, f"Hypotheses filtered: {len(filtered_hypotheses_text.splitlines())} / {len(original_hypotheses_text.splitlines())}")
        except Exception:
            pass
        # include dialog JSON for more grounded selection
        dialog_blob = self._compute_context_blob(chat_id)
        asked = self.followup_questions.get(chat_id, [])
        facts_summary = self._build_facts_summary(sess.answers, asked, self.followup_answers[chat_id])
        # legacy survey selection removed
        final_prompt = ""
        # save prompt snapshot
        try:
            os.makedirs("logs/prompts", exist_ok=True)
            pr_path = Path("logs/prompts") / f"{chat_id}_select_hypothesis.txt"
            pr_path.write_text(final_prompt, encoding="utf-8")
            self._log(chat_id, f"Saved prompt: {pr_path}")
        except Exception:
            pass
        selected_raw = await self._invoke_llm(
            final_prompt,
            system=self._system_strict_json,
        )
        try:
            selected = json.loads(selected_raw)
            hypo = selected.get("hypothesis") or ""
            reason = selected.get("reason") or ""
        except Exception:
            hypo = selected_raw.strip()
            reason = ""
        # Validate against facts; if invalid, pick alternative excluding previous
        try:
            valid_raw = "{\"valid\": true}"
            valid = json.loads(valid_raw)
            if isinstance(valid, dict) and not valid.get("valid", True):
                self._log(chat_id, f"Hypothesis rejected due to: {valid.get('conflicts')}")
                alt_raw = "{}"
                try:
                    alt = json.loads(alt_raw)
                    hypo = alt.get("hypothesis") or hypo
                    reason = alt.get("reason") or reason
                except Exception:
                    pass
        except Exception:
            pass
        # Try: cosine match the best full card from DOCX blocks against the selected hypothesis text
        # Auto-select sub-hypothesis via LLM against subhypotheses.json
        subs = []
        if subs:
            sub_titles = "\n".join([it["title"] for it in subs])
            # Build facts snapshot
            sess = self.surveys.get(chat_id)
            answers_json2 = answers_json
            dialog_blob2 = self._compute_context_blob(chat_id)
            facts_summary2 = self._build_facts_summary(sess.answers, self.followup_questions.get(chat_id, []), self.followup_answers.get(chat_id, [])) if sess else ""
            sub_raw = "{}"
            try:
                sub_json = json.loads(sub_raw)
                sub_name = sub_json.get("sub") or subs[0]["title"]
            except Exception:
                sub_name = subs[0]["title"]
            # find chosen sub and render questions
            chosen = None
            for it in subs:
                if it["title"].strip().lower() == sub_name.strip().lower():
                    chosen = it
                    break
            if chosen is None:
                chosen = subs[0]
            qs_lines = "\n".join([f"• {q}" for q in chosen.get("questions", [])])
            text = (
                f"<b>Гипотеза</b>: <b>{hypo}</b>\n\n"
                f"<b>Побочная</b>: <b>{chosen['title']}</b>\n"
                f"<b>Вопросы:</b>\n{qs_lines}"
            )
            await message.answer(text)
            self._log(chat_id, f"SUB AUTOSELECTED: {chosen['title']}")
            await state.clear()
            return
        # Legacy fallback if no sub-hypotheses configured for this main
        try:
            card_html = get_card_by_hypothesis_title_html(hypo) or get_full_card_formatted_html(hypo) or ""
            desc_html = card_html or ""
        except Exception:
            desc_html = ""
        # Sub-hypothesis block (if available)
        subs = []
        subs_block = ""
        if subs:
            parts = ["\n<b>Побочная гипотеза и вопросы:</b>"]
            # pick first by default; can be extended to interactive choice
            sub = subs[0]
            parts.append(f"<b>{sub['title']}</b>")
            for q in sub["questions"]:
                parts.append(f"• {q}")
            subs_block = "\n".join(parts)

        if desc_html or subs_block:
            subs_part = f"\n{subs_block}" if subs_block else ""
            text = (
                f"<b>Гипотеза</b>: <b>{hypo}</b>\n"
                f"Причина: {reason}\n\n"
                f"<b>Карточка</b>:\n{desc_html}{subs_part}"
            )
        else:
            text = f"<b>Гипотеза</b>: <b>{hypo}</b>\nПричина: {reason}"
        await message.answer(text)
        self._log(chat_id, f"FINAL HYPOTHESIS: {hypo}\nREASON: {reason}")
        await state.clear()

    async def on_sub_select(self, cq: CallbackQuery) -> None:
        chat_id = cq.message.chat.id
        ctx = self.selected_main.get(chat_id)
        if not ctx:
            await cq.answer("Сессия не найдена", show_alert=True)
            return
        main = ctx.get("hypo", "")
        reason = ctx.get("reason", "")
        subs = self.subrepo.get_subs(main)
        try:
            idx = int(cq.data.split(":", 1)[1])
        except Exception:
            idx = 0
        if idx < 0 or idx >= len(subs):
            idx = 0
        sub = subs[idx]
        # Render questions
        questions = "\n".join([f"• {q}" for q in sub.get("questions", [])])
        text = (
            f"<b>Гипотеза</b>: <b>{main}</b>\n"
            f"Причина: {reason}\n\n"
            f"<b>Побочная гипотеза</b>: <b>{sub['title']}</b>\n"
            f"<b>Вопросы:</b>\n{questions}"
        )
        await cq.message.edit_text(text)
        self._log(chat_id, f"SUB SELECTED: {sub['title']}")
        # Clear context to avoid stale state
        self.selected_main.pop(chat_id, None)

    async def _invoke_llm(self, prompt: str, system: str | None = None) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.llm.invoke, prompt, system)

    def _ensure_logging(self) -> None:
        os.makedirs("logs", exist_ok=True)
        logging.basicConfig(
            filename="logs/bot.log",
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )

    def _extract_json_array(self, raw: str):
        try:
            return json.loads(raw)
        except Exception:
            pass
        # strip code fences
        m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, flags=re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        # find first array in text
        m = re.search(r"\[(?:.|\n|\r)*\]", raw)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        # fallback empty
        return []

    def _compute_context_blob(self, chat_id: int) -> str:
        # Merge multiple uploaded materials into a single compact blob
        docs = self.dialog_docs_json_list.get(chat_id) or ([] if self.dialog_docs_json.get(chat_id) is None else [self.dialog_docs_json[chat_id]])
        if not docs:
            return ""
        try:
            parts = [json.dumps(d, ensure_ascii=False) for d in docs]
            blob = "\n\n".join(parts)
            return blob[:8000]
        except Exception:
            return ""

    def _update_hypotheses_pool(self, chat_id: int) -> None:
        sess = self.surveys.get(chat_id)
        if not sess:
            return
        text = Path("data/hypotheses.txt").read_text(encoding="utf-8") if Path("data/hypotheses.txt").exists() else ""
        filtered_text = self._filter_hypotheses_by_facts(text, sess.answers)
        pool = [ln for ln in filtered_text.splitlines() if ln.strip()][:5]
        if not pool:
            pool = [ln for ln in text.splitlines() if ln.strip()][:5]
        self.hypotheses_pool[chat_id] = pool

    async def _ask_next_followup(self, message_or_cq_message: Message, state: FSMContext) -> None:
        chat_id = message_or_cq_message.chat.id
        asked = self.followup_questions.get(chat_id, [])
        sess = self.surveys[chat_id]
        base_answers_json = json.dumps(sess.answers, ensure_ascii=False, indent=2)
        # include previous followups into answers_json for better adaptation
        if asked and self.followup_answers.get(chat_id):
            combined = {
                **sess.answers,
                "followups": [
                    {"question": q, "answer": a}
                    for q, a in zip(asked, self.followup_answers.get(chat_id, []))
                ],
            }
            base_answers_json = json.dumps(combined, ensure_ascii=False, indent=2)

        dialog_blob = self._compute_context_blob(chat_id)
        candidates_text = "\n".join(self.hypotheses_pool.get(chat_id, []))
        prompt = "[]"  # legacy followups removed
        raw = await self._invoke_llm(prompt, system=self._system_strict_json)
        try:
            self._log(chat_id, f"LLM raw (followup next): {raw}")
        except Exception:
            pass
        arr = self._extract_json_array(raw)
        if not isinstance(arr, list) or not arr:
            await message_or_cq_message.answer("Не удалось сгенерировать уточняющий вопрос. Попробуйте /survey ещё раз.")
            return
        qtext = str(arr[0]).strip()
        if not qtext.endswith("?"):
            qtext = qtext.rstrip(". ") + "?"
        asked.append(qtext)
        self.followup_questions[chat_id] = asked
        idx = len(asked)
        # build example answer for UX
        try:
            example_raw = ""
            example = (example_raw or "").strip().strip('`"')
            # trim to first sentence end
            import re as _re
            m = _re.search(r"^(.+?[.!?])\s", example)
            if m:
                example = m.group(1)
            # clamp to 12 words
            words = [w for w in example.split() if w]
            if len(words) > 12:
                example = " ".join(words[:12])
                if not example.endswith(('.', '!', '?')):
                    example += '.'
            tail = f"\n\n<i>Пример ответа:</i> {example}" if example else ""
        except Exception:
            tail = ""
        await message_or_cq_message.answer(f"Уточняющий вопрос {idx}/4:\n{qtext}{tail}")
        self._log(chat_id, f"Q: {qtext}")
        await state.set_state(Flow.waiting_followup_answer)

    def _meta_keyboard(self) -> InlineKeyboardMarkup:
        rows = [
            [
                InlineKeyboardButton(text="Режим: Реальный кейс", callback_data="meta:mode:real"),
                InlineKeyboardButton(text="Режим: Тест", callback_data="meta:mode:test"),
            ],
            [
                InlineKeyboardButton(text="Память: Запоминать", callback_data="meta:remember:yes"),
                InlineKeyboardButton(text="Память: Не запоминать", callback_data="meta:remember:no"),
            ],
        ]
        return InlineKeyboardMarkup(inline_keyboard=rows)

    async def on_meta(self, cq: CallbackQuery) -> None:
        chat_id = cq.message.chat.id
        parts = (cq.data or "").split(":")
        if len(parts) >= 3:
            scope, key, val = parts[0], parts[1], parts[2]
            rec = self.meta.get(chat_id, {})
            if key == "mode":
                rec["mode"] = "real" if val == "real" else "test"
                await cq.answer("Режим установлен")
            elif key == "remember":
                rec["remember"] = "yes" if val == "yes" else "no"
                await cq.answer("Настройка памяти обновлена")
            self.meta[chat_id] = rec
        else:
            await cq.answer()

    def _filter_hypotheses_by_facts(self, hypotheses_text: str, answers: Dict[str, object]) -> str:
        lines = [ln.strip() for ln in hypotheses_text.splitlines() if ln.strip()]
        res: List[str] = []
        revenue = str(answers.get("revenue_trend", "")).lower()
        market = str(answers.get("market_state", "")).lower()
        share = str(answers.get("share_trend", "")).lower()
        for ln in lines:
            low = ln.lower()
            # revenue contradictions
            if "падает" in revenue:
                if "выручк" in low and ("растет" in low or "растёт" in low or "стабил" in low):
                    continue
            if "раст" in revenue or "стабил" in revenue:
                if "выручк" in low and "падает" in low:
                    continue
            # market contradictions
            if "стагнирует" in market:
                if "рынок" in low and ("раст" in low or "падает" in low):
                    continue
            if "раст" in market and ("рынок" in low and "падает" in low):
                continue
            if "падает" in market and ("рынок" in low and "раст" in low):
                continue
            # share contradictions
            if "падает" in share and ("доля" in low and "раст" in low):
                continue
            res.append(ln)
        return "\n".join(res) if res else hypotheses_text

    def _build_facts_summary(self, answers: Dict[str, object], qs: List[str], followup_answers: List[str]) -> str:
        parts: List[str] = []
        m = {
            "revenue_trend": "Выручка",
            "profit_trend": "Прибыль/EBITDA",
            "market_state": "Рынок",
            "share_trend": "Доля рынка",
            "ops_pains": "Операционные проблемы",
            "competitor_actions": "Действия конкурентов",
        }
        for k, title in m.items():
            val = answers.get(k)
            if isinstance(val, list):
                parts.append(f"{title}: {', '.join(val)}")
            elif isinstance(val, str) and val:
                parts.append(f"{title}: {val}")
        # Include brief followups
        for q, a in zip(qs, followup_answers):
            qshort = (q[:120] + "…") if len(q) > 120 else q
            ashort = (a[:120] + "…") if len(a) > 120 else a
            parts.append(f"{qshort} → {ashort}")
        return "\n".join(parts)

    def _ensure_dialog_log(self, chat_id: int, filename: str | None) -> Path:
        if chat_id in self.dialog_log_path:
            return self.dialog_log_path[chat_id]
        os.makedirs("logs/dialogs", exist_ok=True)
        safe = (filename or "no_file").replace("/", "_").replace("\\", "_")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path("logs/dialogs") / f"{chat_id}_{ts}_{safe}.txt"
        header = f"FILE: {filename or '—'}\nCHAT: {chat_id}\nSTARTED: {datetime.now().isoformat()}\n---\n"
        path.write_text(header, encoding="utf-8")
        self.dialog_log_path[chat_id] = path
        return path

    def _log(self, chat_id: int, line: str) -> None:
        try:
            path = self.dialog_log_path.get(chat_id)
            if not path:
                path = self._ensure_dialog_log(chat_id, None)
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            logging.exception("Failed to write dialog log for chat %s", chat_id)


async def run_bot() -> None:
    import os
    load_dotenv()
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
    app = BotApp(token)
    await app.start()


if __name__ == "__main__":
    asyncio.run(run_bot())


