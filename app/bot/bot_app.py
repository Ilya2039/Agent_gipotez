from __future__ import annotations

"""
Назначение: основной класс BotApp — хранит состояние, подключает хендлеры и вызывает потоки.
"""

import asyncio
import json
from typing import Dict, List
import re
import random

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import CommandStart, Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton

from app.llm.client import LLMClient
from app.bot.texts import (
    THEME_PROMPT,
    THEME_START_MSG,
)
from app.prompts.core import build_decide_next_action_prompt
from app.bot.preface import (
    show_preface,
    mark_preface_shown,
    on_preface_step1_answer as preface_step1_handler,
    on_preface_step2_answer as preface_step2_handler,
    on_preface_step3_answer as preface_step3_handler,
)
from app.services.unknowns import classify_unknown
from app.services.examples import generate_short_example
from dotenv import load_dotenv
import logging
import os
from pathlib import Path
import time
from app.parsers.docx_parser import parse_docx_to_json
from app.bot.ui import build_theme_keyboard, build_alt_keyboard, format_question
from app.bot.flows import ask_next_discovery, finalize_dynamic_hypothesis
from app.bot.states import Flow
from app.bot.handlers import register_handlers
from app.bot.files import handle_document
from app.bot.utils import ensure_logging, compute_context_blob, ensure_dialog_log, log_dialog
from app.bot.texts import (
    PREFACE_INTRO,
    PREFACE_STEP1,
    PREFACE_STEP2,
    PREFACE_STEP3,
)


class BotApp:
    def __init__(self, token: str) -> None:
        ensure_logging()
        self.bot = Bot(token=token, default=DefaultBotProperties(parse_mode="HTML"))
        self.dp = Dispatcher()
        self.llm = LLMClient()

        self.dialog_docs_json: Dict[int, dict] = {}
        self.dialog_docs_json_list: Dict[int, List[dict]] = {}
        self.hypotheses_pool: Dict[int, List[str]] = {}
        self.meta: Dict[int, Dict[str, str]] = {}
        self.dialog_log_path: Dict[int, Path] = {}
        self._system_strict_json = "Ты отвечаешь строго в формате JSON без лишнего текста. Не добавляй пояснения, дисклеймеры, списки, markdown. Возвращай только валидный JSON."

        # dynamic flow storages
        self.dynamic_qa: Dict[int, List[Dict[str, str]]] = {}
        self.dynamic_idx: Dict[int, int] = {}
        self.waiting_file_first: Dict[int, bool] = {}
        self.last_result: Dict[int, Dict[str, object]] = {}
        self.avoid_questions_all: Dict[int, List[str]] = {}
        self.avoid_hypotheses: Dict[int, List[str]] = {}
        self.dynamic_theme: Dict[int, str] = {}
        self.last_upload_notice_at: Dict[int, float] = {}
        self.file_debounce_tasks: Dict[int, asyncio.Task] = {}

    async def start(self) -> None:
        """Запускает поллинг и регистрирует хендлеры."""
        register_handlers(self)
        print("Бот запущен")
        await self.dp.start_polling(self.bot)

    async def cmd_start(self, message: Message, state: FSMContext) -> None:
        """Стартовая команда: создаёт лог диалога и просит прислать файлы."""
        ensure_dialog_log(self, message.chat.id, None)
        log_dialog(self, message.chat.id, "== START ==")
        chat_id = message.chat.id
        self.waiting_file_first[chat_id] = True
        upload_prompt = (
            "Сначала пришлите ВСЕ доступные материалы по клиенту (.docx или .json).\n"
            "Я автоматически начну вопросы через пару секунд после последнего файла. Если файлов нет — напишите /skip"
        )
        await message.answer(upload_prompt)

        # <-- изменено: дебаунс, если файлов нет, автоматически стартуем
        async def delayed_start():
            await asyncio.sleep(10)  # ждём 10 секунд после /start
            if self.waiting_file_first.get(chat_id):
                self.waiting_file_first[chat_id] = False
                await self._start_preface_or_goals(message, state)

        self._create_task(delayed_start())

    async def cmd_skip(self, message: Message, state: FSMContext) -> None:
        """Команда /skip: пропускает файлы и переходит к целям/потоку вопросов."""
        chat_id = message.chat.id
        if not self.waiting_file_first.get(chat_id):
            await message.answer("Уже идём по вопросам.")
            return
        self.waiting_file_first[chat_id] = False
        # <-- изменено: сразу стартуем префейс или цели
        await self._start_preface_or_goals(message, state)

    async def _after_first_upload(self, message: Message, state: FSMContext) -> None:
        """Вызывается после первой загрузки: префейс и затем тема или вопросы про цели."""
        chat_id = message.chat.id
        if self.waiting_file_first.get(chat_id, False):
            self.waiting_file_first[chat_id] = False
            await self._start_preface_or_goals(message, state)
    
    async def _start_preface_or_goals(self, message: Message, state: FSMContext) -> None:
        """
        Запускает префейс, если есть файлы, или сразу разговор про цели, если файлов нет.
        """
        chat_id = message.chat.id
        has_files = bool(self.dialog_docs_json_list.get(chat_id))

        # Помечаем, что префейс показан, чтобы не показывать его снова
        mark_preface_shown(self, chat_id)

        if has_files:
            # Файлы есть → показываем префейс
            await show_preface(self, message, state)
        else:
            # Файлов нет → сразу разговор про цели
            await self._ask_goals_prompt(message, state)

    async def _ask_goals_prompt(self, message: Message, state: FSMContext) -> None:
        """Показывает вопрос о целях/беспокойствах (если файлов нет)."""
        await message.answer(
            "Спасибо! Идём дальше! Подскажите, с какими целями (долгосрочными или краткосрочными) клиента вы знакомы?\nО чём сейчас беспокоится ваш клиент?"
        )
        await state.set_state(Flow.preface2)

    # Остальной код остаётся без изменений

    async def on_dynamic_answer(self, message: Message, state: FSMContext) -> None:
        """Обрабатывает ответ пользователя на очередной динамический вопрос."""
        chat_id = message.chat.id
        qa = self.dynamic_qa.get(chat_id) or []
        if qa and not qa[-1].get("answer"):
            qa[-1]["answer"] = message.text or ""
        else:
            qa.append({"question": "(free)", "answer": message.text or ""})
        self.dynamic_qa[chat_id] = qa
        try:
            unknown = await classify_unknown(self._invoke_llm, message.text or "", self._system_strict_json)
            if unknown and qa:
                unknown_list = self.meta.get(chat_id, {}).get("unknown_qs", [])
                unknown_list = list(unknown_list) + [qa[-1].get("question") or ""]
                rec = self.meta.get(chat_id, {})
                rec["unknown_qs"] = unknown_list
                self.meta[chat_id] = rec
        except Exception:
            pass
        log_dialog(self, chat_id, f"A: {message.text or ''}")
        idx = (self.dynamic_idx.get(chat_id) or 1)
        try:
            qa_json_dec = json.dumps(self.dynamic_qa.get(chat_id) or [], ensure_ascii=False, indent=2)
            decision_raw = await self._invoke_llm(
                build_decide_next_action_prompt(self.dynamic_theme.get(chat_id, ""), qa_json_dec, compute_context_blob(self, chat_id), prefer_non_finance=True),
                system=self._system_strict_json,
            )
            decision = json.loads(decision_raw)
        except Exception:
            decision = {}
        act = (decision.get("action") or "").strip().lower() if isinstance(decision, dict) else ""
        if act == "request_file":
            if idx < 5:
                await ask_next_discovery(self, message, state)
            else:
                await finalize_dynamic_hypothesis(self, message, state)
            return
        if act == "hypothesis" or act == "stop":
            await finalize_dynamic_hypothesis(self, message, state)
            return
        if idx < 5:
            await ask_next_discovery(self, message, state)
        else:
            await finalize_dynamic_hypothesis(self, message, state)

    async def _start_discovery(self, message: Message, state: FSMContext) -> None:
        """Инициализирует цикл из максимум 5 уточняющих вопросов."""
        chat_id = message.chat.id
        self.dynamic_qa[chat_id] = []
        self.dynamic_idx[chat_id] = 0
        rec = self.meta.get(chat_id, {})
        rec["unknown_qs"] = []
        self.meta[chat_id] = rec
        await ask_next_discovery(self, message, state)

    async def _ask_next_discovery(self, message: Message, state: FSMContext) -> None:
        """Проксирует генерацию и отправку следующего вопроса."""
        await ask_next_discovery(self, message, state)

    async def _finalize_dynamic_hypothesis(self, message: Message, state: FSMContext) -> None:
        """Формирует 3 гипотезы и 5 вопросов к встрече и отправляет итог."""
        await finalize_dynamic_hypothesis(self, message, state)

    async def _ask_theme(self, message: Message, state: FSMContext) -> None:
        """(Отключено) Раньше спрашивали тему; теперь сразу старт вопросов."""
        await state.clear()
        await self._start_discovery(message, state)

    async def on_theme_answer(self, message: Message, state: FSMContext) -> None:
        """(Отключено) Тема больше не используется; стартуем вопросы."""
        await state.clear()
        await self._start_discovery(message, state)

    async def on_theme_skip(self, cq: CallbackQuery, state: FSMContext) -> None:
        """(Отключено) Тему больше не спрашиваем; стартуем вопросы."""
        try:
            await cq.answer()
        except Exception:
            pass
        await state.clear()
        await self._start_discovery(cq.message, state)

    async def on_alt_more(self, cq: CallbackQuery, state: FSMContext) -> None:
        """Начинает новый раунд: добавляет текущую гипотезу в avoid и возвращается к теме."""
        chat_id = cq.message.chat.id
        prev = self.last_result.get(chat_id) or {}
        prev_hypo = str(prev.get("hypothesis") or "")
        if prev_hypo:
            self.avoid_hypotheses.setdefault(chat_id, []).append(prev_hypo)
        self.dynamic_qa[chat_id] = []
        self.dynamic_idx[chat_id] = 0
        try:
            await cq.answer("Новый раунд")
        except Exception:
            pass
        await self._ask_theme(cq.message, state)

    async def on_document(self, message: Message, state: FSMContext) -> None:
        """Проксирует обработку загрузок документов (.docx/.json)."""
        await handle_document(self, message, state)
        # после первого успешного файла показываем префейс сразу (и отменяем дебаунс)
        chat_id = message.chat.id
        task = self.file_debounce_tasks.pop(chat_id, None)
        if task and not task.done():
            task.cancel()
        # если уже показали префейс — ничего не делаем
        if self.meta.get(chat_id, {}).get("preface_shown"):
            return
        self.waiting_file_first[chat_id] = False
        mark_preface_shown(self, chat_id)
        await show_preface(self, message, state)

    async def _invoke_llm(self, prompt: str, system: str | None = None) -> str:
        """Асинхронно вызывает LLM (синхронный клиент в отдельном пуле)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.llm.invoke, prompt, system)

    # утилиты для flows/files
    def _compute_context_blob(self, chat_id: int) -> str:
        """Возвращает укороченный JSON‑контекст из загруженных файлов для чата."""
        return compute_context_blob(self, chat_id)

    def _log(self, chat_id: int, line: str) -> None:
        """Пишет строку в файл диалога."""
        log_dialog(self, chat_id, line)

    def _ensure_dialog_log(self, chat_id: int, filename: str | None) -> Path:
        """Создаёт файл диалога (если не создан) и возвращает путь."""
        return ensure_dialog_log(self, chat_id, filename)

    async def _sleep(self, seconds: float) -> None:
        """Асинхронная пауза (используется для дебаунса старта)."""
        await asyncio.sleep(seconds)

    def _create_task(self, coro: asyncio.coroutines) -> asyncio.Task:
        """Создаёт таск в фоне (для отложенного старта вопросов)."""
        return asyncio.create_task(coro)

    async def _after_first_upload(self, message: Message, state: FSMContext) -> None:
        """Вызывается после первой загрузки: префейс и затем тема."""
        chat_id = message.chat.id
        if self.waiting_file_first.get(chat_id, False):
            self.waiting_file_first[chat_id] = False
            if not self.meta.get(chat_id, {}).get("preface_shown"):
                self._mark_preface_shown(chat_id)
                await self._show_preface(message, state)

    # _show_preface вынесен в app.bot.preface.show_preface

    # _llm_extract_agreements вынесена в app.bot.preface

    # _mark_preface_shown вынесена в app.bot.preface

    # _llm_extract_overdue_agreements вынесена в app.bot.preface

    async def on_preface_step1_answer(self, message: Message, state: FSMContext) -> None:
        await preface_step1_handler(self, message, state)

    async def on_preface_step2_answer(self, message: Message, state: FSMContext) -> None:
        await preface_step2_handler(self, message, state)

    async def _ask_goals_prompt(self, message: Message, state: FSMContext) -> None:
        """Показывает вопрос о целях/беспокойствах (перед шагом 3/3)."""
        await message.answer(
            "Спасибо! Идём дальше! Подскажите, с какими целями (долгосрочными или краткосрочными) клиента вы знакомы?\nО чём сейчас беспокоится ваш клиент?"
        )
        await state.set_state(Flow.preface2)

    async def on_preface_step3_answer(self, message: Message, state: FSMContext) -> None:
        await preface_step3_handler(self, message, state)


async def run_bot() -> None:
    load_dotenv()
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
    app = BotApp(token)
    await app.start()


if __name__ == "__main__":
    asyncio.run(run_bot())


