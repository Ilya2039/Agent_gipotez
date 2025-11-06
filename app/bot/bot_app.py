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
        self.last_hypotheses: Dict[int, List[Dict[str, str]]] = {}
        self.last_hypotheses_messages: Dict[int, List[str]] = {}

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

    async def cmd_skip(self, message: Message, state: FSMContext) -> None:
        """Команда /skip: пропускает файлы и переходит к целям/потоку вопросов."""
        # Теперь пропуск отключён — ждём загрузки файлов
        upload_prompt = (
            "Сначала пришлите ВСЕ доступные материалы по клиенту (.docx или .json).\n"
            "Я автоматически начну вопросы через пару секунд после последнего файла. Если файлов нет — напишите /skip"
        )
        await message.answer(upload_prompt)
        return

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
                build_decide_next_action_prompt(
                    self.dynamic_theme.get(chat_id, ""), qa_json_dec, self._compute_context_blob(chat_id), prefer_non_finance=True
                ),
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
        """Начинает новый раунд: добавляет все показанные гипотезы в avoid и возвращается к теме."""
        chat_id = cq.message.chat.id
        # учитываем все 3 прошлые гипотезы
        last_hypos = self.last_hypotheses.get(chat_id, [])
        for h in last_hypos:
            title = str(h.get("hypothesis") or "").strip()
            if title:
                self.avoid_hypотезы.setdefault(chat_id, []).append(title)
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
        # Важно: не отменяем дебаунс и не запускаем префейс здесь.
        # files.handle_document сам ставит 2‑секундный debounce, чтобы успели прийти все файлы из одного сообщения.
        # Пусть старт префейса произойдёт из _after_first_upload по таймеру.

    async def _invoke_llm(self, prompt: str, system: str | None = None) -> str:
        """Асинхронно вызывает LLM (синхронный клиент в отдельном пуле)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.llm.invoke, prompt, system)

    # утилиты для flows/files
    def _compute_context_blob(self, chat_id: int, limit: int | None = None) -> str:
        """Возвращает JSON‑контекст из загруженных файлов для чата (ограничение по длине опционально)."""
        return compute_context_blob(self, chat_id, limit if limit is not None else 8000)

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
        """Вызывается после первой загрузки: запускает префейс по готовности."""
        chat_id = message.chat.id
        if self.waiting_file_first.get(chat_id, False):
            self.waiting_file_first[chat_id] = False
            if not self.meta.get(chat_id, {}).get("preface_shown"):
                mark_preface_shown(self, chat_id)
                await show_preface(self, message, state)

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

    async def on_actions_more(self, cq: CallbackQuery, state: FSMContext) -> None:
        """Кнопка 'Получить еще гипотезы' — спрашиваем про интересующую тему."""
        from app.bot.texts import THEME_FOR_CORRECTIONS, BTN_SKIP_THEME
        try:
            await cq.answer()
        except Exception:
            pass
        kb = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text=BTN_SKIP_THEME, callback_data="theme:skip_more")]
        ])
        await cq.message.answer(THEME_FOR_CORRECTIONS, reply_markup=kb)
        await state.set_state(Flow.waiting_theme_for_corrections)

    async def on_actions_correct(self, cq: CallbackQuery, state: FSMContext) -> None:
        """Кнопка 'Скоректировать текущие' — спрашиваем замечания пользователя."""
        from app.bot.texts import ASK_CORRECTIONS
        try:
            await cq.answer()
        except Exception:
            pass
        await cq.message.answer(ASK_CORRECTIONS)
        await state.set_state(Flow.waiting_corrections)

    async def on_actions_agree(self, cq: CallbackQuery, state: FSMContext) -> None:
        """Сохраняет текущие 3 гипотезы (последние показанные сообщения) в txt файл и подтверждает."""
        from pathlib import Path
        chat_id = cq.message.chat.id
        try:
            await cq.answer()
        except Exception:
            pass
        msgs = self.last_hypotheses_messages.get(chat_id) or []
        if not msgs:
            # Нечего сохранять
            await cq.message.answer("Мне нечего сохранить: сначала сформируйте гипотезы.")
            return
        Path("gipotez").mkdir(parents=True, exist_ok=True)
        out_path = Path("gipotez") / "all_gipotez.txt"
        with out_path.open("a", encoding="utf-8") as f:
            for block in msgs[:3]:
                clean = re.sub(r"<[^>]+>", "", block)
                f.write(clean.rstrip() + "\n\n")
        # Сохранили гипотезы — теперь спрашиваем, новый ли это клиент
        from app.bot.texts import NEW_CLIENT_PROMPT, BTN_YES, BTN_NO
        kb = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text=BTN_YES, callback_data="newclient:yes"), InlineKeyboardButton(text=BTN_NO, callback_data="newclient:no")]
        ])
        await cq.message.answer("Согласовано. Сохранила гипотезы.")
        await cq.message.answer(NEW_CLIENT_PROMPT, reply_markup=kb)

    async def on_newclient_yes(self, cq: CallbackQuery, state: FSMContext) -> None:
        """Обработчик кнопки 'Да' для вопроса 'Новый клиент?'. Отправляет 'Текст 1' + гипотезы."""
        chat_id = cq.message.chat.id
        try:
            await cq.answer()
        except Exception:
            pass
        from app.bot.texts import NEW_CLIENT_YES_TEXT
        msgs = self.last_hypotheses_messages.get(chat_id) or []
        hypos = []
        for block in msgs[:3]:
            hypos.append(re.sub(r"<[^>]+>", "", block).rstrip())
        hypos_text = "\n\n".join(hypos) if hypos else ""
        header = "Гипотезы для фокусного обсуждения бизнеса клиента:"
        if hypos_text:
            await cq.message.answer(f"{NEW_CLIENT_YES_TEXT}\n\n{header}\n\n{hypos_text}")
        else:
            await cq.message.answer(NEW_CLIENT_YES_TEXT)

    async def on_newclient_no(self, cq: CallbackQuery, state: FSMContext) -> None:
        """Обработчик кнопки 'Нет' для вопроса 'Новый клиент?'. Отправляет 'Текст 2' + гипотезы."""
        chat_id = cq.message.chat.id
        try:
            await cq.answer()
        except Exception:
            pass
        from app.bot.texts import NEW_CLIENT_NO_TEXT
        msgs = self.last_hypotheses_messages.get(chat_id) or []
        hypos = []
        for block in msgs[:3]:
            hypos.append(re.sub(r"<[^>]+>", "", block).rstrip())
        hypos_text = "\n\n".join(hypos) if hypos else ""
        header = "Гипотезы для фокусного обсуждения бизнеса клиента:"
        if hypos_text:
            await cq.message.answer(f"{NEW_CLIENT_NO_TEXT}\n\n{header}\n\n{hypos_text}")
        else:
            await cq.message.answer(NEW_CLIENT_NO_TEXT)

    async def on_corrections_message(self, message: Message, state: FSMContext) -> None:
        """Получает замечания по гипотезам или тему для новых гипотез."""
        chat_id = message.chat.id
        current = await state.get_state()
        
        if current == Flow.waiting_theme_for_corrections:
            # Пользователь ввёл тему для новых гипотез
            theme = message.text or ""
            # Сохраняем тему и инициализируем состояние как в _start_discovery
            self.dynamic_qa[chat_id] = []
            self.dynamic_idx[chat_id] = 0
            self.dynamic_theme[chat_id] = theme
            # Добавляем последние показанные гипотезы в avoid
            last_hypos = self.last_hypotheses.get(chat_id, [])
            for h in last_hypos:
                title = str(h.get("hypothesis") or "").strip()
                if title:
                    self.avoid_hypотезы.setdefault(chat_id, []).append(title)
            # Инициализируем meta для unknown_qs если нужно
            if chat_id not in self.meta:
                self.meta[chat_id] = {"unknown_qs": []}
            # Устанавливаем состояние для ответов на вопросы
            await state.set_state(Flow.waiting_dynamic_answer)
            # Запускаем первый вопрос
            from app.bot.texts import THEME_START_MSG
            await message.answer(THEME_START_MSG)
            # Запускаем вопрос через ask_next_discovery
            await ask_next_discovery(self, message, state)
        else:
            # Обычный режим корректировок
            # Добавляем исправления в историю для контекста
            qa = self.dynamic_qa.get(chat_id) or []
            qa.append({"question": "Корректировки по гипотезам", "answer": message.text or ""})
            self.dynamic_qa[chat_id] = qa
            from app.bot.texts import CORRECTIONS_DONE
            from app.bot.flows import refine_hypotheses
            await message.answer(CORRECTIONS_DONE)
            # Пересобираем гипотезы с учётом замечаний
            await refine_hypotheses(self, message, state, message.text or "")
            
    async def on_theme_skip_more(self, cq: CallbackQuery, state: FSMContext) -> None:
        """Кнопка 'Пропустить выбор' темы при запросе новых гипотез."""
        try:
            await cq.answer()
        except Exception:
            pass
        # Запускаем обычную генерацию гипотез без учета темы
        await self.on_alt_more(cq, state)

    async def on_theme_skip_corrections(self, cq: CallbackQuery, state: FSMContext) -> None:
        """Кнопка 'Пропустить выбор' темы при корректировках."""
        try:
            await cq.answer()
        except Exception:
            pass
        from app.bot.texts import ASK_CORRECTIONS
        await cq.message.answer(ASK_CORRECTIONS)
        await state.set_state(Flow.waiting_corrections)


async def run_bot() -> None:
    load_dotenv()
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
    app = BotApp(token)
    await app.start()


if __name__ == "__main__":
    asyncio.run(run_bot())


