from __future__ import annotations

"""
Назначение: обработка загрузки файлов (.docx/.json) — сохранение, парсинг, дебаунс старта темы.
"""

import json
import logging
import os
import time
from pathlib import Path

from app.parsers.docx_parser import parse_docx_to_json
FILE_UNSUPPORTED_MSG = ""
FILE_FAIL_MSG = ""
FILE_NOTICE_MSG = ""


async def handle_document(app, message, state) -> None:
    """Обрабатывает документ: сохраняет, парсит, запускает тему после паузы."""
    try:
        doc = message.document
        if not doc:
            return
        file = await app.bot.get_file(doc.file_id)
        os.makedirs("uploads", exist_ok=True)
        local_path = Path("uploads") / f"{message.chat.id}_{doc.file_unique_id}_{doc.file_name}"
        await app.bot.download_file(file.file_path, destination=local_path)
        logging.info("Received file %s -> %s", doc.file_name, local_path)
        app._ensure_dialog_log(message.chat.id, doc.file_name)
        app._log(message.chat.id, f"File: {doc.file_name}")
        if str(local_path).lower().endswith(".docx"):
            parsed = parse_docx_to_json(str(local_path))
        elif str(local_path).lower().endswith(".json"):
            parsed = json.loads(Path(local_path).read_text(encoding="utf-8"))
        else:
            await message.answer(FILE_UNSUPPORTED_MSG)
            return
        app.dialog_docs_json[message.chat.id] = parsed
        app.dialog_docs_json_list.setdefault(message.chat.id, []).append(parsed)
        dump_path = Path("uploads") / f"{message.chat.id}_dialog.json"
        dump_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
        app._log(message.chat.id, f"Parsed JSON saved: {dump_path}")
        # Раньше отправляли промо-сообщение; по требованию отключено
        now = time.time()
        app.last_upload_notice_at[message.chat.id] = now
        chat_id = message.chat.id
        if app.waiting_file_first.get(chat_id, True):
            task = app.file_debounce_tasks.pop(chat_id, None)
            if task and not task.done():
                task.cancel()
            async def _delayed_start():
                try:
                    await app._sleep(2.0)
                    if app.waiting_file_first.get(chat_id, False):
                        await app._after_first_upload(message, state)
                except Exception:
                    logging.exception("debounce start failed")
            app.file_debounce_tasks[chat_id] = app._create_task(_delayed_start())
    except Exception as e:
        logging.exception("Failed to handle document: %s", e)
        await message.answer(FILE_FAIL_MSG)


