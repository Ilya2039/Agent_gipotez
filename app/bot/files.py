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
from app.bot.utils import compute_context_blob
FILE_UNSUPPORTED_MSG = ""
FILE_FAIL_MSG = ""
FILE_NOTICE_MSG = ""


async def handle_document(app, message, state) -> None:
    """Обрабатывает документ: сохраняет, парсит, мерджит JSON и запускает тему после паузы."""
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

        # Парсим документ
        if str(local_path).lower().endswith(".docx"):
            parsed = parse_docx_to_json(str(local_path))
        elif str(local_path).lower().endswith(".json"):
            parsed = json.loads(Path(local_path).read_text(encoding="utf-8"))
        else:
            await message.answer(FILE_UNSUPPORTED_MSG)
            return

        # Сохраняем и мерджим
        app.dialog_docs_json[message.chat.id] = parsed
        app.dialog_docs_json_list.setdefault(message.chat.id, []).append(parsed)

        dump_path = Path("uploads") / f"{message.chat.id}_dialog.json"
        all_docs = app.dialog_docs_json_list[message.chat.id]
        merged_json = {"docs": all_docs}  # Мерджим все файлы в один объект
        dump_path.write_text(json.dumps(merged_json, ensure_ascii=False, indent=2), encoding="utf-8")
        app._log(message.chat.id, f"Merged JSON saved: {dump_path}")
        # Пишем цельный текст всех документов в logs/all_context.txt
        try:
            lines = []
            for d in all_docs:
                for sec in (d.get("sections") or []):
                    h = sec.get("heading")
                    if h:
                        lines.append(str(h))
                    for t in (sec.get("text") or []):
                        lines.append(str(t))
            all_text = ("\n".join(lines)).strip()
            Path("logs").mkdir(parents=True, exist_ok=True)
            Path("logs/all_context.txt").write_text(all_text, encoding="utf-8")
            app._log(message.chat.id, f"ALL_CONTEXT_WRITTEN: {len(all_text)} chars")
        except Exception:
            logging.exception("all_context write failed")
        # Снимок контекста: все документы объединены
        try:
            blob = compute_context_blob(app, message.chat.id, limit=50000)
            Path("logs/examples").mkdir(parents=True, exist_ok=True)
            snap_path = Path("logs/examples") / f"{message.chat.id}_{int(time.time())}.json"
            snap_path.write_text(blob, encoding="utf-8")
            app._log(
                message.chat.id,
                f"EXAMPLES_SNAPSHOT: {snap_path} | docs={len(all_docs)} | bytes={len(blob)}"
            )
            app._log(message.chat.id, "EXAMPLES_TRIM:\n" + (blob[:2000] or "(empty)"))
            # Пишем служебный индекс (кол-во файлов и имена)
            index_path = Path("logs/examples") / f"{message.chat.id}_index.txt"
            names = ", ".join([str(Path(dump_path).name)])
            index_path.write_text(
                f"docs={len(all_docs)}; last_file={doc.file_name}; snapshot={snap_path.name}\n",
                encoding="utf-8",
            )
        except Exception:
            logging.exception("context snapshot failed")

        # Ставим debounce для старта
        chat_id = message.chat.id
        task = app.file_debounce_tasks.pop(chat_id, None)
        if task and not task.done():
            task.cancel()

        async def _delayed_start():
            try:
                await app._sleep(2.0)  # ждем, вдруг придут ещё файлы
                if app.waiting_file_first.get(chat_id, True):
                    await app._after_first_upload(message, state)
            except Exception:
                logging.exception("debounce start failed")

        app.file_debounce_tasks[chat_id] = app._create_task(_delayed_start())

    except Exception as e:
        logging.exception("Failed to handle document: %s", e)
        await message.answer(FILE_FAIL_MSG)
