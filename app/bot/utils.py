from __future__ import annotations

"""
Назначение: утилиты бота — логирование, JSON-хелперы, сбор контекста.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
import re


def ensure_logging() -> None:
    """Инициализирует файловый лог для бота."""
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        filename="logs/bot.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def extract_json_array(raw: str):
    """Аккуратно извлекает JSON-массив из строки (с учётом code fences)."""
    try:
        return json.loads(raw)
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    m = re.search(r"\[(?:.|\n|\r)*\]", raw)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return []


def compute_context_blob(app, chat_id: int, limit: int = 8000) -> str:
    """Собирает компактный JSON‑контекст из ВСЕХ загруженных материалов для чата.
    limit — ограничение длины результата (символов).
    """
    docs = app.dialog_docs_json_list.get(chat_id) or []
    if not docs:
        return ""
    try:
        parts = [json.dumps(d, ensure_ascii=False) for d in docs]
        blob = "\n\n".join(parts)
        return blob[:limit]
    except Exception:
        return ""


def ensure_dialog_log(app, chat_id: int, filename: str | None) -> Path:
    """Убеждается, что файл диалога для чата существует, и возвращает путь."""
    if chat_id in app.dialog_log_path:
        return app.dialog_log_path[chat_id]
    Path("logs/dialogs").mkdir(parents=True, exist_ok=True)
    safe = (filename or "no_file").replace("/", "_").replace("\\", "_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path("logs/dialogs") / f"{chat_id}_{ts}_{safe}.txt"
    header = f"FILE: {filename or '—'}\nCHAT: {chat_id}\nSTARTED: {datetime.now().isoformat()}\n---\n"
    path.write_text(header, encoding="utf-8")
    app.dialog_log_path[chat_id] = path
    return path


def log_dialog(app, chat_id: int, line: str) -> None:
    """Пишет строку в файл диалога чата."""
    try:
        path = app.dialog_log_path.get(chat_id)
        if not path:
            path = ensure_dialog_log(app, chat_id, None)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        logging.exception("Failed to write dialog log for chat %s", chat_id)


