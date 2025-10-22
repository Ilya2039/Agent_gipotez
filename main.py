from __future__ import annotations

import logging
import os
from pathlib import Path
import uvicorn
from app.api.server import create_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# file logging (logs/bot.log)
try:
    Path("logs").mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler("logs/bot.log", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', '').endswith('bot.log') for h in root.handlers):
        root.addHandler(file_handler)
except Exception:
    # fallback to console-only if file handler setup fails
    pass

app = create_app()

if __name__ == "__main__":
    logging.info("Starting API on http://0.0.0.0:8000 (docs at /docs)")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
