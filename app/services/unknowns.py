from __future__ import annotations

import logging

from app.prompts.core import build_is_unknown_answer_prompt


async def classify_unknown(invoke_llm, answer: str, system: str) -> bool:
    try:
        # Классификатор 'да'/'нет' (через LLM), логируем промпт и сырой ответ
        prompt = build_is_unknown_answer_prompt(answer or "")
        logging.info("[unknowns] prompt=%s", prompt.replace("\n", " "))
        raw = await invoke_llm(prompt, system=system)
        text = (raw or "").strip().lower()
        logging.info("[unknowns] raw=%s", text)
        # Accept single-token 'да'/'нет' or short phrases
        if text.startswith("да"):
            return True
        if text.startswith("нет"):
            return False
        return False
    except Exception:
        return False


