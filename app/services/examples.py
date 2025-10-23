from __future__ import annotations

import logging

from app.prompts.core import build_example_answer_prompt


async def generate_short_example(invoke_llm, question: str, answers_json: str, dialog_json: str = "") -> str:
    prompt = build_example_answer_prompt(question, answers_json, dialog_json)
    try:
        sys = "Ответь одной фразой (<=12 слов), без кавычек и дисклеймеров."
        logging.info("[examples] prompt=%s", prompt.replace("\n", " "))
        raw = await invoke_llm(prompt, system=sys)
        logging.info("[examples] raw=%s", (raw or "").strip())
        example = (raw or "").strip().replace("\n", " ")
        # Fallback if model returns empty or too long
        if not example:
            example = "Есть планы, но детали уточняются."
        # Keep it terse
        words = example.split()
        if len(words) > 12:
            example = " ".join(words[:12]).rstrip(",.;:!?")
        logging.info("[examples] final=%s", example)
        return example
    except Exception:
        return "Уточню у команды позже."


