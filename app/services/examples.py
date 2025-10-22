from __future__ import annotations

from app.prompts.core import build_example_answer_prompt


async def generate_short_example(invoke_llm, question: str, answers_json: str, dialog_json: str = "") -> str:
    prompt = build_example_answer_prompt(question, answers_json, dialog_json)
    try:
        raw = await invoke_llm(prompt, system=None)
        example = (raw or "").strip().replace("\n", " ")
        return example
    except Exception:
        return ""


