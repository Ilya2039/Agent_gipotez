from __future__ import annotations

"""
Назначение: генерация коротких примерных ответов (подсказок) для UI.
"""

from app.prompts.core import build_example_answer_prompt


async def generate_short_example(invoke_llm, question: str, answers_json: str, dialog_json: str = "") -> str:
    """Возвращает одно предложение (≤ 12 слов) — пример ответа пользователя.

    invoke_llm: callable(prompt: str, system: str|None) -> str
    """
    prompt = build_example_answer_prompt(question, answers_json, dialog_json)
    try:
        raw = await invoke_llm(prompt, system=None)
        example = (raw or "").strip().replace("\n", " ")
        return example
    except Exception:
        return ""


