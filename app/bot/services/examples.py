from __future__ import annotations

"""
Service to generate short example answers for dynamic questions.
"""

from app.prompts.core import build_example_answer_prompt


async def generate_short_example(invoke_llm, question: str, answers_json: str, dialog_json: str = "") -> str:
    """Return a one-sentence (≤12 words) example answer for UI hinting.

    invoke_llm: callable(prompt: str, system: str|None) -> str
    """
    prompt = build_example_answer_prompt(question, answers_json, dialog_json)
    try:
        raw = await invoke_llm(prompt, system=None)
        example = (raw or "").strip()
        # be conservative: keep it single-line and short
        example = example.replace("\n", " ").strip()
        return example
    except Exception:
        return ""


