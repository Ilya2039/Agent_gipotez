from __future__ import annotations

"""
Service helpers for handling "unknown" user answers.
"""

from app.prompts.core import build_is_unknown_answer_prompt


async def classify_unknown(invoke_llm, answer: str, system: str) -> bool:
    """Return True if the LLM classifies the answer as 'unknown'.

    invoke_llm: callable(prompt: str, system: str|None) -> str
    """
    try:
        raw = await invoke_llm(build_is_unknown_answer_prompt(answer or ""), system=system)
        import json as _json

        data = _json.loads(raw)
        return bool(data.get("unknown", False))
    except Exception:
        return False


