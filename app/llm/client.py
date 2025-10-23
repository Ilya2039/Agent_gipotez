from __future__ import annotations

"""
Клиент LLM (GigaChat) через LangChain.
"""

import os
from typing import Optional
from pydantic import BaseModel
import logging
from dotenv import load_dotenv

try:
    from langchain_community.chat_models.gigachat import GigaChat as LCGigaChat  # type: ignore
    from langchain_core.messages import SystemMessage, HumanMessage  # type: ignore
except Exception:
    LCGigaChat = None  # type: ignore
    SystemMessage = None  # type: ignore
    HumanMessage = None  # type: ignore


class LLMConfig(BaseModel):
    credentials: str
    verify_ssl_certs: bool = False
    scope: str = "GIGACHAT_API_CORP"
    model: str = "GigaChat-2-Reasoning"
    temperature: float = 0.9
    timeout: int = 60
    profanity_check: bool = False


class LLMClient:
    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        load_dotenv()
        cfg = config or LLMConfig(
            credentials=os.getenv("GIGACHAT_AUTH", ""),
            verify_ssl_certs=os.getenv("VERIFY_SSL_CERTS", "false").lower() == "true",
            scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_CORP"),
            model=os.getenv("GIGACHAT_MODEL", "GigaChat-2-Reasoning"),
            temperature=float(os.getenv("GIGACHAT_TEMPERATURE", "0.8")),
            timeout=int(os.getenv("GIGACHAT_TIMEOUT", "60")),
            profanity_check=os.getenv("GIGACHAT_PROFANITY_CHECK", "false").lower() == "true",
        )
        if not LCGigaChat:
            raise RuntimeError("langchain_community GigaChat is not installed.")
        logging.getLogger(__name__).info(
            "GigaChat cfg: model=%s, temp=%s, scope=%s, verify_ssl=%s, timeout=%s, profanity_check=%s",
            cfg.model,
            cfg.temperature,
            cfg.scope,
            cfg.verify_ssl_certs,
            cfg.timeout,
            cfg.profanity_check,
        )
        self._lc = LCGigaChat(
            credentials=cfg.credentials,
            verify_ssl_certs=cfg.verify_ssl_certs,
            scope=cfg.scope,
            model=cfg.model,
            temperature=cfg.temperature,
            timeout=cfg.timeout,
            profanity_check=cfg.profanity_check,
        )

    def invoke(self, prompt: str, system: Optional[str] = None) -> str:
        messages = []
        if system and SystemMessage:
            messages.append(SystemMessage(content=system))
        if HumanMessage:
            messages.append(HumanMessage(content=prompt))
        else:
            messages = prompt  # type: ignore
        resp = self._lc.invoke(messages)
        try:
            return resp.content  # type: ignore[attr-defined]
        except Exception:
            return str(resp)
