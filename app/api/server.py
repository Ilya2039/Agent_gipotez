from __future__ import annotations

"""
FastAPI entrypoint exposing a single /chat endpoint. The dialog engine (LangGraph and
processing) lives in app.api.graph.
"""

from fastapi import FastAPI

from app.api.graph import ChatRequest, ChatResponse, invoke


app = FastAPI(title="Agent Gipotez API", version="1.0")


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    return invoke(req)


