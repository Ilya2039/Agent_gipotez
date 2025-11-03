from __future__ import annotations

"""
FastAPI entrypoint exposing a single /chat endpoint backed by a LangGraph state machine.
The graph replicates the Telegram flow: preface (step 1), difficulties, discovery Q&A (<=5),
hypotheses generation, and actions (correct/more/agree). Hypotheses saving uses the same path
as Telegram: gipotez/all_gipotez.txt
"""

from typing import Dict, Any, List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import re
import json as _json

from app.llm.client import LLMClient
from app.prompts.core import (
    build_generate_discovery_question_prompt,
    build_generate_free_hypothesis_prompt,
    build_generate_alternative_hypothesis_prompt,
    build_generate_meeting_questions_prompt,
    build_refine_hypotheses_prompt,
)
from app.parsers.docx_parser import parse_docx_to_json

# LangGraph (single-node graph wrapping the request processor)
from langgraph.graph import StateGraph, END

app = FastAPI(title="Agent Gipotez API", version="1.0")


class ChatRequest(BaseModel):
    session_id: str
    user_input: Optional[str] = None
    docs: Optional[List[str]] = None  # raw text chunks to build context
    action: Optional[str] = None      # "correct" | "more" | "agree"
    docs_paths: Optional[List[str]] = None  # server reads files from these paths


class ChatResponse(BaseModel):
    messages: List[str]
    stage: str
    idx: int


class SessionState(BaseModel):
    stage: str = "init"
    idx: int = 0
    qa: List[Dict[str, str]] = []
    context_blob: str = ""
    last_hypotheses: List[Dict[str, Any]] = []
    last_hypotheses_messages: List[str] = []
    goals_text: str = ""
    difficulties_text: str = ""


_SESSIONS: Dict[str, SessionState] = {}
_llm = LLMClient()


def _ensure_session(req: ChatRequest) -> SessionState:
    st = _SESSIONS.get(req.session_id)
    if not st:
        st = SessionState()
        _SESSIONS[req.session_id] = st
    blobs: List[str] = []
    if req.docs:
        blobs.extend([s.strip() for s in req.docs if s and s.strip()])
    if req.docs_paths:
        for p in req.docs_paths:
            try:
                pl = str(p).lower()
                if pl.endswith('.docx'):
                    parsed = parse_docx_to_json(p)
                    # извлекаем текст как в bot.files
                    lines: List[str] = []
                    for sec in (parsed.get('sections') or []):
                        h = sec.get('heading')
                        if h:
                            lines.append(str(h))
                        for t in (sec.get('text') or []):
                            lines.append(str(t))
                    txt = ("\n".join(lines)).strip()
                    if txt:
                        blobs.append(txt)
                elif pl.endswith('.json'):
                    data = _json.loads(Path(p).read_text(encoding='utf-8'))
                    lines: List[str] = []
                    for sec in (data.get('sections') or []):
                        h = sec.get('heading')
                        if h:
                            lines.append(str(h))
                        for t in (sec.get('text') or []):
                            lines.append(str(t))
                    txt = ("\n".join(lines)).strip()
                    if txt:
                        blobs.append(txt)
                else:
                    txt = Path(p).read_text(encoding="utf-8", errors="ignore")
                    if txt.strip():
                        blobs.append(txt.strip())
            except Exception:
                continue
    if blobs:
        st.context_blob = "\n\n".join(blobs)[:50000]
    return st


def _ask_discovery_question(st: SessionState) -> str:
    qa_json = __import__("json").dumps(st.qa, ensure_ascii=False, indent=2)
    examples_text = ""
    prompt = build_generate_discovery_question_prompt(
        qa_json=qa_json,
        dialog_json=st.context_blob,
        examples_text=examples_text,
        avoid=[q.get("question", "") for q in st.qa if q.get("question")],
        count=1,
        theme="",
        avoid_unknown=[],
    )
    try:
        raw = _llm.invoke(prompt, system="Ты отвечаешь строго в формате JSON без лишнего текста.")
        arr = __import__("json").loads(raw)
        qtext = str(arr[0]).strip() if isinstance(arr, list) and arr else "Какие ключевые задачи клиента сейчас наиболее актуальны?"
    except Exception:
        qtext = "Какие ключевые задачи клиента сейчас наиболее актуальны?"
    if not qtext.endswith("?"):
        qtext = qtext.rstrip(". ") + "?"
    st.qa.append({"question": qtext, "answer": ""})
    st.idx += 1
    return f"Вопрос {st.idx}:\n{qtext}"


def _finalize_hypotheses(st: SessionState) -> List[str]:
    msgs: List[str] = []
    qa_json = __import__("json").dumps(st.qa, ensure_ascii=False, indent=2)
    examples_text = ""
    main_prompt = build_generate_free_hypothesis_prompt("", qa_json, st.context_blob, examples_text, prefer_non_finance=True)
    main_raw = _llm.invoke(main_prompt, system="Ты отвечаешь строго в формате JSON без лишнего текста.")
    try:
        main_obj = __import__("json").loads(main_raw)
    except Exception:
        main_obj = {"hypothesis": main_raw.strip(), "reason": ""}
    hypos: List[Dict[str, Any]] = []
    if (main_obj.get("hypothesis") or "").strip():
        hypos.append(main_obj)
    avoid_titles = (main_obj.get("hypothesis") or "").strip()
    for _ in range(2):
        alt_prompt = build_generate_alternative_hypothesis_prompt(qa_json, st.context_blob, examples_text, avoid_hypothesis=avoid_titles, prefer_non_finance=True)
        alt_raw = _llm.invoke(alt_prompt, system="Ты отвечаешь строго в формате JSON без лишнего текста.")
        try:
            alt_obj = __import__("json").loads(alt_raw)
        except Exception:
            alt_obj = {"hypothesis": alt_raw.strip(), "reason": ""}
        title = (alt_obj.get("hypothesis") or "").strip()
        if title:
            hypos.append(alt_obj)
            avoid_titles = f"{avoid_titles}; {title}" if avoid_titles else title

    st.last_hypotheses = hypos[:3]
    st.last_hypotheses_messages = []
    for item in st.last_hypotheses:
        title = (item.get("hypothesis") or "").strip()
        reason = (item.get("reason") or "").strip()
        q_prompt = build_generate_meeting_questions_prompt(title, qa_json, st.context_blob, count=3)
        q_raw = _llm.invoke(q_prompt, system="Ты отвечаешь строго в формате JSON без лишнего текста.")
        try:
            q_arr = __import__("json").loads(q_raw)
        except Exception:
            q_arr = []
        qs_list = [str(q).strip() for q in (q_arr[:3] if isinstance(q_arr, list) else [])]
        bullets = "\n".join([f"- {q}" for q in qs_list]) if qs_list else "- —"
        reason_txt = f"Причина: {reason}" if reason else ""
        block = (f"Гипотеза: {title}\n\n" + reason_txt + ("\n\n" + bullets)).strip()
        st.last_hypotheses_messages.append(block)
        msgs.append(block)
    return msgs


def _save_hypotheses(st: SessionState) -> None:
    Path("gipotez").mkdir(parents=True, exist_ok=True)
    out_path = Path("gipotez") / "all_gipotez.txt"
    with out_path.open("a", encoding="utf-8") as f:
        for block in st.last_hypotheses_messages[:3]:
            clean = re.sub(r"<[^>]+>", "", block)
            f.write(clean.rstrip() + "\n\n")


def _refine_hypotheses(st: SessionState, correction_text: str) -> List[str]:
    msgs: List[str] = []
    if not st.last_hypotheses:
        # fallback: если нет прошлых — просто финализируем заново
        return _finalize_hypotheses(st)
    qa_json = __import__("json").dumps(st.qa, ensure_ascii=False, indent=2)
    examples_text = ""
    hypos_json = __import__("json").dumps(st.last_hypotheses, ensure_ascii=False, indent=2)
    refine_prompt = build_refine_hypotheses_prompt(hypos_json, correction_text, qa_json, st.context_blob, examples_text)
    try:
        refine_raw = _llm.invoke(refine_prompt, system="Ты отвечаешь строго в формате JSON без лишнего текста.")
        refined_arr = __import__("json").loads(refine_raw)
        if isinstance(refined_arr, list) and len(refined_arr) == 3:
            hypos = refined_arr
        else:
            hypos = st.last_hypotheses
    except Exception:
        hypos = st.last_hypotheses

    st.last_hypotheses = hypos[:3]
    st.last_hypotheses_messages = []
    for item in st.last_hypotheses:
        title = (item.get("hypothesis") or "").strip()
        reason = (item.get("reason") or "").strip()
        q_prompt = build_generate_meeting_questions_prompt(title, qa_json, st.context_blob, count=3)
        q_raw = _llm.invoke(q_prompt, system="Ты отвечаешь строго в формате JSON без лишнего текста.")
        try:
            q_arr = __import__("json").loads(q_raw)
        except Exception:
            q_arr = []
        qs_list = [str(q).strip() for q in (q_arr[:3] if isinstance(q_arr, list) else [])]
        bullets = "\n".join([f"- {q}" for q in qs_list]) if qs_list else "- —"
        reason_txt = f"Причина: {reason}" if reason else ""
        block = (f"Гипотеза: {title}\n\n" + reason_txt + ("\n\n" + bullets)).strip()
        st.last_hypotheses_messages.append(block)
        msgs.append(block)
    return msgs


def _process_request(req: ChatRequest) -> ChatResponse:
    st = _ensure_session(req)
    messages: List[str] = []

    # Actions that don't require fresh user text
    if req.action == "agree":
        if not st.last_hypotheses_messages:
            messages.append("Мне нечего сохранить: сначала сформируйте гипотезы.")
        else:
            _save_hypotheses(st)
            messages.append("Согласовано. Сохранила гипотезы и сформирую повестку к встрече.")
        return ChatResponse(messages=messages, stage=st.stage, idx=st.idx)

    # Глобальная обработка корректировок: работает и на стадии final
    if req.action == "correct" and (req.user_input or "").strip():
        if st.last_hypotheses:
            messages.extend(_refine_hypotheses(st, req.user_input or ""))
            st.stage = "final"
            return ChatResponse(messages=messages, stage=st.stage, idx=st.idx)
        if st.stage == "discovery":
            st.qa.append({"question": "Корректировки по гипотезам", "answer": req.user_input or ""})
            messages.extend(_finalize_hypotheses(st))
            st.stage = "final"
            return ChatResponse(messages=messages, stage=st.stage, idx=st.idx)
        messages.append("Сначала сформируйте гипотезы, затем пришлите корректировки.")
        return ChatResponse(messages=messages, stage=st.stage, idx=st.idx)

    if st.stage == "init":
        # Preface messages (Step 1) — simplified: we assume docs already provided via req.docs
        messages.append(
            "Я помогу сформулировать гипотезы о развитии бизнеса клиента. После того, как мы определимся с гипотезами, я сформирую повестку встречи с клиентом."
        )
        messages.append("Шаг 1/3\n\nИз протоколов встреч я выделила следующие договорённости:\n\n—")
        messages.append(
            "Я выбрала ключевые договорённости и цели, предлагаю при проведении встречи сфокусироваться на них:\n\n1. —\n2. —\n3. —\n\nПодтверждаете ли вы мой выбор? С какими целями (долгосрочными или краткосрочными) клиента вы знакомы? О чём сейчас беспокоится ваш клиент?"
        )
        st.stage = "await_goals"
        return ChatResponse(messages=messages, stage=st.stage, idx=st.idx)

    if st.stage == "await_goals" and req.user_input:
        st.goals_text = req.user_input
        messages.append("Есть ли у клиента сложности, которые усложнят выполнение договорённостей или целей?")
        st.stage = "await_difficulties"
        return ChatResponse(messages=messages, stage=st.stage, idx=st.idx)

    if st.stage == "await_difficulties" and req.user_input:
        st.difficulties_text = req.user_input
        # move to discovery
        st.stage = "discovery"
        messages.append("Отлично! Начинаю детальный анализ клиента. Мне нужно задать до 5 вопросов.")
        messages.append(_ask_discovery_question(st))
        return ChatResponse(messages=messages, stage=st.stage, idx=st.idx)

    if st.stage == "discovery" and req.action == "more":
        st.qa = []
        st.idx = 0
        messages.append("Новый раунд.")
        messages.append(_ask_discovery_question(st))
        return ChatResponse(messages=messages, stage=st.stage, idx=st.idx)


    if st.stage == "discovery" and req.user_input is not None:
        # record previous answer
        if st.qa and st.qa[-1].get("answer") == "":
            st.qa[-1]["answer"] = req.user_input
        # next or finalize
        if st.idx < 5:
            messages.append(_ask_discovery_question(st))
            return ChatResponse(messages=messages, stage=st.stage, idx=st.idx)
        else:
            messages.extend(_finalize_hypotheses(st))
            st.stage = "final"
            return ChatResponse(messages=messages, stage=st.stage, idx=st.idx)

    # If already final and user asks for more without explicit action, generate more
    if st.stage == "final" and req.action == "more":
        st.qa = []
        st.idx = 0
        st.stage = "discovery"
        messages.append("Новый раунд.")
        messages.append(_ask_discovery_question(st))
        return ChatResponse(messages=messages, stage=st.stage, idx=st.idx)

    # Default fall-through
    messages.append("Не совсем понял запрос. Продолжим?")
    return ChatResponse(messages=messages, stage=st.stage, idx=st.idx)


# Build a minimal LangGraph to process a request through a single node
def _graph_node(state: Dict[str, Any]) -> Dict[str, Any]:
    req_data = state.get("request") or {}
    req = ChatRequest(**req_data)
    resp = _process_request(req)
    return {"response": resp.model_dump()}


_graph = StateGraph(dict)
_graph.add_node("process", _graph_node)
_graph.set_entry_point("process")
_graph.add_edge("process", END)
_compiled = _graph.compile()


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    result = _compiled.invoke({"request": req.model_dump()})
    data = result.get("response") or {}
    return ChatResponse(**data)


