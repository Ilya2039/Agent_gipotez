from __future__ import annotations

from typing import Dict, Any, List, Optional, Sequence
from pydantic import BaseModel
from pathlib import Path
import re
import json

from app.llm.client import LLMClient
from app.prompts.core import (
    build_generate_discovery_question_prompt,
    build_generate_free_hypothesis_prompt,
    build_generate_alternative_hypothesis_prompt,
    build_generate_meeting_questions_prompt,
    build_refine_hypotheses_prompt,
)
from app.parsers.docx_parser import parse_docx_to_json

from langgraph.graph import StateGraph, END


SYSTEM_STRICT_JSON = (
    "Ты отвечаешь строго в формате JSON без лишнего текста. Не добавляй пояснения, дисклеймеры, списки, markdown."
)
MAX_CONTEXT_CHARS = 50000
MAX_DISCOVERY_QUESTIONS = 5


class ChatRequest(BaseModel):
    session_id: str
    user_input: Optional[str] = None
    docs: Optional[List[str]] = None
    action: Optional[str] = None  # "correct" | "more" | "agree"
    docs_paths: Optional[List[str]] = None


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


def _extract_text_from_sections(data: Dict[str, Any]) -> str:
    lines: List[str] = []
    for sec in (data.get("sections") or []):
        heading = sec.get("heading")
        if heading:
            lines.append(str(heading))
        for paragraph in (sec.get("text") or []):
            lines.append(str(paragraph))
    return ("\n".join(lines)).strip()


def _build_context_from_paths(paths: Sequence[str]) -> str:
    blobs: List[str] = []
    for p in paths:
        try:
            lower = str(p).lower()
            if lower.endswith(".docx"):
                parsed = parse_docx_to_json(p)
                txt = _extract_text_from_sections(parsed)
                if txt:
                    blobs.append(txt)
            elif lower.endswith(".json"):
                data = json.loads(Path(p).read_text(encoding="utf-8"))
                txt = _extract_text_from_sections(data)
                if txt:
                    blobs.append(txt)
            else:
                txt = Path(p).read_text(encoding="utf-8", errors="ignore").strip()
                if txt:
                    blobs.append(txt)
        except Exception:
            continue
    return ("\n\n".join(blobs))[:MAX_CONTEXT_CHARS] if blobs else ""


class DialogEngine:
    def __init__(self, llm: Optional[LLMClient] = None) -> None:
        self.llm = llm or LLMClient()
        self.sessions: Dict[str, SessionState] = {}
        self._compiled = self._build_graph()

    def _build_graph(self):
        def node(state: Dict[str, Any]) -> Dict[str, Any]:
            req = ChatRequest(**(state.get("request") or {}))
            resp = self._process(req)
            return {"response": resp.model_dump()}

        graph = StateGraph(dict)
        graph.add_node("process", node)
        graph.set_entry_point("process")
        graph.add_edge("process", END)
        return graph.compile()

    def invoke(self, req: ChatRequest) -> ChatResponse:
        result = self._compiled.invoke({"request": req.model_dump()})
        data = result.get("response") or {}
        return ChatResponse(**data)

    def _get_or_create_session(self, req: ChatRequest) -> SessionState:
        state = self.sessions.get(req.session_id)
        if not state:
            state = SessionState()
            self.sessions[req.session_id] = state
        context_parts: List[str] = []
        if req.docs:
            context_parts.extend([s.strip() for s in req.docs if s and s.strip()])
        if req.docs_paths:
            ctx = _build_context_from_paths(req.docs_paths)
            if ctx:
                context_parts.append(ctx)
        if context_parts:
            state.context_blob = ("\n\n".join(context_parts))[:MAX_CONTEXT_CHARS]
        return state

    def _ask_discovery_question(self, st: SessionState) -> str:
        qa_json = json.dumps(st.qa, ensure_ascii=False, indent=2)
        prompt = build_generate_discovery_question_prompt(
            qa_json=qa_json,
            dialog_json=st.context_blob,
            examples_text="",
            avoid=[q.get("question", "") for q in st.qa if q.get("question")],
            count=1,
            theme="",
            avoid_unknown=[],
        )
        try:
            raw = self.llm.invoke(prompt, system=SYSTEM_STRICT_JSON)
            arr = json.loads(raw)
            question = str(arr[0]).strip() if isinstance(arr, list) and arr else "Какие ключевые задачи клиента сейчас наиболее актуальны?"
        except Exception:
            question = "Какие ключевые задачи клиента сейчас наиболее актуальны?"
        if not question.endswith("?"):
            question = question.rstrip(". ") + "?"
        st.qa.append({"question": question, "answer": ""})
        st.idx += 1
        return f"Вопрос {st.idx}:\n{question}"

    def _generate_hypotheses_blocks(self, st: SessionState) -> List[str]:
        qa_json = json.dumps(st.qa, ensure_ascii=False, indent=2)
        blocks: List[str] = []

        main_prompt = build_generate_free_hypothesis_prompt("", qa_json, st.context_blob, "", prefer_non_finance=True)
        main_raw = self.llm.invoke(main_prompt, system=SYSTEM_STRICT_JSON)
        try:
            main_obj = json.loads(main_raw)
        except Exception:
            main_obj = {"hypothesis": main_raw.strip(), "reason": ""}

        hypos: List[Dict[str, Any]] = []
        if (main_obj.get("hypothesis") or "").strip():
            hypos.append(main_obj)
        avoid_titles = (main_obj.get("hypothesis") or "").strip()

        for _ in range(2):
            alt_prompt = build_generate_alternative_hypothesis_prompt(qa_json, st.context_blob, "", avoid_hypothesis=avoid_titles, prefer_non_finance=True)
            alt_raw = self.llm.invoke(alt_prompt, system=SYSTEM_STRICT_JSON)
            try:
                alt_obj = json.loads(alt_raw)
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
            q_raw = self.llm.invoke(q_prompt, system=SYSTEM_STRICT_JSON)
            try:
                q_arr = json.loads(q_raw)
            except Exception:
                q_arr = []
            meeting_questions = [str(q).strip() for q in (q_arr[:3] if isinstance(q_arr, list) else [])]
            bullets = "\n".join([f"- {q}" for q in meeting_questions]) if meeting_questions else "- —"
            reason_txt = f"Причина: {reason}" if reason else ""
            block = (f"Гипотеза: {title}\n\n" + reason_txt + ("\n\n" + bullets)).strip()
            st.last_hypotheses_messages.append(block)
            blocks.append(block)
        return blocks

    def _refine_hypotheses(self, st: SessionState, correction_text: str) -> List[str]:
        if not st.last_hypotheses:
            return self._generate_hypotheses_blocks(st)
        qa_json = json.dumps(st.qa, ensure_ascii=False, indent=2)
        hypos_json = json.dumps(st.last_hypotheses, ensure_ascii=False, indent=2)
        refine_prompt = build_refine_hypotheses_prompt(hypos_json, correction_text, qa_json, st.context_blob, "")
        try:
            refine_raw = self.llm.invoke(refine_prompt, system=SYSTEM_STRICT_JSON)
            refined_arr = json.loads(refine_raw)
            hypos = refined_arr if isinstance(refined_arr, list) and len(refined_arr) == 3 else st.last_hypotheses
        except Exception:
            hypos = st.last_hypotheses

        st.last_hypotheses = hypos[:3]
        st.last_hypotheses_messages = []
        blocks: List[str] = []
        for item in st.last_hypotheses:
            title = (item.get("hypothesis") or "").strip()
            reason = (item.get("reason") or "").strip()
            q_prompt = build_generate_meeting_questions_prompt(title, qa_json, st.context_blob, count=3)
            q_raw = self.llm.invoke(q_prompt, system=SYSTEM_STRICT_JSON)
            try:
                q_arr = json.loads(q_raw)
            except Exception:
                q_arr = []
            meeting_questions = [str(q).strip() for q in (q_arr[:3] if isinstance(q_arr, list) else [])]
            bullets = "\n".join([f"- {q}" for q in meeting_questions]) if meeting_questions else "- —"
            reason_txt = f"Причина: {reason}" if reason else ""
            block = (f"Гипотеза: {title}\n\n" + reason_txt + ("\n\n" + bullets)).strip()
            st.last_hypotheses_messages.append(block)
            blocks.append(block)
        return blocks

    def _save_hypotheses(self, st: SessionState) -> None:
        Path("gipotez").mkdir(parents=True, exist_ok=True)
        out_path = Path("gipotez") / "all_gipotez.txt"
        with out_path.open("a", encoding="utf-8") as f:
            for block in st.last_hypotheses_messages[:3]:
                clean = re.sub(r"<[^>]+>", "", block)
                f.write(clean.rstrip() + "\n\n")

    def _process(self, req: ChatRequest) -> ChatResponse:
        st = self._get_or_create_session(req)
        messages: List[str] = []

        if req.action == "agree":
            if not st.last_hypotheses_messages:
                messages.append("Мне нечего сохранить: сначала сформируйте гипотезы.")
            else:
                self._save_hypotheses(st)
                messages.append("Согласовано. Сохранила гипотезы и сформирую повестку к встрече.")
            return ChatResponse(messages=messages, stage=st.stage, idx=st.idx)

        if req.action == "correct" and (req.user_input or "").strip():
            if st.last_hypotheses:
                messages.extend(self._refine_hypotheses(st, req.user_input or ""))
                st.stage = "final"
                return ChatResponse(messages=messages, stage=st.stage, idx=st.idx)
            if st.stage == "discovery":
                st.qa.append({"question": "Корректировки по гипотезам", "answer": req.user_input or ""})
                messages.extend(self._generate_hypotheses_blocks(st))
                st.stage = "final"
                return ChatResponse(messages=messages, stage=st.stage, idx=st.idx)
            messages.append("Сначала сформируйте гипотезы, затем пришлите корректировки.")
            return ChatResponse(messages=messages, stage=st.stage, idx=st.idx)

        if st.stage == "init":
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
            st.stage = "discovery"
            messages.append("Отлично! Начинаю детальный анализ клиента. Мне нужно задать до 5 вопросов.")
            messages.append(self._ask_discovery_question(st))
            return ChatResponse(messages=messages, stage=st.stage, idx=st.idx)

        if st.stage == "discovery" and req.action == "more":
            st.qa = []
            st.idx = 0
            messages.append("Новый раунд.")
            messages.append(self._ask_discovery_question(st))
            return ChatResponse(messages=messages, stage=st.stage, idx=st.idx)

        if st.stage == "discovery" and req.user_input is not None:
            if st.qa and st.qa[-1].get("answer") == "":
                st.qa[-1]["answer"] = req.user_input
            if st.idx < MAX_DISCOVERY_QUESTIONS:
                messages.append(self._ask_discovery_question(st))
                return ChatResponse(messages=messages, stage=st.stage, idx=st.idx)
            messages.extend(self._generate_hypotheses_blocks(st))
            st.stage = "final"
            return ChatResponse(messages=messages, stage=st.stage, idx=st.idx)

        if st.stage == "final" and req.action == "more":
            st.qa = []
            st.idx = 0
            st.stage = "discovery"
            messages.append("Новый раунд.")
            messages.append(self._ask_discovery_question(st))
            return ChatResponse(messages=messages, stage=st.stage, idx=st.idx)

        messages.append("Не совсем понял запрос. Продолжим?")
        return ChatResponse(messages=messages, stage=st.stage, idx=st.idx)


# Singleton engine and public invoke API
ENGINE = DialogEngine()


def invoke(req: ChatRequest) -> ChatResponse:
    return ENGINE.invoke(req)


