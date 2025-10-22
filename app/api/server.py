from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.llm.client import LLMClient
from app.engine.graph import GraphEngine
from app.parsers.docx_parser import parse_docx_to_json
from app.prompts.core import (
    build_generate_discovery_question_prompt,
    build_decide_next_action_prompt,
    build_generate_free_hypothesis_prompt,
    build_generate_alternative_hypothesis_prompt,
    build_generate_meeting_questions_prompt,
)
from app.services.unknowns import classify_unknown
from app.services.examples import generate_short_example


# Pydantic models MUST be module-level for FastAPI to resolve annotations
class AskResponse(BaseModel):
    question: str
    example: str = ""
    idx: int


class AnswerIn(BaseModel):
    session_id: str
    answer: str


class Session(BaseModel):
    theme: str = ""
    qa: List[Dict[str, str]] = []
    idx: int = 0
    context_docs: List[dict] = []
    unknown_qs: List[str] = []
    avoid_questions_all: List[str] = []
    avoid_hypotheses: List[str] = []
    awaiting_theme: bool = True


class APIApp:
    def __init__(self) -> None:
        self.llm = LLMClient()
        self.engine = GraphEngine(self.llm)
        self.sessions: Dict[str, Session] = {}
        self.dialog_paths: Dict[str, Path] = {}

        self.app = FastAPI(title="Hypothesis Agent API")
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self._register_routes()

    def _invoke_llm(self, prompt: str, system: Optional[str] = None) -> str:
        return self.llm.invoke(prompt, system)

    def _get_session(self, sid: str) -> Session:
        if sid not in self.sessions:
            self.sessions[sid] = Session()
        return self.sessions[sid]

    def _compute_context_blob(self, sess: Session) -> str:
        try:
            data = {"docs": sess.context_docs[-5:]}
            return json.dumps(data, ensure_ascii=False, indent=2)[:8000]
        except Exception:
            return ""

    def _ensure_dialog_log(self, sid: str) -> Path:
        try:
            base = Path("logs/dialogs")
            base.mkdir(parents=True, exist_ok=True)
            p = base / f"{sid}.log"
            if sid not in self.dialog_paths:
                self.dialog_paths[sid] = p
                if not p.exists():
                    p.write_text("", encoding="utf-8")
            return p
        except Exception:
            return Path("/dev/null")

    def _log_dialog(self, sid: str, line: str) -> None:
        try:
            p = self._ensure_dialog_log(sid)
            with p.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    def _register_routes(self) -> None:
        app = self.app

        @app.post("/session/start")
        def start_session(session_id: str = Form(...), theme: str = Form("")):
            sess = self._get_session(session_id)
            sess.theme = theme.strip()
            sess.qa = []
            sess.idx = 0
            sess.unknown_qs = []
            sess.awaiting_theme = (sess.theme == "")
            logging.info("[session] start sid=%s theme=%s", session_id, sess.theme)
            self._log_dialog(session_id, f"== START ==\nTheme: {sess.theme}")
            return {"ok": True}

        @app.post("/session/upload")
        async def upload(session_id: str = Form(...), file: UploadFile = File(...)):
            sess = self._get_session(session_id)
            os_path = Path("uploads")
            os_path.mkdir(parents=True, exist_ok=True)
            dst = os_path / f"{session_id}_{file.filename}"
            content = await file.read()
            dst.write_bytes(content)
            if file.filename.lower().endswith(".docx"):
                parsed = parse_docx_to_json(str(dst))
            elif file.filename.lower().endswith(".json"):
                parsed = json.loads(dst.read_text(encoding="utf-8"))
            else:
                raise HTTPException(400, "Only .docx or .json supported")
            sess.context_docs.append(parsed)
            total = len(sess.context_docs)
            logging.info("[upload] sid=%s file=%s parsed_keys=%s total=%d", session_id, file.filename, list(parsed.keys())[:5] if isinstance(parsed, dict) else type(parsed), total)
            self._log_dialog(session_id, f"[FILE] {file.filename}")
            return {"ok": True, "files": total}

        @app.post("/qa/next", response_model=AskResponse)
        async def next_question(session_id: str = Form(...)):
            sess = self._get_session(session_id)
            # limit 5 questions
            if sess.idx >= 5:
                raise HTTPException(409, "limit reached")
            # theme-first flow
            if sess.awaiting_theme:
                theme_prompt = (
                    "Какая тема вам интересна для анализа состояния клиента?\n\n"
                    "Примеры:\n"
                    "• Состояние относительно конкурентов\n"
                    "• Организационные ситуации\n"
                    "• Операцонные ситуации\n"
                    "• Финансовое положение"
                )
                logging.info("[qa] sid=%s THEME_PROMPT", session_id)
                self._log_dialog(session_id, "THEME?: " + theme_prompt.replace("\n", " "))
                return AskResponse(question=theme_prompt, example="", idx=0)
            try:
                qtext = str(self.engine.next_question(sess).get("question") or "Уточните ключевой аспект по теме?").strip()
            except Exception as e:
                logging.exception("[qa] engine next_question failed: %s", e)
                theme_txt = (sess.theme or "деятельности компании").strip()
                qtext = f"Какие ключевые задачи по теме {theme_txt} сейчас наиболее актуальны?"
            if not qtext.endswith("?"):
                qtext = qtext.rstrip(". ") + "?"
            sess.qa.append({"question": qtext, "answer": ""})
            if qtext not in sess.avoid_questions_all:
                sess.avoid_questions_all.append(qtext)
            qa_json = json.dumps(sess.qa, ensure_ascii=False, indent=2)
            example = await generate_short_example(self._invoke_llm, qtext, qa_json, self._compute_context_blob(sess))
            sess.idx += 1
            logging.info("[qa] sid=%s Q%d: %s", session_id, sess.idx, qtext)
            self._log_dialog(session_id, f"Q{sess.idx}: {qtext}")
            if example:
                self._log_dialog(session_id, f"EXAMPLE: {example}")
            return AskResponse(question=qtext, example=example or "", idx=sess.idx)

        @app.post("/qa/answer")
        async def post_answer(payload: AnswerIn):
            sess = self._get_session(payload.session_id)
            # handle theme first
            if sess.awaiting_theme:
                raw = (payload.answer or "").strip()
                # treat skip
                if raw.lower() in {"", "skip", "пропустить", "не важно", "без темы"}:
                    sess.theme = ""
                else:
                    sess.theme = raw
                sess.awaiting_theme = False
                logging.info("[qa] sid=%s THEME_SET: %s", payload.session_id, sess.theme)
                self._log_dialog(payload.session_id, f"THEME= {sess.theme}")
                return {"next": "ask"}
            if not sess.qa or sess.qa[-1].get("answer"):
                raise HTTPException(400, "no pending question")
            sess.qa[-1]["answer"] = payload.answer or ""
            # track unknowns
            try:
                unknown = await classify_unknown(self._invoke_llm, payload.answer or "", "Возвращай только JSON.")
                if unknown:
                    sess.unknown_qs.append(sess.qa[-1].get("question") or "")
            except Exception:
                pass
            # decide next action
            try:
                act = self.engine.decide_next(sess)
            except Exception as e:
                logging.exception("[qa] engine decide_next failed: %s", e)
                act = "ask"
            logging.info("[qa] sid=%s A: %s | act=%s", payload.session_id, (payload.answer or "").strip()[:80], act)
            self._log_dialog(payload.session_id, f"A{sess.idx}: {payload.answer or ''}")
            self._log_dialog(payload.session_id, f"ACT: {act}")
            if act in {"hypothesis", "stop"} or sess.idx >= 5:
                return {"next": "finalize"}
            return {"next": "ask"}

        @app.post("/finalize")
        def finalize(session_id: str = Form(...)):
            sess = self._get_session(session_id)
            try:
                res = self.engine.finalize(sess)
                # strip tags if LLM returned them
                try:
                    for h in (res.get("hypotheses") or []):
                        if isinstance(h, dict) and "tags" in h:
                            h.pop("tags", None)
                except Exception:
                    pass
            except Exception as e:
                logging.exception("[finalize] engine finalize failed: %s", e)
                # minimal fallback
                theme_txt = (self._get_session(session_id).theme or "клиент").strip()
                res = {
                    "hypotheses": [
                        {"hypothesis": f"Предварительная гипотеза по теме: {theme_txt}", "reason": "Недостаточно данных для точной формулировки"}
                    ],
                    "meeting_questions": [
                        "Какие ключевые метрики вы используете для оценки прогресса?",
                        "Какие ограничения мешают достижению целей сейчас?",
                        "Какие изменения в отрасли сильнее всего на вас влияют?",
                        "Какие гипотезы уже проверялись и с каким результатом?",
                        "Какие ресурсы доступны для быстрых экспериментов?",
                    ],
                }
            unknowns = list(sess.unknown_qs)
            logging.info("[finalize] sid=%s hypos=%s | qs=%d", session_id, [ (h.get('hypothesis') or '') for h in res.get('hypotheses', []) ], len(res.get('meeting_questions', [])))
            res["unknowns"] = unknowns
            # write dialog summary
            self._log_dialog(session_id, "== FINAL ==")
            for i, h in enumerate(res.get("hypotheses", [])[:3], start=1):
                self._log_dialog(session_id, f"H{i}: {(h.get('hypothesis') or '')}")
            for q in res.get("meeting_questions", [])[:5]:
                self._log_dialog(session_id, f"MEET_Q: {q}")
            if unknowns:
                self._log_dialog(session_id, "UNKNOWN:")
                for u in unknowns:
                    self._log_dialog(session_id, f"- {u}")
            return res


def create_app() -> FastAPI:
    return APIApp().app


