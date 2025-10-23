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
    last_hypotheses: List[str] = []


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

    async def _invoke_llm(self, prompt: str, system: Optional[str] = None) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.llm.invoke, prompt, system)

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
        def start_session(session_id: str = Form(...)):
            sess = self._get_session(session_id)
            sess.theme = ""
            sess.qa = []
            sess.idx = 0
            sess.unknown_qs = []
            sess.awaiting_theme = True
            logging.info("[session] start sid=%s", session_id)
            self._log_dialog(session_id, "== START ==")
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

        @app.post("/session/theme")
        def set_theme(session_id: str = Form(...), theme: str = Form("")):
            sess = self._get_session(session_id)
            sess.theme = (theme or "").strip()
            sess.awaiting_theme = False
            logging.info("[session] theme sid=%s -> %s", session_id, sess.theme)
            self._log_dialog(session_id, f"THEME= {sess.theme}")
            return {"ok": True, "theme": sess.theme}

        @app.post("/qa/next", response_model=AskResponse)
        async def next_question(session_id: str = Form(...)):
            sess = self._get_session(session_id)
            # максимум 5 вопросов
            if sess.idx >= 5:
                raise HTTPException(409, "limit reached")
            # если тема не задана и пользователь не вызывал /session/theme
            if sess.awaiting_theme:
                theme_prompt = (
                    "Какая тема вам интересна для анализа состояния клиента?\n\n"
                    "Примеры:\n"
                    "• Доля на рынке\n"
                    "• Организационные ситуации\n"
                    "• Операцонные ситуации\n"
                    "• Финансовое положение\n"
                    "• Состояние относительно конкурентов"
                )
                logging.info("[qa] sid=%s THEME_PROMPT", session_id)
                self._log_dialog(session_id, "THEME?: " + theme_prompt.replace("\n", " "))
                return AskResponse(question=theme_prompt, example="", idx=0)
            # генерируем уточняющий вопрос напрямую через промпт
            asked = [it.get("question", "") for it in sess.qa if it.get("question")]
            qa_json = json.dumps(sess.qa, ensure_ascii=False, indent=2)
            dialog_blob = self._compute_context_blob(sess)
            examples_text = Path("data/hypotheses.txt").read_text(encoding="utf-8") if Path("data/hypotheses.txt").exists() else ""
            prompt = build_generate_discovery_question_prompt(
                qa_json=qa_json,
                dialog_json=dialog_blob,
                examples_text=examples_text,
                avoid=list(set(asked + (sess.avoid_questions_all or []))),
                count=1,
                theme=sess.theme,
                avoid_unknown=sess.unknown_qs,
            )
            try:
                raw = await self._invoke_llm(prompt, system=("Ты отвечаешь строго JSON массивом строк."))
                logging.info("[qa] sid=%s GEN_Q_PROMPT=%s", session_id, prompt.replace("\n", " "))
                logging.info("[qa] sid=%s GEN_Q_RAW=%s", session_id, (raw or "").strip())
                arr = json.loads(raw)
            except Exception as e:
                logging.exception("[qa] discovery prompt failed: %s", e)
                arr = []
            if not isinstance(arr, list) or not arr:
                theme_txt = (sess.theme or "деятельности компании").strip()
                qtext = f"Какие ключевые задачи по теме {theme_txt} сейчас наиболее актуальны?"
            else:
                qtext = str(arr[0]).strip()
            if not qtext.endswith("?"):
                qtext = qtext.rstrip(". ") + "?"
            # жёстко избегаем дубликатов, если LLM проигнорировал avoid
            avoid_all = set((sess.avoid_questions_all or []) + [q.get("question", "") for q in sess.qa])
            if qtext in avoid_all:
                theme_txt = (sess.theme or "текущей теме").strip()
                qtext = f"Назовите другой аспект по {theme_txt}, который мы ещё не затрагивали?"
            sess.qa.append({"question": qtext, "answer": ""})
            if qtext not in sess.avoid_questions_all:
                sess.avoid_questions_all.append(qtext)
            qa_json = json.dumps(sess.qa, ensure_ascii=False, indent=2)
            example = await generate_short_example(self._invoke_llm, qtext, qa_json, self._compute_context_blob(sess))
            sess.idx += 1
            logging.info("[qa] sid=%s Q%d: %s", session_id, sess.idx, qtext)
            self._log_dialog(session_id, f"Q{sess.idx}: {qtext}")
            if example:
                logging.info("[qa] sid=%s EXAMPLE: %s", session_id, example)
                self._log_dialog(session_id, f"EXAMPLE: {example}")
            return AskResponse(question=qtext, example=example or "", idx=sess.idx)

        @app.post("/qa/answer")
        async def post_answer(payload: AnswerIn):
            sess = self._get_session(payload.session_id)
            # сначала обрабатываем тему
            if sess.awaiting_theme:
                raw = (payload.answer or "").strip()
                # пропуск темы
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
            # определяем 'unknown' промптом LLM (да/нет)
            try:
                unknown = await classify_unknown(
                    self._invoke_llm,
                    payload.answer or "",
                    "Ответь одним словом: 'да' если ответ означает 'не знаю/нет данных', иначе 'нет'.",
                )
                logging.info("[qa] sid=%s UNKNOWN_DECISION=%s | answer=%.120s | question=%s", payload.session_id, unknown, (payload.answer or ""), (sess.qa[-1].get("question") or ""))
                self._log_dialog(payload.session_id, f"UNKNOWN_DECISION={unknown} :: Q={(sess.qa[-1].get('question') or '').strip()}")
                if unknown:
                    sess.unknown_qs.append(sess.qa[-1].get("question") or "")
            except Exception:
                logging.exception("[qa] unknown classification failed")
            # решаем следующий шаг
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
                # убираем tags, если модель их вернула
                try:
                    for h in (res.get("hypotheses") or []):
                        if isinstance(h, dict) and "tags" in h:
                            h.pop("tags", None)
                except Exception:
                    pass
            except Exception as e:
                logging.exception("[finalize] engine finalize failed: %s", e)
                # минимальный фолбэк
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
            # запоминаем последние гипотезы для избегания повторов
            try:
                sess.last_hypotheses = [ (h.get("hypothesis") or "").strip() for h in (res.get("hypotheses") or []) ]
            except Exception:
                sess.last_hypotheses = []
            # итог диалога в лог
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

        @app.post("/alt/start")
        def alt_start(session_id: str = Form(...)):
            sess = self._get_session(session_id)
            # добавляем последние гипотезы в список avoid
            merged = set(sess.avoid_hypotheses or []) | set(sess.last_hypotheses or [])
            sess.avoid_hypotheses = sorted({h for h in merged if h})
            # сбрасываем состояние раунда
            sess.qa = []
            sess.idx = 0
            sess.unknown_qs = []
            logging.info("[alt] start sid=%s avoids=%d", session_id, len(sess.avoid_hypotheses))
            self._log_dialog(session_id, "== ALT ROUND ==")
            return {"ok": True, "avoids": len(sess.avoid_hypotheses)}


def create_app() -> FastAPI:
    return APIApp().app


