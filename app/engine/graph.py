from __future__ import annotations

import json
from typing import TypedDict, List, Dict, Optional

from langgraph.graph import StateGraph, START, END

from app.prompts.core import (
    build_generate_discovery_question_prompt,
    build_decide_next_action_prompt,
    build_generate_free_hypothesis_prompt,
    build_generate_alternative_hypothesis_prompt,
    build_generate_meeting_questions_prompt,
)


class SessionSnapshot(TypedDict, total=False):
    theme: str
    qa: List[Dict[str, str]]
    idx: int
    context_json: str
    examples_text: str
    unknown_qs: List[str]
    avoid_questions_all: List[str]
    action: str
    question: str
    example: str
    hypotheses: List[Dict[str, object]]
    meeting_questions: List[str]


class GraphEngine:
    def __init__(self, llm_client) -> None:
        self._llm = llm_client
        self._strict_json = (
            "Ты отвечаешь строго в формате JSON без лишнего текста. "
            "Не добавляй пояснения, дисклеймеры, списки, markdown. Возвращай только валидный JSON."
        )
        self._graph = self._build_graph()

    def _invoke_llm(self, prompt: str, system: Optional[str] = None) -> str:
        return self._llm.invoke(prompt, system)

    def _build_graph(self):
        g = StateGraph(SessionSnapshot)

        def decide(state: SessionSnapshot) -> SessionSnapshot:
            qa_json = json.dumps(state.get("qa") or [], ensure_ascii=False, indent=2)
            prompt = build_decide_next_action_prompt(
                state.get("theme", ""), qa_json, state.get("context_json", ""), prefer_non_finance=True
            )
            raw = self._invoke_llm(prompt, system=self._strict_json)
            try:
                obj = json.loads(raw)
            except Exception:
                obj = {}
            state["action"] = (obj.get("action") or "ask").strip()
            qtext = (obj.get("question") or "").strip()
            if qtext and not qtext.endswith("?"):
                qtext = qtext.rstrip(". ") + "?"
            if qtext:
                state["question"] = qtext
            return state

        def ask(state: SessionSnapshot) -> SessionSnapshot:
            # If question already proposed by decide, reuse; else generate discovery question
            qtext = (state.get("question") or "").strip()
            if not qtext:
                qa_json = json.dumps(state.get("qa") or [], ensure_ascii=False, indent=2)
                avoid = (state.get("avoid_questions_all") or [])
                unknowns = (state.get("unknown_qs") or [])
                prompt = build_generate_discovery_question_prompt(
                    qa_json=qa_json,
                    dialog_json=state.get("context_json", ""),
                    examples_text=state.get("examples_text", ""),
                    avoid=avoid,
                    count=1,
                    theme=state.get("theme", ""),
                    avoid_unknown=unknowns,
                )
                raw = self._invoke_llm(prompt, system=self._strict_json)
                try:
                    arr = json.loads(raw)
                except Exception:
                    arr = []
                qtext = (str(arr[0]).strip() if arr else "Уточните ключевой аспект по теме?")
                if not qtext.endswith("?"):
                    qtext = qtext.rstrip(". ") + "?"
            state["question"] = qtext
            return state

        def finalize(state: SessionSnapshot) -> SessionSnapshot:
            qa_json = json.dumps(state.get("qa") or [], ensure_ascii=False, indent=2)
            dialog_blob = state.get("context_json", "")
            examples_text = state.get("examples_text", "")
            # build 3 hypos
            main_raw = self._invoke_llm(
                build_generate_free_hypothesis_prompt(state.get("theme", ""), qa_json, dialog_blob, examples_text, prefer_non_finance=True),
                system=self._strict_json,
            )
            try:
                main_obj = json.loads(main_raw)
            except Exception:
                main_obj = {"hypothesis": main_raw.strip(), "reason": ""}
            hypos: List[Dict[str, object]] = []
            if (main_obj.get("hypothesis") or "").strip():
                hypos.append(main_obj)
            avoid_title = (main_obj.get("hypothesis") or "")
            for _ in range(2):
                alt_raw = self._invoke_llm(
                    build_generate_alternative_hypothesis_prompt(
                        qa_json, dialog_blob, examples_text, avoid_hypothesis=avoid_title, prefer_non_finance=True
                    ),
                    system=self._strict_json,
                )
                try:
                    alt_obj = json.loads(alt_raw)
                except Exception:
                    alt_obj = {"hypothesis": alt_raw.strip(), "reason": ""}
                title = (alt_obj.get("hypothesis") or "").strip()
                if title:
                    hypos.append(alt_obj)
                    avoid_title = f"{avoid_title}; {title}" if avoid_title else title
            state["hypotheses"] = hypos[:3]
            # shared 5 questions
            combined_title = "Гипотезы:\n" + "\n".join(
                [f"{i+1}) {(h.get('hypothesis') or '').strip()}" for i, h in enumerate(hypos[:3])]
            )
            q_raw = self._invoke_llm(
                build_generate_meeting_questions_prompt(combined_title, qa_json, dialog_blob, count=5),
                system=self._strict_json,
            )
            try:
                q_arr = json.loads(q_raw)
            except Exception:
                q_arr = []
            state["meeting_questions"] = (q_arr[:5] if isinstance(q_arr, list) else [])
            return state

        g.add_node("decide", decide)
        g.add_node("ask", ask)
        g.add_node("finalize", finalize)
        g.add_edge(START, "decide")

        def router(state: SessionSnapshot) -> str:
            act = (state.get("action") or "ask").lower()
            if act in {"hypothesis", "stop"}:
                return "finalize"
            # treat request_file as ask in our product logic
            return "ask"

        g.add_conditional_edges("decide", router, {"ask": "ask", "finalize": "finalize"})
        g.add_edge("ask", END)
        g.add_edge("finalize", END)
        return g.compile()

    # Convenience wrappers used by REST API
    def next_question(self, sess) -> Dict[str, object]:
        state: SessionSnapshot = {
            "theme": sess.theme,
            "qa": sess.qa,
            "idx": sess.idx,
            "context_json": self._compute_context(sess),
            "examples_text": self._examples_text(),
            "unknown_qs": sess.unknown_qs,
            "avoid_questions_all": sess.avoid_questions_all,
        }
        out = self._graph.invoke(state)
        question = (out.get("question") or "").strip()
        return {"question": question}

    def decide_next(self, sess) -> str:
        state: SessionSnapshot = {
            "theme": sess.theme,
            "qa": sess.qa,
            "idx": sess.idx,
            "context_json": self._compute_context(sess),
        }
        out = self._graph.invoke(state)
        return (out.get("action") or "ask").lower()

    def finalize(self, sess) -> Dict[str, object]:
        # Compute finalization directly to avoid routing through decide node
        qa_json = json.dumps(sess.qa or [], ensure_ascii=False, indent=2)
        dialog_blob = self._compute_context(sess)
        examples_text = self._examples_text()
        # 3 hypotheses (1 main + 2 alternatives)
        main_raw = self._invoke_llm(
            build_generate_free_hypothesis_prompt(sess.theme or "", qa_json, dialog_blob, examples_text, prefer_non_finance=True),
            system=self._strict_json,
        )
        try:
            main_obj = json.loads(main_raw)
        except Exception:
            main_obj = {"hypothesis": (main_raw or "").strip(), "reason": ""}
        hypos: List[Dict[str, object]] = []
        if (main_obj.get("hypothesis") or "").strip():
            hypos.append(main_obj)
        avoid_title = (main_obj.get("hypothesis") or "")
        for _ in range(2):
            alt_raw = self._invoke_llm(
                build_generate_alternative_hypothesis_prompt(qa_json, dialog_blob, examples_text, avoid_hypothesis=avoid_title, prefer_non_finance=True),
                system=self._strict_json,
            )
            try:
                alt_obj = json.loads(alt_raw)
            except Exception:
                alt_obj = {"hypothesis": (alt_raw or "").strip(), "reason": ""}
            title = (alt_obj.get("hypothesis") or "").strip()
            if title:
                hypos.append(alt_obj)
                avoid_title = f"{avoid_title}; {title}" if avoid_title else title
        # Shared 5 meeting questions across 3 hypos
        combined_title = "Гипотезы:\n" + "\n".join([f"{i+1}) {(h.get('hypothesis') or '').strip()}" for i, h in enumerate(hypos[:3])])
        q_raw = self._invoke_llm(
            build_generate_meeting_questions_prompt(combined_title, qa_json, dialog_blob, count=5),
            system=self._strict_json,
        )
        try:
            q_arr = json.loads(q_raw)
        except Exception:
            q_arr = []
        return {"hypotheses": hypos[:3], "meeting_questions": (q_arr[:5] if isinstance(q_arr, list) else [])}

    def _compute_context(self, sess) -> str:
        import json as _json
        try:
            data = {"docs": sess.context_docs[-5:]}
            return _json.dumps(data, ensure_ascii=False, indent=2)[:8000]
        except Exception:
            return ""

    def _examples_text(self) -> str:
        from pathlib import Path
        p = Path("data/hypotheses.txt")
        return p.read_text(encoding="utf-8") if p.exists() else ""


