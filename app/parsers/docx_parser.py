from __future__ import annotations

from typing import Any, Dict, List

from docx import Document  # type: ignore


def parse_docx_to_json(path: str) -> Dict[str, Any]:
    doc = Document(path)

    sections: List[Dict[str, Any]] = []
    current: Dict[str, Any] | None = None

    def push_current() -> None:
        nonlocal current
        if current:
            # drop empty
            if current.get("text") or current.get("tables"):
                sections.append(current)
        current = None

    for p in doc.paragraphs:
        text = (p.text or "").strip()
        if not text:
            continue
        style = (getattr(p.style, "name", "") or "").lower()
        if "heading" in style or style.startswith("заголовок"):
            push_current()
            current = {"heading": text, "text": [], "tables": []}
        else:
            if current is None:
                current = {"heading": None, "text": [], "tables": []}
            current["text"].append(text)

    # tables captured after paragraphs to preserve order loosely
    # We do not merge table positions with paragraphs, only attach to nearest current
    for t in doc.tables:
        rows: List[List[str]] = []
        for row in t.rows:
            rows.append([cell.text.strip() for cell in row.cells])
        if current is None:
            current = {"heading": None, "text": [], "tables": []}
        current["tables"].append({"rows": rows})

    push_current()

    meta = {
        "paragraph_count": len(doc.paragraphs),
        "table_count": len(doc.tables),
    }

    return {"meta": meta, "sections": sections}



