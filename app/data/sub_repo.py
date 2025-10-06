from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, TypedDict


DATA_PATH = Path("data/subhypotheses.json")


class SubItem(TypedDict):
    title: str
    questions: List[str]


class SubRepo:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DATA_PATH
        self._data: Dict[str, List[SubItem]] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self._data = {}

    def get_subs(self, main_title: str) -> List[SubItem]:
        # exact
        if main_title in self._data:
            return self._data[main_title]
        # normalized
        mt = self._norm(main_title)
        for k, v in self._data.items():
            if self._norm(k) == mt:
                return v
        return []

    def list_main_titles(self) -> List[str]:
        return list(self._data.keys())

    def _norm(self, s: str) -> str:
        import re

        t = s.strip().lower().replace("ё", "е").replace("—", "-")
        t = re.sub(r"\s+", " ", t)
        return t


