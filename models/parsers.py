from __future__ import annotations

from typing import List, Optional, Tuple
from pathlib import Path


def parse_reclist_line(line: str) -> Optional[Tuple[str, Optional[str]]]:
    raw = line.strip()
    if not raw:
        return None
    if raw.startswith("#") or raw.startswith(";"):
        return None

    if "\t" in raw:
        parts = [p.strip() for p in raw.split("\t") if p.strip()]
    elif "," in raw:
        parts = [p.strip() for p in raw.split(",") if p.strip()]
    else:
        parts = [raw]

    if not parts:
        return None

    alias = parts[0]
    note = parts[1] if len(parts) > 1 else None
    return alias, note


def parse_reclist_text(text: str) -> List[Tuple[str, Optional[str]]]:
    items: List[Tuple[str, Optional[str]]] = []
    for line in text.splitlines():
        parsed = parse_reclist_line(line)
        if parsed:
            items.append(parsed)
    return items


def read_text_guess(path: Path) -> str:
    encodings = ["utf-8-sig", "utf-8", "cp932", "shift_jis"]
    for enc in encodings:
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")
