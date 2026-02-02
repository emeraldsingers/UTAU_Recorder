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
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        try:
            return raw.decode("utf-16")
        except UnicodeDecodeError:
            pass
    if raw.startswith(b"\xef\xbb\xbf"):
        try:
            return raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            pass

    encodings = [
        "utf-8",
        "utf-8-sig",
        "utf-16",
        "utf-16-le",
        "utf-16-be",
        "cp932",
        "shift_jis",
        "euc_jp",
        "gbk",
        "cp936",
        "big5",
        "cp950",
        "euc_kr",
    ]
    for enc in encodings:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")
