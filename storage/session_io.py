from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from models.session import Session


SESSION_FILENAME = "session.json"


def save_session(session: Session, path: Optional[Path] = None) -> Path:
    session_dir = path if path else session.session_dir()
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "Recordings").mkdir(parents=True, exist_ok=True)
    data = session.to_dict()
    out_path = session_dir / SESSION_FILENAME
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def load_session(path: Path) -> Session:
    if path.is_dir():
        path = path / SESSION_FILENAME
    data = json.loads(path.read_text(encoding="utf-8"))
    return Session.from_dict(data)


def export_recordings_json(session: Session, out_path: Path) -> None:
    data = [{
        "id": item.id,
        "alias": item.alias,
        "note": item.note,
        "status": item.status.value,
        "wav_path": item.wav_path,
        "duration_sec": item.duration_sec,
    } for item in session.items]
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
