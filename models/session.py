from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import List, Optional
import uuid


class ItemStatus(str, Enum):
    PENDING = "pending"
    RECORDED = "recorded"
    VERIFIED = "verified"


@dataclass
class Item:
    id: str
    alias: str
    note: Optional[str] = None
    romaji: Optional[str] = None
    status: ItemStatus = ItemStatus.PENDING
    wav_path: Optional[str] = None
    notes: str = ""
    duration_sec: Optional[float] = None

    @staticmethod
    def new(alias: str, note: Optional[str] = None) -> "Item":
        return Item(id=str(uuid.uuid4())[:8], alias=alias, note=note)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @staticmethod
    def from_dict(data: dict) -> "Item":
        return Item(
            id=data["id"],
            alias=data["alias"],
            note=data.get("note"),
            romaji=data.get("romaji"),
            status=ItemStatus(data.get("status", ItemStatus.PENDING.value)),
            wav_path=data.get("wav_path"),
            notes=data.get("notes", ""),
            duration_sec=data.get("duration_sec"),
        )


@dataclass
class Session:
    name: str
    singer: str
    base_path: Path
    sample_rate: int = 44100
    bit_depth: int = 16
    channels: int = 1
    items: List[Item] = field(default_factory=list)
    voicebank_path: Optional[Path] = None
    voicebank_paths: List[str] = field(default_factory=list)
    voicebank_prefix: str = ""
    voicebank_suffix: str = ""
    voicebank_use_bgm: bool = False
    output_prefix: str = ""
    output_suffix: str = ""
    bgm_wav_path: Optional[str] = None
    bgm_note: Optional[str] = None
    bgm_overlay_enabled: bool = False
    bgm_overlay_note: Optional[str] = None
    bgm_overlay_duration: Optional[float] = None
    bgm_override: bool = False

    def session_dir(self) -> Path:
        return self.base_path

    def recordings_dir(self) -> Path:
        return self.session_dir() / "Recordings"

    def add_item(self, alias: str, note: Optional[str] = None, romaji: Optional[str] = None) -> Item:
        item = Item.new(alias, note)
        item.romaji = romaji
        self.items.append(item)
        return item

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "singer": self.singer,
            "base_path": str(self.base_path),
            "sample_rate": self.sample_rate,
            "bit_depth": self.bit_depth,
            "channels": self.channels,
            "voicebank_path": str(self.voicebank_path) if self.voicebank_path else None,
            "voicebank_paths": self.voicebank_paths,
            "voicebank_prefix": self.voicebank_prefix,
            "voicebank_suffix": self.voicebank_suffix,
            "voicebank_use_bgm": self.voicebank_use_bgm,
            "output_prefix": self.output_prefix,
            "output_suffix": self.output_suffix,
            "bgm_wav_path": self.bgm_wav_path,
            "bgm_note": self.bgm_note,
            "bgm_overlay_enabled": self.bgm_overlay_enabled,
            "bgm_overlay_note": self.bgm_overlay_note,
            "bgm_overlay_duration": self.bgm_overlay_duration,
            "bgm_override": self.bgm_override,
            "items": [item.to_dict() for item in self.items],
        }

    @staticmethod
    def from_dict(data: dict) -> "Session":
        session = Session(
            name=data["name"],
            singer=data.get("singer", ""),
            base_path=Path(data["base_path"]),
            sample_rate=data.get("sample_rate", 44100),
            bit_depth=data.get("bit_depth", 16),
            channels=data.get("channels", 1),
            voicebank_path=Path(data["voicebank_path"]) if data.get("voicebank_path") else None,
            voicebank_paths=data.get("voicebank_paths", []),
            voicebank_prefix=data.get("voicebank_prefix", ""),
            voicebank_suffix=data.get("voicebank_suffix", ""),
            voicebank_use_bgm=data.get("voicebank_use_bgm", False),
            output_prefix=data.get("output_prefix", ""),
            output_suffix=data.get("output_suffix", ""),
            bgm_wav_path=data.get("bgm_wav_path"),
            bgm_note=data.get("bgm_note"),
            bgm_overlay_enabled=data.get("bgm_overlay_enabled", False),
            bgm_overlay_note=data.get("bgm_overlay_note"),
            bgm_overlay_duration=data.get("bgm_overlay_duration"),
            bgm_override=data.get("bgm_override", False),
        )
        session.items = [Item.from_dict(item) for item in data.get("items", [])]
        return session
