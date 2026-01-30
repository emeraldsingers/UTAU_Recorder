from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from models.parsers import read_text_guess


def parse_oto_ini(path: Path) -> List[Tuple[str, str]]:
    entries: List[Tuple[str, str]] = []
    if not path.exists():
        return entries
    for line in read_text_guess(path).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        wav_name, rest = line.split("=", 1)
        alias = rest.split(",", 1)[0].strip()
        wav_name = wav_name.strip()
        if not alias:
            alias = Path(wav_name).stem
        entries.append((alias, wav_name))
    return entries


def import_voicebank(folder: Path, prefix: str = "", suffix: str = "") -> List[str]:
    oto_path = folder / "oto.ini"
    if oto_path.exists():
        wavs = [Path(wav_name).stem for _, wav_name in parse_oto_ini(oto_path)]
        wavs = [_strip_affixes(name, prefix, suffix) for name in wavs]
        return _dedupe_preserve([name for name in wavs if name])

    wavs = sorted(folder.glob("*.wav"))
    names = [wav.stem for wav in wavs]
    names = [_strip_affixes(name, prefix, suffix) for name in names]
    return _dedupe_preserve([name for name in names if name])


def _dedupe_preserve(values: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _strip_affixes(name: str, prefix: str, suffix: str) -> str:
    if prefix and name.startswith(prefix):
        name = name[len(prefix):]
    if suffix and name.endswith(suffix):
        name = name[: -len(suffix)]
    return name.strip()
