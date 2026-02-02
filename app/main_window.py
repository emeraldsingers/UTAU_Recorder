from __future__ import annotations

import logging
import sys
import uuid
from collections import OrderedDict
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
from typing import Optional
import urllib.request
import urllib.error
import zipfile

import numpy as np
import soundfile as sf
import sounddevice as sd
import shutil
from datetime import datetime
from PyQt6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

pg.setConfigOptions(antialias=True)

from audio.engine import AudioEngine
from audio.dsp import (
    compute_fft,
    compute_rms,
    estimate_f0,
    note_from_f0,
    note_to_freq,
    compute_f0_contour,
    compute_f0_contour_yin,
    compute_power_db,
    compute_mel_spectrogram,
    f0_to_midi,
    midi_to_note,
)
from audio.ring_buffer import RingBuffer
from models.parsers import parse_reclist_text, read_text_guess
from models.romaji import kana_to_romaji_tokens, needs_romaji
from models.session import Session, Item, ItemStatus
from models.voicebank import import_voicebank, parse_oto_ini
from storage.session_io import save_session, load_session, export_recordings_json
from app.vst_batch import VstBatchDialog
from app.voicebank_config_dialog import VoicebankConfigDialog


logger = logging.getLogger(__name__)

APP_NAME = "AsoCorder"
APP_VERSION = "0.4.1"
GITHUB_OWNER = "emeraldsingers"
GITHUB_REPO = "UTAU_Recorder"
GITHUB_PROFILE_URL = "https://github.com/emeraldsingers"
GITHUB_RELEASES_URL = "https://github.com/emeraldsingers/UTAU_Recorder/releases"
GITHUB_URL = "https://github.com/emeraldsingers/UTAU_Recorder"
YOUTUBE_URL = "https://www.youtube.com/@asoqwer"


def _parse_version(value: str) -> tuple[int, ...]:
    cleaned = value.strip().lstrip("vV")
    parts: list[int] = []
    for piece in cleaned.split("."):
        try:
            parts.append(int(piece))
        except ValueError:
            break
    return tuple(parts) if parts else (0,)


def _is_version_newer(candidate: str, current: str) -> bool:
    cand = _parse_version(candidate)
    cur = _parse_version(current)
    length = max(len(cand), len(cur))
    cand += (0,) * (length - len(cand))
    cur += (0,) * (length - len(cur))
    return cand > cur

def _analysis_cache_key(path: Path) -> str:
    return hashlib.sha1(str(path).encode("utf-8", errors="ignore")).hexdigest()


def _analysis_cache_paths(cache_dir: Path, key: str) -> tuple[Path, Path]:
    return cache_dir / f"{key}.json", cache_dir / f"{key}.npz"

def _note_from_f0s(f0s: np.ndarray) -> str:
    if f0s.size == 0:
        return "--"
    mids = [f0_to_midi(float(f)) for f in f0s if f and f > 0]
    mids = [m for m in mids if m is not None]
    if not mids:
        return "--"
    avg_midi = float(np.mean(mids))
    return midi_to_note(avg_midi)


def _analyze_note_task(
    path: str,
    target_sr: int,
    algo: str,
    ref_bits: int,
    cache_dir: Optional[str],
) -> Optional[tuple[str, float, str]]:
    try:
        file_path = Path(path)
        if not file_path.exists():
            return None
        audio, sr = sf.read(str(file_path), dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != target_sr:
            audio = AudioEngine._resample(audio, sr, target_sr)
            sr = target_sr
        if algo == "yin":
            times, f0s = compute_f0_contour_yin(audio, sr)
        else:
            times, f0s = compute_f0_contour(audio, sr)
        note = _note_from_f0s(f0s)
        if cache_dir:
            _save_analysis_cache_to_disk(
                Path(cache_dir),
                file_path,
                file_path.stat().st_mtime,
                algo,
                sr,
                ref_bits,
                times,
                f0s,
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
                merge_existing=True,
                note=note,
            )
        return (str(file_path), file_path.stat().st_mtime, note)
    except Exception:
        logger.exception("Failed to analyze note for %s", path)
        return None


def _load_analysis_cache_from_disk(
    cache_dir: Path,
    abs_path: Path,
    mtime: float,
    algo: str,
    sr: int,
    ref_bits: int,
) -> Optional[tuple]:
    key = _analysis_cache_key(abs_path)
    meta_path, data_path = _analysis_cache_paths(cache_dir, key)
    if not meta_path.exists() or not data_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("path") != str(abs_path):
            return None
        if float(meta.get("mtime", -1)) != float(mtime):
            return None
        if meta.get("algo") != algo:
            return None
        if int(meta.get("sr", -1)) != int(sr):
            return None
        if int(meta.get("ref_bits", -1)) != int(ref_bits):
            return None
        data = np.load(str(data_path))
        times = data.get("times", np.array([], dtype=np.float32))
        f0s = data.get("f0s", np.array([], dtype=np.float32))
        mel_db = data.get("mel_db", np.array([], dtype=np.float32))
        mel_times = data.get("mel_times", np.array([], dtype=np.float32))
        power_times = data.get("power_times", np.array([], dtype=np.float32))
        power_db = data.get("power_db", np.array([], dtype=np.float32))
        has_pitch = bool(meta.get("has_pitch", False))
        has_mel = bool(meta.get("has_mel", False))
        has_power = bool(meta.get("has_power", False))
        note = meta.get("note")
        return (times, f0s, mel_db, mel_times, power_times, power_db, has_pitch, has_mel, has_power, note)
    except Exception:
        return None


def _save_analysis_cache_to_disk(
    cache_dir: Path,
    abs_path: Path,
    mtime: float,
    algo: str,
    sr: int,
    ref_bits: int,
    times: np.ndarray,
    f0s: np.ndarray,
    mel_db: np.ndarray,
    mel_times: np.ndarray,
    power_times: np.ndarray,
    power_db: np.ndarray,
    merge_existing: bool = True,
    note: Optional[str] = None,
) -> None:
    key = _analysis_cache_key(abs_path)
    meta_path, data_path = _analysis_cache_paths(cache_dir, key)
    has_pitch = bool(times.size and f0s.size)
    has_mel = bool(mel_db.size and mel_times.size)
    has_power = bool(power_times.size and power_db.size)

    if merge_existing and meta_path.exists() and data_path.exists():
        existing = _load_analysis_cache_from_disk(cache_dir, abs_path, mtime, algo, sr, ref_bits)
        if existing:
            ex_times, ex_f0s, ex_mel, ex_mel_times, ex_power_t, ex_power = existing[:6]
            ex_has_pitch, ex_has_mel, ex_has_power, ex_note = existing[6:]
            if not has_pitch and ex_has_pitch:
                times, f0s = ex_times, ex_f0s
                has_pitch = True
            if not has_mel and ex_has_mel:
                mel_db, mel_times = ex_mel, ex_mel_times
                has_mel = True
            if not has_power and ex_has_power:
                power_times, power_db = ex_power_t, ex_power
                has_power = True
            if note is None and ex_note:
                note = ex_note

    meta = {
        "path": str(abs_path),
        "mtime": float(mtime),
        "algo": str(algo),
        "sr": int(sr),
        "ref_bits": int(ref_bits),
        "has_pitch": bool(has_pitch),
        "has_mel": bool(has_mel),
        "has_power": bool(has_power),
    }
    if note:
        meta["note"] = note
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(
        str(data_path),
        times=times.astype(np.float32, copy=False),
        f0s=f0s.astype(np.float32, copy=False),
        mel_db=mel_db.astype(np.float32, copy=False),
        mel_times=mel_times.astype(np.float32, copy=False),
        power_times=power_times.astype(np.float32, copy=False),
        power_db=power_db.astype(np.float32, copy=False),
    )

TRANSLATIONS = {
    "English": {
        "app_title": "UTAU Voicebank Recorder",
        "file": "File",
        "import": "Import",
        "settings": "Settings",
        "edit": "Edit",
        "tools": "Tools",
        "vst_tools": "VST Tools",
        "help": "Help",
        "about": "About",
        "check_updates": "Check for updates...",
        "about_title": "About AsoCorder",
        "about_version": "Version",
        "about_github": "Authors GitHub",
        "about_youtube": "Authors YouTube",
        "about_releases": "Releases",
        "about_check_updates": "Check for updates",
        "about_download_update": "Download update",
        "about_checking": "Checking for updates...",
        "about_latest": "Latest",
        "about_up_to_date": "Up to date",
        "about_update_available": "Update available",
        "about_update_failed": "Update check failed",
        "about_current_newer": "Current version is newer than latest release.",
        "about_ignore_until_next": "Don't bother until next update",
        "about_no_assets": "No release archives found.",
        "about_download_done": "Update downloaded.",
        "about_download_failed": "Download failed.",
        "about_extracting": "Extracting update...",
        "about_extract_failed": "Failed to extract update.",
        "about_restart_required": "Restart the app to apply the update.",
        "session_settings": "Session Settings",
        "session_settings_title": "Edit Session Settings",
        "session_settings_note": "Changes apply to future recordings.",
        "session_settings_busy": "Stop recording/preview before editing session settings.",
        "rename_files_title": "Rename recordings",
        "rename_files_prompt": "Rename existing files to match new prefix/suffix?",
        "rename_files_failed": "Failed to rename one or more files.",
        "new_session": "New Session",
        "open_session": "Open Session",
        "save_session": "Save Session",
        "save_as": "Save Session As",
        "export_recordings": "Export Recordings JSON",
        "open_folder": "Open Session Folder",
        "export_voicebank": "Export Voicebank",
        "edit_voicebank": "Edit Voicebank",
        "export_voicebank_title": "Export Voicebank",
        "export_voicebank_done": "Voicebank exported.",
        "voicebank_edit_title": "Edit Voicebank",
        "voicebank_name": "Name",
        "voicebank_author": "Author",
        "voicebank_voice": "Voice",
        "voicebank_web": "Web",
        "voicebank_version": "Version",
        "voicebank_image": "Image",
        "voicebank_sample": "Sample",
        "voicebank_portrait": "Portrait",
        "voicebank_portrait_opacity": "Portrait opacity",
        "voicebank_portrait_height": "Portrait height",
        "voicebank_text_encoding": "Text encoding",
        "voicebank_txt_export": "TXT export",
        "voicebank_singer_type": "Singer type",
        "voicebank_phonemizer": "Default phonemizer",
        "voicebank_use_filename": "Use filename as alias",
        "voicebank_other_info": "Other info",
        "voicebank_localized_names": "Localized names",
        "voicebank_add": "Add",
        "voicebank_remove": "Remove",
        "voicebank_pick_image": "Select image",
        "voicebank_pick_sample": "Select sample",
        "back_exit": "Back / Exit",
        "save_reclist_to": "Save Reclist To",
        "import_reclist": "Import Reclist",
        "import_oremo_comment": "Import OREMO Comment",
        "import_voicebank": "Import Voicebank",
        "import_bgm": "Import BGM WAV",
        "generate_bgm": "Generate BGM Note",
        "audio_devices": "Audio Devices",
        "language": "Language",
        "ui_settings": "UI Settings",
        "pitch_algorithm": "Pitch algorithm (charts + note)",
        "pitch_algo_classic": "Classic (autocorr)",
        "pitch_algo_yin": "New (YIN)",
        "note_workers": "Note analysis workers",
        "hold_to_record": "Hold R to record (OREMO style)",
        "theme": "Theme",
        "theme_light": "Light",
        "theme_dark": "Dark",
        "undo": "Undo",
        "apply_vst": "Apply VST Plugins...",
        "vst_batch_title": "VST Batch Processor",
        "vst_select_audio": "Select audio files from this session",
        "vst_search_placeholder": "Search by name or file...",
        "vst_reload": "Reload from session",
        "vst_select_all": "Select all",
        "vst_clear_selection": "Clear selection",
        "vst_use": "Use",
        "vst_file": "File",
        "vst_chain_title": "Plugin chain",
        "vst_chain_presets": "Chain presets",
        "vst_load_preset": "Load",
        "vst_save_preset": "Save",
        "vst_delete_preset": "Delete",
        "vst_preset_none": "(None)",
        "vst_preset_name": "Preset name",
        "vst_plugin_path": "Plugin path",
        "vst_plugin_preset": "Preset file",
        "vst_bypass": "Bypass",
        "vst_add_plugin": "Add plugin",
        "vst_remove_plugin": "Remove plugin",
        "vst_move_up": "Move up",
        "vst_move_down": "Move down",
        "vst_browse_preset": "Browse preset",
        "vst_open_ui": "Open plugin UI",
        "vst_host_cli": "VST host (CLI)",
        "vst_host_gui": "VST host (GUI)",
        "vst_tools_settings": "VST Tools Settings",
        "vst_host_cli_path": "CLI host path",
        "vst_host_gui_path": "GUI host path",
        "vst_tools_note": "Paths are stored in settings and used by the batch processor.",
        "vst_browse": "Browse",
        "vst_host_note": "Requires external VST hosts: CLI accepts --input/--output/--chain; GUI opens plugin UI to save presets. Set paths in Settings > VST Tools.",
        "vst_workers": "Parallel jobs",
        "vst_back": "Back",
        "vst_next": "Next",
        "vst_process": "Process",
        "vst_cancel": "Cancel",
        "vst_no_files": "Select at least one audio file.",
        "vst_no_chain": "Add at least one plugin to the chain.",
        "vst_no_host": "Set a VST host (CLI) first.",
        "vst_no_gui_host": "Set a VST host (GUI) first.",
        "vst_no_plugin": "Select a plugin row first.",
        "vst_preset_prompt": "Save preset as...",
        "vst_gui_failed": "GUI host exited early. Check the path and build output.",
        "vst_no_session": "Open or create a session first.",
        "vst_backup_note": "Backup is created before processing.",
        "vst_done": "Processing completed.",
        "start_title": "Start",
        "start_new": "New",
        "start_open": "Open",
        "start_recent": "Recent",
        "recent_sessions": "Recent Sessions",
        "current_item": "Current item: --",
        "current_item_prefix": "Current item: ",
        "current_note": "Current note: --",
        "current_note_prefix": "Current note: ",
        "record": "Record",
        "stop": "Stop",
        "rerecord": "Re-record",
        "preview_bgm": "Preview BGM",
        "preview_overlay": "Preview Overlay",
        "bgm_during": "BGM during recording",
        "auto_next": "Auto next item",
        "bgm_level": "BGM level",
        "bgm_overlay_level": "BGM overlay level",
        "pre_roll": "Pre-roll (ms)",
        "cut_selection": "Cut Selection",
        "select_region": "Select Region",
        "bgm_timing": "BGM timing",
        "bpm": "BPM",
        "mora_count": "Mora count",
        "waveform": "Waveform",
        "spectrum": "Spectrum",
        "power": "Power",
        "f0": "F0",
        "recorded_f0": "Recorded F0",
        "mel": "Mel",
        "table_status": "Status",
        "table_alias": "Alias",
        "table_romaji": "Romaji",
        "table_note": "Note",
        "table_comment": "Comment",
        "table_duration": "Duration",
        "table_file": "File",
        "recompute_note": "Recompute Note",
        "new_session_title": "New Recording Session",
        "session_name": "Session name",
        "singer": "Singer",
        "project_path": "Project path",
        "browse": "Browse",
        "sample_rate": "Sample rate",
        "bit_depth": "Bit depth",
        "channels": "Channels",
        "output_prefix": "Output prefix",
        "output_suffix": "Output suffix",
        "session_note": "Target note",
        "missing_data": "Missing data",
        "name_required": "Name is required.",
        "audio_devices_title": "Audio Devices",
        "input_device": "Input device",
        "output_device": "Output device",
        "default": "Default",
        "language_title": "Language",
        "voicebank_options": "Voicebank Import Options",
        "use_vb_bgm": "Use voicebank samples as BGM (per alias)",
        "copy_oto": "Copy oto.ini to session",
        "strip_oto_alias": "Strip prefix/suffix from oto aliases",
        "bgm_mode_title": "BGM Note",
        "bgm_note": "Note (e.g. A4)",
        "bgm_duration": "Duration (sec)",
        "bgm_mode": "Mode",
        "bgm_replace": "Replace BGM",
        "bgm_add": "Add overlay",
        "bgm_metronome": "Metronome",
        "import_reclist_title": "Import Reclist",
        "import_reclist_question": "Replace current reclist or add new entries?",
        "import_oremo_title": "Import OREMO Comment",
        "import_oremo_question": "Replace current reclist or add/update comments?",
        "add_entry": "Add Entry",
        "delete_entry": "Delete Entry",
        "delete_selected_title": "Delete recordings",
        "delete_selected_prompt": "Delete {count} selected entries and their files?",
        "alias": "Alias",
        "note_optional": "Note (optional)",
    },
    "Русский": {
        "app_title": "UTAU Voicebank Recorder",
        "file": "Файл",
        "import": "Импорт",
        "settings": "Настройки",
        "help": "Справка",
        "about": "О программе",
        "check_updates": "Проверить обновления...",
        "about_title": "О программе AsoCorder",
        "about_version": "Версия",
        "about_github": "GitHub автора",
        "about_youtube": "YouTube автора",
        "about_releases": "Релизы",
        "about_check_updates": "Проверить обновления",
        "about_download_update": "Скачать обновление",
        "about_checking": "Проверяю обновления...",
        "about_latest": "Последняя",
        "about_up_to_date": "Актуальная версия",
        "about_update_available": "Доступна новая версия",
        "about_update_failed": "Не удалось проверить обновления",
        "about_current_newer": "Текущая версия новее, чем в релизах.",
        "about_ignore_until_next": "Не беспокоить до следующего обновления",
        "about_no_assets": "Архивы релиза не найдены.",
        "about_download_done": "Обновление скачано.",
        "about_download_failed": "Не удалось скачать обновление.",
        "about_extracting": "Распаковка обновления...",
        "about_extract_failed": "Не удалось распаковать обновление.",
        "about_restart_required": "Перезапусти приложение для применения обновления.",
        "session_settings": "Настройки сессии",
        "session_settings_title": "Изменить настройки сессии",
        "session_settings_note": "Изменения применяются к будущим записям.",
        "session_settings_busy": "Остановите запись/прослушивание перед изменением настроек сессии.",
        "rename_files_title": "Переименовать записи",
        "rename_files_prompt": "Переименовать существующие файлы под новый префикс/суффикс?",
        "rename_files_failed": "Не удалось переименовать один или несколько файлов.",
        "edit": "Правка",
        "new_session": "Новая сессия",
        "open_session": "Открыть сессию",
        "save_session": "Сохранить сессию",
        "save_as": "Сохранить как",
        "export_recordings": "Экспорт записей JSON",
        "open_folder": "Открыть папку сессии",
        "export_voicebank": "Экспорт войсбанка",
        "edit_voicebank": "Редактировать войсбанк",
        "export_voicebank_title": "Экспорт войсбанка",
        "export_voicebank_done": "Войсбанк экспортирован.",
        "voicebank_edit_title": "Редактировать войсбанк",
        "voicebank_name": "Имя",
        "voicebank_author": "Автор",
        "voicebank_voice": "Голос",
        "voicebank_web": "Сайт",
        "voicebank_version": "Версия",
        "voicebank_image": "Изображение",
        "voicebank_sample": "Семпл",
        "voicebank_portrait": "Портрет",
        "voicebank_portrait_opacity": "Прозрачность портрета",
        "voicebank_portrait_height": "Высота портрета",
        "voicebank_text_encoding": "Кодировка текста",
        "voicebank_txt_export": "Экспорт TXT",
        "voicebank_singer_type": "Тип войсбанка",
        "voicebank_phonemizer": "Фонемайзер по умолчанию",
        "voicebank_use_filename": "Использовать имя файла как алиас",
        "voicebank_other_info": "Прочее",
        "voicebank_localized_names": "Локализованные имена",
        "voicebank_add": "Добавить",
        "voicebank_remove": "Удалить",
        "voicebank_pick_image": "Выбрать изображение",
        "voicebank_pick_sample": "Выбрать семпл",
        "back_exit": "Назад / Выход",
        "save_reclist_to": "Сохранить реклист как",
        "import_reclist": "Импорт реклиста",
        "import_oremo_comment": "Импорт OREMO Comment",
        "import_voicebank": "Импорт войсбанка",
        "import_bgm": "Импорт BGM WAV",
        "generate_bgm": "Сгенерировать BGM ноту",
        "audio_devices": "Аудиоустройства",
        "language": "Язык",
        "ui_settings": "Настройки UI",
        "pitch_algorithm": "Алгоритм высоты (графики + нота)",
        "pitch_algo_classic": "Классический (автокорр.)",
        "pitch_algo_yin": "Новый (YIN)",
        "note_workers": "Потоки анализа нот",
        "hold_to_record": "Запись удержанием R (как в OREMO)",
        "theme": "Тема",
        "theme_light": "Светлая",
        "theme_dark": "Темная",
        "undo": "Отменить",
        "start_title": "Старт",
        "start_new": "Новая",
        "start_open": "Открыть",
        "start_recent": "Недавние",
        "recent_sessions": "Последние сессии",
        "current_item": "Текущий элемент: --",
        "current_item_prefix": "Текущий элемент: ",
        "current_note": "Текущая нота: --",
        "current_note_prefix": "Текущая нота: ",
        "record": "Запись",
        "stop": "Стоп",
        "rerecord": "Перезапись",
        "preview_bgm": "Прослушать BGM",
        "preview_overlay": "Прослушать оверлей",
        "bgm_during": "BGM при записи",
        "auto_next": "Автопереход",
        "bgm_level": "Уровень BGM",
        "bgm_overlay_level": "Уровень оверлея",
        "pre_roll": "Предролл (мс)",
        "cut_selection": "Вырезать выделение",
        "select_region": "Выделить участок",
        "bgm_timing": "Тайминг BGM",
        "bpm": "BPM",
        "mora_count": "Кол-во мор",
        "waveform": "Вейвформа",
        "spectrum": "Спектр",
        "power": "Мощность",
        "f0": "F0",
        "recorded_f0": "F0 записи",
        "mel": "Мел-спектр",
        "table_status": "Статус",
        "table_alias": "Алиас",
        "table_romaji": "Ромадзи",
        "table_note": "Нота",
        "table_comment": "Комментарий",
        "table_duration": "Длительность",
        "table_file": "Файл",
        "recompute_note": "Пересчитать ноту",
        "new_session_title": "Новая сессия записи",
        "session_name": "Имя сессии",
        "singer": "Певец",
        "project_path": "Путь проекта",
        "browse": "Обзор",
        "sample_rate": "Частота",
        "bit_depth": "Битность",
        "channels": "Каналы",
        "output_prefix": "Префикс файла",
        "output_suffix": "Суффикс файла",
        "session_note": "Целевая нота",
        "missing_data": "Недостаточно данных",
        "name_required": "Имя обязательно.",
        "audio_devices_title": "Аудиоустройства",
        "input_device": "Входное устройство",
        "output_device": "Выходное устройство",
        "default": "По умолчанию",
        "language_title": "Язык",
        "voicebank_options": "Опции импорта voicebank",
        "use_vb_bgm": "Использовать семплы voicebank как BGM (по алиасу)",
        "copy_oto": "Копировать oto.ini в сессию",
        "strip_oto_alias": "Убрать префикс/суффикс у алиасов oto",
        "bgm_mode_title": "BGM нота",
        "bgm_note": "Нота (например A4)",
        "bgm_duration": "Длительность (сек)",
        "bgm_mode": "Режим",
        "bgm_replace": "Заменить BGM",
        "bgm_add": "Добавить поверх",
        "bgm_metronome": "Метроном",
        "import_reclist_title": "Импорт реклиста",
        "import_reclist_question": "Заменить текущий реклист или добавить новые?",
        "import_oremo_title": "Импорт OREMO Comment",
        "import_oremo_question": "Заменить текущий реклист или добавить/обновить комментарии?",
        "add_entry": "Добавить",
        "delete_entry": "Удалить",
        "delete_selected_title": "Удалить записи",
        "delete_selected_prompt": "Удалить выбранные записи ({count}) и их файлы?",
        "alias": "Алиас",
        "note_optional": "Нота (опц.)",
    },
    "日本語": {
        "app_title": "UTAU Voicebank Recorder",
        "file": "ファイル",
        "import": "インポート",
        "settings": "設定",
        "help": "ヘルプ",
        "about": "このソフトについて",
        "check_updates": "更新を確認...",
        "about_title": "AsoCorder について",
        "about_version": "バージョン",
        "about_github": "GitHub",
        "about_youtube": "YouTube",
        "about_releases": "リリース",
        "about_check_updates": "更新を確認",
        "about_download_update": "更新をダウンロード",
        "about_checking": "更新を確認中...",
        "about_latest": "最新",
        "about_up_to_date": "最新です",
        "about_update_available": "更新があります",
        "about_update_failed": "更新確認に失敗しました",
        "about_current_newer": "現在のバージョンは最新リリースより新しいです。",
        "about_ignore_until_next": "次の更新まで通知しない",
        "about_no_assets": "アーカイブが見つかりません。",
        "about_download_done": "更新をダウンロードしました。",
        "about_download_failed": "ダウンロードに失敗しました。",
        "about_extracting": "更新を展開中...",
        "about_extract_failed": "更新の展開に失敗しました。",
        "about_restart_required": "更新を適用するには再起動してください。",
        "session_settings": "セッション設定",
        "session_settings_title": "セッション設定を編集",
        "session_settings_note": "変更は今後の録音に適用されます。",
        "session_settings_busy": "録音/プレビューを停止してからセッション設定を変更してください。",
        "rename_files_title": "録音ファイルの名前変更",
        "rename_files_prompt": "既存のファイルを新しいプレフィックス/サフィックスに合わせて名前変更しますか？",
        "rename_files_failed": "一部のファイルの名前変更に失敗しました。",
        "edit": "編集",
        "new_session": "新規セッション",
        "open_session": "セッションを開く",
        "save_session": "セッションを保存",
        "save_as": "名前を付けて保存",
        "export_recordings": "録音一覧JSONを書き出し",
        "open_folder": "セッションフォルダを開く",
        "export_voicebank": "音源を書き出す",
        "edit_voicebank": "音源を編集",
        "export_voicebank_title": "音源を書き出す",
        "export_voicebank_done": "音源を書き出しました。",
        "voicebank_edit_title": "音源を編集",
        "voicebank_name": "名前",
        "voicebank_author": "作者",
        "voicebank_voice": "声",
        "voicebank_web": "ウェブ",
        "voicebank_version": "バージョン",
        "voicebank_image": "画像",
        "voicebank_sample": "サンプル",
        "voicebank_portrait": "ポートレート",
        "voicebank_portrait_opacity": "ポートレート透明度",
        "voicebank_portrait_height": "ポートレート高さ",
        "voicebank_text_encoding": "テキストエンコード",
        "voicebank_txt_export": "TXTエクスポート",
        "voicebank_singer_type": "音源タイプ",
        "voicebank_phonemizer": "デフォルト音素化",
        "voicebank_use_filename": "ファイル名をエイリアスに使う",
        "voicebank_other_info": "その他",
        "voicebank_localized_names": "ローカライズ名",
        "voicebank_add": "追加",
        "voicebank_remove": "削除",
        "voicebank_pick_image": "画像を選択",
        "voicebank_pick_sample": "サンプルを選択",
        "back_exit": "戻る / 終了",
        "save_reclist_to": "レコリストを書き出し",
        "import_reclist": "レコリストをインポート",
        "import_oremo_comment": "OREMOコメントをインポート",
        "import_voicebank": "音源をインポート",
        "import_bgm": "BGM WAVをインポート",
        "generate_bgm": "BGMノート生成",
        "audio_devices": "オーディオデバイス",
        "language": "言語",
        "ui_settings": "UI設定",
        "pitch_algorithm": "ピッチアルゴリズム（グラフ＋ノート）",
        "pitch_algo_classic": "クラシック（自己相関）",
        "pitch_algo_yin": "新しい（YIN）",
        "note_workers": "ノート解析ワーカー数",
        "hold_to_record": "Rキー長押しで録音（OREMO風）",
        "theme": "テーマ",
        "theme_light": "ライト",
        "theme_dark": "ダーク",
        "undo": "元に戻す",
        "start_title": "スタート",
        "start_new": "新規",
        "start_open": "開く",
        "start_recent": "最近",
        "recent_sessions": "最近のセッション",
        "current_item": "現在の項目: --",
        "current_item_prefix": "現在の項目: ",
        "current_note": "現在のノート: --",
        "current_note_prefix": "現在のノート: ",
        "record": "録音",
        "stop": "停止",
        "rerecord": "再録音",
        "preview_bgm": "BGMプレビュー",
        "preview_overlay": "オーバーレイプレビュー",
        "bgm_during": "録音中BGM",
        "auto_next": "自動で次へ",
        "bgm_level": "BGMレベル",
        "bgm_overlay_level": "BGMオーバーレイ",
        "pre_roll": "プリロール (ms)",
        "cut_selection": "選択範囲を削除",
        "select_region": "範囲を選択",
        "bgm_timing": "BGMタイミング",
        "bpm": "BPM",
        "mora_count": "モーラ数",
        "waveform": "波形",
        "spectrum": "スペクトル",
        "power": "パワー",
        "f0": "F0",
        "recorded_f0": "録音F0",
        "mel": "メル",
        "table_status": "状態",
        "table_alias": "エイリアス",
        "table_romaji": "ローマ字",
        "table_note": "ノート",
        "table_comment": "コメント",
        "table_duration": "長さ",
        "table_file": "ファイル",
        "recompute_note": "ノートを再計算",
        "new_session_title": "新規録音セッション",
        "session_name": "セッション名",
        "singer": "歌い手",
        "project_path": "プロジェクトパス",
        "browse": "参照",
        "sample_rate": "サンプルレート",
        "bit_depth": "ビット深度",
        "channels": "チャンネル",
        "output_prefix": "出力プレフィックス",
        "output_suffix": "出力サフィックス",
        "session_note": "ターゲットノート",
        "missing_data": "データ不足",
        "name_required": "名前が必要です。",
        "audio_devices_title": "オーディオデバイス",
        "input_device": "入力デバイス",
        "output_device": "出力デバイス",
        "default": "既定",
        "language_title": "言語",
        "voicebank_options": "音源インポート設定",
        "use_vb_bgm": "音源サンプルをBGMとして使用 (エイリアス毎)",
        "copy_oto": "oto.iniをセッションにコピー",
        "strip_oto_alias": "otoのエイリアスから接頭/接尾を除去",
        "bgm_mode_title": "BGMノート",
        "bgm_note": "ノート (例 A4)",
        "bgm_duration": "長さ (秒)",
        "bgm_mode": "モード",
        "bgm_replace": "BGMを置換",
        "bgm_add": "上に追加",
        "bgm_metronome": "メトロノーム",
        "import_reclist_title": "レコリストをインポート",
        "import_reclist_question": "既存を置換しますか？それとも追加しますか？",
        "import_oremo_title": "OREMOコメントをインポート",
        "import_oremo_question": "既存を置換しますか？それともコメントを追加・更新しますか？",
        "add_entry": "追加",
        "delete_entry": "削除",
        "delete_selected_title": "録音を削除",
        "delete_selected_prompt": "選択した{count}件とファイルを削除しますか？",
        "alias": "エイリアス",
        "note_optional": "ノート (任意)",
    },
}


def tr(lang: str, key: str) -> str:
    table = TRANSLATIONS.get(lang) or TRANSLATIONS["English"]
    return table.get(key, TRANSLATIONS["English"].get(key, key))


class NewSessionDialog(QtWidgets.QDialog):
    def __init__(self, lang: str, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.lang = lang
        self.setWindowTitle(tr(self.lang, "new_session_title"))
        layout = QtWidgets.QFormLayout(self)

        self.name_edit = QtWidgets.QLineEdit()
        self.singer_edit = QtWidgets.QLineEdit()
        self.path_edit = QtWidgets.QLineEdit()
        browse_btn = QtWidgets.QPushButton(tr(self.lang, "browse"))
        browse_btn.clicked.connect(self._browse)

        path_layout = QtWidgets.QHBoxLayout()
        path_layout.addWidget(self.path_edit)
        path_layout.addWidget(browse_btn)

        self.sr_spin = QtWidgets.QSpinBox()
        self.sr_spin.setRange(8000, 192000)
        self.sr_spin.setValue(44100)

        self.bit_depth_combo = QtWidgets.QComboBox()
        self.bit_depth_combo.addItems(["16", "24", "32"])
        self.channels_combo = QtWidgets.QComboBox()
        self.channels_combo.addItems(["1", "2"])
        self.output_prefix_edit = QtWidgets.QLineEdit()
        self.output_suffix_edit = QtWidgets.QLineEdit()
        self.note_edit = QtWidgets.QLineEdit()
        self.note_edit.setPlaceholderText("A4")

        layout.addRow(tr(self.lang, "session_name"), self.name_edit)
        layout.addRow(tr(self.lang, "singer"), self.singer_edit)
        layout.addRow(tr(self.lang, "project_path"), path_layout)
        layout.addRow(tr(self.lang, "sample_rate"), self.sr_spin)
        layout.addRow(tr(self.lang, "bit_depth"), self.bit_depth_combo)
        layout.addRow(tr(self.lang, "channels"), self.channels_combo)
        layout.addRow(tr(self.lang, "output_prefix"), self.output_prefix_edit)
        layout.addRow(tr(self.lang, "output_suffix"), self.output_suffix_edit)
        layout.addRow(tr(self.lang, "session_note"), self.note_edit)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Project Folder")
        if path:
            self.path_edit.setText(path)

    def get_data(self) -> Optional[dict]:
        if self.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        name = self.name_edit.text().strip()
        path = self.path_edit.text().strip()
        if not name:
            QtWidgets.QMessageBox.warning(self, tr(self.lang, "missing_data"), tr(self.lang, "name_required"))
            return None
        singer = self.singer_edit.text().strip() or "Unknown"
        if not path:
            path = str(Path.cwd() / "recordings" / singer / name)
        return {
            "name": name,
            "singer": singer,
            "path": Path(path),
            "sample_rate": self.sr_spin.value(),
            "bit_depth": int(self.bit_depth_combo.currentText()),
            "channels": int(self.channels_combo.currentText()),
            "output_prefix": self.output_prefix_edit.text().strip(),
            "output_suffix": self.output_suffix_edit.text().strip(),
            "target_note": self.note_edit.text().strip() or None,
        }


class SessionSettingsDialog(QtWidgets.QDialog):
    def __init__(self, lang: str, session: Session, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.lang = lang
        self.setWindowTitle(tr(self.lang, "session_settings_title"))
        layout = QtWidgets.QFormLayout(self)

        self.sr_spin = QtWidgets.QSpinBox()
        self.sr_spin.setRange(8000, 192000)
        self.sr_spin.setValue(session.sample_rate)

        self.bit_depth_combo = QtWidgets.QComboBox()
        self.bit_depth_combo.addItems(["16", "24", "32"])
        self.bit_depth_combo.setCurrentText(str(session.bit_depth))

        self.channels_combo = QtWidgets.QComboBox()
        self.channels_combo.addItems(["1", "2"])
        self.channels_combo.setCurrentText(str(session.channels))

        self.output_prefix_edit = QtWidgets.QLineEdit(session.output_prefix)
        self.output_suffix_edit = QtWidgets.QLineEdit(session.output_suffix)

        self.note_edit = QtWidgets.QLineEdit(session.target_note or "")
        self.note_edit.setPlaceholderText("A4")

        layout.addRow(tr(self.lang, "sample_rate"), self.sr_spin)
        layout.addRow(tr(self.lang, "bit_depth"), self.bit_depth_combo)
        layout.addRow(tr(self.lang, "channels"), self.channels_combo)
        layout.addRow(tr(self.lang, "output_prefix"), self.output_prefix_edit)
        layout.addRow(tr(self.lang, "output_suffix"), self.output_suffix_edit)
        layout.addRow(tr(self.lang, "session_note"), self.note_edit)

        note_label = QtWidgets.QLabel(tr(self.lang, "session_settings_note"))
        note_label.setWordWrap(True)
        layout.addRow(note_label)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_data(self) -> Optional[dict]:
        if self.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        return {
            "sample_rate": self.sr_spin.value(),
            "bit_depth": int(self.bit_depth_combo.currentText()),
            "channels": int(self.channels_combo.currentText()),
            "output_prefix": self.output_prefix_edit.text().strip(),
            "output_suffix": self.output_suffix_edit.text().strip(),
            "target_note": self.note_edit.text().strip() or None,
        }


class NoteAxis(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        return [midi_to_note(v) for v in values]


class NoteTableItem(QtWidgets.QTableWidgetItem):
    def __init__(self, text: str, priority: int) -> None:
        super().__init__(text)
        self._priority = priority

    def __lt__(self, other: QtWidgets.QTableWidgetItem) -> bool:
        if isinstance(other, NoteTableItem) and self._priority != other._priority:
            return self._priority < other._priority
        return super().__lt__(other)


class AudioSettingsDialog(QtWidgets.QDialog):
    def __init__(self, lang: str, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.lang = lang
        self.setWindowTitle(tr(self.lang, "audio_devices_title"))
        layout = QtWidgets.QFormLayout(self)

        self.input_combo = QtWidgets.QComboBox()
        self.output_combo = QtWidgets.QComboBox()
        self._populate_devices()

        layout.addRow(tr(self.lang, "input_device"), self.input_combo)
        layout.addRow(tr(self.lang, "output_device"), self.output_combo)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _populate_devices(self) -> None:
        self.input_combo.addItem(tr(self.lang, "default"), None)
        self.output_combo.addItem(tr(self.lang, "default"), None)
        devices = sd.query_devices()
        for idx, dev in enumerate(devices):
            name = dev["name"]
            if dev["max_input_channels"] > 0:
                self.input_combo.addItem(f"{idx}: {name}", idx)
            if dev["max_output_channels"] > 0:
                self.output_combo.addItem(f"{idx}: {name}", idx)

    def set_selected(self, input_dev: Optional[int], output_dev: Optional[int]) -> None:
        self._select_combo(self.input_combo, input_dev)
        self._select_combo(self.output_combo, output_dev)

    def get_selected(self) -> tuple[Optional[int], Optional[int]]:
        return self.input_combo.currentData(), self.output_combo.currentData()

    @staticmethod
    def _select_combo(combo: QtWidgets.QComboBox, value: Optional[int]) -> None:
        for i in range(combo.count()):
            if combo.itemData(i) == value:
                combo.setCurrentIndex(i)
                return


def _default_tool_path(name: str) -> str:
    exe = f"{name}.exe" if sys.platform == "win32" else name
    return str(Path.cwd() / "tools" / exe)


class VstToolsDialog(QtWidgets.QDialog):
    def __init__(self, lang: str, settings: QtCore.QSettings, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.lang = lang
        self.settings = settings
        self.setWindowTitle(tr(self.lang, "vst_tools_settings"))
        layout = QtWidgets.QFormLayout(self)

        self.cli_edit = QtWidgets.QLineEdit(
            self.settings.value("vst_host_cli", _default_tool_path("utau_vst_host"))
        )
        self.gui_edit = QtWidgets.QLineEdit(
            self.settings.value("vst_host_gui", _default_tool_path("utau_vst_host_gui"))
        )

        cli_row = QtWidgets.QHBoxLayout()
        cli_row.addWidget(self.cli_edit)
        self.cli_browse_btn = QtWidgets.QPushButton(tr(self.lang, "vst_browse"))
        self.cli_browse_btn.clicked.connect(lambda: self._browse(self.cli_edit))
        cli_row.addWidget(self.cli_browse_btn)

        gui_row = QtWidgets.QHBoxLayout()
        gui_row.addWidget(self.gui_edit)
        self.gui_browse_btn = QtWidgets.QPushButton(tr(self.lang, "vst_browse"))
        self.gui_browse_btn.clicked.connect(lambda: self._browse(self.gui_edit))
        gui_row.addWidget(self.gui_browse_btn)

        layout.addRow(tr(self.lang, "vst_host_cli_path"), cli_row)
        layout.addRow(tr(self.lang, "vst_host_gui_path"), gui_row)
        note = QtWidgets.QLabel(tr(self.lang, "vst_tools_note"))
        note.setStyleSheet("color: #666666;")
        note.setWordWrap(True)
        layout.addRow(note)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def _browse(self, target: QtWidgets.QLineEdit) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, tr(self.lang, "vst_browse"), "", "Executable (*)")
        if path:
            target.setText(path)

    def save(self) -> None:
        self.settings.setValue("vst_host_cli", self.cli_edit.text().strip())
        self.settings.setValue("vst_host_gui", self.gui_edit.text().strip())


class UiSettingsDialog(QtWidgets.QDialog):
    def __init__(
        self,
        lang: str,
        current_lang: str,
        current_theme_key: str,
        current_pitch_algo: str,
        current_note_workers: int,
        current_hold_to_record: bool,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.lang = lang
        self.setWindowTitle(tr(self.lang, "ui_settings"))
        layout = QtWidgets.QFormLayout(self)
        self.lang_combo = QtWidgets.QComboBox()
        self.lang_combo.addItems(["English", "Русский", "日本語"])
        if current_lang:
            idx = self.lang_combo.findText(current_lang)
            if idx >= 0:
                self.lang_combo.setCurrentIndex(idx)
        self.theme_combo = QtWidgets.QComboBox()
        self.theme_combo.addItem(tr(self.lang, "theme_light"), "light")
        self.theme_combo.addItem(tr(self.lang, "theme_dark"), "dark")
        if current_theme_key:
            for i in range(self.theme_combo.count()):
                if self.theme_combo.itemData(i) == current_theme_key:
                    self.theme_combo.setCurrentIndex(i)
                    break

        self.pitch_combo = QtWidgets.QComboBox()
        self.pitch_combo.addItem(tr(self.lang, "pitch_algo_classic"), "classic")
        self.pitch_combo.addItem(tr(self.lang, "pitch_algo_yin"), "yin")
        if current_pitch_algo:
            for i in range(self.pitch_combo.count()):
                if self.pitch_combo.itemData(i) == current_pitch_algo:
                    self.pitch_combo.setCurrentIndex(i)
                    break

        self.note_workers_spin = QtWidgets.QSpinBox()
        self.note_workers_spin.setRange(1, 16)
        if current_note_workers:
            self.note_workers_spin.setValue(int(current_note_workers))

        self.hold_to_record_check = QtWidgets.QCheckBox(tr(self.lang, "hold_to_record"))
        self.hold_to_record_check.setChecked(bool(current_hold_to_record))

        layout.addRow(tr(self.lang, "language"), self.lang_combo)
        layout.addRow(tr(self.lang, "theme"), self.theme_combo)
        layout.addRow(tr(self.lang, "pitch_algorithm"), self.pitch_combo)
        layout.addRow(tr(self.lang, "note_workers"), self.note_workers_spin)
        layout.addRow(self.hold_to_record_check)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_data(self) -> tuple[str, str, str, int, bool]:
        return (
            self.lang_combo.currentText(),
            str(self.theme_combo.currentData()),
            str(self.pitch_combo.currentData()),
            int(self.note_workers_spin.value()),
            bool(self.hold_to_record_check.isChecked()),
        )


class VoicebankImportDialog(QtWidgets.QDialog):
    def __init__(self, lang: str, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.lang = lang
        self.setWindowTitle(tr(self.lang, "voicebank_options"))
        layout = QtWidgets.QVBoxLayout(self)
        self.use_bgm_checkbox = QtWidgets.QCheckBox(tr(self.lang, "use_vb_bgm"))
        self.use_bgm_checkbox.setChecked(True)
        layout.addWidget(self.use_bgm_checkbox)
        self.copy_oto_checkbox = QtWidgets.QCheckBox(tr(self.lang, "copy_oto"))
        self.copy_oto_checkbox.setChecked(True)
        layout.addWidget(self.copy_oto_checkbox)
        self.strip_oto_checkbox = QtWidgets.QCheckBox(tr(self.lang, "strip_oto_alias"))
        self.strip_oto_checkbox.setChecked(False)
        layout.addWidget(self.strip_oto_checkbox)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def use_bgm(self) -> bool:
        return self.use_bgm_checkbox.isChecked()

    def copy_oto(self) -> bool:
        return self.copy_oto_checkbox.isChecked()

    def strip_oto_aliases(self) -> bool:
        return self.strip_oto_checkbox.isChecked()


class StartDialog(QtWidgets.QDialog):
    def __init__(self, lang: str, recent: list[dict], parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.lang = lang
        self.setWindowTitle(tr(self.lang, "recent_sessions"))
        self.resize(860, 540)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(18)

        menubar = QtWidgets.QMenuBar(self)
        file_menu = menubar.addMenu(tr(self.lang, "file"))
        new_action = file_menu.addAction(tr(self.lang, "new_session"))
        open_action = file_menu.addAction(tr(self.lang, "open_session"))
        file_menu.addSeparator()
        close_action = file_menu.addAction(tr(self.lang, "stop"))
        settings_menu = menubar.addMenu(tr(self.lang, "settings"))
        ui_action = settings_menu.addAction(tr(self.lang, "ui_settings"))
        help_menu = menubar.addMenu(tr(self.lang, "help"))
        about_action = help_menu.addAction(tr(self.lang, "about"))
        layout.setMenuBar(menubar)

        title = QtWidgets.QLabel(tr(self.lang, "start_title"))
        title.setStyleSheet("font-size: 26px; font-weight: 700;")
        subtitle = QtWidgets.QLabel(tr(self.lang, "recent_sessions"))
        subtitle.setStyleSheet("color: #808080;")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        cards = QtWidgets.QHBoxLayout()
        cards.setSpacing(16)
        self.new_btn = QtWidgets.QPushButton(tr(self.lang, "start_new"))
        self.open_btn = QtWidgets.QPushButton(tr(self.lang, "start_open"))
        for btn in (self.new_btn, self.open_btn):
            btn.setMinimumHeight(64)
            btn.setStyleSheet(
                "QPushButton { font-size: 16px; padding: 12px 18px; }"
            )
        cards.addWidget(self.new_btn)
        cards.addWidget(self.open_btn)
        layout.addLayout(cards)

        recent_title = QtWidgets.QLabel(tr(self.lang, "start_recent"))
        recent_title.setStyleSheet("font-size: 16px; font-weight: 600;")
        layout.addWidget(recent_title)
        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setAlternatingRowColors(True)
        self.list_widget.setMinimumHeight(260)
        self.list_widget.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.list_widget.setMouseTracking(True)
        self.list_widget.setStyleSheet(
            "QListWidget::item { padding: 2px; }"
            "QListWidget::item:hover { background: #e6f0ff; }"
            "QListWidget::item:selected { background: #cfe3ff; }"
        )
        for entry in recent:
            item = QtWidgets.QListWidgetItem()
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEnabled)
            widget = QtWidgets.QWidget()
            v = QtWidgets.QVBoxLayout(widget)
            v.setContentsMargins(8, 4, 8, 4)
            title_label = QtWidgets.QLabel(entry["title"])
            title_label.setStyleSheet("font-weight: 600;")
            path_label = QtWidgets.QLabel(entry["path"])
            path_label.setStyleSheet("color: #808080; font-size: 11px;")
            v.addWidget(title_label)
            v.addWidget(path_label)
            item.setSizeHint(widget.sizeHint())
            self.list_widget.addItem(item)
            self.list_widget.setItemWidget(item, widget)
        self.list_widget.setCurrentRow(-1)
        self.list_widget.clearSelection()
        layout.addWidget(self.list_widget, 1)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

        footer = QtWidgets.QLabel(
            f'{APP_NAME} (v{APP_VERSION}) <a href="{GITHUB_URL}">{GITHUB_URL}</a>'
        )
        footer.setStyleSheet("color: #888888; font-size: 10px;")
        footer.setTextFormat(QtCore.Qt.TextFormat.RichText)
        footer.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextBrowserInteraction)
        footer.setOpenExternalLinks(True)
        footer.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(footer)


        self.new_btn.clicked.connect(self._new_clicked)
        self.open_btn.clicked.connect(self._open_clicked)
        self.list_widget.itemDoubleClicked.connect(self._recent_clicked)
        new_action.triggered.connect(self._new_clicked)
        open_action.triggered.connect(self._open_clicked)
        close_action.triggered.connect(self.reject)
        ui_action.triggered.connect(self._ui_clicked)
        about_action.triggered.connect(self._about_clicked)

        self.action: Optional[str] = None
        self.selected_path: Optional[str] = None
        self._recent_entries = recent
        self.closed_via_x = False

    def _new_clicked(self) -> None:
        self.action = "new"
        self.accept()

    def _open_clicked(self) -> None:
        self.action = "open"
        self.accept()

    def _recent_clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        self.action = "recent"
        row = self.list_widget.row(item)
        if 0 <= row < len(self._recent_entries):
            self.selected_path = self._recent_entries[row]["full_path"]
        self.accept()

    def _ui_clicked(self) -> None:
        self.action = "settings"
        self.accept()

    def _about_clicked(self) -> None:
        dialog = AboutDialog(self.lang, self)
        dialog.exec()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.closed_via_x = True
        super().closeEvent(event)


class UpdateCheckWorker(QtCore.QThread):
    result = QtCore.pyqtSignal(str, str, str, str)

    def run(self) -> None:
        url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest"
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": f"{APP_NAME}/{APP_VERSION}"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            tag = str(payload.get("tag_name", "")).strip()
            assets = payload.get("assets") or []
            asset_url = ""
            asset_name = ""
            for asset in assets:
                name = str(asset.get("name", ""))
                url = str(asset.get("browser_download_url", ""))
                if not name or not url:
                    continue
                lower = name.lower()
                if lower.endswith(".zip") or lower.endswith(".rar"):
                    asset_name = name
                    asset_url = url
                    break
            self.result.emit(tag, asset_name, asset_url, "")
        except Exception as exc:
            self.result.emit("", "", "", str(exc))


class UpdateDownloadWorker(QtCore.QThread):
    finished = QtCore.pyqtSignal(bool, str)

    def __init__(self, url: str, out_path: Path, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self.url = url
        self.out_path = out_path

    def run(self) -> None:
        try:
            req = urllib.request.Request(
                self.url,
                headers={"User-Agent": f"{APP_NAME}/{APP_VERSION}"},
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = resp.read()
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            self.out_path.write_bytes(data)
            self.finished.emit(True, str(self.out_path))
        except Exception as exc:
            self.finished.emit(False, str(exc))


class AboutDialog(QtWidgets.QDialog):
    def __init__(self, lang: str, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.lang = lang
        self.setWindowTitle(tr(self.lang, "about_title"))
        self.setMinimumWidth(420)
        self._worker: Optional[UpdateCheckWorker] = None
        self._download_worker: Optional[UpdateDownloadWorker] = None
        self._latest_tag: str = ""
        self._latest_asset_name: str = ""
        self._latest_asset_url: str = ""
        self._manual_update_worker: Optional[UpdateCheckWorker] = None
        self._last_download_path: Optional[Path] = None

        layout = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel(APP_NAME)
        title.setStyleSheet("font-size: 22px; font-weight: 700;")
        layout.addWidget(title)

        version = QtWidgets.QLabel(f"{tr(self.lang, 'about_version')}: v{APP_VERSION}")
        layout.addWidget(version)

        link_row = QtWidgets.QHBoxLayout()
        self.github_btn = QtWidgets.QPushButton(tr(self.lang, "about_github"))
        self.releases_btn = QtWidgets.QPushButton(tr(self.lang, "about_releases"))
        self.youtube_btn = QtWidgets.QPushButton(tr(self.lang, "about_youtube"))
        link_row.addWidget(self.github_btn)
        link_row.addWidget(self.releases_btn)
        link_row.addWidget(self.youtube_btn)
        layout.addLayout(link_row)

        self.status_label = QtWidgets.QLabel("")
        self.status_label.setStyleSheet("color: #666666;")
        layout.addWidget(self.status_label)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

        self.github_btn.clicked.connect(lambda: self._open_url(GITHUB_PROFILE_URL))
        self.releases_btn.clicked.connect(lambda: self._open_url(GITHUB_RELEASES_URL))
        self.youtube_btn.clicked.connect(lambda: self._open_url(YOUTUBE_URL))

        if not YOUTUBE_URL:
            self.youtube_btn.setEnabled(False)
            self.youtube_btn.setToolTip("Set YOUTUBE_URL in app/main_window.py")

    def _open_url(self, url: str) -> None:
        if not url:
            return
        QtGui.QDesktopServices.openUrl(QtCore.QUrl(url))

class BgmNoteDialog(QtWidgets.QDialog):
    def __init__(self, lang: str, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.lang = lang
        self.setWindowTitle(tr(self.lang, "bgm_mode_title"))
        layout = QtWidgets.QFormLayout(self)

        self.note_edit = QtWidgets.QLineEdit()
        self.duration_spin = QtWidgets.QDoubleSpinBox()
        self.duration_spin.setRange(0.2, 10.0)
        self.duration_spin.setSingleStep(0.1)
        self.duration_spin.setValue(2.0)
        self.timing_combo = QtWidgets.QComboBox()
        self.timing_combo.addItems([tr(self.lang, "bgm_timing"), tr(self.lang, "bgm_duration")])
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems([
            tr(self.lang, "bgm_replace"),
            tr(self.lang, "bgm_add"),
            tr(self.lang, "bgm_metronome"),
        ])
        self.bpm_spin = QtWidgets.QSpinBox()
        self.bpm_spin.setRange(40, 240)
        self.bpm_spin.setValue(60)
        self.mora_spin = QtWidgets.QSpinBox()
        self.mora_spin.setRange(1, 32)
        self.mora_spin.setValue(4)

        self.note_label = QtWidgets.QLabel(tr(self.lang, "bgm_note"))
        self.mode_label = QtWidgets.QLabel(tr(self.lang, "bgm_mode"))
        self.timing_label = QtWidgets.QLabel(tr(self.lang, "bgm_timing"))
        self.duration_label = QtWidgets.QLabel(tr(self.lang, "bgm_duration"))
        self.bpm_label = QtWidgets.QLabel(tr(self.lang, "bpm"))
        self.mora_label = QtWidgets.QLabel(tr(self.lang, "mora_count"))

        layout.addRow(self.note_label, self.note_edit)
        layout.addRow(self.mode_label, self.mode_combo)
        layout.addRow(self.timing_label, self.timing_combo)
        layout.addRow(self.duration_label, self.duration_spin)
        layout.addRow(self.bpm_label, self.bpm_spin)
        layout.addRow(self.mora_label, self.mora_spin)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.timing_combo.currentIndexChanged.connect(self._update_visibility)
        self.mode_combo.currentIndexChanged.connect(self._update_visibility)
        self._update_visibility()

    def _update_visibility(self) -> None:
        timing = self.timing_combo.currentText()
        is_timing = timing == tr(self.lang, "bgm_timing")
        is_metronome = self.mode_combo.currentText() == tr(self.lang, "bgm_metronome")
        self.duration_label.setVisible(not is_timing)
        self.duration_spin.setVisible(not is_timing)
        self.bpm_label.setVisible(is_timing)
        self.bpm_spin.setVisible(is_timing)
        self.mora_label.setVisible(is_timing)
        self.mora_spin.setVisible(is_timing)
        self.note_label.setVisible(not is_metronome)
        self.note_edit.setVisible(not is_metronome)
        if is_metronome:
            self.timing_combo.setVisible(False)
            self.timing_label.setVisible(False)
            self.duration_label.setVisible(True)
            self.duration_spin.setVisible(True)
            self.bpm_label.setVisible(True)
            self.bpm_spin.setVisible(True)
            self.mora_label.setVisible(False)
            self.mora_spin.setVisible(False)
        else:
            self.timing_combo.setVisible(True)
            self.timing_label.setVisible(True)

    def get_data(self) -> Optional[tuple[str, float, str, int, int, str]]:
        if self.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        mode = self.mode_combo.currentText()
        note = self.note_edit.text().strip()
        if mode != tr(self.lang, "bgm_metronome") and not note:
            return None
        return (
            note,
            float(self.duration_spin.value()),
            mode,
            int(self.bpm_spin.value()),
            int(self.mora_spin.value()),
            self.timing_combo.currentText(),
        )


class NoteAnalysisWorker(QtCore.QThread):
    result = QtCore.pyqtSignal(str, float, str)
    progress = QtCore.pyqtSignal(int, int)
    finished = QtCore.pyqtSignal()

    def __init__(
        self,
        files: list[str],
        target_sr: int,
        algo: str,
        cache_dir: Optional[Path],
        ref_bits: int,
        max_workers: int,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.files = files
        self.target_sr = target_sr
        self.algo = algo
        self.cache_dir = cache_dir
        self.ref_bits = ref_bits
        self.max_workers = max(1, int(max_workers))

    def run(self) -> None:
        total = len(self.files)
        self.progress.emit(0, total)
        done = 0
        cache_dir = str(self.cache_dir) if self.cache_dir is not None else None
        try:
            if self.max_workers <= 1 or total <= 1:
                for path in self.files:
                    if self.isInterruptionRequested():
                        break
                    result = _analyze_note_task(
                        path,
                        self.target_sr,
                        self.algo,
                        self.ref_bits,
                        cache_dir,
                    )
                    if result:
                        self.result.emit(*result)
                    done += 1
                    self.progress.emit(done, total)
            else:
                from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
                try:
                    with ProcessPoolExecutor(
                        max_workers=self.max_workers,
                        mp_context=mp.get_context("spawn"),
                    ) as executor:
                        futures = {
                            executor.submit(
                                _analyze_note_task,
                                p,
                                self.target_sr,
                                self.algo,
                                self.ref_bits,
                                cache_dir,
                            ): p
                            for p in self.files
                        }
                        for future in as_completed(futures):
                            if self.isInterruptionRequested():
                                break
                            try:
                                result = future.result()
                            except Exception:
                                logger.exception("Note analysis worker failed")
                                result = None
                            if result:
                                self.result.emit(*result)
                            done += 1
                            self.progress.emit(done, total)
                except Exception:
                    logger.exception("Process pool failed, falling back to threads")
                    with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                        futures = {
                            executor.submit(
                                _analyze_note_task,
                                p,
                                self.target_sr,
                                self.algo,
                                self.ref_bits,
                                cache_dir,
                            ): p
                            for p in self.files
                        }
                        for future in as_completed(futures):
                            if self.isInterruptionRequested():
                                break
                            try:
                                result = future.result()
                            except Exception:
                                logger.exception("Note analysis worker failed")
                                result = None
                            if result:
                                self.result.emit(*result)
                            done += 1
                            self.progress.emit(done, total)
        finally:
            self.finished.emit()


class RecordedAnalysisWorker(QtCore.QThread):
    result = QtCore.pyqtSignal(int, object, object, object, object, object, object, object, bool, bool, bool)
    finished = QtCore.pyqtSignal()

    def __init__(
        self,
        audio: np.ndarray,
        sr: int,
        algo: str,
        ref_bits: int,
        token: int,
        cache_key: Optional[tuple],
        cached_pitch: Optional[tuple],
        cached_mel: Optional[tuple],
        cached_power: Optional[tuple],
        compute_pitch: bool,
        compute_mel: bool,
        compute_power: bool,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.audio = audio
        self.sr = sr
        self.algo = algo
        self.ref_bits = ref_bits
        self.token = token
        self.cache_key = cache_key
        self.cached_pitch = cached_pitch
        self.cached_mel = cached_mel
        self.cached_power = cached_power
        self.compute_pitch = compute_pitch
        self.compute_mel = compute_mel
        self.compute_power = compute_power

    def run(self) -> None:
        try:
            audio = self.audio
            if audio.size == 0:
                return
            if self.isInterruptionRequested():
                return
            pitch_done = False
            mel_done = False
            power_done = False
            if self.compute_pitch:
                if self.algo == "yin":
                    times, f0s = compute_f0_contour_yin(audio, self.sr)
                else:
                    times, f0s = compute_f0_contour(audio, self.sr)
                pitch_done = True
            elif self.cached_pitch:
                times, f0s = self.cached_pitch
                pitch_done = True
            else:
                times, f0s = np.array([], dtype=np.float32), np.array([], dtype=np.float32)
            if self.isInterruptionRequested():
                return
            if self.compute_mel:
                mel_db, mel_times = compute_mel_spectrogram(audio, self.sr)
                mel_done = True
            elif self.cached_mel:
                mel_db, mel_times = self.cached_mel
                mel_done = True
            else:
                mel_db, mel_times = np.array([], dtype=np.float32), np.array([], dtype=np.float32)
            if self.isInterruptionRequested():
                return
            if self.compute_power:
                power_times, power_db = compute_power_db(audio, self.sr, ref_bits=self.ref_bits)
                power_done = True
            elif self.cached_power:
                power_times, power_db = self.cached_power
                power_done = True
            else:
                power_times, power_db = np.array([], dtype=np.float32), np.array([], dtype=np.float32)
            if self.isInterruptionRequested():
                return
            self.result.emit(
                self.token,
                self.cache_key,
                times,
                f0s,
                mel_db,
                mel_times,
                power_times,
                power_db,
                pitch_done,
                mel_done,
                power_done,
            )
        except Exception:
            logger.exception("Failed to update recorded analysis (worker)")
        finally:
            self.finished.emit()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.resize(1200, 800)

        self.settings = QtCore.QSettings("UtauRecorder", "UtauRecorder")
        self.ui_language = self.settings.value("ui_language", "English")
        self.ui_theme = self.settings.value("ui_theme", "light")
        self.pitch_algo = self.settings.value("pitch_algo", "classic")
        self.note_workers = int(self.settings.value("note_workers", 2))
        self.hold_to_record = bool(self.settings.value("hold_to_record", False))
        self.recent_sessions: list[str] = list(self.settings.value("recent_sessions", []))
        self.setWindowTitle(tr(self.ui_language, "app_title"))
        icon_path = Path(__file__).resolve().parent.parent / "icon" / "icon.ico"
        if icon_path.exists():
            self.setWindowIcon(QtGui.QIcon(str(icon_path)))


        self.session: Optional[Session] = None
        self.current_item: Optional[Item] = None
        self.session_path: Optional[Path] = None
        self.record_start_time: Optional[float] = None
        self.voicebank_samples: dict[str, Path] = {}
        self._sung_note_cache: dict[str, tuple[float, str]] = {}
        self._note_worker: Optional[NoteAnalysisWorker] = None
        self._note_analysis_pending: set[str] = set()
        self._analysis_pitch_worker: Optional[RecordedAnalysisWorker] = None
        self._analysis_spectro_worker: Optional[RecordedAnalysisWorker] = None
        self._analysis_token = 0
        self._analysis_cache_limit = 8
        self._recorded_analysis_cache: OrderedDict[tuple, tuple] = OrderedDict()
        self._current_analysis_key: Optional[tuple] = None
        self._current_analysis_meta: Optional[tuple] = None
        self.note_progress: Optional[QtWidgets.QProgressBar] = None
        self._update_check_worker: Optional[UpdateCheckWorker] = None
        self._update_download_worker: Optional[UpdateDownloadWorker] = None
        self._last_update_tag: str = ""
        self._last_update_asset_name: str = ""
        self._last_update_asset_url: str = ""
        self._manual_update_worker: Optional[UpdateCheckWorker] = None
        self._hold_record_active = False

        self.audio = AudioEngine()
        self.audio.error.connect(self._show_error)
        self.audio.status.connect(self._set_status)

        self._build_ui()
        self._build_menu()
        self._connect_actions()
        self._apply_language()
        self._apply_theme()
        self._maybe_show_start_dialog()
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

        self.visual_timer = QtCore.QTimer(self)
        self.visual_timer.setInterval(80)
        self.visual_timer.timeout.connect(self._update_visuals)
        self.visual_timer.start()

        self.play_timer = QtCore.QTimer(self)
        self.play_timer.setInterval(30)
        self.play_timer.timeout.connect(self._update_playhead)
        self.play_timer.start()

        self.autosave_timer = QtCore.QTimer(self)
        self.autosave_timer.setInterval(10000)
        self.autosave_timer.timeout.connect(self._autosave)
        self.autosave_timer.start()

        self.power_history = []
        self.history_size = 200
        self.playing = False
        self.play_start_time: Optional[QtCore.QElapsedTimer] = None
        self.play_start_pos = 0.0
        self.play_duration = 0.0
        self._paused_audio: Optional[np.ndarray] = None
        self._paused_pos = 0.0
        self._paused_sr = 0
        self.selected_audio: Optional[np.ndarray] = None
        self.undo_stack: list[list[dict]] = []
        self._suppress_item_changed = False
        self._note_sort_state = 0

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        main_layout = QtWidgets.QVBoxLayout(central)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter, 1)

        self.table = QtWidgets.QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels([
            tr(self.ui_language, "table_status"),
            tr(self.ui_language, "table_alias"),
            tr(self.ui_language, "table_romaji"),
            tr(self.ui_language, "table_note"),
            tr(self.ui_language, "table_comment"),
            tr(self.ui_language, "table_duration"),
            tr(self.ui_language, "table_file"),
        ])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionsMovable(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.horizontalHeader().sectionClicked.connect(self._table_header_clicked)
        splitter.addWidget(self.table)

        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)

        self.current_label = QtWidgets.QLabel(tr(self.ui_language, "current_item"))
        self.note_label = QtWidgets.QLabel(tr(self.ui_language, "current_note"))
        right_layout.addWidget(self.current_label)
        right_layout.addWidget(self.note_label)

        self.record_btn = QtWidgets.QPushButton(tr(self.ui_language, "record"))
        self.stop_btn = QtWidgets.QPushButton(tr(self.ui_language, "stop"))
        self.rerecord_btn = QtWidgets.QPushButton(tr(self.ui_language, "rerecord"))
        self.preview_btn = QtWidgets.QPushButton(tr(self.ui_language, "preview_bgm"))
        self.preview_overlay_btn = QtWidgets.QPushButton(tr(self.ui_language, "preview_overlay"))
        self.cut_btn = QtWidgets.QPushButton(tr(self.ui_language, "cut_selection"))
        self.select_btn = QtWidgets.QPushButton(tr(self.ui_language, "select_region"))

        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addWidget(self.record_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addWidget(self.rerecord_btn)
        right_layout.addLayout(btn_layout)
        preview_row = QtWidgets.QHBoxLayout()
        preview_row.addWidget(self.preview_btn)
        preview_row.addWidget(self.preview_overlay_btn)
        right_layout.addLayout(preview_row)
        select_cut_layout = QtWidgets.QHBoxLayout()
        select_cut_layout.addWidget(self.select_btn)
        select_cut_layout.addWidget(self.cut_btn)
        right_layout.addLayout(select_cut_layout)

        self.bgm_checkbox = QtWidgets.QCheckBox(tr(self.ui_language, "bgm_during"))
        self.bgm_checkbox.setChecked(True)
        self.auto_next_checkbox = QtWidgets.QCheckBox(tr(self.ui_language, "auto_next"))
        self.auto_next_checkbox.setChecked(True)
        right_layout.addWidget(self.bgm_checkbox)
        right_layout.addWidget(self.auto_next_checkbox)

        self.bgm_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.bgm_slider.setRange(0, 100)
        self.bgm_slider.setValue(50)
        self.bgm_level_label = QtWidgets.QLabel(tr(self.ui_language, "bgm_level"))
        self.bgm_overlay_label = QtWidgets.QLabel(tr(self.ui_language, "bgm_overlay_level"))
        level_row = QtWidgets.QHBoxLayout()
        level_row.addWidget(self.bgm_level_label)
        level_row.addWidget(self.bgm_slider, 1)

        self.bgm_overlay_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.bgm_overlay_slider.setRange(0, 100)
        self.bgm_overlay_slider.setValue(50)
        level_row.addWidget(self.bgm_overlay_label)
        level_row.addWidget(self.bgm_overlay_slider, 1)
        right_layout.addLayout(level_row)

        self.pre_roll_spin = QtWidgets.QSpinBox()
        self.pre_roll_spin.setRange(0, 2000)
        self.pre_roll_spin.setValue(300)
        self.pre_roll_label = QtWidgets.QLabel(tr(self.ui_language, "pre_roll"))
        right_layout.addWidget(self.pre_roll_label)
        right_layout.addWidget(self.pre_roll_spin)

        self.bpm_spin = None
        self.mora_spin = None
        self.bgm_timing_label = None
        self.bpm_label = None
        self.mora_label = None

        right_layout.addStretch(1)
        splitter.addWidget(right_panel)

        self.plot_tabs = QtWidgets.QTabWidget()
        main_layout.addWidget(self.plot_tabs, 1)

        self.wave_plot = pg.PlotWidget(title=tr(self.ui_language, "waveform"))
        self.wave_curve = self.wave_plot.plot(pen="c")
        self.playhead = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("w", width=1))
        self.playhead.setVisible(False)
        self.wave_plot.addItem(self.playhead)
        self.selection_region = pg.LinearRegionItem(values=(0, 0), movable=True, brush=(200, 200, 255, 50))
        self.selection_region.setVisible(False)
        self.wave_plot.addItem(self.selection_region)
        self.plot_tabs.addTab(self.wave_plot, tr(self.ui_language, "waveform"))

        self.spec_plot = pg.PlotWidget(title=tr(self.ui_language, "spectrum"))
        self.spec_curve = self.spec_plot.plot(pen="m")
        self.spec_playhead = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("w", width=1))
        self.spec_playhead.setVisible(False)
        self.spec_plot.addItem(self.spec_playhead)
        self.spec_plot.setXLink(self.wave_plot)
        self.plot_tabs.addTab(self.spec_plot, tr(self.ui_language, "spectrum"))

        self.power_plot = pg.PlotWidget(title=tr(self.ui_language, "power"))
        self.power_curve = self.power_plot.plot(pen="y")
        self.power_playhead = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("w", width=1))
        self.power_playhead.setVisible(False)
        self.power_plot.addItem(self.power_playhead)
        self.power_plot.setXLink(self.wave_plot)
        self.plot_tabs.addTab(self.power_plot, tr(self.ui_language, "power"))

        self.recorded_f0_plot = pg.PlotWidget(
            title=f"{tr(self.ui_language, 'recorded_f0')} (Piano Roll)",
            axisItems={"left": NoteAxis(orientation="left")},
        )
        self.recorded_f0_plot.showGrid(x=True, y=True, alpha=0.2)
        self.recorded_f0_curve = self.recorded_f0_plot.plot(
            pen=pg.mkPen("#22c55e", width=2),
            connect="finite",
        )
        self.recorded_f0_playhead = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#ffcc00", width=1))
        self.recorded_f0_playhead.setVisible(False)
        self.recorded_f0_playhead.setZValue(10)
        self.recorded_f0_plot.addItem(self.recorded_f0_playhead)
        self.recorded_f0_plot.setXLink(self.wave_plot)
        self.plot_tabs.addTab(self.recorded_f0_plot, tr(self.ui_language, "recorded_f0"))

        self.mel_plot = pg.PlotWidget(title=tr(self.ui_language, "mel"))
        self.mel_img = pg.ImageItem()
        self.mel_plot.addItem(self.mel_img)
        self.mel_plot.setLabel("left", "Mel bins")
        self.mel_plot.setLabel("bottom", "Time (s)")
        self.mel_playhead = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("w", width=1))
        self.mel_playhead.setVisible(False)
        self.mel_plot.addItem(self.mel_playhead)
        self.mel_plot.setXLink(self.wave_plot)
        self.plot_tabs.addTab(self.mel_plot, tr(self.ui_language, "mel"))

        self.status_bar = self.statusBar()
        self.note_progress = QtWidgets.QProgressBar()
        self.note_progress.setVisible(False)
        self.note_progress.setMaximumHeight(12)
        self.note_progress.setTextVisible(True)
        self.status_bar.addPermanentWidget(self.note_progress, 1)

        for plot in (
            self.wave_plot,
            self.spec_plot,
            self.power_plot,
            self.recorded_f0_plot,
            self.mel_plot,
        ):
            plot.scene().sigMouseClicked.connect(lambda event, p=plot: self._plot_clicked(p, event))

    def _build_menu(self) -> None:
        menu = self.menuBar()
        self.file_menu = menu.addMenu(tr(self.ui_language, "file"))

        self.new_action = self.file_menu.addAction(tr(self.ui_language, "new_session"))
        self.open_action = self.file_menu.addAction(tr(self.ui_language, "open_session"))
        self.save_action = self.file_menu.addAction(tr(self.ui_language, "save_session"))
        self.save_as_action = QtGui.QAction(tr(self.ui_language, "save_as"), self)
        self.save_reclist_action = self.file_menu.addAction(tr(self.ui_language, "save_reclist_to"))
        self.file_menu.addSeparator()

        self.edit_voicebank_action = self.file_menu.addAction(tr(self.ui_language, "edit_voicebank"))
        self.export_voicebank_action = self.file_menu.addAction(tr(self.ui_language, "export_voicebank"))
        self.export_action = self.file_menu.addAction(tr(self.ui_language, "export_recordings"))
        self.open_folder_action = self.file_menu.addAction(tr(self.ui_language, "open_folder"))
        self.file_menu.addSeparator()
        self.recent_menu = self.file_menu.addMenu(tr(self.ui_language, "recent_sessions"))
        self._rebuild_recent_menu()
        self.file_menu.addSeparator()
        self.back_action = self.file_menu.addAction(tr(self.ui_language, "back_exit"))

        self.import_menu = menu.addMenu(tr(self.ui_language, "import"))
        self.import_reclist_action = self.import_menu.addAction(tr(self.ui_language, "import_reclist"))
        self.import_oremo_action = self.import_menu.addAction(tr(self.ui_language, "import_oremo_comment"))
        self.import_voicebank_action = self.import_menu.addAction(tr(self.ui_language, "import_voicebank"))
        self.import_bgm_action = self.import_menu.addAction(tr(self.ui_language, "import_bgm"))
        self.generate_bgm_action = self.import_menu.addAction(tr(self.ui_language, "generate_bgm"))

        self.tools_menu = menu.addMenu(tr(self.ui_language, "tools"))
        self.vst_batch_action = self.tools_menu.addAction(tr(self.ui_language, "apply_vst"))

        self.settings_menu = menu.addMenu(tr(self.ui_language, "settings"))
        self.session_settings_action = self.settings_menu.addAction(tr(self.ui_language, "session_settings"))
        self.audio_settings_action = self.settings_menu.addAction(tr(self.ui_language, "audio_devices"))
        self.vst_tools_action = self.settings_menu.addAction(tr(self.ui_language, "vst_tools"))
        self.ui_settings_action = self.settings_menu.addAction(tr(self.ui_language, "ui_settings"))

        self.edit_menu = menu.addMenu(tr(self.ui_language, "edit"))
        self.undo_action = self.edit_menu.addAction(tr(self.ui_language, "undo"))
        self.undo_action.setShortcut(QtGui.QKeySequence.StandardKey.Undo)
        self.new_action.setShortcut(QtGui.QKeySequence.StandardKey.New)
        self.save_action.setShortcut(QtGui.QKeySequence.StandardKey.Save)
        self.import_reclist_action.setShortcut(QtGui.QKeySequence("Ctrl+I"))
        self.open_action.setShortcut(QtGui.QKeySequence("Ctrl+Shift+O"))
        self.open_folder_action.setShortcut(QtGui.QKeySequence("Ctrl+O"))
        self.session_settings_action.setShortcut(QtGui.QKeySequence("Ctrl+R"))
        self.export_voicebank_action.setShortcut(QtGui.QKeySequence("Ctrl+Shift+E"))
        self.back_action.setShortcut(QtGui.QKeySequence("Ctrl+B"))

        self.help_menu = menu.addMenu(tr(self.ui_language, "help"))
        self.about_action = self.help_menu.addAction(tr(self.ui_language, "about"))
        self.check_updates_action = self.help_menu.addAction(tr(self.ui_language, "check_updates"))

    def _connect_actions(self) -> None:
        self.new_action.triggered.connect(self._new_session)
        self.open_action.triggered.connect(self._open_session)
        self.open_folder_action.triggered.connect(self._open_session_folder)
        self.save_action.triggered.connect(self._save_session)
        self.save_as_action.triggered.connect(self._save_as_session)
        self.export_action.triggered.connect(self._export_recordings)
        self.export_voicebank_action.triggered.connect(self._export_voicebank)
        self.edit_voicebank_action.triggered.connect(self._edit_voicebank)
        self.save_reclist_action.triggered.connect(self._save_reclist_as)
        self.back_action.triggered.connect(self._back_or_exit)

        self.import_reclist_action.triggered.connect(self._import_reclist)
        self.import_oremo_action.triggered.connect(self._import_oremo_comment)
        self.import_voicebank_action.triggered.connect(self._import_voicebank)
        self.import_bgm_action.triggered.connect(self._import_bgm)
        self.generate_bgm_action.triggered.connect(self._generate_bgm)
        self.vst_batch_action.triggered.connect(self._open_vst_batch)
        self.session_settings_action.triggered.connect(self._open_session_settings)
        self.audio_settings_action.triggered.connect(self._open_audio_settings)
        self.vst_tools_action.triggered.connect(self._open_vst_tools)
        self.ui_settings_action.triggered.connect(self._open_ui_settings)
        self.undo_action.triggered.connect(self._undo)
        self.about_action.triggered.connect(self._open_about)
        self.check_updates_action.triggered.connect(self._manual_check_updates)

        self.table.itemSelectionChanged.connect(self._select_item)
        self.table.itemChanged.connect(self._item_changed)
        self.table.customContextMenuRequested.connect(self._table_context_menu)
        self.record_btn.clicked.connect(self._record)
        self.stop_btn.clicked.connect(self._stop)
        self.rerecord_btn.clicked.connect(self._rerecord)
        self.preview_btn.clicked.connect(self._toggle_preview)
        self.preview_overlay_btn.clicked.connect(self._toggle_preview_overlay)
        self.cut_btn.clicked.connect(self._cut_selection)
        self.select_btn.clicked.connect(self._toggle_selection)

        self.bgm_slider.valueChanged.connect(self._update_bgm_level)
        self.bgm_overlay_slider.valueChanged.connect(self._update_bgm_overlay_level)
        self.pre_roll_spin.valueChanged.connect(self._update_pre_roll)

    def _new_session(self) -> None:
        dialog = NewSessionDialog(self.ui_language, self)
        data = dialog.get_data()
        if not data:
            return
        self._stop_note_worker()
        self._sung_note_cache.clear()
        self.session = Session(
            name=data["name"],
            singer=data["singer"],
            base_path=data["path"],
            sample_rate=data["sample_rate"],
            bit_depth=data["bit_depth"],
            channels=data["channels"],
            output_prefix=data.get("output_prefix", ""),
            output_suffix=data.get("output_suffix", ""),
            target_note=data.get("target_note"),
        )
        self.audio.sample_rate = self.session.sample_rate
        self.audio.channels = self.session.channels
        self._refresh_table()
        self._set_status("Session created")
        self._save_session()
        QtWidgets.QMessageBox.information(
            self,
            "Session Created",
            f"Name: {self.session.name}\n"
            f"Singer: {self.session.singer}\n"
            f"Path: {self.session.session_dir()}\n"
            f"Sample rate: {self.session.sample_rate}\n"
            f"Channels: {self.session.channels}\n"
            f"Output prefix: {self.session.output_prefix or '--'}\n"
            f"Output suffix: {self.session.output_suffix or '--'}",
        )

    def _open_session(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Session", "", "Session (session.json)")
        if not path:
            return
        try:
            self._stop_note_worker()
            self._sung_note_cache.clear()
            self.session = load_session(Path(path))
            self._add_recent_session(path)
            self.audio.sample_rate = self.session.sample_rate
            self.audio.channels = self.session.channels
            self._restore_session_assets()
            self._refresh_table()
            self._start_note_analysis()
            self._set_status("Session loaded")
            bgm_info = self.session.bgm_wav_path or (self.session.bgm_note or "--")
            QtWidgets.QMessageBox.information(
                self,
                "Session Opened",
                f"Name: {self.session.name}\n"
                f"Singer: {self.session.singer}\n"
                f"Path: {self.session.session_dir()}\n"
                f"Output prefix: {self.session.output_prefix or '--'}\n"
                f"Output suffix: {self.session.output_suffix or '--'}\n"
                f"Voicebank BGM: {bgm_info}",
            )
        except Exception as exc:
            logger.exception("Failed to open session")
            self._show_error(str(exc))

    def _save_session(self) -> None:
        if not self.session:
            return
        try:
            self.session_path = save_session(self.session)
            if self.session_path:
                self._add_recent_session(str(self.session_path))
            self._set_status("Session saved")
        except Exception as exc:
            logger.exception("Failed to save session")
            self._show_error(str(exc))

    def _save_as_session(self) -> None:
        if not self.session:
            return
        base = QtWidgets.QFileDialog.getExistingDirectory(self, "Select New Base Folder")
        if not base:
            return
        name, ok = QtWidgets.QInputDialog.getText(self, "New Name", "Session name", text=self.session.name)
        if not ok or not name.strip():
            return
        self.session.base_path = Path(base)
        self.session.name = name.strip()
        self._save_session()

    def _autosave(self) -> None:
        if self.session:
            try:
                save_session(self.session)
            except Exception:
                logger.exception("Autosave failed")

    def _import_reclist(self) -> None:
        if not self.session:
            self._create_temp_session()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import Reclist", "", "Text Files (*.txt)")
        if not path:
            return
        try:
            text = read_text_guess(Path(path))
            choice = QtWidgets.QMessageBox.question(
                self,
                tr(self.ui_language, "import_reclist_title"),
                tr(self.ui_language, "import_reclist_question"),
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            replace_all = choice == QtWidgets.QMessageBox.StandardButton.Yes
            if replace_all:
                self._push_undo_state()
                self.session.items.clear()
            existing = {item.alias for item in self.session.items}
            parsed = [entry for entry in parse_reclist_text(text) if entry[0] not in existing]
            if parsed and not replace_all:
                self._push_undo_state()
            for alias, note, comment in parsed:
                romaji = "_".join(kana_to_romaji_tokens(alias)) if needs_romaji(alias) else None
                item = self.session.add_item(alias, note, romaji=romaji)
                if comment:
                    item.notes = comment
            self._save_reclist_copy(text)
            self._log_event("import_reclist", Path(path).name)
            self._refresh_table()
            self._save_session()
        except Exception as exc:
            logger.exception("Failed to import reclist")
            self._show_error(str(exc))

    def _import_oremo_comment(self) -> None:
        if not self.session:
            self._create_temp_session()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import OREMO Comment", "", "Text Files (*.txt)")
        if not path:
            return
        try:
            text = read_text_guess(Path(path))
            choice = QtWidgets.QMessageBox.question(
                self,
                tr(self.ui_language, "import_oremo_title"),
                tr(self.ui_language, "import_oremo_question"),
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            replace_all = choice == QtWidgets.QMessageBox.StandardButton.Yes
            if replace_all:
                self._push_undo_state()
                self.session.items.clear()
            existing = {item.alias: item for item in self.session.items}
            added = False
            for line in text.splitlines():
                raw = line.strip()
                if not raw or raw.startswith("#") or raw.startswith(";"):
                    continue
                if "\t" in raw:
                    alias, comment = raw.split("\t", 1)
                elif "/t" in raw:
                    alias, comment = raw.split("/t", 1)
                else:
                    alias, comment = raw, ""
                alias = alias.strip()
                comment = comment.strip()
                if not alias:
                    continue
                if alias in existing:
                    if comment:
                        existing[alias].notes = comment
                else:
                    romaji = "_".join(kana_to_romaji_tokens(alias)) if needs_romaji(alias) else None
                    item = self.session.add_item(alias, None, romaji=romaji)
                    if comment:
                        item.notes = comment
                    added = True
            if added and not replace_all:
                self._push_undo_state()
            self._save_reclist_copy(text)
            self._log_event("import_oremo_comment", Path(path).name)
            self._refresh_table()
            self._save_session()
        except Exception as exc:
            logger.exception("Failed to import OREMO comment")
            self._show_error(str(exc))

    def _import_voicebank(self) -> None:
        if not self.session:
            self._create_temp_session()
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Voicebank Folder")
        if not folder:
            return
        prefix, ok = QtWidgets.QInputDialog.getText(self, "Remove Prefix", "Prefix to remove (optional)")
        if not ok:
            return
        suffix, ok = QtWidgets.QInputDialog.getText(self, "Remove Suffix", "Suffix to remove (optional)")
        if not ok:
            return
        dialog = VoicebankImportDialog(self.ui_language, self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        use_bgm = dialog.use_bgm()
        copy_oto = dialog.copy_oto()
        strip_oto_aliases = dialog.strip_oto_aliases()
        try:
            prefix = prefix.strip()
            suffix = suffix.strip()
            folder_path = Path(folder)
            names = import_voicebank(folder_path, prefix=prefix, suffix=suffix)
            existing = {item.alias for item in self.session.items}
            new_names = [name for name in names if name not in existing]
            if new_names:
                self._push_undo_state()
            for name in new_names:
                romaji = "_".join(kana_to_romaji_tokens(name)) if needs_romaji(name) else None
                self.session.add_item(name, romaji=romaji)
            self._refresh_table()
            self.session.voicebank_path = folder_path
            if str(folder_path) not in self.session.voicebank_paths:
                self.session.voicebank_paths.append(str(folder_path))
            self.session.voicebank_prefix = prefix
            self.session.voicebank_suffix = suffix
            self.session.voicebank_use_bgm = use_bgm
            self.session.voicebank_oto_strip_aliases = strip_oto_aliases
            if use_bgm:
                self.session.bgm_override = False
            self._save_session()
            if use_bgm:
                self.voicebank_samples = self._build_voicebank_map(
                    [Path(p) for p in self.session.voicebank_paths],
                    prefix,
                    suffix,
                )
            else:
                self.voicebank_samples = {}
            if use_bgm and copy_oto:
                self._copy_and_adjust_oto(
                    folder_path / "oto.ini",
                    self.session.session_dir() / "oto.ini",
                    remove_prefix=prefix,
                    remove_suffix=suffix,
                    add_prefix=self.session.output_prefix,
                    add_suffix=self.session.output_suffix,
                    strip_aliases=strip_oto_aliases,
                )
            self._log_event("import_voicebank", folder_path.name)
        except Exception as exc:
            logger.exception("Voicebank import failed")
            self._show_error(str(exc))

    def _import_bgm(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import BGM WAV", "", "WAV Files (*.wav)")
        if not path:
            return
        try:
            copied = self._copy_to_session(Path(path), "BGM")
            self.audio.load_bgm_wav(copied)
            if self.session:
                self.session.bgm_wav_path = str(Path("BGM") / copied.name)
                self.session.bgm_note = None
                self.session.bgm_override = True
                self._save_session()
                self._log_event("import_bgm", copied.name)
        except Exception as exc:
            logger.exception("Failed to load BGM")
            self._show_error(str(exc))

    def _generate_bgm(self) -> None:
        dialog = BgmNoteDialog(self.ui_language, self)
        data = dialog.get_data()
        if not data:
            return
        note, dur, mode_text, bpm, mora, timing_mode = data
        try:
            if mode_text == tr(self.ui_language, "bgm_add"):
                self.audio.set_bgm_overlay(note.strip(), dur)
                if self.session:
                    self.session.bgm_overlay_note = note.strip()
                    self.session.bgm_overlay_duration = dur
                    self.session.bgm_overlay_enabled = True
                    self._save_session()
                    self._log_event("bgm_overlay", note.strip())
                    self._refresh_table()
                    self._start_note_analysis()
                return

            if mode_text == tr(self.ui_language, "bgm_metronome"):
                self.audio.generate_metronome(float(bpm), float(dur))
                if self.session:
                    self.session.bgm_note = None
                    self.session.bgm_wav_path = None
                    self.session.bgm_overlay_enabled = False
                    self.session.bgm_overlay_note = None
                    self.session.bgm_overlay_duration = None
                    self.session.bgm_override = True
                    self._save_session()
                    self._log_event("bgm_metronome", str(bpm))
                    self._save_generated_bgm("metronome", dur)
                    self._refresh_table()
                    self._start_note_analysis()
                return

            if timing_mode == tr(self.ui_language, "bgm_timing") and not (self.session and self.session.voicebank_use_bgm):
                gap = min(0.06, 0.35 * (60.0 / bpm))
                self.audio.generate_bgm_mora(
                    note.strip(),
                    bpm=bpm,
                    mora_count=mora,
                    gap_sec=gap,
                    pre_silence_sec=gap,
                    post_silence_sec=gap,
                )
            else:
                self.audio.generate_bgm(note.strip(), dur)
            if self.session:
                self.session.bgm_note = note.strip()
                self.session.bgm_wav_path = None
                self.session.bgm_overlay_enabled = False
                self.session.bgm_overlay_note = None
                self.session.bgm_overlay_duration = None
                self.session.bgm_override = True
                self._save_session()
                self._log_event("bgm_generate", note.strip())
                self._save_generated_bgm(note.strip(), dur)
                self._refresh_table()
                self._start_note_analysis()
        except Exception as exc:
            self._show_error(str(exc))

    def _export_recordings(self) -> None:
        if not self.session:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export Recordings", "recordings.json", "JSON (*.json)")
        if not path:
            return
        try:
            export_recordings_json(self.session, Path(path))
            self._set_status("Exported recordings JSON")
        except Exception as exc:
            self._show_error(str(exc))

    def _save_reclist_as(self) -> None:
        if not self.session:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Reclist", "reclist.txt", "Text Files (*.txt)")
        if not path:
            return
        try:
            lines = []
            for item in self.session.items:
                if item.notes:
                    lines.append(f"{item.alias}\\t{item.notes}")
                elif item.note:
                    lines.append(f"{item.alias}\\t{item.note}")
                else:
                    lines.append(item.alias)
            Path(path).write_text("\n".join(lines), encoding="utf-8")
            self._set_status("Reclist saved")
        except Exception as exc:
            self._show_error(str(exc))

    def _restore_session_assets(self) -> None:
        if not self.session:
            return
        if self.session.voicebank_use_bgm:
            paths = []
            if self.session.voicebank_paths:
                paths = [Path(p) for p in self.session.voicebank_paths if Path(p).exists()]
            elif self.session.voicebank_path and self.session.voicebank_path.exists():
                paths = [self.session.voicebank_path]
            if paths:
                self.voicebank_samples = self._build_voicebank_map(
                    paths,
                    self.session.voicebank_prefix,
                    self.session.voicebank_suffix,
                )
            else:
                self.voicebank_samples = {}
        else:
            self.voicebank_samples = {}
        if self.session.bgm_wav_path:
            try:
                bgm_path = Path(self.session.bgm_wav_path)
                if not bgm_path.is_absolute():
                    bgm_path = self.session.session_dir() / bgm_path
                self.audio.load_bgm_wav(bgm_path)
            except Exception:
                logger.exception("Failed to load saved BGM WAV")
        elif self.session.bgm_note:
            try:
                self.audio.generate_bgm(self.session.bgm_note, 2.0)
            except Exception:
                logger.exception("Failed to generate saved BGM note")
        if self.session.bgm_overlay_enabled and self.session.bgm_overlay_note:
            try:
                self.audio.set_bgm_overlay(
                    self.session.bgm_overlay_note,
                    float(self.session.bgm_overlay_duration or 2.0),
                )
            except Exception:
                logger.exception("Failed to set saved BGM overlay")

    def _open_session_settings(self) -> None:
        if not self.session:
            self._show_error(tr(self.ui_language, "vst_no_session"))
            return
        if self.audio.recording or self.audio.preview:
            QtWidgets.QMessageBox.warning(
                self,
                tr(self.ui_language, "session_settings_title"),
                tr(self.ui_language, "session_settings_busy"),
            )
            return
        dialog = SessionSettingsDialog(self.ui_language, self.session, self)
        data = dialog.get_data()
        if not data:
            return
        old_prefix = self.session.output_prefix
        old_suffix = self.session.output_suffix
        prefix_changed = data["output_prefix"] != old_prefix
        suffix_changed = data["output_suffix"] != old_suffix

        rename_files = False
        if (prefix_changed or suffix_changed) and any(item.wav_path for item in self.session.items):
            result = QtWidgets.QMessageBox.question(
                self,
                tr(self.ui_language, "rename_files_title"),
                tr(self.ui_language, "rename_files_prompt"),
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No
                | QtWidgets.QMessageBox.StandardButton.Cancel,
            )
            if result == QtWidgets.QMessageBox.StandardButton.Cancel:
                return
            rename_files = result == QtWidgets.QMessageBox.StandardButton.Yes

        sample_rate_changed = data["sample_rate"] != self.session.sample_rate
        channels_changed = data["channels"] != self.session.channels

        self.session.sample_rate = data["sample_rate"]
        self.session.bit_depth = data["bit_depth"]
        self.session.channels = data["channels"]
        self.session.output_prefix = data["output_prefix"]
        self.session.output_suffix = data["output_suffix"]
        self.session.target_note = data.get("target_note")

        if sample_rate_changed or channels_changed:
            self.audio.sample_rate = self.session.sample_rate
            self.audio.channels = self.session.channels
            if self.audio.stream:
                self.audio.stream.stop()
                self.audio.stream.close()
                self.audio.stream = None
            self.audio._ring = RingBuffer(size=self.session.sample_rate * 5)
            self.audio.set_pre_roll_ms(self.pre_roll_spin.value())

        if rename_files:
            ok, mapping = self._rename_recordings_for_prefix_suffix(
                self.session.output_prefix,
                self.session.output_suffix,
            )
            if not ok:
                QtWidgets.QMessageBox.warning(
                    self,
                    tr(self.ui_language, "rename_files_title"),
                    tr(self.ui_language, "rename_files_failed"),
                )
            for old_abs, new_abs in mapping.items():
                cached = self._sung_note_cache.pop(old_abs, None)
                if cached:
                    self._sung_note_cache[new_abs] = cached

        self._save_session()
        self._refresh_table()
        self._start_note_analysis()

    def _open_audio_settings(self) -> None:
        dialog = AudioSettingsDialog(self.ui_language, self)
        current = self.audio.device
        if current is not None:
            dialog.set_selected(current[0], current[1])
        else:
            default_in, default_out = sd.default.device
            dialog.set_selected(default_in, default_out)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        input_dev, output_dev = dialog.get_selected()
        try:
            self.audio.set_devices(input_dev, output_dev)
            self._set_status("Audio devices updated")
        except Exception as exc:
            self._show_error(str(exc))

    def _open_ui_settings(self) -> None:
        dialog = UiSettingsDialog(
            self.ui_language,
            str(self.ui_language),
            str(self.ui_theme),
            str(self.pitch_algo),
            int(self.note_workers),
            bool(self.hold_to_record),
            self,
        )
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        lang, theme, pitch_algo, note_workers, hold_to_record = dialog.get_data()
        self.ui_language = lang
        self.ui_theme = theme
        self.pitch_algo = pitch_algo
        self.note_workers = int(note_workers)
        self.hold_to_record = bool(hold_to_record)
        self.settings.setValue("ui_language", lang)
        self.settings.setValue("ui_theme", theme)
        self.settings.setValue("pitch_algo", pitch_algo)
        self.settings.setValue("note_workers", int(note_workers))
        self.settings.setValue("hold_to_record", bool(hold_to_record))
        self._apply_language()
        self._apply_theme()
        self._sung_note_cache.clear()
        self._start_note_analysis()
        self._update_recorded_analysis()

    def _open_about(self) -> None:
        dialog = AboutDialog(self.ui_language, self)
        dialog.exec()

    def _manual_check_updates(self) -> None:
        if self._manual_update_worker and self._manual_update_worker.isRunning():
            return
        self._manual_update_worker = UpdateCheckWorker(self)
        self._manual_update_worker.result.connect(self._on_manual_update_result)
        self._manual_update_worker.start()

    def _on_manual_update_result(self, tag: str, asset_name: str, asset_url: str, error: str) -> None:
        if error or not tag:
            QtWidgets.QMessageBox.warning(
                self,
                tr(self.ui_language, "check_updates"),
                tr(self.ui_language, "about_update_failed"),
            )
            return
        if _is_version_newer(APP_VERSION, tag):
            QtWidgets.QMessageBox.information(
                self,
                tr(self.ui_language, "check_updates"),
                tr(self.ui_language, "about_current_newer"),
            )
            return
        if not _is_version_newer(tag, APP_VERSION):
            QtWidgets.QMessageBox.information(
                self,
                tr(self.ui_language, "check_updates"),
                f"{tr(self.ui_language, 'about_up_to_date')} (v{APP_VERSION})",
            )
            return
        self._last_update_tag = tag
        self._last_update_asset_name = asset_name
        self._last_update_asset_url = asset_url
        msg = f"{tr(self.ui_language, 'about_update_available')}: {tag}"
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle(tr(self.ui_language, "check_updates"))
        box.setText(msg)
        download_btn = box.addButton(tr(self.ui_language, "about_download_update"), QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        if not asset_url:
            download_btn.setEnabled(False)
        box.exec()
        if box.clickedButton() == download_btn:
            self._download_update(asset_url, asset_name, tag)

    def _init_update_check(self) -> None:
        QtCore.QTimer.singleShot(800, self._auto_check_updates)

    def _auto_check_updates(self) -> None:
        if self._update_check_worker and self._update_check_worker.isRunning():
            return
        self._update_check_worker = UpdateCheckWorker(self)
        self._update_check_worker.result.connect(self._on_auto_update_result)
        self._update_check_worker.start()

    def _on_auto_update_result(self, tag: str, asset_name: str, asset_url: str, error: str) -> None:
        if error or not tag:
            return
        if not _is_version_newer(tag, APP_VERSION):
            return
        ignored = str(self.settings.value("update_ignore_tag", ""))
        if ignored and ignored == tag:
            return
        self._last_update_tag = tag
        self._last_update_asset_name = asset_name
        self._last_update_asset_url = asset_url

        msg = f"{tr(self.ui_language, 'about_update_available')}: {tag}"
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle(tr(self.ui_language, "about_check_updates"))
        box.setText(msg)
        download_btn = box.addButton(tr(self.ui_language, "about_download_update"), QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        later_btn = box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        ignore_btn = box.addButton(tr(self.ui_language, "about_ignore_until_next"), QtWidgets.QMessageBox.ButtonRole.DestructiveRole)
        if not asset_url:
            download_btn.setEnabled(False)
        box.exec()
        clicked = box.clickedButton()
        if clicked == ignore_btn:
            self.settings.setValue("update_ignore_tag", tag)
            return
        if clicked != download_btn:
            return
        self._download_update(asset_url, asset_name, tag)

    def _download_update(self, asset_url: str, asset_name: str, tag: str) -> None:
        suggested = asset_name or f"{APP_NAME}_{tag}.zip"
        target, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            tr(self.ui_language, "about_download_update"),
            suggested,
            "Archives (*.zip *.rar);;All Files (*)",
        )
        if not target:
            return
        self._update_download_worker = UpdateDownloadWorker(asset_url, Path(target), self)
        self._update_download_worker.finished.connect(self._on_auto_download_done)
        self._update_download_worker.start()

    def _on_auto_download_done(self, ok: bool, info: str) -> None:
        if not ok:
            QtWidgets.QMessageBox.warning(
                self,
                tr(self.ui_language, "about_download_update"),
                tr(self.ui_language, "about_download_failed"),
            )
            return
        path = Path(info)
        if path.suffix.lower() == ".zip":
            try:
                extract_dir = path.with_suffix("")
                extract_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(path, "r") as zf:
                    zf.extractall(extract_dir)
            except Exception:
                QtWidgets.QMessageBox.warning(
                    self,
                    tr(self.ui_language, "about_download_update"),
                    tr(self.ui_language, "about_extract_failed"),
                )
                return
        QtWidgets.QMessageBox.information(
            self,
            tr(self.ui_language, "about_download_update"),
            tr(self.ui_language, "about_restart_required"),
        )

    def _open_vst_tools(self) -> None:
        dialog = VstToolsDialog(self.ui_language, self.settings, self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        dialog.save()

    def _open_vst_batch(self) -> None:
        if not self.session:
            QtWidgets.QMessageBox.warning(self, tr(self.ui_language, "apply_vst"), tr(self.ui_language, "vst_no_session"))
            return
        dialog = VstBatchDialog(lambda key: tr(self.ui_language, key), self.settings, self.session, self)
        dialog.exec()

    def _back_or_exit(self) -> None:
        self.close()

    def _open_session_folder(self) -> None:
        if not self.session:
            return
        folder = self.session.session_dir()
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(folder)))

    @staticmethod
    def _adjust_name(value: str, remove_prefix: str, remove_suffix: str, add_prefix: str, add_suffix: str) -> str:
        base = value
        if remove_prefix and base.startswith(remove_prefix):
            base = base[len(remove_prefix):]
        if remove_suffix and base.endswith(remove_suffix):
            base = base[: -len(remove_suffix)]
        return f"{add_prefix}{base}{add_suffix}"

    @staticmethod
    def _detect_text_encoding(path: Path) -> str:
        raw = path.read_bytes()
        if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
            return "utf-16"
        if raw.startswith(b"\xef\xbb\xbf"):
            return "utf-8-sig"
        for enc in (
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
        ):
            try:
                raw.decode(enc)
                return enc
            except UnicodeDecodeError:
                continue
        return "utf-8"

    def _copy_and_adjust_oto(
        self,
        src_path: Path,
        dst_path: Path,
        remove_prefix: str,
        remove_suffix: str,
        add_prefix: str,
        add_suffix: str,
        strip_aliases: bool,
    ) -> None:
        if not src_path.exists():
            return
        encoding_in = self._detect_text_encoding(src_path)
        lines = src_path.read_text(encoding=encoding_in, errors="replace").splitlines()
        out_lines = []
        for line in lines:
            raw = line.strip()
            if not raw or raw.startswith("#") or "=" not in raw:
                out_lines.append(line)
                continue
            left, right = raw.split("=", 1)
            filename = left.strip()
            rest = right
            alias = ""
            if "," in rest:
                alias, rest_tail = rest.split(",", 1)
                alias = alias.strip()
                rest = rest_tail
            stem = Path(filename).stem
            ext = Path(filename).suffix
            new_stem = self._adjust_name(stem, remove_prefix, remove_suffix, add_prefix, add_suffix)
            new_filename = f"{new_stem}{ext}"
            if strip_aliases and alias:
                alias = self._adjust_name(alias, remove_prefix, remove_suffix, "", "")
            new_line = f"{new_filename}={alias}"
            if rest:
                new_line = f"{new_line},{rest}"
            out_lines.append(new_line)
        try:
            dst_path.write_text("\n".join(out_lines), encoding=encoding_in, errors="strict")
        except UnicodeEncodeError:
            dst_path.write_text("\n".join(out_lines), encoding="utf-8", errors="replace")

    def _copy_and_adjust_oto_alias(
        self,
        src_path: Path,
        dst_path: Path,
        add_prefix: str,
        add_suffix: str,
    ) -> None:
        if not src_path.exists():
            return
        encoding_in = self._detect_text_encoding(src_path)
        lines = src_path.read_text(encoding=encoding_in, errors="replace").splitlines()
        out_lines = []
        for line in lines:
            raw = line.strip()
            if not raw or raw.startswith("#") or "=" not in raw:
                out_lines.append(line)
                continue
            left, right = raw.split("=", 1)
            filename = left.strip()
            rest = right
            alias = ""
            if "," in rest:
                alias, rest_tail = rest.split(",", 1)
                alias = alias.strip()
                rest = rest_tail
            if alias:
                alias = self._adjust_name(alias, "", "", add_prefix, add_suffix)
            new_line = f"{filename}={alias}"
            if rest:
                new_line = f"{new_line},{rest}"
            out_lines.append(new_line)
        try:
            dst_path.write_text("\n".join(out_lines), encoding=encoding_in, errors="strict")
        except UnicodeEncodeError:
            dst_path.write_text("\n".join(out_lines), encoding="utf-8", errors="replace")

    @staticmethod
    def _sanitize_folder_name(name: str) -> str:
        bad = '<>:"/\\|?*'
        cleaned = "".join("_" if ch in bad else ch for ch in name).strip()
        return cleaned or "voicebank"

    def _export_voicebank(self) -> None:
        if not self.session:
            return
        base = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            tr(self.ui_language, "export_voicebank_title"),
            str(self.session.session_dir()),
        )
        if not base:
            return
        folder_name = self._sanitize_folder_name(f"{self.session.singer} {self.session.name}")
        out_dir = Path(base) / folder_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for item in self.session.items:
            if not item.wav_path:
                continue
            src = Path(item.wav_path)
            if not src.is_absolute():
                src = self.session.session_dir() / src
            if not src.exists():
                continue
            shutil.copy2(src, out_dir / src.name)

        src_vb = self.session.session_dir()
        if src_vb and src_vb.exists():
            oto_src = self.session.session_dir() / "oto.ini"
            if oto_src.exists():
                if self.session.voicebank_oto_strip_aliases:
                    self._copy_and_adjust_oto_alias(
                        oto_src,
                        out_dir / "oto.ini",
                        add_prefix=self.session.output_prefix,
                        add_suffix=self.session.output_suffix,
                    )
                else:
                    shutil.copy2(oto_src, out_dir / "oto.ini")
            for pattern in ("character.txt", "character.yaml", "character.yml"):
                src_file = src_vb / pattern
                if src_file.exists():
                    shutil.copy2(src_file, out_dir / src_file.name)
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif", "*.webp"):
                for img in src_vb.glob(ext):
                    shutil.copy2(img, out_dir / img.name)
        else:
            # character.txt (minimal)
            char_path = out_dir / "character.txt"
            char_path.write_text(
                f"name={self.session.singer}\n"
                f"author={self.session.singer}\n"
                f"comment={self.session.name}\n",
                encoding="utf-8",
            )

        QtWidgets.QMessageBox.information(
            self,
            tr(self.ui_language, "export_voicebank_title"),
            tr(self.ui_language, "export_voicebank_done"),
        )

    def _edit_voicebank(self) -> None:
        if not self.session:
            return
        folder = self.session.session_dir()
        if not folder or not folder.exists():
            folder_str = QtWidgets.QFileDialog.getExistingDirectory(
                self,
                tr(self.ui_language, "edit_voicebank"),
                str(self.session.session_dir()),
            )
            if not folder_str:
                return
            folder = Path(folder_str)
        dialog = VoicebankConfigDialog(folder, lambda k: tr(self.ui_language, k), self.ui_language, self)
        dialog.exec()

    def _maybe_show_start_dialog(self) -> None:
        recent_entries = []
        for path in self.recent_sessions[:10]:
            try:
                session = load_session(Path(path))
                recorded = sum(1 for item in session.items if item.status != ItemStatus.PENDING)
                total = len(session.items)
                title = f"{session.singer} — {session.name}  [{recorded}/{total}]"
            except Exception:
                title = Path(path).stem
            try:
                rel_path = Path(path).resolve().relative_to(Path.cwd().resolve())
                rel_text = str(rel_path)
            except Exception:
                rel_text = str(Path(path))
            recent_entries.append({
                "title": title,
                "path": rel_text,
                "full_path": path,
            })
        dialog = StartDialog(self.ui_language, recent_entries, self)
        dialog.show()
        QtCore.QTimer.singleShot(600, self._auto_check_updates)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            if dialog.closed_via_x:
                self.close()
            return
        if dialog.action == "new":
            self._new_session()
        elif dialog.action == "open":
            self._open_session()
        elif dialog.action == "recent" and dialog.selected_path:
            self._open_recent_path(dialog.selected_path)
        elif dialog.action == "settings":
            self._open_ui_settings()

    def _open_recent_path(self, path: str) -> None:
        try:
            self._stop_note_worker()
            self._sung_note_cache.clear()
            self.session = load_session(Path(path))
            self._add_recent_session(path)
            self.audio.sample_rate = self.session.sample_rate
            self.audio.channels = self.session.channels
            self._restore_session_assets()
            self._refresh_table()
            self._start_note_analysis()
            self._set_status("Session loaded")
        except Exception as exc:
            logger.exception("Failed to open recent session")
            self._show_error(str(exc))

    def _add_recent_session(self, path: str) -> None:
        path = str(Path(path))
        if path in self.recent_sessions:
            self.recent_sessions.remove(path)
        self.recent_sessions.insert(0, path)
        self.recent_sessions = self.recent_sessions[:10]
        self.settings.setValue("recent_sessions", self.recent_sessions)
        self._rebuild_recent_menu()

    def _rebuild_recent_menu(self) -> None:
        if not hasattr(self, "recent_menu"):
            return
        self.recent_menu.clear()
        if not self.recent_sessions:
            empty = QtGui.QAction("-", self)
            empty.setEnabled(False)
            self.recent_menu.addAction(empty)
            return
        for path in self.recent_sessions[:10]:
            action = QtGui.QAction(path, self)
            action.triggered.connect(lambda checked=False, p=path: self._open_recent_path(p))
            self.recent_menu.addAction(action)

    def _create_temp_session(self) -> None:
        if self.session:
            return
        name = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        singer = "Unknown"
        base_path = Path.cwd() / "recordings" / singer / name
        self.session = Session(
            name=name,
            singer=singer,
            base_path=base_path,
            sample_rate=44100,
            bit_depth=16,
            channels=1,
        )
        self.audio.sample_rate = self.session.sample_rate
        self.audio.channels = self.session.channels
        self._refresh_table()
        self._save_session()
        if self.session_path:
            self._add_recent_session(str(self.session_path))

    def _save_reclist_copy(self, text: str) -> None:
        if not self.session:
            return
        try:
            path = self.session.session_dir() / "reclist.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        except Exception:
            logger.exception("Failed to save reclist copy")

    def _copy_to_session(self, src: Path, folder: str) -> Path:
        if not self.session:
            return src
        dst_dir = self.session.session_dir() / folder
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        shutil.copy2(src, dst)
        return dst

    def _save_generated_bgm(self, note: str, dur: float) -> None:
        if not self.session:
            return
        try:
            audio = self.audio.get_bgm_data()
            if audio is None or audio.size == 0:
                return
            safe_note = note.replace(" ", "_")
            name = f"bgm_{safe_note}.wav"
            out_path = self.session.session_dir() / "BGM" / name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out_path), audio, self.audio.sample_rate)
            self.session.bgm_wav_path = str(Path("BGM") / name)
            self._save_session()
        except Exception:
            logger.exception("Failed to save generated BGM")

    def _log_event(self, event: str, detail: str = "") -> None:
        if not self.session:
            return
        try:
            log_path = self.session.session_dir() / "event_log.txt"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().isoformat(timespec="seconds")
            line = f"{timestamp}\t{event}\t{detail}\n"
            with log_path.open("a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            logger.exception("Failed to write event log")

    def _apply_language(self) -> None:
        self.setWindowTitle(tr(self.ui_language, "app_title"))
        icon_path = Path(__file__).resolve().parent.parent / "icon" / "icon.ico"
        if icon_path.exists():
            self.setWindowIcon(QtGui.QIcon(str(icon_path)))

        self.file_menu.setTitle(tr(self.ui_language, "file"))
        self.import_menu.setTitle(tr(self.ui_language, "import"))
        self.tools_menu.setTitle(tr(self.ui_language, "tools"))
        self.settings_menu.setTitle(tr(self.ui_language, "settings"))
        self.edit_menu.setTitle(tr(self.ui_language, "edit"))
        if hasattr(self, "help_menu"):
            self.help_menu.setTitle(tr(self.ui_language, "help"))

        self.new_action.setText(tr(self.ui_language, "new_session"))
        self.open_action.setText(tr(self.ui_language, "open_session"))
        self.save_action.setText(tr(self.ui_language, "save_session"))
        self.save_as_action.setText(tr(self.ui_language, "save_as"))
        self.export_action.setText(tr(self.ui_language, "export_recordings"))
        self.open_folder_action.setText(tr(self.ui_language, "open_folder"))
        self.export_voicebank_action.setText(tr(self.ui_language, "export_voicebank"))
        self.edit_voicebank_action.setText(tr(self.ui_language, "edit_voicebank"))
        self.save_reclist_action.setText(tr(self.ui_language, "save_reclist_to"))
        self.recent_menu.setTitle(tr(self.ui_language, "recent_sessions"))
        if hasattr(self, "back_action"):
            self.back_action.setText(tr(self.ui_language, "back_exit"))
        self.import_reclist_action.setText(tr(self.ui_language, "import_reclist"))
        self.import_oremo_action.setText(tr(self.ui_language, "import_oremo_comment"))
        self.import_voicebank_action.setText(tr(self.ui_language, "import_voicebank"))
        self.import_bgm_action.setText(tr(self.ui_language, "import_bgm"))
        self.generate_bgm_action.setText(tr(self.ui_language, "generate_bgm"))
        self.vst_batch_action.setText(tr(self.ui_language, "apply_vst"))
        self.session_settings_action.setText(tr(self.ui_language, "session_settings"))
        self.audio_settings_action.setText(tr(self.ui_language, "audio_devices"))
        self.vst_tools_action.setText(tr(self.ui_language, "vst_tools"))
        self.ui_settings_action.setText(tr(self.ui_language, "ui_settings"))
        self.undo_action.setText(tr(self.ui_language, "undo"))
        if hasattr(self, "about_action"):
            self.about_action.setText(tr(self.ui_language, "about"))
        if hasattr(self, "check_updates_action"):
            self.check_updates_action.setText(tr(self.ui_language, "check_updates"))

        if self.current_item:
            duration = ""
            if self.current_item.duration_sec:
                duration = f" ({self.current_item.duration_sec:.2f}s)"
            self.current_label.setText(
                f"{tr(self.ui_language, 'current_item_prefix')}{self.current_item.alias}{duration}"
            )
        else:
            self.current_label.setText(tr(self.ui_language, "current_item"))
        if self.note_label.text().endswith("--"):
            self.note_label.setText(tr(self.ui_language, "current_note"))
        self.record_btn.setText(tr(self.ui_language, "record"))
        self.stop_btn.setText(tr(self.ui_language, "stop"))
        self.rerecord_btn.setText(tr(self.ui_language, "rerecord"))
        self.preview_btn.setText(tr(self.ui_language, "preview_bgm"))
        self.preview_overlay_btn.setText(tr(self.ui_language, "preview_overlay"))
        self.cut_btn.setText(tr(self.ui_language, "cut_selection"))
        self.select_btn.setText(tr(self.ui_language, "select_region"))
        self.bgm_checkbox.setText(tr(self.ui_language, "bgm_during"))
        self.auto_next_checkbox.setText(tr(self.ui_language, "auto_next"))
        self.bgm_level_label.setText(tr(self.ui_language, "bgm_level"))
        self.bgm_overlay_label.setText(tr(self.ui_language, "bgm_overlay_level"))
        self.pre_roll_label.setText(tr(self.ui_language, "pre_roll"))

        self.table.setHorizontalHeaderLabels([
            tr(self.ui_language, "table_status"),
            tr(self.ui_language, "table_alias"),
            tr(self.ui_language, "table_romaji"),
            tr(self.ui_language, "table_note"),
            tr(self.ui_language, "table_comment"),
            tr(self.ui_language, "table_duration"),
            tr(self.ui_language, "table_file"),
        ])

        self.wave_plot.setTitle(tr(self.ui_language, "waveform"))
        self.spec_plot.setTitle(tr(self.ui_language, "spectrum"))
        self.power_plot.setTitle(tr(self.ui_language, "power"))
        self.recorded_f0_plot.setTitle(f"{tr(self.ui_language, 'recorded_f0')} (Piano Roll)")
        self.mel_plot.setTitle(tr(self.ui_language, "mel"))

        if self.plot_tabs.count() >= 5:
            self.plot_tabs.setTabText(0, tr(self.ui_language, "waveform"))
            self.plot_tabs.setTabText(1, tr(self.ui_language, "spectrum"))
            self.plot_tabs.setTabText(2, tr(self.ui_language, "power"))
            self.plot_tabs.setTabText(3, tr(self.ui_language, "recorded_f0"))
            self.plot_tabs.setTabText(4, tr(self.ui_language, "mel"))

    def _apply_theme(self) -> None:
        if self.ui_theme == "dark":
            self.setStyleSheet(
                """
                QWidget { background: #1e1e1e; color: #e6e6e6; }
                QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QListView {
                    background: #2a2a2a; color: #e6e6e6; border: 1px solid #3a3a3a;
                }
                QPushButton { background: #2f2f2f; border: 1px solid #3a3a3a; padding: 4px; }
                QPushButton:hover { background: #3a3a3a; }
                QTableWidget { gridline-color: #3a3a3a; }
                QHeaderView::section { background: #2a2a2a; color: #e6e6e6; border: 1px solid #3a3a3a; }
                QTabWidget::pane { border: 1px solid #3a3a3a; }
                """
            )
        else:
            self.setStyleSheet("")

    def _select_item(self) -> None:
        row = self.table.currentRow()
        if row < 0 or not self.session:
            self.current_item = None
            return
        index_item = self.table.item(row, 0)
        if index_item is not None:
            model_index = index_item.data(QtCore.Qt.ItemDataRole.UserRole)
        else:
            model_index = None
        if model_index is None:
            model_index = row
        try:
            model_index = int(model_index)
        except (TypeError, ValueError):
            model_index = row
        if model_index < 0 or model_index >= len(self.session.items):
            model_index = row
        self.current_item = self.session.items[model_index]
        duration = ""
        if self.current_item.duration_sec:
            duration = f" ({self.current_item.duration_sec:.2f}s)"
        self.current_label.setText(
            f"{tr(self.ui_language, 'current_item_prefix')}{self.current_item.alias}{duration}"
        )
        if self.selection_region:
            self.selection_region.setVisible(False)
        self._analyze_selected_item()

    def _record(self) -> None:
        if not self.session:
            self._show_error("Create or open a session first")
            return
        if not self.current_item:
            self._show_error("Select an item to record")
            return
        self.selected_audio = None
        self._clear_analysis()
        self._stop_playback()
        out_dir = self.session.recordings_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = self.session.output_prefix if self.session else ""
        suffix = self.session.output_suffix if self.session else ""
        out_path = out_dir / f"{prefix}{self.current_item.alias}{suffix}.wav"
        try:
            if (
                self.bgm_checkbox.isChecked()
                and self.session
                and self.session.voicebank_use_bgm
                and not self.session.bgm_override
                and self.current_item.alias in self.voicebank_samples
            ):
                self.audio.load_bgm_wav(self.voicebank_samples[self.current_item.alias])
            self.audio.set_pre_roll_ms(self.pre_roll_spin.value())
            self.audio.start_recording(out_path, self.bgm_checkbox.isChecked())
            self._log_event("record_start", self.current_item.alias)
        except Exception as exc:
            logger.exception("Recording failed")
            self._show_error(str(exc))

    def _stop(self) -> None:
        if self.audio.recording:
            self.audio.stop_recording()
            self._finalize_recording()
            self._hold_record_active = False
        elif self.audio.preview:
            self.audio.stop_bgm()
        elif self.playing:
            self._stop_playback()

    def _rerecord(self) -> None:
        if self.current_item:
            self._log_event("retake", self.current_item.alias)
        self._record()

    def _toggle_preview(self) -> None:
        try:
            if self.audio.preview:
                self.audio.stop_bgm()
                self.audio.set_overlay_enabled(True)
            else:
                self._stop_playback()
                self.audio.set_overlay_enabled(False)
                if (
                    self.current_item
                    and self.session
                    and self.session.voicebank_use_bgm
                    and not self.session.bgm_override
                    and self.current_item.alias in self.voicebank_samples
                ):
                    self.audio.load_bgm_wav(self.voicebank_samples[self.current_item.alias])
                self.audio.play_bgm()
        except Exception as exc:
            self._show_error(str(exc))

    def _toggle_preview_overlay(self) -> None:
        try:
            if self.audio.preview:
                self.audio.stop_bgm()
                self.audio.set_overlay_enabled(True)
                return
            self._stop_playback()
            if self.audio._bgm_overlay is None or getattr(self.audio._bgm_overlay, "size", 0) == 0:
                self._show_error("No overlay BGM set")
                return
            self.audio._bgm_data = np.zeros(int(self.audio.sample_rate * 2), dtype=np.float32)
            self.audio._bgm_pos = 0
            self.audio._bgm_overlay_pos = 0
            self.audio.set_overlay_enabled(True)
            self.audio.play_bgm()
        except Exception as exc:
            self._show_error(str(exc))

    def _finalize_recording(self) -> None:
        if not self.session or not self.current_item:
            return
        prefix = self.session.output_prefix if self.session else ""
        suffix = self.session.output_suffix if self.session else ""
        rel_path = Path("Recordings") / f"{prefix}{self.current_item.alias}{suffix}.wav"
        abs_path = self.session.session_dir() / rel_path
        if abs_path.exists():
            info = sf.info(str(abs_path))
            self.current_item.duration_sec = info.frames / info.samplerate
            self.current_item.wav_path = str(rel_path)
            self.current_item.status = ItemStatus.RECORDED
            self._refresh_table()
            self._save_session()
            self._analyze_selected_item()
            self._start_note_analysis()
            if self.playhead:
                self.playhead.setVisible(True)
                self.playhead.setPos(0.0)
            self._log_event("record_stop", self.current_item.alias)
            if self.auto_next_checkbox.isChecked():
                self._select_next()

    def _select_next(self) -> None:
        row = self.table.currentRow()
        if row < 0:
            return
        next_row = min(row + 1, self.table.rowCount() - 1)
        self.table.selectRow(next_row)

    def _update_bgm_level(self) -> None:
        self.audio.set_bgm_gain(self.bgm_slider.value() / 100.0)

    def _update_bgm_overlay_level(self) -> None:
        self.audio.set_overlay_gain(self.bgm_overlay_slider.value() / 100.0)

    def _update_pre_roll(self) -> None:
        self.audio.set_pre_roll_ms(self.pre_roll_spin.value())

    def _refresh_table(self) -> None:
        if not self.session:
            self.table.setRowCount(0)
            return
        note_sort_state = getattr(self, "_note_sort_state", 0)
        self._suppress_item_changed = True
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(self.session.items))
        target_note = self._target_bgm_note()
        for row, item in enumerate(self.session.items):
            status_item = QtWidgets.QTableWidgetItem(item.status.value)
            status_item.setData(QtCore.Qt.ItemDataRole.UserRole, row)
            self.table.setItem(row, 0, status_item)

            alias_item = QtWidgets.QTableWidgetItem(item.alias)
            alias_item.setData(QtCore.Qt.ItemDataRole.UserRole, row)
            self.table.setItem(row, 1, alias_item)
            romaji_text = ""
            if item.romaji is None and needs_romaji(item.alias):
                item.romaji = "_".join(kana_to_romaji_tokens(item.alias))
            if item.romaji:
                romaji_text = item.romaji.replace(" ", "_")
            romaji_item = QtWidgets.QTableWidgetItem(romaji_text)
            romaji_item.setData(QtCore.Qt.ItemDataRole.UserRole, row)
            self.table.setItem(row, 2, romaji_item)
            note_text = item.note or ""
            if target_note and item.wav_path:
                sung_note = self._get_cached_sung_note(item.wav_path)
                if sung_note is None:
                    note_text = "..."
                else:
                    note_text = self._format_note_check(target_note, sung_note)
            note_item = NoteTableItem(note_text, self._note_sort_priority(note_text))
            note_item.setData(QtCore.Qt.ItemDataRole.UserRole, row)
            self.table.setItem(row, 3, note_item)
            comment_item = QtWidgets.QTableWidgetItem(item.notes or "")
            comment_item.setData(QtCore.Qt.ItemDataRole.UserRole, row)
            self.table.setItem(row, 4, comment_item)
            duration = f"{item.duration_sec:.2f}" if item.duration_sec else ""
            duration_item = QtWidgets.QTableWidgetItem(duration)
            duration_item.setData(QtCore.Qt.ItemDataRole.UserRole, row)
            self.table.setItem(row, 5, duration_item)

            file_item = QtWidgets.QTableWidgetItem(item.wav_path or "")
            file_item.setData(QtCore.Qt.ItemDataRole.UserRole, row)
            self.table.setItem(row, 6, file_item)
            for col in range(7):
                item_widget = self.table.item(row, col)
                if item_widget is None:
                    continue
                flags = item_widget.flags()
                if col in (1, 4):
                    item_widget.setFlags(flags | QtCore.Qt.ItemFlag.ItemIsEditable)
                else:
                    item_widget.setFlags(flags & ~QtCore.Qt.ItemFlag.ItemIsEditable)
        self._suppress_item_changed = False
        if note_sort_state != 0:
            order = (
                QtCore.Qt.SortOrder.AscendingOrder
                if note_sort_state > 0
                else QtCore.Qt.SortOrder.DescendingOrder
            )
            self.table.setSortingEnabled(True)
            self.table.sortItems(3, order)

    def _target_bgm_note(self) -> str:
        if not self.session:
            return ""
        if self.session.target_note:
            return self.session.target_note.strip()
        note = (self.session.bgm_note or "").strip()
        if note:
            return note
        if self.session.bgm_overlay_enabled and self.session.bgm_overlay_note:
            return self.session.bgm_overlay_note.strip()
        return ""

    @staticmethod
    def _normalize_note(note: str) -> str:
        return note.strip().upper().replace(" ", "")

    @staticmethod
    def _note_to_midi(note: str) -> Optional[int]:
        freq = note_to_freq(note)
        if not freq:
            return None
        midi = f0_to_midi(freq)
        if midi is None:
            return None
        return int(round(midi))

    def _format_note_check(self, target: str, sung: str) -> str:
        target_midi = self._note_to_midi(target)
        sung_midi = self._note_to_midi(sung)
        if target_midi is None or sung_midi is None:
            is_match = self._normalize_note(target) == self._normalize_note(sung)
            mark = "✅" if is_match else "❌"
            return f"{mark} ({sung})"

        diff = abs(target_midi - sung_midi)
        if diff == 0:
            mark = "✅"
        elif diff <= 2:
            mark = "⚠️"
        else:
            mark = "❌"
        return f"{mark} ({sung})"

    @staticmethod
    def _note_sort_priority(note_text: str) -> int:
        if not note_text:
            return 3
        if note_text.startswith("❌"):
            return 0
        if note_text.startswith("⚠"):
            return 1
        if note_text.startswith("✅"):
            return 2
        return 3

    def _get_cached_sung_note(self, wav_rel: str) -> Optional[str]:
        if not self.session:
            return None
        abs_path = Path(wav_rel)
        if not abs_path.is_absolute():
            abs_path = self.session.session_dir() / abs_path
        cache_key = str(abs_path)
        cached = self._sung_note_cache.get(cache_key)
        if not cached:
            disk_note = self._load_note_from_disk_cache(abs_path)
            if disk_note is None:
                return None
            self._sung_note_cache[cache_key] = (abs_path.stat().st_mtime, disk_note)
            return disk_note
        return cached[1]

    def _recompute_note_for_selection(self) -> None:
        if not self.session or not self.current_item or not self.current_item.wav_path:
            return
        abs_path = Path(self.current_item.wav_path)
        if not abs_path.is_absolute():
            abs_path = self.session.session_dir() / abs_path
        if not abs_path.exists():
            return
        cache_key = str(abs_path)
        if cache_key in self._sung_note_cache:
            self._sung_note_cache.pop(cache_key, None)
        cache_dir = self._analysis_cache_dir()
        if cache_dir:
            key = _analysis_cache_key(abs_path)
            meta_path, data_path = _analysis_cache_paths(cache_dir, key)
            try:
                meta_path.unlink(missing_ok=True)
            except Exception:
                pass
            try:
                data_path.unlink(missing_ok=True)
            except Exception:
                pass
        self._start_note_analysis_files([str(abs_path)])

    def _collect_note_analysis_files(self) -> list[str]:
        if not self.session:
            return []
        files: list[str] = []
        for item in self.session.items:
            if not item.wav_path:
                continue
            abs_path = Path(item.wav_path)
            if not abs_path.is_absolute():
                abs_path = self.session.session_dir() / abs_path
            if not abs_path.exists():
                continue
            try:
                mtime = abs_path.stat().st_mtime
            except OSError:
                continue
            cache_key = str(abs_path)
            cached = self._sung_note_cache.get(cache_key)
            if cached and cached[0] == mtime:
                continue
            disk_note = self._load_note_from_disk_cache(abs_path)
            if disk_note is not None:
                self._sung_note_cache[cache_key] = (mtime, disk_note)
                continue
            files.append(str(abs_path))
        return files

    def _start_note_analysis_files(self, files: list[str]) -> None:
        if not self.session or not files:
            return
        if self._note_worker and self._note_worker.isRunning():
            self._note_analysis_pending.update(files)
            return
        cache_dir = self._analysis_cache_dir()
        ref_bits = self.session.bit_depth if self.session else 16
        self._note_worker = NoteAnalysisWorker(
            files,
            self.session.sample_rate,
            self.pitch_algo,
            cache_dir,
            ref_bits,
            self.note_workers,
            self,
        )
        self._note_worker.result.connect(self._on_note_analysis_result)
        self._note_worker.progress.connect(self._on_note_analysis_progress)
        self._note_worker.finished.connect(self._on_note_analysis_finished)
        self._update_note_progress(0, len(files))
        self._note_worker.start()

    def _load_note_from_disk_cache(self, abs_path: Path) -> Optional[str]:
        if not self.session:
            return None
        cache_dir = self._analysis_cache_dir()
        if cache_dir is None:
            return None
        try:
            mtime = abs_path.stat().st_mtime
        except OSError:
            return None
        ref_bits = self.session.bit_depth if self.session else 16
        cached = _load_analysis_cache_from_disk(
            cache_dir,
            abs_path,
            mtime,
            self.pitch_algo,
            self.session.sample_rate,
            ref_bits,
        )
        if not cached:
            return None
        times, f0s, _mel, _mel_times, _pt, _pd, has_pitch, _has_mel, _has_power, note = cached
        if note:
            return str(note)
        if has_pitch and f0s.size:
            computed = _note_from_f0s(f0s)
            _save_analysis_cache_to_disk(
                cache_dir,
                abs_path,
                mtime,
                self.pitch_algo,
                self.session.sample_rate,
                ref_bits,
                times,
                f0s,
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
                merge_existing=True,
                note=computed,
            )
            return computed
        return None

    def _start_note_analysis(self) -> None:
        target_note = self._target_bgm_note()
        if not target_note or not self.session:
            return
        files = self._collect_note_analysis_files()
        if not files:
            self._update_note_progress(0, 0, hide=True)
            return
        self._start_note_analysis_files(files)

    def _stop_note_worker(self) -> None:
        if self._note_worker and self._note_worker.isRunning():
            self._note_worker.requestInterruption()
            self._note_worker.wait(2000)
        self._note_worker = None
        self._note_analysis_pending.clear()
        self._update_note_progress(0, 0, hide=True)

    def _stop_recorded_worker(self) -> None:
        for attr in ("_analysis_pitch_worker", "_analysis_spectro_worker"):
            worker = getattr(self, attr)
            if worker and worker.isRunning():
                worker.requestInterruption()
                worker.wait(2000)
            setattr(self, attr, None)

    def _analysis_cache_dir(self) -> Optional[Path]:
        if not self.session:
            return None
        path = self.session.session_dir() / "_analysis_cache"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _on_note_analysis_result(self, path: str, mtime: float, note: str) -> None:
        self._sung_note_cache[path] = (mtime, note)
        if not self.session:
            return
        target_note = self._target_bgm_note()
        if not target_note:
            return
        for row, item in enumerate(self.session.items):
            if not item.wav_path:
                continue
            abs_path = Path(item.wav_path)
            if not abs_path.is_absolute():
                abs_path = self.session.session_dir() / abs_path
            if str(abs_path) == path:
                note_text = self._format_note_check(target_note, note)
                note_item = NoteTableItem(note_text, self._note_sort_priority(note_text))
                self.table.setItem(row, 3, note_item)
                break

    def _on_note_analysis_finished(self) -> None:
        if self._note_analysis_pending:
            files = list(self._note_analysis_pending)
            self._note_analysis_pending.clear()
            if self.session:
                cache_dir = self._analysis_cache_dir()
                ref_bits = self.session.bit_depth if self.session else 16
                self._note_worker = NoteAnalysisWorker(
                    files,
                    self.session.sample_rate,
                    self.pitch_algo,
                    cache_dir,
                    ref_bits,
                    self.note_workers,
                    self,
                )
                self._note_worker.result.connect(self._on_note_analysis_result)
                self._note_worker.progress.connect(self._on_note_analysis_progress)
                self._note_worker.finished.connect(self._on_note_analysis_finished)
                self._update_note_progress(0, len(files))
                self._note_worker.start()
        else:
            self._update_note_progress(0, 0, hide=True)

    def _on_note_analysis_progress(self, done: int, total: int) -> None:
        self._update_note_progress(done, total)

    def _update_note_progress(self, done: int, total: int, hide: bool = False) -> None:
        if not self.note_progress:
            return
        if hide or total <= 0:
            self.note_progress.setVisible(False)
            self.note_progress.setRange(0, 1)
            self.note_progress.setValue(0)
            return
        self.note_progress.setVisible(True)
        self.note_progress.setRange(0, total)
        self.note_progress.setValue(done)

    def _update_visuals(self) -> None:
        if not self.audio.is_active():
            return
        if self.audio.preview and not self.audio.recording:
            buffer = self.audio.get_preview_audio()
        else:
            buffer = self.audio.get_latest_audio(2048)
        if buffer.size == 0:
            return

        if self.audio.preview and not self.audio.recording:
            wave = buffer
            wave_sr = self.audio.sample_rate
        elif self.audio.recording:
            wave = self.audio.get_waveform_audio()
            if wave.size == 0:
                wave = buffer
                wave_sr = self.audio.sample_rate
            else:
                wave_sr = self.audio.get_waveform_sample_rate()
        elif self.selected_audio is not None and self.selected_audio.size:
            wave = self.selected_audio
            wave_sr = self.audio.sample_rate
        else:
            wave = self.audio.get_waveform_audio()
            if wave.size == 0:
                wave = buffer
                wave_sr = self.audio.sample_rate
            else:
                wave_sr = self.audio.get_waveform_sample_rate()
        wave_x = np.linspace(0, len(wave) / wave_sr, len(wave))
        if len(wave) > 20000:
            step = max(1, len(wave) // 20000)
            wave = wave[::step]
            wave_x = wave_x[::step]
        self.wave_curve.setData(wave_x, wave)
        self.wave_plot.enableAutoRange(axis="xy", enable=True)
        self.wave_plot.plotItem.vb.autoRange()

        freqs, mag = compute_fft(buffer, self.audio.sample_rate)
        self.spec_curve.setData(freqs, mag)

        rms = compute_rms(buffer)
        self.power_history.append(rms)
        if len(self.power_history) > self.history_size:
            self.power_history.pop(0)
        power_x = np.arange(len(self.power_history)) * (self.visual_timer.interval() / 1000.0)
        self.power_curve.setData(power_x, self.power_history)

        f0 = estimate_f0(buffer, self.audio.sample_rate)
        note, cents = note_from_f0(f0)
        self.note_label.setText(f"{tr(self.ui_language, 'current_note_prefix')}{note} ({cents:+.1f} cents)")

    def _plot_clicked(self, plot: pg.PlotWidget, event: QtCore.QEvent) -> None:
        if not self.playhead:
            return
        if self.audio.recording or self.audio.preview:
            return
        if hasattr(event, "button") and event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        view_pos = plot.plotItem.vb.mapSceneToView(event.scenePos())
        x = float(view_pos.x())
        if x < 0:
            x = 0.0
        self._play_recorded_from(x)

    def _should_handle_shortcuts(self) -> bool:
        widget = QtWidgets.QApplication.focusWidget()
        if widget is None:
            return True
        if self.table:
            if self.table.state() == QtWidgets.QAbstractItemView.State.EditingState:
                return False
            if widget and self.table.isAncestorOf(widget):
                if isinstance(
                    widget,
                    (
                        QtWidgets.QLineEdit,
                        QtWidgets.QTextEdit,
                        QtWidgets.QPlainTextEdit,
                        QtWidgets.QSpinBox,
                        QtWidgets.QDoubleSpinBox,
                        QtWidgets.QComboBox,
                    ),
                ):
                    return False
        if isinstance(
            widget,
            (
                QtWidgets.QLineEdit,
                QtWidgets.QTextEdit,
                QtWidgets.QPlainTextEdit,
                QtWidgets.QSpinBox,
                QtWidgets.QDoubleSpinBox,
                QtWidgets.QComboBox,
            ),
        ):
            return False
        return True

    def _handle_key_press(self, event: QtGui.QKeyEvent) -> bool:
        if event.key() == QtCore.Qt.Key.Key_Escape:
            if self.table:
                editor = self.table.focusWidget()
                if editor and editor.parent() is self.table:
                    self.table.closeEditor(editor, QtWidgets.QAbstractItemDelegate.EndEditHint.NoHint)
                self.table.clearSelection()
                self.table.setCurrentItem(None)
                self.table.clearFocus()
            event.accept()
            return True
        if not self._should_handle_shortcuts():
            return False
        key = event.key()
        text = (event.text() or "").lower()
        if key == QtCore.Qt.Key.Key_Space:
            if not self.audio.recording and not self.audio.preview:
                if self.playing:
                    if self.play_start_time is not None:
                        elapsed = self.play_start_time.elapsed() / 1000.0
                        self._paused_pos = max(0.0, self.play_start_pos + elapsed)
                    else:
                        self._paused_pos = 0.0
                    self._paused_audio = (
                        self.selected_audio if self.selected_audio is not None else self.audio.get_waveform_audio()
                    )
                    self._paused_sr = self.audio.sample_rate
                    sd.stop()
                    self.playing = False
                    self.playhead.setPos(self._paused_pos)
                    event.accept()
                    return True
                if self._paused_audio is not None and self._paused_sr:
                    duration = len(self._paused_audio) / self._paused_sr if self._paused_audio.size else 0.0
                    if duration and self._paused_pos >= max(0.0, duration - 0.01):
                        self._paused_audio = None
                        self._paused_pos = 0.0
                        self._paused_sr = 0
                    else:
                        self._play_audio_from(self._paused_audio, self._paused_pos, self._paused_sr)
                        event.accept()
                        return True
                    event.accept()
                    return True
                if self.current_item and self.current_item.wav_path:
                    if self.selected_audio is None or getattr(self.selected_audio, "size", 0) == 0:
                        self._analyze_selected_item()
                    self._play_recorded_from(0.0)
                event.accept()
                return True
        if key == QtCore.Qt.Key.Key_Delete:
            delete_files = bool(
                QtWidgets.QApplication.keyboardModifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier
            )
            self._delete_selected_entries(delete_files=delete_files)
            event.accept()
            return True
        if (key == QtCore.Qt.Key.Key_R or text in ("r", "к")) and self.hold_to_record:
            if not self.audio.recording and not self.audio.preview:
                if self.table:
                    editor = self.table.focusWidget()
                    if editor and editor.parent() is self.table:
                        self.table.closeEditor(
                            editor,
                            QtWidgets.QAbstractItemDelegate.EndEditHint.NoHint,
                        )
                self._hold_record_active = True
                self._record()
                event.accept()
                return True
        return False

    def _handle_key_release(self, event: QtGui.QKeyEvent) -> bool:
        key = event.key()
        text = (event.text() or "").lower()
        if (key == QtCore.Qt.Key.Key_R or text in ("r", "к")) and self.hold_to_record:
            if self._hold_record_active:
                self._hold_record_active = False
                if self.audio.recording:
                    self._stop()
            event.accept()
            return True
        return False

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.isAutoRepeat():
            super().keyPressEvent(event)
            return
        if self._handle_key_press(event):
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.isAutoRepeat():
            super().keyReleaseEvent(event)
            return
        if self._handle_key_release(event):
            return
        super().keyReleaseEvent(event)

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.Type.KeyPress:
            if not event.isAutoRepeat() and self._handle_key_press(event):
                return True
        elif event.type() == QtCore.QEvent.Type.KeyRelease:
            if not event.isAutoRepeat() and self._handle_key_release(event):
                return True
        return super().eventFilter(obj, event)

    def _play_recorded_from(self, start_sec: float) -> None:
        audio = self.selected_audio if self.selected_audio is not None else self.audio.get_waveform_audio()
        self._paused_audio = None
        self._paused_pos = 0.0
        self._paused_sr = 0
        self._play_audio_from(audio, start_sec, self.audio.sample_rate)

    def _play_audio_from(self, audio: np.ndarray, start_sec: float, sr: int) -> None:
        if audio.size == 0:
            return
        start_sample = int(start_sec * sr)
        start_sample = max(0, min(start_sample, len(audio) - 1))
        output_device = None
        if self.audio.device is not None:
            output_device = self.audio.device[1]
        try:
            sd.stop()
            sd.play(audio[start_sample:], samplerate=sr, device=output_device)
            self.playing = True
            self.play_start_time = QtCore.QElapsedTimer()
            self.play_start_time.start()
            self.play_start_pos = start_sample / sr
            self.play_duration = (len(audio) - start_sample) / sr
            self.playhead.setVisible(True)
            self.playhead.setPos(self.play_start_pos)
        except Exception as exc:
            self._show_error(str(exc))

    def _update_playhead(self) -> None:
        if not self.playing or not self.playhead or self.play_start_time is None:
            return
        elapsed = self.play_start_time.elapsed() / 1000.0
        pos = self.play_start_pos + elapsed
        self.playhead.setPos(pos)
        if hasattr(self, "spec_playhead"):
            self.spec_playhead.setVisible(True)
            self.spec_playhead.setPos(pos)
        if hasattr(self, "power_playhead"):
            self.power_playhead.setVisible(True)
            self.power_playhead.setPos(pos)
        if hasattr(self, "recorded_f0_playhead"):
            self.recorded_f0_playhead.setVisible(True)
            self.recorded_f0_playhead.setPos(pos)
        if hasattr(self, "mel_playhead"):
            self.mel_playhead.setVisible(True)
            self.mel_playhead.setPos(pos)
        if elapsed >= self.play_duration:
            self.playing = False
            self._paused_audio = None
            self._paused_pos = 0.0
            self._paused_sr = 0
            if hasattr(self, "spec_playhead"):
                self.spec_playhead.setVisible(False)
            if hasattr(self, "power_playhead"):
                self.power_playhead.setVisible(False)
            if hasattr(self, "recorded_f0_playhead"):
                self.recorded_f0_playhead.setVisible(False)
            if hasattr(self, "mel_playhead"):
                self.mel_playhead.setVisible(False)

    def _stop_playback(self) -> None:
        if self.playing:
            sd.stop()
        self.playing = False
        self._paused_audio = None
        self._paused_pos = 0.0
        self._paused_sr = 0
        if hasattr(self, "spec_playhead"):
            self.spec_playhead.setVisible(False)
        if hasattr(self, "power_playhead"):
            self.power_playhead.setVisible(False)
        if hasattr(self, "recorded_f0_playhead"):
            self.recorded_f0_playhead.setVisible(False)
        if hasattr(self, "mel_playhead"):
            self.mel_playhead.setVisible(False)
        if hasattr(self, "mel_playhead"):
            self.mel_playhead.setVisible(False)

    def _update_recorded_analysis(self) -> None:
        audio = self.selected_audio if self.selected_audio is not None else self.audio.get_waveform_audio()
        if audio.size == 0:
            return
        self._start_recorded_analysis(audio)

    def _start_recorded_analysis(
        self,
        audio: np.ndarray,
        cache_key: Optional[tuple] = None,
    ) -> None:
        self._current_analysis_key = cache_key
        self._analysis_token += 1
        token = self._analysis_token
        self._stop_recorded_worker()
        ref_bits = self.session.bit_depth if self.session else 16
        cache_dir = self._analysis_cache_dir()
        empty = np.array([], dtype=np.float32)
        times = empty
        f0s = empty
        mel_db = empty
        mel_times = empty
        power_times = empty
        power_db = empty
        has_pitch = False
        has_mel = False
        has_power = False
        if cache_key and cache_key in self._recorded_analysis_cache:
            cached_entry = self._recorded_analysis_cache[cache_key]
            self._recorded_analysis_cache.move_to_end(cache_key)
            (
                times,
                f0s,
                mel_db,
                mel_times,
                power_times,
                power_db,
                has_pitch,
                has_mel,
                has_power,
            ) = cached_entry
        if cache_dir and cache_key:
            cached = _load_analysis_cache_from_disk(
                cache_dir,
                Path(cache_key[0]),
                cache_key[1],
                cache_key[2],
                cache_key[3],
                cache_key[4],
            )
            if cached:
                (
                    cached_times,
                    cached_f0s,
                    cached_mel_db,
                    cached_mel_times,
                    cached_power_times,
                    cached_power_db,
                    cached_has_pitch,
                    cached_has_mel,
                    cached_has_power,
                    _note_value,
                ) = cached
                if not has_pitch and cached_has_pitch:
                    times, f0s, has_pitch = cached_times, cached_f0s, True
                if not has_mel and cached_has_mel:
                    mel_db, mel_times, has_mel = cached_mel_db, cached_mel_times, True
                if not has_power and cached_has_power:
                    power_times, power_db, has_power = cached_power_times, cached_power_db, True

        if cache_key:
            entry = (times, f0s, mel_db, mel_times, power_times, power_db, has_pitch, has_mel, has_power)
            self._recorded_analysis_cache[cache_key] = entry
            self._recorded_analysis_cache.move_to_end(cache_key)
            while len(self._recorded_analysis_cache) > self._analysis_cache_limit:
                self._recorded_analysis_cache.popitem(last=False)

        if has_pitch or has_mel or has_power:
            self._apply_recorded_analysis(
                times if has_pitch else empty,
                f0s if has_pitch else empty,
                mel_db if has_mel else empty,
                mel_times if has_mel else empty,
                power_times if has_power else empty,
                power_db if has_power else empty,
            )

        if has_pitch and has_mel and has_power:
            return

        compute_pitch = not has_pitch
        compute_mel = not has_mel
        compute_power = not has_power

        self._current_analysis_meta = (
            Path(cache_key[0]) if cache_key else None,
            cache_key[1] if cache_key else None,
            cache_key[2] if cache_key else None,
            cache_key[3] if cache_key else None,
            cache_key[4] if cache_key else None,
        )
        audio_copy = audio.astype(np.float32, copy=True)
        if compute_pitch:
            self._analysis_pitch_worker = RecordedAnalysisWorker(
                audio_copy,
                self.audio.sample_rate,
                self.pitch_algo,
                ref_bits,
                token,
                cache_key,
                None,
                None,
                None,
                True,
                False,
                False,
                self,
            )
            self._analysis_pitch_worker.result.connect(self._on_recorded_analysis_result)
            self._analysis_pitch_worker.start()
        if compute_mel or compute_power:
            self._analysis_spectro_worker = RecordedAnalysisWorker(
                audio_copy,
                self.audio.sample_rate,
                self.pitch_algo,
                ref_bits,
                token,
                cache_key,
                None,
                None,
                None,
                False,
                compute_mel,
                compute_power,
                self,
            )
            self._analysis_spectro_worker.result.connect(self._on_recorded_analysis_result)
            self._analysis_spectro_worker.start()

    def _on_recorded_analysis_result(
        self,
        token: int,
        cache_key: Optional[tuple],
        times: np.ndarray,
        f0s: np.ndarray,
        mel_db: np.ndarray,
        mel_times: np.ndarray,
        power_times: np.ndarray,
        power_db: np.ndarray,
        pitch_done: bool,
        mel_done: bool,
        power_done: bool,
    ) -> None:
        if token != self._analysis_token:
            return
        if cache_key is not None and cache_key != self._current_analysis_key:
            return

        empty = np.array([], dtype=np.float32)
        existing = self._recorded_analysis_cache.get(cache_key) if cache_key else None
        if existing:
            (
                ex_times,
                ex_f0s,
                ex_mel_db,
                ex_mel_times,
                ex_power_times,
                ex_power_db,
                ex_has_pitch,
                ex_has_mel,
                ex_has_power,
            ) = existing
        else:
            ex_times = empty
            ex_f0s = empty
            ex_mel_db = empty
            ex_mel_times = empty
            ex_power_times = empty
            ex_power_db = empty
            ex_has_pitch = False
            ex_has_mel = False
            ex_has_power = False

        if pitch_done:
            ex_times = times
            ex_f0s = f0s
            ex_has_pitch = bool(times.size and f0s.size)
        if mel_done:
            ex_mel_db = mel_db
            ex_mel_times = mel_times
            ex_has_mel = bool(mel_db.size and mel_times.size)
        if power_done:
            ex_power_times = power_times
            ex_power_db = power_db
            ex_has_power = bool(power_times.size and power_db.size)

        if cache_key is not None:
            entry = (
                ex_times,
                ex_f0s,
                ex_mel_db,
                ex_mel_times,
                ex_power_times,
                ex_power_db,
                ex_has_pitch,
                ex_has_mel,
                ex_has_power,
            )
            self._recorded_analysis_cache[cache_key] = entry
            self._recorded_analysis_cache.move_to_end(cache_key)
            while len(self._recorded_analysis_cache) > self._analysis_cache_limit:
                self._recorded_analysis_cache.popitem(last=False)
            if (pitch_done or mel_done or power_done) and self._current_analysis_meta and self._current_analysis_meta[0] is not None:
                cache_dir = self._analysis_cache_dir()
                if cache_dir:
                    _save_analysis_cache_to_disk(
                        cache_dir,
                        self._current_analysis_meta[0],
                        float(self._current_analysis_meta[1]),
                        str(self._current_analysis_meta[2]),
                        int(self._current_analysis_meta[3]),
                        int(self._current_analysis_meta[4]),
                        times,
                        f0s,
                        mel_db,
                        mel_times,
                        power_times,
                        power_db,
                        merge_existing=True,
                    )

        self._apply_recorded_analysis(
            ex_times if ex_has_pitch else empty,
            ex_f0s if ex_has_pitch else empty,
            ex_mel_db if ex_has_mel else empty,
            ex_mel_times if ex_has_mel else empty,
            ex_power_times if ex_has_power else empty,
            ex_power_db if ex_has_power else empty,
        )

    def _apply_recorded_analysis(
        self,
        times: np.ndarray,
        f0s: np.ndarray,
        mel_db: np.ndarray,
        mel_times: np.ndarray,
        power_times: np.ndarray,
        power_db: np.ndarray,
    ) -> None:
        if times.size and f0s.size:
            midi_vals = np.array([f0_to_midi(f) if f > 0 else np.nan for f in f0s], dtype=np.float32)
            self.recorded_f0_curve.setData(times, midi_vals)
        else:
            self.recorded_f0_curve.setData([], [])
        if mel_db.size:
            self.mel_img.setImage(mel_db.T, autoLevels=True)
            self.mel_img.setRect(QtCore.QRectF(0, 0, float(mel_times[-1]) if mel_times.size else 1.0, mel_db.shape[0]))
            if mel_times.size:
                self.mel_plot.setLimits(xMin=0, xMax=float(mel_times[-1]))
        else:
            self.mel_img.clear()
        if power_times.size:
            self.power_curve.setData(power_times, power_db)
            self.power_plot.setLimits(xMin=0, xMax=float(power_times[-1]))
        else:
            self.power_curve.setData([], [])

    def _analyze_selected_item(self) -> None:
        if not self.session or not self.current_item or not self.current_item.wav_path:
            self.selected_audio = None
            self._clear_analysis()
            return
        self._paused_audio = None
        self._paused_pos = 0.0
        self._paused_sr = 0
        abs_path = self.session.session_dir() / self.current_item.wav_path
        if not abs_path.exists():
            self.selected_audio = None
            self._clear_analysis()
            return
        try:
            audio, sr = sf.read(str(abs_path), dtype="float32")
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            if sr != self.audio.sample_rate:
                audio = self.audio._resample(audio, sr, self.audio.sample_rate)
            self.selected_audio = audio
            self._render_waveform(audio)
            snippet = audio[-2048:] if audio.size >= 2048 else audio
            freqs, mag = compute_fft(snippet, self.audio.sample_rate)
            self.spec_curve.setData(freqs, mag)
            rms = compute_rms(snippet)
            self.power_history = [rms]
            self.power_curve.setData([0.0], self.power_history)
            f0 = estimate_f0(snippet, self.audio.sample_rate)
            note, cents = note_from_f0(f0)
            self.note_label.setText(f"{tr(self.ui_language, 'current_note_prefix')}{note} ({cents:+.1f} cents)")
            mtime = abs_path.stat().st_mtime
            ref_bits = self.session.bit_depth if self.session else 16
            cache_key = (
                str(abs_path),
                mtime,
                str(self.pitch_algo),
                int(self.audio.sample_rate),
                int(ref_bits),
            )
            self._start_recorded_analysis(audio, cache_key)
        except Exception as exc:
            logger.exception("Failed to analyze selected item")
            self._show_error(str(exc))

    def _table_header_clicked(self, index: int) -> None:
        if index != 3:
            if self._note_sort_state != 0:
                self._note_sort_state = 0
                self.table.setSortingEnabled(False)
                self._refresh_table()
            return
        if self._note_sort_state == 0:
            self._note_sort_state = 1
        elif self._note_sort_state == 1:
            self._note_sort_state = -1
        else:
            self._note_sort_state = 0
        if self._note_sort_state == 0:
            self.table.setSortingEnabled(False)
            self._refresh_table()
            return
        order = (
            QtCore.Qt.SortOrder.AscendingOrder
            if self._note_sort_state > 0
            else QtCore.Qt.SortOrder.DescendingOrder
        )
        self.table.setSortingEnabled(True)
        self.table.sortItems(3, order)

    def _table_context_menu(self, pos: QtCore.QPoint) -> None:
        menu = QtWidgets.QMenu(self)
        add_action = menu.addAction(tr(self.ui_language, "add_entry"))
        delete_action = menu.addAction(tr(self.ui_language, "delete_entry"))
        recompute_action = menu.addAction(tr(self.ui_language, "recompute_note"))
        action = menu.exec(self.table.viewport().mapToGlobal(pos))
        if action == add_action:
            self._add_entry()
        elif action == delete_action:
            delete_files = bool(
                QtWidgets.QApplication.keyboardModifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier
            )
            self._delete_selected_entries(delete_files=delete_files)
        elif action == recompute_action:
            self._recompute_note_for_selection()

    def _add_entry(self) -> None:
        if not self.session:
            self._create_temp_session()
        alias, ok = QtWidgets.QInputDialog.getText(self, tr(self.ui_language, "add_entry"), tr(self.ui_language, "alias"))
        if not ok or not alias.strip():
            return
        note, _ = QtWidgets.QInputDialog.getText(self, tr(self.ui_language, "add_entry"), tr(self.ui_language, "note_optional"))
        alias = alias.strip()
        if any(item.alias == alias for item in self.session.items):
            self._show_error("Alias already exists")
            return
        romaji = "_".join(kana_to_romaji_tokens(alias)) if needs_romaji(alias) else None
        self._push_undo_state()
        self.session.add_item(alias, note.strip() or None, romaji=romaji)
        self._log_event("add_entry", alias)
        self._refresh_table()
        self._save_session()

    def _delete_entry(self) -> None:
        if not self.session:
            return
        row = self.table.currentRow()
        if row < 0:
            return
        self._push_undo_state()
        alias = self.session.items[row].alias
        self.session.items.pop(row)
        self._log_event("delete_entry", alias)
        self._refresh_table()
        self._save_session()

    def _selected_model_indices(self) -> list[int]:
        if not self.session:
            return []
        model_rows: list[int] = []
        selection = self.table.selectionModel()
        if selection is None:
            return []
        for index in selection.selectedRows():
            row = index.row()
            item = self.table.item(row, 0)
            if item is not None:
                model_index = item.data(QtCore.Qt.ItemDataRole.UserRole)
            else:
                model_index = row
            try:
                model_index = int(model_index)
            except (TypeError, ValueError):
                model_index = row
            if 0 <= model_index < len(self.session.items):
                model_rows.append(model_index)
        return sorted(set(model_rows))

    def _delete_selected_entries(self, delete_files: bool = False) -> None:
        if not self.session:
            return
        indices = self._selected_model_indices()
        if not indices:
            return
        if delete_files:
            count = len(indices)
            prompt = tr(self.ui_language, "delete_selected_prompt").format(count=count)
            reply = QtWidgets.QMessageBox.question(
                self,
                tr(self.ui_language, "delete_selected_title"),
                prompt,
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        self._push_undo_state()
        for idx in sorted(indices, reverse=True):
            if idx < 0 or idx >= len(self.session.items):
                continue
            item = self.session.items[idx]
            if delete_files and item.wav_path:
                abs_path = Path(item.wav_path)
                if not abs_path.is_absolute():
                    abs_path = self.session.session_dir() / abs_path
                try:
                    abs_path.unlink(missing_ok=True)
                except Exception:
                    pass
                self._sung_note_cache.pop(str(abs_path), None)
            self._log_event("delete_entry", item.alias)
            self.session.items.pop(idx)
        if self.current_item and self.current_item not in self.session.items:
            self.current_item = None
            self.selected_audio = None
            self._clear_analysis()
        self._refresh_table()
        self._save_session()

    def _item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if self._suppress_item_changed or not self.session:
            return
        row = item.row()
        col = item.column()
        if row < 0 or row >= len(self.session.items):
            return
        current = self.session.items[row]
        if col == 1:
            new_alias = item.text().strip()
            if not new_alias:
                return
            if any(it.alias == new_alias for i, it in enumerate(self.session.items) if i != row):
                self._show_error("Alias already exists")
                self._refresh_table()
                return
            self._push_undo_state()
            current.alias = new_alias
            current.romaji = "_".join(kana_to_romaji_tokens(new_alias)) if needs_romaji(new_alias) else None
            self._refresh_table()
            self._save_session()
        elif col == 4:
            self._push_undo_state()
            current.notes = item.text()
            self._save_session()

    def _push_undo_state(self) -> None:
        if not self.session:
            return
        snapshot = [item.to_dict() for item in self.session.items]
        self.undo_stack.append(snapshot)
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)

    def _undo(self) -> None:
        if not self.session or not self.undo_stack:
            return
        snapshot = self.undo_stack.pop()
        self.session.items = [Item.from_dict(data) for data in snapshot]
        self._refresh_table()
        self._save_session()

    def _clear_analysis(self) -> None:
        self._stop_recorded_worker()
        self._current_analysis_key = None
        self.wave_curve.setData([], [])
        self.spec_curve.setData([], [])
        self.power_history = []
        self.power_curve.setData([], [])
        self.recorded_f0_curve.setData([], [])
        self.mel_img.clear()
        self.note_label.setText(tr(self.ui_language, "current_note"))
        if self.playhead:
            self.playhead.setVisible(False)
        if self.selection_region:
            self.selection_region.setVisible(False)
        if hasattr(self, "spec_playhead"):
            self.spec_playhead.setVisible(False)
        if hasattr(self, "power_playhead"):
            self.power_playhead.setVisible(False)
        if hasattr(self, "recorded_f0_playhead"):
            self.recorded_f0_playhead.setVisible(False)

    def _progress_text(self) -> str:
        if not self.session:
            return "-- / --"
        recorded = sum(1 for item in self.session.items if item.status != ItemStatus.PENDING)
        total = len(self.session.items)
        return f"{recorded} / {total}"

    def _render_waveform(self, audio: np.ndarray) -> None:
        if audio.size == 0:
            return
        wave = audio
        wave_x = np.linspace(0, len(wave) / self.audio.sample_rate, len(wave))
        if len(wave) > 20000:
            step = max(1, len(wave) // 20000)
            wave = wave[::step]
            wave_x = wave_x[::step]
        self.wave_curve.setData(wave_x, wave)
        self.wave_plot.enableAutoRange(axis="xy", enable=True)
        self.wave_plot.plotItem.vb.autoRange()
        if self.selection_region:
            self.selection_region.setVisible(False)
        if self.playhead:
            self.playhead.setVisible(True)
            self.playhead.setPos(0.0)

    def _cut_selection(self) -> None:
        if not self.session or not self.current_item or not self.current_item.wav_path:
            return
        if self.audio.recording or self.audio.preview or self.playing:
            return
        if not self.selection_region or not self.selection_region.isVisible():
            return
        abs_path = self.session.session_dir() / self.current_item.wav_path
        if not abs_path.exists():
            return
        try:
            audio, sr = sf.read(str(abs_path), dtype="float32")
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            if sr != self.audio.sample_rate:
                audio = self.audio._resample(audio, sr, self.audio.sample_rate)
                sr = self.audio.sample_rate
            start_sec, end_sec = self.selection_region.getRegion()
            if end_sec <= start_sec:
                return
            start_s = max(0, int(start_sec * sr))
            end_s = min(len(audio), int(end_sec * sr))
            if end_s <= start_s:
                return
            new_audio = np.concatenate((audio[:start_s], audio[end_s:]))
            sf.write(str(abs_path), new_audio, sr, subtype="PCM_16")
            self.selected_audio = new_audio
            self.current_item.duration_sec = len(new_audio) / sr if len(new_audio) else 0.0
            self._save_session()
            self._analyze_selected_item()
            self._refresh_table()
            if self.current_item:
                duration = f" ({self.current_item.duration_sec:.2f}s)" if self.current_item.duration_sec else ""
                self.current_label.setText(
                    f"{tr(self.ui_language, 'current_item_prefix')}{self.current_item.alias}{duration}"
                )
        except Exception as exc:
            logger.exception("Failed to cut selection")
            self._show_error(str(exc))

    def _toggle_selection(self) -> None:
        if not self.selection_region:
            return
        if self.audio.recording or self.audio.preview or self.playing:
            return
        if self.selection_region.isVisible():
            self.selection_region.setVisible(False)
            return
        audio = self.selected_audio
        if audio is None or audio.size == 0:
            return
        end = len(audio) / self.audio.sample_rate
        start = max(0.0, end - 0.3)
        self.selection_region.setRegion((start, end))
        self.selection_region.setVisible(True)

    def _build_voicebank_map(self, folders: list[Path], prefix: str, suffix: str) -> dict[str, Path]:
        mapping: dict[str, Path] = {}
        for folder in folders:
            oto_path = folder / "oto.ini"
            if oto_path.exists():
                entries = parse_oto_ini(oto_path)
                for alias, wav_name in entries:
                    wav_path = folder / wav_name
                    if not wav_path.exists():
                        continue
                    key = self._normalize_alias(Path(wav_name).stem, prefix, suffix)
                    if not key:
                        continue
                    if key not in mapping:
                        mapping[key] = wav_path
                continue
            for wav in sorted(folder.glob("*.wav")):
                key = self._normalize_alias(wav.stem, prefix, suffix)
                if not key or key in mapping:
                    continue
                mapping[key] = wav
        return mapping

    def _normalize_alias(self, name: str, prefix: str, suffix: str) -> str:
        if prefix and name.startswith(prefix):
            name = name[len(prefix):]
        if suffix and name.endswith(suffix):
            name = name[: -len(suffix)]
        if self.session:
            if self.session.output_prefix and name.startswith(self.session.output_prefix):
                name = name[len(self.session.output_prefix):]
            if self.session.output_suffix and name.endswith(self.session.output_suffix):
                name = name[: -len(self.session.output_suffix)]
        return name.strip()

    def _rename_recordings_for_prefix_suffix(
        self,
        new_prefix: str,
        new_suffix: str,
    ) -> tuple[bool, dict[str, str]]:
        if not self.session:
            return True, {}
        mapping: dict[str, str] = {}
        errors = False
        temp_entries: list[tuple[Path, Path, Item, Path, Path]] = []
        base = self.session.session_dir()
        for item in self.session.items:
            if not item.wav_path:
                continue
            old_rel = Path(item.wav_path)
            old_abs = old_rel if old_rel.is_absolute() else base / old_rel
            if not old_abs.exists():
                continue
            new_name = f"{new_prefix}{item.alias}{new_suffix}{old_abs.suffix}"
            if old_rel.is_absolute():
                new_rel = old_abs.with_name(new_name)
                new_abs = new_rel
            else:
                new_rel = old_rel.with_name(new_name)
                new_abs = base / new_rel
            if new_abs == old_abs:
                item.wav_path = str(new_rel)
                mapping[str(old_abs)] = str(new_abs)
                continue
            if new_abs.exists():
                errors = True
                continue
            temp_path = old_abs.with_name(f".tmp_rename_{uuid.uuid4().hex}{old_abs.suffix}")
            try:
                old_abs.replace(temp_path)
            except Exception:
                errors = True
                continue
            temp_entries.append((temp_path, new_abs, item, new_rel, old_abs))

        for temp_path, new_abs, item, new_rel, old_abs in temp_entries:
            try:
                new_abs.parent.mkdir(parents=True, exist_ok=True)
                if new_abs.exists():
                    errors = True
                    temp_path.replace(old_abs)
                    continue
                temp_path.replace(new_abs)
                item.wav_path = str(new_rel)
                mapping[str(old_abs)] = str(new_abs)
            except Exception:
                errors = True
                try:
                    temp_path.replace(old_abs)
                except Exception:
                    pass
        return (not errors), mapping

    def _set_status(self, message: str) -> None:
        self.status_bar.showMessage(message, 5000)

    def _show_error(self, message: str) -> None:
        QtWidgets.QMessageBox.critical(self, "Error", message)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        try:
            if self.audio.recording:
                self.audio.stop_recording()
            if self.audio.preview:
                self.audio.stop_bgm()
        except Exception:
            pass
        self._stop_note_worker()
        self._stop_recorded_worker()
        self._autosave()
        super().closeEvent(event)
