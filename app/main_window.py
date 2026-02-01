from __future__ import annotations

import logging
import sys
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import sounddevice as sd
import shutil
from datetime import datetime
from PyQt6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from audio.engine import AudioEngine
from audio.dsp import (
    compute_fft,
    compute_rms,
    estimate_f0,
    note_from_f0,
    compute_f0_contour,
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


logger = logging.getLogger(__name__)


TRANSLATIONS = {
    "English": {
        "app_title": "UTAU Voicebank Recorder",
        "file": "File",
        "import": "Import",
        "settings": "Settings",
        "edit": "Edit",
        "tools": "Tools",
        "vst_tools": "VST Tools",
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
        "save_reclist_to": "Save Reclist To",
        "import_reclist": "Import Reclist",
        "import_voicebank": "Import Voicebank",
        "import_bgm": "Import BGM WAV",
        "generate_bgm": "Generate BGM Note",
        "audio_devices": "Audio Devices",
        "language": "Language",
        "ui_settings": "UI Settings",
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
        "table_duration": "Duration",
        "table_file": "File",
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
        "bgm_mode_title": "BGM Note",
        "bgm_note": "Note (e.g. A4)",
        "bgm_duration": "Duration (sec)",
        "bgm_mode": "Mode",
        "bgm_replace": "Replace BGM",
        "bgm_add": "Add overlay",
        "bgm_duration": "Duration (sec)",
        "import_reclist_title": "Import Reclist",
        "import_reclist_question": "Replace current reclist or add new entries?",
        "add_entry": "Add Entry",
        "delete_entry": "Delete Entry",
        "alias": "Alias",
        "note_optional": "Note (optional)",
    },
    "Русский": {
        "app_title": "UTAU Voicebank Recorder",
        "file": "Файл",
        "import": "Импорт",
        "settings": "Настройки",
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
        "save_reclist_to": "Сохранить реклист как",
        "import_reclist": "Импорт реклиста",
        "import_voicebank": "Импорт voicebank",
        "import_bgm": "Импорт BGM WAV",
        "generate_bgm": "Сгенерировать BGM ноту",
        "audio_devices": "Аудиоустройства",
        "language": "Язык",
        "ui_settings": "Настройки UI",
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
        "table_duration": "Длительность",
        "table_file": "Файл",
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
        "bgm_mode_title": "BGM нота",
        "bgm_note": "Нота (например A4)",
        "bgm_duration": "Длительность (сек)",
        "bgm_mode": "Режим",
        "bgm_replace": "Заменить BGM",
        "bgm_add": "Добавить поверх",
        "bgm_duration": "Длительность (сек)",
        "import_reclist_title": "Импорт реклиста",
        "import_reclist_question": "Заменить текущий реклист или добавить новые?",
        "add_entry": "Добавить",
        "delete_entry": "Удалить",
        "alias": "Алиас",
        "note_optional": "Нота (опц.)",
    },
    "日本語": {
        "app_title": "UTAU Voicebank Recorder",
        "file": "ファイル",
        "import": "インポート",
        "settings": "設定",
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
        "save_reclist_to": "レコリストを書き出し",
        "import_reclist": "レコリストをインポート",
        "import_voicebank": "音源をインポート",
        "import_bgm": "BGM WAVをインポート",
        "generate_bgm": "BGMノート生成",
        "audio_devices": "オーディオデバイス",
        "language": "言語",
        "ui_settings": "UI設定",
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
        "table_duration": "長さ",
        "table_file": "ファイル",
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
        "bgm_mode_title": "BGMノート",
        "bgm_note": "ノート (例 A4)",
        "bgm_duration": "長さ (秒)",
        "bgm_mode": "モード",
        "bgm_replace": "BGMを置換",
        "bgm_add": "上に追加",
        "bgm_duration": "長さ (秒)",
        "import_reclist_title": "レコリストをインポート",
        "import_reclist_question": "既存を置換しますか？それとも追加しますか？",
        "add_entry": "追加",
        "delete_entry": "削除",
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

        layout.addRow(tr(self.lang, "language"), self.lang_combo)
        layout.addRow(tr(self.lang, "theme"), self.theme_combo)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_data(self) -> tuple[str, str]:
        return self.lang_combo.currentText(), str(self.theme_combo.currentData())


class VoicebankImportDialog(QtWidgets.QDialog):
    def __init__(self, lang: str, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.lang = lang
        self.setWindowTitle(tr(self.lang, "voicebank_options"))
        layout = QtWidgets.QVBoxLayout(self)
        self.use_bgm_checkbox = QtWidgets.QCheckBox(tr(self.lang, "use_vb_bgm"))
        self.use_bgm_checkbox.setChecked(True)
        layout.addWidget(self.use_bgm_checkbox)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def use_bgm(self) -> bool:
        return self.use_bgm_checkbox.isChecked()


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

        self.new_btn.clicked.connect(self._new_clicked)
        self.open_btn.clicked.connect(self._open_clicked)
        self.list_widget.itemDoubleClicked.connect(self._recent_clicked)
        new_action.triggered.connect(self._new_clicked)
        open_action.triggered.connect(self._open_clicked)
        close_action.triggered.connect(self.reject)
        ui_action.triggered.connect(self._ui_clicked)

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

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.closed_via_x = True
        super().closeEvent(event)
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
        self.mode_combo.addItems([tr(self.lang, "bgm_replace"), tr(self.lang, "bgm_add")])
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
        self._update_visibility()

    def _update_visibility(self) -> None:
        timing = self.timing_combo.currentText()
        is_timing = timing == tr(self.lang, "bgm_timing")
        self.duration_label.setVisible(not is_timing)
        self.duration_spin.setVisible(not is_timing)
        self.bpm_label.setVisible(is_timing)
        self.bpm_spin.setVisible(is_timing)
        self.mora_label.setVisible(is_timing)
        self.mora_spin.setVisible(is_timing)

    def get_data(self) -> Optional[tuple[str, float, str, int, int, str]]:
        if self.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        note = self.note_edit.text().strip()
        if not note:
            return None
        return (
            note,
            float(self.duration_spin.value()),
            self.mode_combo.currentText(),
            int(self.bpm_spin.value()),
            int(self.mora_spin.value()),
            self.timing_combo.currentText(),
        )


class NoteAnalysisWorker(QtCore.QThread):
    result = QtCore.pyqtSignal(str, float, str)
    finished = QtCore.pyqtSignal()

    def __init__(self, files: list[str], target_sr: int, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self.files = files
        self.target_sr = target_sr

    def run(self) -> None:
        for path in self.files:
            if self.isInterruptionRequested():
                break
            try:
                file_path = Path(path)
                if not file_path.exists():
                    continue
                audio, sr = sf.read(str(file_path), dtype="float32")
                if audio.ndim > 1:
                    audio = np.mean(audio, axis=1)
                if sr != self.target_sr:
                    audio = AudioEngine._resample(audio, sr, self.target_sr)
                    sr = self.target_sr
                _, f0s = compute_f0_contour(audio, sr)
                if f0s.size == 0:
                    note = "--"
                else:
                    mids = [f0_to_midi(float(f)) for f in f0s if f and f > 0]
                    mids = [m for m in mids if m is not None]
                    if not mids:
                        note = "--"
                    else:
                        avg_midi = float(np.mean(mids))
                        note = midi_to_note(avg_midi)
                mtime = file_path.stat().st_mtime
                self.result.emit(str(file_path), mtime, note)
            except Exception:
                logger.exception("Failed to analyze note for %s", path)
        self.finished.emit()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.resize(1200, 800)

        self.settings = QtCore.QSettings("UtauRecorder", "UtauRecorder")
        self.ui_language = self.settings.value("ui_language", "English")
        self.ui_theme = self.settings.value("ui_theme", "light")
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

        self.audio = AudioEngine()
        self.audio.error.connect(self._show_error)
        self.audio.status.connect(self._set_status)

        self._build_ui()
        self._build_menu()
        self._connect_actions()
        self._apply_language()
        self._apply_theme()
        self._maybe_show_start_dialog()

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
        self.f0_values = []
        self.history_size = 200
        self.playing = False
        self.play_start_time: Optional[QtCore.QElapsedTimer] = None
        self.play_start_pos = 0.0
        self.play_duration = 0.0
        self.selected_audio: Optional[np.ndarray] = None
        self.undo_stack: list[list[dict]] = []
        self._suppress_item_changed = False

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        main_layout = QtWidgets.QVBoxLayout(central)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter, 1)

        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels([
            tr(self.ui_language, "table_status"),
            tr(self.ui_language, "table_alias"),
            tr(self.ui_language, "table_romaji"),
            tr(self.ui_language, "table_note"),
            tr(self.ui_language, "table_duration"),
            tr(self.ui_language, "table_file"),
        ])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
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

        self.f0_plot = pg.PlotWidget(
            title=f"{tr(self.ui_language, 'f0')} (Piano Roll)",
            axisItems={"left": NoteAxis(orientation="left")},
        )
        self.f0_curve = self.f0_plot.plot(pen="g")
        self.f0_playhead = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("w", width=1))
        self.f0_playhead.setVisible(False)
        self.f0_plot.addItem(self.f0_playhead)
        self.f0_plot.setXLink(self.wave_plot)
        self.plot_tabs.addTab(self.f0_plot, tr(self.ui_language, "f0"))

        self.recorded_f0_plot = pg.PlotWidget(
            title=f"{tr(self.ui_language, 'recorded_f0')} (Piano Roll)",
            axisItems={"left": NoteAxis(orientation="left")},
        )
        self.recorded_f0_curve = self.recorded_f0_plot.plot(pen="g")
        self.recorded_f0_playhead = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("w", width=1))
        self.recorded_f0_playhead.setVisible(False)
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

        self.wave_plot.scene().sigMouseClicked.connect(self._wave_plot_clicked)

    def _build_menu(self) -> None:
        menu = self.menuBar()
        self.file_menu = menu.addMenu(tr(self.ui_language, "file"))

        self.new_action = self.file_menu.addAction(tr(self.ui_language, "new_session"))
        self.open_action = self.file_menu.addAction(tr(self.ui_language, "open_session"))
        self.save_action = self.file_menu.addAction(tr(self.ui_language, "save_session"))
        self.save_as_action = self.file_menu.addAction(tr(self.ui_language, "save_as"))
        self.export_action = self.file_menu.addAction(tr(self.ui_language, "export_recordings"))
        self.save_reclist_action = self.file_menu.addAction(tr(self.ui_language, "save_reclist_to"))
        self.file_menu.addSeparator()
        self.recent_menu = self.file_menu.addMenu(tr(self.ui_language, "recent_sessions"))
        self._rebuild_recent_menu()

        self.import_menu = menu.addMenu(tr(self.ui_language, "import"))
        self.import_reclist_action = self.import_menu.addAction(tr(self.ui_language, "import_reclist"))
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

    def _connect_actions(self) -> None:
        self.new_action.triggered.connect(self._new_session)
        self.open_action.triggered.connect(self._open_session)
        self.save_action.triggered.connect(self._save_session)
        self.save_as_action.triggered.connect(self._save_as_session)
        self.export_action.triggered.connect(self._export_recordings)
        self.save_reclist_action.triggered.connect(self._save_reclist_as)

        self.import_reclist_action.triggered.connect(self._import_reclist)
        self.import_voicebank_action.triggered.connect(self._import_voicebank)
        self.import_bgm_action.triggered.connect(self._import_bgm)
        self.generate_bgm_action.triggered.connect(self._generate_bgm)
        self.vst_batch_action.triggered.connect(self._open_vst_batch)
        self.session_settings_action.triggered.connect(self._open_session_settings)
        self.audio_settings_action.triggered.connect(self._open_audio_settings)
        self.vst_tools_action.triggered.connect(self._open_vst_tools)
        self.ui_settings_action.triggered.connect(self._open_ui_settings)
        self.undo_action.triggered.connect(self._undo)

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
            for alias, note in parsed:
                romaji = "_".join(kana_to_romaji_tokens(alias)) if needs_romaji(alias) else None
                self.session.add_item(alias, note, romaji=romaji)
            self._save_reclist_copy(text)
            self._log_event("import_reclist", Path(path).name)
            self._refresh_table()
            self._save_session()
        except Exception as exc:
            logger.exception("Failed to import reclist")
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
                if item.note:
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
        dialog = UiSettingsDialog(self.ui_language, str(self.ui_language), str(self.ui_theme), self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        lang, theme = dialog.get_data()
        self.ui_language = lang
        self.ui_theme = theme
        self.settings.setValue("ui_language", lang)
        self.settings.setValue("ui_theme", theme)
        self._apply_language()
        self._apply_theme()

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
            name = f"bgm_{note}.wav"
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

        self.new_action.setText(tr(self.ui_language, "new_session"))
        self.open_action.setText(tr(self.ui_language, "open_session"))
        self.save_action.setText(tr(self.ui_language, "save_session"))
        self.save_as_action.setText(tr(self.ui_language, "save_as"))
        self.export_action.setText(tr(self.ui_language, "export_recordings"))
        self.save_reclist_action.setText(tr(self.ui_language, "save_reclist_to"))
        self.recent_menu.setTitle(tr(self.ui_language, "recent_sessions"))
        self.import_reclist_action.setText(tr(self.ui_language, "import_reclist"))
        self.import_voicebank_action.setText(tr(self.ui_language, "import_voicebank"))
        self.import_bgm_action.setText(tr(self.ui_language, "import_bgm"))
        self.generate_bgm_action.setText(tr(self.ui_language, "generate_bgm"))
        self.vst_batch_action.setText(tr(self.ui_language, "apply_vst"))
        self.session_settings_action.setText(tr(self.ui_language, "session_settings"))
        self.audio_settings_action.setText(tr(self.ui_language, "audio_devices"))
        self.vst_tools_action.setText(tr(self.ui_language, "vst_tools"))
        self.ui_settings_action.setText(tr(self.ui_language, "ui_settings"))
        self.undo_action.setText(tr(self.ui_language, "undo"))

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
            tr(self.ui_language, "table_duration"),
            tr(self.ui_language, "table_file"),
        ])

        self.wave_plot.setTitle(tr(self.ui_language, "waveform"))
        self.spec_plot.setTitle(tr(self.ui_language, "spectrum"))
        self.power_plot.setTitle(tr(self.ui_language, "power"))
        self.f0_plot.setTitle(f"{tr(self.ui_language, 'f0')} (Piano Roll)")
        self.recorded_f0_plot.setTitle(f"{tr(self.ui_language, 'recorded_f0')} (Piano Roll)")
        self.mel_plot.setTitle(tr(self.ui_language, "mel"))

        if self.plot_tabs.count() >= 5:
            self.plot_tabs.setTabText(0, tr(self.ui_language, "waveform"))
            self.plot_tabs.setTabText(1, tr(self.ui_language, "spectrum"))
            self.plot_tabs.setTabText(2, tr(self.ui_language, "power"))
            self.plot_tabs.setTabText(3, tr(self.ui_language, "f0"))
            self.plot_tabs.setTabText(4, tr(self.ui_language, "recorded_f0"))
            self.plot_tabs.setTabText(5, tr(self.ui_language, "mel"))

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
        self.current_item = self.session.items[row]
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
            self._update_recorded_analysis()
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
        self._suppress_item_changed = True
        self.table.setRowCount(len(self.session.items))
        target_note = self._target_bgm_note()
        for row, item in enumerate(self.session.items):
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(item.status.value))
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(item.alias))
            romaji_text = ""
            if item.romaji is None and needs_romaji(item.alias):
                item.romaji = "_".join(kana_to_romaji_tokens(item.alias))
            if item.romaji:
                romaji_text = item.romaji.replace(" ", "_")
            self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(romaji_text))
            note_text = item.note or ""
            if target_note and item.wav_path:
                sung_note = self._get_cached_sung_note(item.wav_path)
                if sung_note is None:
                    note_text = "..."
                else:
                    note_text = self._format_note_check(target_note, sung_note)
            self.table.setItem(row, 3, QtWidgets.QTableWidgetItem(note_text))
            duration = f"{item.duration_sec:.2f}" if item.duration_sec else ""
            self.table.setItem(row, 4, QtWidgets.QTableWidgetItem(duration))
            self.table.setItem(row, 5, QtWidgets.QTableWidgetItem(item.wav_path or ""))
            for col in range(6):
                item_widget = self.table.item(row, col)
                if item_widget is None:
                    continue
                flags = item_widget.flags()
                if col == 1:
                    item_widget.setFlags(flags | QtCore.Qt.ItemFlag.ItemIsEditable)
                else:
                    item_widget.setFlags(flags & ~QtCore.Qt.ItemFlag.ItemIsEditable)
        self._suppress_item_changed = False

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

    def _format_note_check(self, target: str, sung: str) -> str:
        is_match = self._normalize_note(target) == self._normalize_note(sung)
        mark = "✅" if is_match else "❌"
        return f"{mark} ({sung})"

    def _get_cached_sung_note(self, wav_rel: str) -> Optional[str]:
        if not self.session:
            return None
        abs_path = Path(wav_rel)
        if not abs_path.is_absolute():
            abs_path = self.session.session_dir() / abs_path
        cache_key = str(abs_path)
        cached = self._sung_note_cache.get(cache_key)
        if not cached:
            return None
        return cached[1]

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
            files.append(str(abs_path))
        return files

    def _start_note_analysis(self) -> None:
        target_note = self._target_bgm_note()
        if not target_note or not self.session:
            return
        files = self._collect_note_analysis_files()
        if not files:
            return
        if self._note_worker and self._note_worker.isRunning():
            self._note_analysis_pending.update(files)
            return
        self._note_worker = NoteAnalysisWorker(files, self.session.sample_rate, self)
        self._note_worker.result.connect(self._on_note_analysis_result)
        self._note_worker.finished.connect(self._on_note_analysis_finished)
        self._note_worker.start()

    def _stop_note_worker(self) -> None:
        if self._note_worker and self._note_worker.isRunning():
            self._note_worker.requestInterruption()
            self._note_worker.wait(2000)
        self._note_worker = None
        self._note_analysis_pending.clear()

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
                self.table.setItem(row, 3, QtWidgets.QTableWidgetItem(self._format_note_check(target_note, note)))
                break

    def _on_note_analysis_finished(self) -> None:
        if self._note_analysis_pending:
            files = list(self._note_analysis_pending)
            self._note_analysis_pending.clear()
            if self.session:
                self._note_worker = NoteAnalysisWorker(files, self.session.sample_rate, self)
                self._note_worker.result.connect(self._on_note_analysis_result)
                self._note_worker.finished.connect(self._on_note_analysis_finished)
                self._note_worker.start()

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
        self.power_curve.setData(self.power_history)

        f0 = estimate_f0(buffer, self.audio.sample_rate)
        note, cents = note_from_f0(f0)
        self.note_label.setText(f"{tr(self.ui_language, 'current_note_prefix')}{note} ({cents:+.1f} cents)")

        midi_val = f0_to_midi(f0)
        if midi_val is not None:
            self.f0_values.append(midi_val)
            if len(self.f0_values) > self.history_size:
                self.f0_values.pop(0)
            xs = np.arange(len(self.f0_values)) * (self.visual_timer.interval() / 1000.0)
            self.f0_curve.setData(xs, self.f0_values)

    def _wave_plot_clicked(self, event: QtCore.QEvent) -> None:
        if not self.playhead:
            return
        if self.audio.recording or self.audio.preview:
            return
        view_pos = self.wave_plot.plotItem.vb.mapSceneToView(event.scenePos())
        x = float(view_pos.x())
        if x < 0:
            x = 0.0
        self._play_recorded_from(x)

    def _play_recorded_from(self, start_sec: float) -> None:
        audio = self.selected_audio if self.selected_audio is not None else self.audio.get_waveform_audio()
        if audio.size == 0:
            return
        start_sample = int(start_sec * self.audio.sample_rate)
        start_sample = max(0, min(start_sample, len(audio) - 1))
        output_device = None
        if self.audio.device is not None:
            output_device = self.audio.device[1]
        try:
            sd.stop()
            sd.play(audio[start_sample:], samplerate=self.audio.sample_rate, device=output_device)
            self.playing = True
            self.play_start_time = QtCore.QElapsedTimer()
            self.play_start_time.start()
            self.play_start_pos = start_sample / self.audio.sample_rate
            self.play_duration = (len(audio) - start_sample) / self.audio.sample_rate
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
        if hasattr(self, "f0_playhead"):
            self.f0_playhead.setVisible(True)
            self.f0_playhead.setPos(pos)
        if hasattr(self, "recorded_f0_playhead"):
            self.recorded_f0_playhead.setVisible(True)
            self.recorded_f0_playhead.setPos(pos)
        if hasattr(self, "mel_playhead"):
            self.mel_playhead.setVisible(True)
            self.mel_playhead.setPos(pos)
        if elapsed >= self.play_duration:
            self.playing = False
            if hasattr(self, "spec_playhead"):
                self.spec_playhead.setVisible(False)
            if hasattr(self, "power_playhead"):
                self.power_playhead.setVisible(False)
            if hasattr(self, "f0_playhead"):
                self.f0_playhead.setVisible(False)
            if hasattr(self, "recorded_f0_playhead"):
                self.recorded_f0_playhead.setVisible(False)
            if hasattr(self, "mel_playhead"):
                self.mel_playhead.setVisible(False)

    def _stop_playback(self) -> None:
        if self.playing:
            sd.stop()
        self.playing = False
        if hasattr(self, "spec_playhead"):
            self.spec_playhead.setVisible(False)
        if hasattr(self, "power_playhead"):
            self.power_playhead.setVisible(False)
        if hasattr(self, "f0_playhead"):
            self.f0_playhead.setVisible(False)
        if hasattr(self, "recorded_f0_playhead"):
            self.recorded_f0_playhead.setVisible(False)
        if hasattr(self, "mel_playhead"):
            self.mel_playhead.setVisible(False)

    def _update_recorded_analysis(self) -> None:
        try:
            audio = self.selected_audio if self.selected_audio is not None else self.audio.get_waveform_audio()
            if audio.size == 0:
                return
            times, f0s = compute_f0_contour(audio, self.audio.sample_rate)
            if times.size and f0s.size:
                midi_vals = np.array([f0_to_midi(f) if f > 0 else np.nan for f in f0s], dtype=np.float32)
                self.recorded_f0_curve.setData(times, midi_vals)
            mel_db, mel_times = compute_mel_spectrogram(audio, self.audio.sample_rate)
            if mel_db.size:
                self.mel_img.setImage(mel_db.T, autoLevels=True)
                self.mel_img.setRect(QtCore.QRectF(0, 0, float(mel_times[-1]) if mel_times.size else 1.0, mel_db.shape[0]))
                if mel_times.size:
                    self.mel_plot.setLimits(xMin=0, xMax=float(mel_times[-1]))
        except Exception as exc:
            logger.exception("Failed to update recorded analysis")
            self._show_error(str(exc))

    def _analyze_selected_item(self) -> None:
        if not self.session or not self.current_item or not self.current_item.wav_path:
            self.selected_audio = None
            self._clear_analysis()
            return
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
            self.power_curve.setData(self.power_history)
            f0 = estimate_f0(snippet, self.audio.sample_rate)
            note, cents = note_from_f0(f0)
            self.note_label.setText(f"{tr(self.ui_language, 'current_note_prefix')}{note} ({cents:+.1f} cents)")
            self._update_recorded_analysis()
        except Exception as exc:
            logger.exception("Failed to analyze selected item")
            self._show_error(str(exc))

    def _table_context_menu(self, pos: QtCore.QPoint) -> None:
        menu = QtWidgets.QMenu(self)
        add_action = menu.addAction(tr(self.ui_language, "add_entry"))
        delete_action = menu.addAction(tr(self.ui_language, "delete_entry"))
        action = menu.exec(self.table.viewport().mapToGlobal(pos))
        if action == add_action:
            self._add_entry()
        elif action == delete_action:
            self._delete_entry()

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

    def _item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if self._suppress_item_changed or not self.session:
            return
        row = item.row()
        col = item.column()
        if row < 0 or row >= len(self.session.items):
            return
        if col != 1:
            return
        new_alias = item.text().strip()
        if not new_alias:
            return
        current = self.session.items[row]
        if any(it.alias == new_alias for i, it in enumerate(self.session.items) if i != row):
            self._show_error("Alias already exists")
            self._refresh_table()
            return
        self._push_undo_state()
        current.alias = new_alias
        current.romaji = "_".join(kana_to_romaji_tokens(new_alias)) if needs_romaji(new_alias) else None
        self._refresh_table()
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
        self.wave_curve.setData([], [])
        self.spec_curve.setData([], [])
        self.power_history = []
        self.power_curve.setData([], [])
        self.f0_values = []
        self.f0_curve.setData([], [])
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
        if hasattr(self, "f0_playhead"):
            self.f0_playhead.setVisible(False)
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
        self._autosave()
        super().closeEvent(event)
