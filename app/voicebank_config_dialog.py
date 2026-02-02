from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

from PyQt6 import QtCore, QtGui, QtWidgets
import yaml

from models.parsers import read_text_guess


class VoicebankConfigDialog(QtWidgets.QDialog):
    def __init__(
        self,
        voicebank_dir: Path,
        tr: Optional[callable] = None,
        lang: str = "English",
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.voicebank_dir = voicebank_dir
        self.lang = lang
        self._t = tr or (lambda key: key)
        self.setWindowTitle(self._t("voicebank_edit_title"))
        self.resize(700, 520)

        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        layout.addLayout(form)

        self.name_edit = QtWidgets.QLineEdit()
        self.author_edit = QtWidgets.QLineEdit()
        self.voice_edit = QtWidgets.QLineEdit()
        self.web_edit = QtWidgets.QLineEdit()
        self.version_edit = QtWidgets.QLineEdit()
        self.other_edit = QtWidgets.QTextEdit()
        self.other_edit.setFixedHeight(80)

        self.image_label = QtWidgets.QLabel("")
        self.sample_label = QtWidgets.QLabel("")
        self.portrait_label = QtWidgets.QLabel("")
        self.portrait_opacity_edit = QtWidgets.QLineEdit()
        self.portrait_height_edit = QtWidgets.QLineEdit()

        self.text_encoding_combo = QtWidgets.QComboBox()
        self.text_encoding_combo.addItems(
            ["", "shift_jis", "utf-8", "gbk", "big5", "euc_jp", "cp932", "cp936", "cp950", "euc_kr"]
        )

        self.txt_export_combo = QtWidgets.QComboBox()
        self.txt_export_combo.addItems(["Shift_JIS", "Specified encoding", "No export"])

        self.singer_type_edit = QtWidgets.QLineEdit()
        self.phonemizer_edit = QtWidgets.QLineEdit()
        self.use_filename_check = QtWidgets.QCheckBox("Use filename as alias")

        form.addRow(self._t("voicebank_name"), self.name_edit)
        form.addRow(self._t("voicebank_author"), self.author_edit)
        form.addRow(self._t("voicebank_voice"), self.voice_edit)
        form.addRow(self._t("voicebank_web"), self.web_edit)
        form.addRow(self._t("voicebank_version"), self.version_edit)

        form.addRow(self._t("voicebank_image"), self._file_row(self.image_label, "image"))
        form.addRow(self._t("voicebank_sample"), self._file_row(self.sample_label, "sample"))
        form.addRow(self._t("voicebank_portrait"), self._file_row(self.portrait_label, "portrait"))
        form.addRow(self._t("voicebank_portrait_opacity"), self.portrait_opacity_edit)
        form.addRow(self._t("voicebank_portrait_height"), self.portrait_height_edit)

        form.addRow(self._t("voicebank_text_encoding"), self.text_encoding_combo)
        form.addRow(self._t("voicebank_txt_export"), self.txt_export_combo)
        form.addRow(self._t("voicebank_singer_type"), self.singer_type_edit)
        form.addRow(self._t("voicebank_phonemizer"), self.phonemizer_edit)
        self.use_filename_check.setText(self._t("voicebank_use_filename"))
        form.addRow(self.use_filename_check)
        form.addRow(self._t("voicebank_other_info"), self.other_edit)

        self.localized_table = QtWidgets.QTableWidget(0, 2)
        self.localized_table.setHorizontalHeaderLabels(["Language", "Name"])
        self.localized_table.horizontalHeader().setStretchLastSection(True)
        self.localized_table.setMinimumHeight(160)
        layout.addWidget(QtWidgets.QLabel(self._t("voicebank_localized_names")))
        layout.addWidget(self.localized_table)

        row_btns = QtWidgets.QHBoxLayout()
        self.add_lang_btn = QtWidgets.QPushButton(self._t("voicebank_add"))
        self.remove_lang_btn = QtWidgets.QPushButton(self._t("voicebank_remove"))
        row_btns.addWidget(self.add_lang_btn)
        row_btns.addWidget(self.remove_lang_btn)
        row_btns.addStretch(1)
        layout.addLayout(row_btns)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Save | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        layout.addWidget(buttons)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        self.add_lang_btn.clicked.connect(self._add_localized_row)
        self.remove_lang_btn.clicked.connect(self._remove_localized_row)

        self._load()

    def _file_row(self, label: QtWidgets.QLabel, tag: str) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        label.setMinimumWidth(300)
        row.addWidget(label, 1)
        open_btn = QtWidgets.QPushButton("Open")
        clear_btn = QtWidgets.QPushButton("Clear")
        open_btn.clicked.connect(lambda: self._open_file(tag))
        clear_btn.clicked.connect(lambda: self._clear_file(tag))
        row.addWidget(open_btn)
        row.addWidget(clear_btn)
        return container

    def _open_file(self, tag: str) -> None:
        if tag in ("image", "portrait"):
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                self._t("voicebank_pick_image"),
                "",
                "Images (*.png *.jpg *.jpeg *.bmp *.gif)",
            )
        else:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                self._t("voicebank_pick_sample"),
                "",
                "Audio (*.wav *.mp3 *.ogg *.flac);;All Files (*)",
            )
        if not path:
            return
        src = Path(path)
        dest = src
        try:
            src_rel = src.resolve()
            vb_rel = self.voicebank_dir.resolve()
            inside = src_rel.is_relative_to(vb_rel)
        except Exception:
            inside = False
        if not inside:
            dest = self.voicebank_dir / src.name
            try:
                shutil.copy2(src, dest)
            except Exception:
                dest = src
        rel = dest.name
        if tag == "image":
            self.image_label.setText(rel)
        elif tag == "portrait":
            self.portrait_label.setText(rel)
        elif tag == "sample":
            self.sample_label.setText(rel)

    def _clear_file(self, tag: str) -> None:
        if tag == "image":
            self.image_label.setText("")
        elif tag == "portrait":
            self.portrait_label.setText("")
        elif tag == "sample":
            self.sample_label.setText("")

    def _add_localized_row(self) -> None:
        row = self.localized_table.rowCount()
        self.localized_table.insertRow(row)
        self.localized_table.setItem(row, 0, QtWidgets.QTableWidgetItem(""))
        self.localized_table.setItem(row, 1, QtWidgets.QTableWidgetItem(""))

    def _remove_localized_row(self) -> None:
        rows = sorted({idx.row() for idx in self.localized_table.selectionModel().selectedRows()}, reverse=True)
        for row in rows:
            self.localized_table.removeRow(row)

    def _load(self) -> None:
        txt_path = self.voicebank_dir / "character.txt"
        if txt_path.exists():
            text = read_text_guess(txt_path)
            other_lines = []
            for line in text.splitlines():
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip().lower()
                    value = value.strip()
                    if key == "name":
                        self.name_edit.setText(value)
                    elif key == "author":
                        self.author_edit.setText(value)
                    elif key in ("image", "icon"):
                        self.image_label.setText(value)
                    elif key == "sample":
                        self.sample_label.setText(value)
                    elif key in ("web", "site"):
                        self.web_edit.setText(value)
                    elif key == "voice":
                        self.voice_edit.setText(value)
                    elif key == "version":
                        self.version_edit.setText(value)
                    else:
                        other_lines.append(line)
                else:
                    other_lines.append(line)
            if other_lines:
                self.other_edit.setPlainText("\n".join(other_lines).strip())

        yaml_path = self.voicebank_dir / "character.yaml"
        if yaml_path.exists():
            try:
                data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            except Exception:
                data = None
            if isinstance(data, dict):
                localized = data.get("localized_names") or data.get("LocalizedNames") or data.get("localizedNames") or {}
                if isinstance(localized, dict):
                    for lang, name in localized.items():
                        row = self.localized_table.rowCount()
                        self.localized_table.insertRow(row)
                        self.localized_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(lang)))
                        self.localized_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(name)))
                self.singer_type_edit.setText(str(data.get("singer_type", data.get("SingerType", "")) or ""))
                self.text_encoding_combo.setCurrentText(str(data.get("text_file_encoding", data.get("TextFileEncoding", "")) or ""))
                portrait = data.get("portrait", data.get("Portrait", "")) or ""
                self.portrait_label.setText(str(portrait))
                image = data.get("image", data.get("Image", "")) or ""
                if image:
                    self.image_label.setText(str(image))
                if data.get("portrait_opacity", data.get("PortraitOpacity")) is not None:
                    self.portrait_opacity_edit.setText(str(data.get("portrait_opacity", data.get("PortraitOpacity"))))
                if data.get("portrait_height", data.get("PortraitHeight")) is not None:
                    self.portrait_height_edit.setText(str(data.get("portrait_height", data.get("PortraitHeight"))))
                self.phonemizer_edit.setText(str(data.get("default_phonemizer", data.get("DefaultPhonemizer", "")) or ""))
                self.use_filename_check.setChecked(bool(data.get("use_filename_as_alias", data.get("UseFilenameAsAlias", False))))
        if not self.singer_type_edit.text().strip():
            self.singer_type_edit.setText("classic")

    def _save(self) -> None:
        self.voicebank_dir.mkdir(parents=True, exist_ok=True)

        localized = {}
        for row in range(self.localized_table.rowCount()):
            lang_item = self.localized_table.item(row, 0)
            name_item = self.localized_table.item(row, 1)
            lang = lang_item.text().strip() if lang_item else ""
            name = name_item.text().strip() if name_item else ""
            if lang and name:
                localized[lang] = name

        yaml_data = {
            "name": self.name_edit.text().strip() or None,
            "singer_type": self.singer_type_edit.text().strip() or None,
            "text_file_encoding": self.text_encoding_combo.currentText().strip() or None,
            "image": self.image_label.text().strip() or None,
            "portrait": self.portrait_label.text().strip() or None,
            "portrait_opacity": self._float_or_none(self.portrait_opacity_edit.text().strip()),
            "portrait_height": self._int_or_none(self.portrait_height_edit.text().strip()),
            "default_phonemizer": self.phonemizer_edit.text().strip() or None,
            "localized_names": localized or None,
            "use_filename_as_alias": True if self.use_filename_check.isChecked() else None,
        }
        yaml_path = self.voicebank_dir / "character.yaml"
        yaml_path.write_text(yaml.safe_dump(yaml_data, allow_unicode=True, sort_keys=False), encoding="utf-8")

        # character.txt
        export_mode = self.txt_export_combo.currentIndex()
        if export_mode != 2:
            encoding = "shift_jis"
            if export_mode == 1:
                encoding = self.text_encoding_combo.currentText().strip() or "shift_jis"
            lines = []
            self._add_txt_line(lines, "name", self.name_edit.text().strip())
            self._add_txt_line(lines, "image", self.image_label.text().strip())
            self._add_txt_line(lines, "icon", self.image_label.text().strip())
            self._add_txt_line(lines, "author", self.author_edit.text().strip())
            self._add_txt_line(lines, "web", self.web_edit.text().strip())
            self._add_txt_line(lines, "site", self.web_edit.text().strip())
            self._add_txt_line(lines, "version", self.version_edit.text().strip())
            other = self.other_edit.toPlainText().strip()
            if other:
                lines.append(other)
            txt_path = self.voicebank_dir / "character.txt"
            txt_path.write_text("\n".join(lines), encoding=encoding, errors="ignore")

        self.accept()

    @staticmethod
    def _add_txt_line(lines: list[str], key: str, value: str) -> None:
        if value:
            lines.append(f"{key}={value}")

    @staticmethod
    def _float_or_none(value: str) -> Optional[float]:
        try:
            return float(value)
        except ValueError:
            return None

    @staticmethod
    def _int_or_none(value: str) -> Optional[int]:
        try:
            return int(value)
        except ValueError:
            return None
