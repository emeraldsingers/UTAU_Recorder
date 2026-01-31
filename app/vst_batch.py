from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from PyQt6 import QtCore, QtWidgets

from models.session import Session

logger = logging.getLogger(__name__)


@dataclass
class VstPluginSlot:
    path: str
    preset: str = ""
    bypass: bool = False


class VstPresetStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._data = {"version": 1, "presets": {}}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            self._data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            self._data = {"version": 1, "presets": {}}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8")

    def list_presets(self) -> list[str]:
        return sorted(self._data.get("presets", {}).keys())

    def get_preset(self, name: str) -> Optional[list[dict]]:
        return self._data.get("presets", {}).get(name)

    def save_preset(self, name: str, chain: list[dict]) -> None:
        self._data.setdefault("presets", {})[name] = chain
        self._save()

    def delete_preset(self, name: str) -> None:
        if name in self._data.get("presets", {}):
            self._data["presets"].pop(name, None)
            self._save()


def _default_tool_path(name: str) -> str:
    exe = f"{name}.exe" if sys.platform == "win32" else name
    return str(Path.cwd() / "tools" / exe)


class VstBatchWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, int, str)
    status = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)
    finished_ok = QtCore.pyqtSignal()

    def __init__(
        self,
        files: list[Path],
        chain: list[dict],
        host_cmd: str,
        max_workers: int = 1,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.files = files
        self.chain = chain
        self.host_cmd = host_cmd or ""
        self.max_workers = max(1, int(max_workers))

    def run(self) -> None:
        if not self.files:
            self.error.emit("No files selected.")
            return
        if not self.host_cmd.strip():
            self.error.emit("VST host command is not set.")
            return

        backup_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dirs: dict[Path, Path] = {}

        chain_file = None
        try:
            chain_payload = {"version": 1, "plugins": self.chain}
            chain_fd, chain_path = tempfile.mkstemp(suffix=".json")
            chain_file = Path(chain_path)
            with os.fdopen(chain_fd, "w", encoding="utf-8") as f:
                json.dump(chain_payload, f, indent=2, ensure_ascii=False)

            base_cmd = self._split_cmd(self.host_cmd)
            if not base_cmd:
                self.error.emit("VST host command is not valid.")
                return

            total = len(self.files)
            for file_path in self.files:
                backup_dir = backup_dirs.get(file_path.parent)
                if backup_dir is None:
                    backup_dir = file_path.parent / f"_backup_vst_{backup_stamp}"
                    backup_dir.mkdir(parents=True, exist_ok=True)
                    backup_dirs[file_path.parent] = backup_dir

            def process_one(file_path: Path) -> tuple[bool, str]:
                if not file_path.exists():
                    return False, f"File not found: {file_path}"
                backup_path = backup_dirs[file_path.parent] / file_path.name
                shutil.copy2(file_path, backup_path)
                tmp_out = file_path.with_name(f"{file_path.stem}.vsttmp{file_path.suffix}")
                try:
                    if tmp_out.exists():
                        tmp_out.unlink()
                    cmd = base_cmd + ["--input", str(file_path), "--output", str(tmp_out), "--chain", str(chain_file)]
                    self.status.emit(f"Processing {file_path.name}...")
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        check=False,
                    )
                    if result.returncode != 0:
                        stderr = result.stderr.strip()
                        msg = stderr or result.stdout.strip() or "Unknown VST host error."
                        return False, msg
                    if not tmp_out.exists():
                        return False, "VST host did not produce output file."
                    os.replace(tmp_out, file_path)
                    return True, ""
                finally:
                    if tmp_out.exists():
                        tmp_out.unlink(missing_ok=True)

            if self.max_workers <= 1:
                for idx, file_path in enumerate(self.files, start=1):
                    self.progress.emit(idx - 1, total, str(file_path))
                    ok, msg = process_one(file_path)
                    if not ok:
                        self.error.emit(msg)
                        return
                    self.progress.emit(idx, total, str(file_path))
            else:
                done = 0
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = {executor.submit(process_one, fp): fp for fp in self.files}
                    for future in as_completed(futures):
                        file_path = futures[future]
                        ok, msg = future.result()
                        done += 1
                        self.progress.emit(done, total, str(file_path))
                        if not ok:
                            self.error.emit(msg)
                            return

            self.status.emit("Done.")
            self.finished_ok.emit()
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            if chain_file and chain_file.exists():
                chain_file.unlink(missing_ok=True)

    @staticmethod
    def _split_cmd(cmd: str) -> list[str]:
        try:
            return shlex.split(cmd, posix=os.name != "nt")
        except ValueError:
            return []


class VstFileSelectionPage(QtWidgets.QWidget):
    def __init__(
        self,
        t: Callable[[str], str],
        session: Session,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._t = t
        self.session = session
        layout = QtWidgets.QVBoxLayout(self)

        title = QtWidgets.QLabel(self._t("vst_select_audio"))
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        layout.addWidget(title)

        search_layout = QtWidgets.QHBoxLayout()
        self.search_edit = QtWidgets.QLineEdit()
        self.search_edit.setPlaceholderText(self._t("vst_search_placeholder"))
        search_layout.addWidget(self.search_edit)
        layout.addLayout(search_layout)

        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels([self._t("vst_use"), self._t("alias"), self._t("vst_file")])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table, 1)

        btn_layout = QtWidgets.QHBoxLayout()
        self.reload_btn = QtWidgets.QPushButton(self._t("vst_reload"))
        self.select_all_btn = QtWidgets.QPushButton(self._t("vst_select_all"))
        self.clear_selection_btn = QtWidgets.QPushButton(self._t("vst_clear_selection"))
        btn_layout.addWidget(self.reload_btn)
        btn_layout.addWidget(self.select_all_btn)
        btn_layout.addWidget(self.clear_selection_btn)
        layout.addLayout(btn_layout)

        self.reload_btn.clicked.connect(self._load_session_files)
        self.select_all_btn.clicked.connect(self._select_all)
        self.clear_selection_btn.clicked.connect(self._clear_selection)
        self.search_edit.textChanged.connect(self._apply_filter)
        self._load_session_files()

    def _load_session_files(self) -> None:
        self.table.setRowCount(0)
        base = self.session.session_dir()
        for item in self.session.items:
            if not item.wav_path:
                continue
            full_path = base / item.wav_path
            if not full_path.exists():
                continue
            try:
                display_path = str(full_path.relative_to(base))
            except ValueError:
                display_path = str(full_path)
            row = self.table.rowCount()
            self.table.insertRow(row)
            check_item = QtWidgets.QTableWidgetItem()
            check_item.setFlags(check_item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            check_item.setCheckState(QtCore.Qt.CheckState.Checked)
            alias_item = QtWidgets.QTableWidgetItem(item.alias)
            file_item = QtWidgets.QTableWidgetItem(display_path)
            file_item.setData(QtCore.Qt.ItemDataRole.UserRole, str(full_path))
            self.table.setItem(row, 0, check_item)
            self.table.setItem(row, 1, alias_item)
            self.table.setItem(row, 2, file_item)
        self._apply_filter()

    def _select_all(self) -> None:
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item:
                item.setCheckState(QtCore.Qt.CheckState.Checked)

    def _clear_selection(self) -> None:
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item:
                item.setCheckState(QtCore.Qt.CheckState.Unchecked)

    def _apply_filter(self) -> None:
        needle = self.search_edit.text().strip().lower()
        for row in range(self.table.rowCount()):
            alias_item = self.table.item(row, 1)
            file_item = self.table.item(row, 2)
            if not alias_item or not file_item:
                continue
            text = f"{alias_item.text()} {file_item.text()}".lower()
            self.table.setRowHidden(row, needle not in text)

    def selected_files(self) -> list[Path]:
        files: list[Path] = []
        for row in range(self.table.rowCount()):
            check_item = self.table.item(row, 0)
            path_item = self.table.item(row, 2)
            if not check_item or not path_item:
                continue
            if check_item.checkState() == QtCore.Qt.CheckState.Checked:
                raw_path = path_item.data(QtCore.Qt.ItemDataRole.UserRole) or path_item.text()
                files.append(Path(raw_path))
        return files


class VstMixerPage(QtWidgets.QWidget):
    def __init__(
        self,
        t: Callable[[str], str],
        settings: QtCore.QSettings,
        store: VstPresetStore,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._t = t
        self.settings = settings
        self.store = store
        self._gui_processes: list[subprocess.Popen] = []

        layout = QtWidgets.QVBoxLayout(self)
        header = QtWidgets.QLabel(self._t("vst_chain_title"))
        header.setStyleSheet("font-size: 16px; font-weight: 600;")
        layout.addWidget(header)

        preset_row = QtWidgets.QHBoxLayout()
        preset_row.addWidget(QtWidgets.QLabel(self._t("vst_chain_presets")))
        self.preset_combo = QtWidgets.QComboBox()
        self._reload_presets()
        preset_row.addWidget(self.preset_combo, 1)
        self.load_btn = QtWidgets.QPushButton(self._t("vst_load_preset"))
        self.save_btn = QtWidgets.QPushButton(self._t("vst_save_preset"))
        self.delete_btn = QtWidgets.QPushButton(self._t("vst_delete_preset"))
        preset_row.addWidget(self.load_btn)
        preset_row.addWidget(self.save_btn)
        preset_row.addWidget(self.delete_btn)
        layout.addLayout(preset_row)

        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(
            [self._t("vst_plugin_path"), self._t("vst_plugin_preset"), self._t("vst_bypass")]
        )
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.AllEditTriggers)
        layout.addWidget(self.table, 1)

        btn_row = QtWidgets.QHBoxLayout()
        self.add_plugin_btn = QtWidgets.QPushButton(self._t("vst_add_plugin"))
        self.remove_plugin_btn = QtWidgets.QPushButton(self._t("vst_remove_plugin"))
        self.move_up_btn = QtWidgets.QPushButton(self._t("vst_move_up"))
        self.move_down_btn = QtWidgets.QPushButton(self._t("vst_move_down"))
        self.open_ui_btn = QtWidgets.QPushButton(self._t("vst_open_ui"))
        self.browse_preset_btn = QtWidgets.QPushButton(self._t("vst_browse_preset"))
        btn_row.addWidget(self.add_plugin_btn)
        btn_row.addWidget(self.remove_plugin_btn)
        btn_row.addWidget(self.move_up_btn)
        btn_row.addWidget(self.move_down_btn)
        btn_row.addWidget(self.open_ui_btn)
        btn_row.addWidget(self.browse_preset_btn)
        layout.addLayout(btn_row)

        cli_row = QtWidgets.QHBoxLayout()
        cli_row.addWidget(QtWidgets.QLabel(self._t("vst_host_cli")))
        self.cli_edit = QtWidgets.QLineEdit(
            self.settings.value("vst_host_cli", _default_tool_path("utau_vst_host"))
        )
        cli_row.addWidget(self.cli_edit, 1)
        self.cli_browse_btn = QtWidgets.QPushButton(self._t("vst_browse"))
        cli_row.addWidget(self.cli_browse_btn)
        layout.addLayout(cli_row)

        gui_row = QtWidgets.QHBoxLayout()
        gui_row.addWidget(QtWidgets.QLabel(self._t("vst_host_gui")))
        self.gui_edit = QtWidgets.QLineEdit(
            self.settings.value("vst_host_gui", _default_tool_path("utau_vst_host_gui"))
        )
        gui_row.addWidget(self.gui_edit, 1)
        self.gui_browse_btn = QtWidgets.QPushButton(self._t("vst_browse"))
        gui_row.addWidget(self.gui_browse_btn)
        layout.addLayout(gui_row)

        workers_row = QtWidgets.QHBoxLayout()
        workers_row.addWidget(QtWidgets.QLabel(self._t("vst_workers")))
        self.workers_spin = QtWidgets.QSpinBox()
        self.workers_spin.setRange(1, 32)
        default_workers = int(self.settings.value("vst_workers", os.cpu_count() or 1))
        self.workers_spin.setValue(max(1, min(default_workers, 32)))
        workers_row.addWidget(self.workers_spin)
        workers_row.addStretch(1)
        layout.addLayout(workers_row)

        note = QtWidgets.QLabel(self._t("vst_host_note"))
        note.setStyleSheet("color: #666666;")
        note.setWordWrap(True)
        layout.addWidget(note)

        self.load_btn.clicked.connect(self._load_preset)
        self.save_btn.clicked.connect(self._save_preset)
        self.delete_btn.clicked.connect(self._delete_preset)
        self.add_plugin_btn.clicked.connect(self._add_plugin)
        self.remove_plugin_btn.clicked.connect(self._remove_plugin)
        self.move_up_btn.clicked.connect(self._move_up)
        self.move_down_btn.clicked.connect(self._move_down)
        self.open_ui_btn.clicked.connect(self._open_plugin_ui)
        self.browse_preset_btn.clicked.connect(self._browse_preset_for_selected)
        self.cli_browse_btn.clicked.connect(self._browse_cli_host)
        self.gui_browse_btn.clicked.connect(self._browse_gui_host)
        self.preset_combo.currentIndexChanged.connect(self._preset_selection_changed)
        self.cli_edit.editingFinished.connect(self._store_host_paths)
        self.gui_edit.editingFinished.connect(self._store_host_paths)
        self.workers_spin.valueChanged.connect(self._store_host_paths)

        last_preset = self.settings.value("vst_last_preset", "")
        if last_preset:
            idx = self.preset_combo.findText(last_preset)
            if idx >= 0:
                self.preset_combo.setCurrentIndex(idx)

    def _reload_presets(self) -> None:
        self.preset_combo.clear()
        presets = self.store.list_presets()
        self.preset_combo.addItem(self._t("vst_preset_none"))
        self.preset_combo.addItems(presets)

    def _preset_selection_changed(self) -> None:
        name = self.preset_combo.currentText()
        if name and name != self._t("vst_preset_none"):
            self.settings.setValue("vst_last_preset", name)

    def _load_preset(self) -> None:
        name = self.preset_combo.currentText()
        if not name or name == self._t("vst_preset_none"):
            return
        chain = self.store.get_preset(name) or []
        self._set_chain(chain)

    def _save_preset(self) -> None:
        name, ok = QtWidgets.QInputDialog.getText(self, self._t("vst_save_preset"), self._t("vst_preset_name"))
        if not ok or not name.strip():
            return
        chain = self.chain()
        self.store.save_preset(name.strip(), chain)
        self._reload_presets()
        idx = self.preset_combo.findText(name.strip())
        if idx >= 0:
            self.preset_combo.setCurrentIndex(idx)

    def _delete_preset(self) -> None:
        name = self.preset_combo.currentText()
        if not name or name == self._t("vst_preset_none"):
            return
        if QtWidgets.QMessageBox.question(self, self._t("vst_delete_preset"), name) != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self.store.delete_preset(name)
        self._reload_presets()

    def _add_plugin(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self._t("vst_add_plugin"),
            "",
            "VST Plugins (*.vst3 *.dll *.so *.dylib);;All Files (*)",
        )
        if not path:
            return
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(path))
        self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(""))
        bypass_item = QtWidgets.QTableWidgetItem()
        bypass_item.setFlags(bypass_item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
        bypass_item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.table.setItem(row, 2, bypass_item)

    def _remove_plugin(self) -> None:
        rows = sorted({idx.row() for idx in self.table.selectionModel().selectedRows()}, reverse=True)
        for row in rows:
            self.table.removeRow(row)

    def _move_up(self) -> None:
        row = self._selected_row()
        if row is None or row == 0:
            return
        self._swap_rows(row, row - 1)
        self.table.selectRow(row - 1)

    def _move_down(self) -> None:
        row = self._selected_row()
        if row is None or row >= self.table.rowCount() - 1:
            return
        self._swap_rows(row, row + 1)
        self.table.selectRow(row + 1)

    def _browse_preset_for_selected(self) -> None:
        row = self._selected_row()
        if row is None:
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self._t("vst_browse_preset"),
            "",
            "VST Preset (*.vstpreset *.fxp);;All Files (*)",
        )
        if not path:
            return
        self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(path))

    def _browse_cli_host(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, self._t("vst_browse"), "", "Executable (*)")
        if not path:
            return
        self.cli_edit.setText(path)
        self._store_host_paths()

    def _browse_gui_host(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, self._t("vst_browse"), "", "Executable (*)")
        if not path:
            return
        self.gui_edit.setText(path)
        self._store_host_paths()

    def _open_plugin_ui(self) -> None:
        row = self._selected_row()
        if row is None:
            QtWidgets.QMessageBox.warning(self, self._t("vst_batch_title"), self._t("vst_no_plugin"))
            return
        path_item = self.table.item(row, 0)
        preset_item = self.table.item(row, 1)
        plugin_path = path_item.text().strip() if path_item else ""
        if not plugin_path:
            QtWidgets.QMessageBox.warning(self, self._t("vst_batch_title"), self._t("vst_no_plugin"))
            return
        gui_host = self.gui_cmd()
        if not gui_host:
            QtWidgets.QMessageBox.warning(self, self._t("vst_batch_title"), self._t("vst_no_gui_host"))
            return

        preset_path = preset_item.text().strip() if preset_item else ""

        cmd = [gui_host, "--plugin", plugin_path]
        if preset_path:
            cmd += ["--preset", preset_path, "--save", preset_path]
        try:
            logger.info("Launching VST GUI host: %s", " ".join(cmd))
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            self._gui_processes.append(proc)
            QtCore.QTimer.singleShot(500, lambda: self._check_gui_process(proc))
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, self._t("vst_batch_title"), str(exc))

    def _check_gui_process(self, proc: subprocess.Popen) -> None:
        if proc.poll() is None:
            return
        try:
            stdout, stderr = proc.communicate(timeout=0)
        except Exception:
            stdout, stderr = "", ""
        if proc.returncode not in (0, None):
            details = "\n".join(line for line in [stderr.strip(), stdout.strip()] if line)
            message = self._t("vst_gui_failed")
            if details:
                message = f"{message}\n\n{details}"
            QtWidgets.QMessageBox.critical(self, self._t("vst_batch_title"), message)

    def _store_host_paths(self) -> None:
        self.settings.setValue("vst_host_cli", self.cli_cmd())
        self.settings.setValue("vst_host_gui", self.gui_cmd())
        self.settings.setValue("vst_workers", self.workers_spin.value())

    def _selected_row(self) -> Optional[int]:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return None
        return rows[0].row()

    def _swap_rows(self, row_a: int, row_b: int) -> None:
        for col in range(self.table.columnCount()):
            item_a = self.table.takeItem(row_a, col)
            item_b = self.table.takeItem(row_b, col)
            self.table.setItem(row_a, col, item_b)
            self.table.setItem(row_b, col, item_a)

    def _set_chain(self, chain: list[dict]) -> None:
        self.table.setRowCount(0)
        for slot in chain:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(slot.get("path", "")))
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(slot.get("preset", "")))
            bypass_item = QtWidgets.QTableWidgetItem()
            bypass_item.setFlags(bypass_item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            bypass_item.setCheckState(QtCore.Qt.CheckState.Checked if slot.get("bypass") else QtCore.Qt.CheckState.Unchecked)
            self.table.setItem(row, 2, bypass_item)

    def chain(self) -> list[dict]:
        chain: list[dict] = []
        for row in range(self.table.rowCount()):
            path_item = self.table.item(row, 0)
            preset_item = self.table.item(row, 1)
            bypass_item = self.table.item(row, 2)
            path = path_item.text().strip() if path_item else ""
            if not path:
                continue
            chain.append({
                "path": path,
                "preset": preset_item.text().strip() if preset_item else "",
                "bypass": bypass_item.checkState() == QtCore.Qt.CheckState.Checked if bypass_item else False,
            })
        return chain

    def cli_cmd(self) -> str:
        return self.cli_edit.text().strip()

    def gui_cmd(self) -> str:
        return self.gui_edit.text().strip()

    def worker_count(self) -> int:
        return int(self.workers_spin.value())


class VstBatchDialog(QtWidgets.QDialog):
    def __init__(
        self,
        t: Callable[[str], str],
        settings: QtCore.QSettings,
        session: Session,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._t = t
        self.settings = settings
        self.session = session
        self.setWindowTitle(self._t("vst_batch_title"))
        self.resize(980, 640)

        store_path = Path.cwd() / "storage" / "vst_chains.json"
        self.store = VstPresetStore(store_path)

        layout = QtWidgets.QVBoxLayout(self)
        self.stack = QtWidgets.QStackedWidget()
        self.files_page = VstFileSelectionPage(self._t, self.session)
        self.mixer_page = VstMixerPage(self._t, self.settings, self.store)
        self.stack.addWidget(self.files_page)
        self.stack.addWidget(self.mixer_page)
        layout.addWidget(self.stack, 1)

        self.progress_label = QtWidgets.QLabel("")
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.progress_bar)

        btn_row = QtWidgets.QHBoxLayout()
        self.back_btn = QtWidgets.QPushButton(self._t("vst_back"))
        self.next_btn = QtWidgets.QPushButton(self._t("vst_next"))
        self.cancel_btn = QtWidgets.QPushButton(self._t("vst_cancel"))
        btn_row.addWidget(self.back_btn)
        btn_row.addWidget(self.next_btn)
        btn_row.addWidget(self.cancel_btn)
        layout.addLayout(btn_row)

        self.back_btn.clicked.connect(self._back)
        self.next_btn.clicked.connect(self._next_or_process)
        self.cancel_btn.clicked.connect(self.reject)
        self._update_buttons()

        self.worker: Optional[VstBatchWorker] = None

    def _back(self) -> None:
        idx = self.stack.currentIndex()
        if idx > 0:
            self.stack.setCurrentIndex(idx - 1)
        self._update_buttons()

    def _next_or_process(self) -> None:
        idx = self.stack.currentIndex()
        if idx == 0:
            files = self.files_page.selected_files()
            if not files:
                QtWidgets.QMessageBox.warning(self, self._t("vst_batch_title"), self._t("vst_no_files"))
                return
            self.stack.setCurrentIndex(1)
            self._update_buttons()
            return

        self._start_processing()

    def _update_buttons(self) -> None:
        idx = self.stack.currentIndex()
        self.back_btn.setEnabled(idx > 0)
        self.next_btn.setText(self._t("vst_process") if idx == 1 else self._t("vst_next"))

    def _start_processing(self) -> None:
        files = self.files_page.selected_files()
        chain = self.mixer_page.chain()
        host_cmd = self.mixer_page.cli_cmd()
        if not files:
            QtWidgets.QMessageBox.warning(self, self._t("vst_batch_title"), self._t("vst_no_files"))
            return
        if not chain:
            QtWidgets.QMessageBox.warning(self, self._t("vst_batch_title"), self._t("vst_no_chain"))
            return
        if not host_cmd:
            QtWidgets.QMessageBox.warning(self, self._t("vst_batch_title"), self._t("vst_no_host"))
            return

        self.settings.setValue("vst_host_cli", host_cmd)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.progress_label.setText(self._t("vst_backup_note"))
        self._set_controls_enabled(False)

        self.worker = VstBatchWorker(files, chain, host_cmd, self.mixer_page.worker_count(), self)
        self.worker.progress.connect(self._on_progress)
        self.worker.status.connect(self._on_status)
        self.worker.error.connect(self._on_error)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.start()

    def _set_controls_enabled(self, enabled: bool) -> None:
        self.back_btn.setEnabled(enabled and self.stack.currentIndex() > 0)
        self.next_btn.setEnabled(enabled)
        self.cancel_btn.setEnabled(enabled)
        self.files_page.setEnabled(enabled)
        self.mixer_page.setEnabled(enabled)

    def _on_progress(self, current: int, total: int, path: str) -> None:
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.progress_label.setText(f"{current}/{total} - {path}")

    def _on_status(self, message: str) -> None:
        self.progress_label.setText(message)

    def _on_error(self, message: str) -> None:
        self._set_controls_enabled(True)
        self.progress_bar.setVisible(False)
        QtWidgets.QMessageBox.critical(self, self._t("vst_batch_title"), message)

    def _on_finished(self) -> None:
        self._set_controls_enabled(True)
        self.progress_bar.setVisible(False)
        QtWidgets.QMessageBox.information(self, self._t("vst_batch_title"), self._t("vst_done"))
