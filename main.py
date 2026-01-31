from __future__ import annotations

import logging
import sys
from pathlib import Path

from PyQt6 import QtWidgets, QtGui

from app.main_window import MainWindow


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def main() -> int:
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("UtauRecorder")
        except Exception:
            logging.exception("Failed to set Windows AppUserModelID")
    app = QtWidgets.QApplication(sys.argv)
    icon_path = Path(__file__).parent / "icon" / "icon.ico"
    if icon_path.exists():
        icon = QtGui.QIcon(str(icon_path))
        app.setWindowIcon(icon)
        QtGui.QGuiApplication.setWindowIcon(icon)
    window = MainWindow()
    if icon_path.exists():
        window.setWindowIcon(app.windowIcon())
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
