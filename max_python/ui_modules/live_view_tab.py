from typing import Optional
from PyQt5 import QtWidgets, QtGui, QtCore
import numpy as np

from modules.capture_backends import (
    list_windows, WinInfo, CaptureController
)

class LiveViewTab(QtWidgets.QWidget):
    """
    Live capture: choose window, choose backend (DXCAM default), preview.
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        self.cap = CaptureController()

        layout = QtWidgets.QVBoxLayout(self)

        # Controls row
        row = QtWidgets.QHBoxLayout()
        self.backend = QtWidgets.QComboBox()
        self.backend.addItems(["DXCAM", "WGC"])
        self.backend.setCurrentText("DXCAM")
        self.backend.currentTextChanged.connect(self.on_backend_changed)

        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.populate_windows)

        self.window_combo = QtWidgets.QComboBox()
        self.show_all = QtWidgets.QCheckBox("Show all (not just Java)")
        self.show_all.setChecked(False)
        self.show_all.stateChanged.connect(lambda _: self.populate_windows())

        self.start_btn = QtWidgets.QPushButton("Start")
        self.start_btn.clicked.connect(self.start_capture)
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_capture)

        row.addWidget(QtWidgets.QLabel("Backend:"))
        row.addWidget(self.backend, 0)
        row.addSpacing(8)
        row.addWidget(self.refresh_btn, 0)
        row.addWidget(self.window_combo, 1)
        row.addWidget(self.show_all, 0)
        row.addSpacing(8)
        row.addWidget(self.start_btn, 0)
        row.addWidget(self.stop_btn, 0)
        layout.addLayout(row)

        # Preview
        self.preview = QtWidgets.QLabel("(preview)")
        self.preview.setAlignment(QtCore.Qt.AlignCenter)
        self.preview.setMinimumHeight(360)
        layout.addWidget(self.preview, 1)

        self._wins = []
        self._last_frame = None

        # UI timer (~30 FPS)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)

        self.populate_windows(initial=True)

    def on_backend_changed(self, name: str):
        self.cap.set_backend(name or "DXCAM")

    def populate_windows(self, initial: bool = False):
        self.window_combo.blockSignals(True)
        self.window_combo.clear()
        wins = list_windows()
        if not self.show_all.isChecked():
            wins = [w for w in wins if "java" in (w.exe or "")]
        # prefer exact "Graphwar"
        wins.sort(key=lambda w: (w.title != "Graphwar", w.exe != "javaw.exe", w.title.lower()))
        self._wins = wins
        for w in wins:
            self.window_combo.addItem(f"{w.title}  [{w.exe}]  (HWND {w.hwnd})", userData=w)
        # auto-pick Graphwar
        if wins:
            idx = 0
            for i, w in enumerate(wins):
                if w.title == "Graphwar":
                    idx = i; break
            self.window_combo.setCurrentIndex(idx)
        self.window_combo.blockSignals(False)

    def _selected_win(self) -> Optional[WinInfo]:
        return self.window_combo.currentData()

    def start_capture(self):
        w = self._selected_win()
        if w is None:
            QtWidgets.QMessageBox.information(self, "Select window", "Pick a window first.")
            return
        try:
            self.cap.start(w)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Capture error", str(e))

    def stop_capture(self):
        self.cap.stop()

    def _tick(self):
        frm = self.cap.latest()
        if frm is None:
            self.preview.setText("(no frame yet — if WGC, ensure window is not minimized)")
            return
        h, w, _ = frm.shape
        pw, ph = max(200, self.preview.width()), max(200, self.preview.height())
        scale = min(pw / w, ph / h)
        tw, th = max(1, int(w * scale)), max(1, int(h * scale))
        qimg = QtGui.QImage(frm.data, w, h, w * 3, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(tw, th, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.preview.setPixmap(pix)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        return self.cap.latest()

    def shutdown(self):
        self.cap.stop()
