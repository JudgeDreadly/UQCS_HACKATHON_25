# ui_modules/process_tab.py
from typing import Callable, Optional
from PyQt5 import QtWidgets, QtCore, QtGui
import numpy as np
import cv2

from modules.solver_engine import solve_from_bgr  # unified entrypoint

class ProcessTab(QtWidgets.QWidget):
    """
    Button-triggered processing (no background runs).
    Shows two previews:
      - Overlay (on ROI)
      - Critical grid (no background image; obstacles + tolerance + path + actors)
    """
    def __init__(self, get_frame_func: Callable[[], Optional[np.ndarray]],
                 set_output_func: Callable[[str], None],
                 get_save_dir_func: Callable[[], str],
                 parent=None):
        super().__init__(parent)
        self.get_frame = get_frame_func
        self.set_output = set_output_func
        self.get_save_dir = get_save_dir_func

        root = QtWidgets.QVBoxLayout(self)

        # Controls row 1
        row1 = QtWidgets.QHBoxLayout()
        self.process_btn = QtWidgets.QPushButton("Process Latest Frame")
        self.process_btn.clicked.connect(self.run_processing)
        row1.addWidget(self.process_btn)

        row1.addSpacing(12)
        self.xmin = QtWidgets.QDoubleSpinBox(); self.xmin.setRange(-1000,1000); self.xmin.setDecimals(2); self.xmin.setValue(-25.0)
        self.xmax = QtWidgets.QDoubleSpinBox(); self.xmax.setRange(-1000,1000); self.xmax.setDecimals(2); self.xmax.setValue(25.0)
        self.ymin = QtWidgets.QDoubleSpinBox(); self.ymin.setRange(-1000,1000); self.ymin.setDecimals(2); self.ymin.setValue(-15.0)
        self.ymax = QtWidgets.QDoubleSpinBox(); self.ymax.setRange(-1000,1000); self.ymax.setDecimals(2); self.ymax.setValue(15.0)
        self.min_area = QtWidgets.QSpinBox(); self.min_area.setRange(1, 10000); self.min_area.setValue(60)

        row1.addWidget(QtWidgets.QLabel("X:"))
        row1.addWidget(self.xmin); row1.addWidget(self.xmax)
        row1.addSpacing(8)
        row1.addWidget(QtWidgets.QLabel("Y:"))
        row1.addWidget(self.ymin); row1.addWidget(self.ymax)
        row1.addSpacing(8)
        row1.addWidget(QtWidgets.QLabel("Min area:"))
        row1.addWidget(self.min_area)
        row1.addStretch(1)
        root.addLayout(row1)

        # Controls row 2 (solver options)
        row2 = QtWidgets.QHBoxLayout()
        self.solver_mode = QtWidgets.QComboBox(); self.solver_mode.addItems(["moves", "continuous"])
        self.moves_algo = QtWidgets.QComboBox(); self.moves_algo.addItems(["A*", "Dijkstra", "Greedy", "BFS"])
        self.grid_dx = QtWidgets.QDoubleSpinBox(); self.grid_dx.setRange(0.05, 2.0); self.grid_dx.setSingleStep(0.05); self.grid_dx.setValue(0.25)
        self.grid_dy = QtWidgets.QDoubleSpinBox(); self.grid_dy.setRange(0.05, 2.0); self.grid_dy.setSingleStep(0.05); self.grid_dy.setValue(0.25)
        self.safety_px = QtWidgets.QSpinBox(); self.safety_px.setRange(0, 60); self.safety_px.setValue(8)
        self.flex_hearts = QtWidgets.QCheckBox("Flex hearts (continuous)"); self.flex_hearts.setChecked(False)
        self.solver_mode.currentTextChanged.connect(self._on_solver_change)

        row2.addWidget(QtWidgets.QLabel("Solver:"))
        row2.addWidget(self.solver_mode)
        row2.addSpacing(8)
        row2.addWidget(QtWidgets.QLabel("Algo:"))
        row2.addWidget(self.moves_algo)
        row2.addSpacing(12)
        row2.addWidget(QtWidgets.QLabel("Grid dx/dy:"))
        row2.addWidget(self.grid_dx); row2.addWidget(self.grid_dy)
        row2.addSpacing(12)
        row2.addWidget(QtWidgets.QLabel("Tolerance px:"))
        row2.addWidget(self.safety_px)
        row2.addStretch(1)
        row2.addWidget(self.flex_hearts)
        root.addLayout(row2)

        # Previews: Overlay + Critical Grid (bigger and auto-expanding)
        panels = QtWidgets.QHBoxLayout()
        self.overlay_label = QtWidgets.QLabel("(overlay)")
        self.overlay_label.setAlignment(QtCore.Qt.AlignCenter)
        self.overlay_label.setMinimumHeight(420)
        self.overlay_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        panels.addWidget(self.overlay_label, 1)

        self.crit_label = QtWidgets.QLabel("(critical grid)")
        self.crit_label.setAlignment(QtCore.Qt.AlignCenter)
        self.crit_label.setMinimumHeight(420)
        self.crit_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        panels.addWidget(self.crit_label, 1)

        root.addLayout(panels, 1)

        self.status = QtWidgets.QLabel("")
        self.status.setStyleSheet("color:#666")
        root.addWidget(self.status)

        # initial state
        self._on_solver_change(self.solver_mode.currentText())

    def _on_solver_change(self, mode: str):
        is_moves = (mode.lower() == "moves")
        self.moves_algo.setEnabled(is_moves)
        self.grid_dx.setEnabled(is_moves)
        self.grid_dy.setEnabled(is_moves)
        self.safety_px.setEnabled(is_moves)
        self.flex_hearts.setEnabled(not is_moves)

    def _show_np_rgb(self, img: Optional[np.ndarray], label: QtWidgets.QLabel):
        if img is None:
            label.setText("(none)"); return
        # Ensure RGB ndarray
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        h, w = img.shape[:2]
        pw, ph = max(200, label.width()), max(200, label.height())
        scale = min(pw / w, ph / h)
        tw, th = max(1, int(w * scale)), max(1, int(h * scale))
        qimg = QtGui.QImage(img.data, w, h, w * 3, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(tw, th, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        label.setPixmap(pix)

    def run_processing(self):
        rgb = self.get_frame()
        if rgb is None:
            QtWidgets.QMessageBox.information(self, "No frame", "No frame available. Start capture first.")
            return
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        save_dir = self.get_save_dir() or "."
        overlay_path = f"{save_dir}/graphwar_overlay.png"

        mode = self.solver_mode.currentText().lower()
        kwargs = dict(
            x_range=(float(self.xmin.value()), float(self.xmax.value())),
            y_range=(float(self.ymin.value()), float(self.ymax.value())),
            min_area=int(self.min_area.value()),
            overlay_path=overlay_path,
        )
        if mode == "moves":
            kwargs.update(
                solver_mode="moves",
                moves_algo=self.moves_algo.currentText(),
                grid_dx=float(self.grid_dx.value()),
                grid_dy=float(self.grid_dy.value()),
                moves_safety_px=int(self.safety_px.value()),
                flex=False,
            )
        else:
            kwargs.update(solver_mode="continuous", flex=bool(self.flex_hearts.isChecked()))

        try:
            result = solve_from_bgr(bgr, **kwargs)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Solver error", str(e))
            return

        expr = result.get("expr", "0")
        self.set_output(expr)

        overlay_bgr = result.get("overlay_bgr")
        if overlay_bgr is not None:
            overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
            self._show_np_rgb(overlay_rgb, self.overlay_label)
        else:
            self.overlay_label.setText("(no overlay)")

        crit_bgr = result.get("crit_overlay_bgr")
        if crit_bgr is not None:
            crit_rgb = cv2.cvtColor(crit_bgr, cv2.COLOR_BGR2RGB)
            self._show_np_rgb(crit_rgb, self.crit_label)
        else:
            self.crit_label.setText("(no critical grid)")

        self.status.setText(f"{mode.title()} solved. Expression length: {len(expr)} | Overlay saved: {overlay_path}")
