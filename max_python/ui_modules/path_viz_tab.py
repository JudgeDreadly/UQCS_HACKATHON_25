# ui_modules/path_viz_tab.py
from typing import Optional, Callable, List, Tuple
from PyQt5 import QtWidgets, QtGui, QtCore
import numpy as np

from modules.moves_solver import (
    GridSpec, plan_moves_expression, grid_from_capture, PLANNERS
)
from modules.solver_engine import Board  # for typing

CELL_SZ = 10  # pixels per grid cell in the viz

class GridCanvas(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.occ: Optional[np.ndarray] = None    # (rows, cols) uint8
        self.path_cells: List[Tuple[int,int]] = []
        self.open_set: set = set()
        self.closed_set: set = set()
        self.setMinimumSize(600, 400)

    def paintEvent(self, e: QtGui.QPaintEvent):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(30,30,30))
        if self.occ is None: 
            p.end(); return
        rows, cols = self.occ.shape
        w = cols * CELL_SZ
        h = rows * CELL_SZ
        # center
        offx = max(0, (self.width()-w)//2)
        offy = max(0, (self.height()-h)//2)

        # draw grid
        for r in range(rows):
            for c in range(cols):
                x = offx + c*CELL_SZ
                y = offy + r*CELL_SZ
                rect = QtCore.QRect(x, y, CELL_SZ-1, CELL_SZ-1)
                if self.occ[r, c] == 1:
                    p.fillRect(rect, QtGui.QColor(60,60,60))
                else:
                    p.fillRect(rect, QtGui.QColor(240,240,240))
        # open/closed
        pen = QtGui.QPen(QtGui.QColor(0,120,255)); pen.setWidth(1); p.setPen(pen)
        for (c,r) in self.open_set:
            x = offx + c*CELL_SZ; y = offy + r*CELL_SZ
            p.fillRect(QtCore.QRect(x, y, CELL_SZ-1, CELL_SZ-1), QtGui.QColor(140,200,255,120))
        pen = QtGui.QPen(QtGui.QColor(255,180,0)); pen.setWidth(1); p.setPen(pen)
        for (c,r) in self.closed_set:
            x = offx + c*CELL_SZ; y = offy + r*CELL_SZ
            p.fillRect(QtCore.QRect(x, y, CELL_SZ-1, CELL_SZ-1), QtGui.QColor(255,210,90,120))
        # path
        pen = QtGui.QPen(QtGui.QColor(255,0,80)); pen.setWidth(2); p.setPen(pen)
        for i in range(1, len(self.path_cells)):
            c1,r1 = self.path_cells[i-1]; c2,r2 = self.path_cells[i]
            x1 = offx + c1*CELL_SZ + CELL_SZ//2
            y1 = offy + r1*CELL_SZ + CELL_SZ//2
            x2 = offx + c2*CELL_SZ + CELL_SZ//2
            y2 = offy + r2*CELL_SZ + CELL_SZ//2
            p.drawLine(x1,y1,x2,y2)
        p.end()

class PathVizTab(QtWidgets.QWidget):
    def __init__(self, get_frame_func: Callable[[], Optional[np.ndarray]], parent=None):
        super().__init__(parent)
        self.get_frame = get_frame_func
        self.gs = GridSpec(dx_units=0.5, dy_units=0.5, max_cols=160, max_rows=120)
        self.occ: Optional[np.ndarray] = None
        self.board: Optional[Board] = None
        self.start_xy: Optional[Tuple[float,float]] = None
        self.goal_xy: Optional[Tuple[float,float]] = None
        self.timer = QtCore.QTimer(self); self.timer.setInterval(50)
        self.timer.timeout.connect(self._tick)

        layout = QtWidgets.QVBoxLayout(self)

        # Controls row
        row = QtWidgets.QHBoxLayout()
        self.algo = QtWidgets.QComboBox(); self.algo.addItems(list(PLANNERS.keys()))
        self.rand_btn = QtWidgets.QPushButton("Random grid")
        self.from_cap_btn = QtWidgets.QPushButton("Use latest capture")
        self.play_btn = QtWidgets.QPushButton("Run")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.export_btn = QtWidgets.QPushButton("Export expression")

        row.addWidget(QtWidgets.QLabel("Algorithm:"))
        row.addWidget(self.algo, 0)
        row.addSpacing(12)
        row.addWidget(self.rand_btn, 0)
        row.addWidget(self.from_cap_btn, 0)
        row.addStretch(1)
        row.addWidget(self.play_btn, 0)
        row.addWidget(self.stop_btn, 0)
        row.addSpacing(12)
        row.addWidget(self.export_btn, 0)
        layout.addLayout(row)

        self.canvas = GridCanvas()
        layout.addWidget(self.canvas, 1)

        # status
        self.status = QtWidgets.QLabel("")
        self.status.setStyleSheet("color:#666")
        layout.addWidget(self.status)

        # signals
        self.rand_btn.clicked.connect(self._make_random)
        self.from_cap_btn.clicked.connect(self._from_capture)
        self.play_btn.clicked.connect(self._run_once)
        self.stop_btn.clicked.connect(self._stop)
        self.export_btn.clicked.connect(self._export_expr)

    def _make_random(self):
        rows, cols = 60, 100
        occ = np.zeros((rows, cols), dtype=np.uint8)
        # add some random blocks
        rng = np.random.default_rng(42)
        for _ in range(600):
            r = rng.integers(0, rows)
            c = rng.integers(5, cols)  # keep left edge emptier
            occ[r, c] = 1
        self.occ = occ; self.board=None
        # start ~left-middle, goal ~right-middle
        self.start_xy = (-25.0, 0.0); self.goal_xy = (25.0, 0.0)
        self.canvas.occ = occ; self.canvas.path_cells = []; self.canvas.update()
        self.status.setText("Random grid ready. Choose an algorithm and Run.")

    def _from_capture(self):
        rgb = self.get_frame()
        if rgb is None:
            QtWidgets.QMessageBox.information(self, "No frame", "Start capture first, then try again.")
            return
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        try:
            occ, board, start_xy, goal_xy = grid_from_capture(bgr, self.gs)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Extract error", str(e))
            return
        self.occ = occ; self.board = board
        self.start_xy = start_xy; self.goal_xy = goal_xy
        self.canvas.occ = occ; self.canvas.path_cells = []; self.canvas.update()
        self.status.setText("Grid built from latest capture. Choose an algorithm and Run.")

    def _run_once(self):
        if self.occ is None or self.start_xy is None or self.goal_xy is None:
            QtWidgets.QMessageBox.information(self, "No grid", "Create a grid first.")
            return
        # fake “animation”: we compute path, then reveal it cell-by-cell
        algo = self.algo.currentText()
        # synthesize a Board if missing (random mode)
        if self.board is None:
            class Dummy: pass
            self.board = Dummy()  # minimal Board shim
            self.board.x_range = (-25.0, 25.0)
            self.board.y_range = (-15.0, 15.0)
        res = plan_moves_expression(self.occ, self.board, self.start_xy, self.goal_xy, self.gs, algo=algo)
        self._path_cells_full = res.cells
        self._moves = res.moves
        self._expr = res.expr
        self._reveal_idx = 1
        self.canvas.path_cells = [self._path_cells_full[0]]
        self.timer.start()
        self.status.setText(f"{algo}: {len(self._moves)} moves")

    def _tick(self):
        if not hasattr(self, "_path_cells_full"): 
            self.timer.stop(); return
        if self._reveal_idx >= len(self._path_cells_full):
            self.timer.stop()
            self.canvas.update()
            return
        self.canvas.path_cells = self._path_cells_full[: self._reveal_idx+1]
        self._reveal_idx += 1
        self.canvas.update()

    def _stop(self):
        self.timer.stop()

    def _export_expr(self):
        if not hasattr(self, "_expr"):
            QtWidgets.QMessageBox.information(self, "Nothing to export", "Run an algorithm first.")
            return
        # copy to clipboard and show a toast
        cb = QtWidgets.QApplication.clipboard()
        cb.setText(self._expr)
        QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), "Expression copied!", self)
