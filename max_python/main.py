# main.py
import sys
from PyQt5 import QtWidgets, QtCore, QtGui

from ui_modules.live_view_tab import LiveViewTab
from ui_modules.process_tab import ProcessTab
from ui_modules.shot_output_tab import ShotOutputTab


class GraphwarBotUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Graphwar Bot — Capture + Solver")
        self.resize(1280, 860)

        # Tabs
        tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(tabs)

        # Create tabs
        self.shot_output = ShotOutputTab()
        self.live_view = LiveViewTab()  # pull-based: exposes get_latest_frame()

        self.process_tab = ProcessTab(
            get_frame_func=self.live_view.get_latest_frame,
            set_output_func=self.shot_output.set_text,   # ShotOutputTab API
            get_save_dir_func=self._get_default_save_dir
        )

        tabs.addTab(self.live_view, "Live View")
        tabs.addTab(self.process_tab, "Process")
        tabs.addTab(self.shot_output, "Shot Output")

    def _get_default_save_dir(self) -> str:
        return "."

    def closeEvent(self, e):
        try:
            self.live_view.shutdown()
        except Exception:
            pass
        super().closeEvent(e)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = GraphwarBotUI()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
