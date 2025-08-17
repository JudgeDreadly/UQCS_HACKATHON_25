from PyQt5 import QtWidgets, QtCore, QtGui

class ShotOutputTab(QtWidgets.QWidget):
    """
    Shows the final command text with one-click Copy.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)

        self.text = QtWidgets.QPlainTextEdit()
        self.text.setPlaceholderText("Your command/function output will appear here after processing.")
        layout.addWidget(self.text, 1)

        row = QtWidgets.QHBoxLayout()
        self.copy_btn = QtWidgets.QPushButton("Copy to Clipboard")
        self.copy_btn.clicked.connect(self.copy_all)
        row.addStretch(1)
        row.addWidget(self.copy_btn, 0)
        layout.addLayout(row)

    def set_text(self, s: str):
        self.text.setPlainText(s)

    def copy_all(self):
        self.text.selectAll()
        self.text.copy()
        QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), "Copied!", self)
