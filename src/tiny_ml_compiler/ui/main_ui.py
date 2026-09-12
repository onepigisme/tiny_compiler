"""TinyML Compiler 主界面。

提供模型文件选择和编译选项的图形化操作。
"""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

# 支持的模型格式
MODEL_FILTER = "模型文件 (*.onnx *.h5 *.tflite);;ONNX (*.onnx);;Keras/HDF5 (*.h5);;TFLite (*.tflite)"


class MainWindow(QMainWindow):
    """TinyML Compiler 主窗口。"""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("TinyML Compiler")
        self.setMinimumWidth(520)
        self._build_ui()

    # ---- UI 构建 ----

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        # 1. 模型路径输入行
        layout.addWidget(QLabel("模型文件路径:"))
        path_row = QHBoxLayout()
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("选择 .onnx / .h5 / .tflite 文件...")
        self.path_input.setReadOnly(False)
        browse_btn = QPushButton("浏览...")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self._on_browse)
        path_row.addWidget(self.path_input)
        path_row.addWidget(browse_btn)
        layout.addLayout(path_row)

        # 2. 加速选项勾选框
        self.accel_checkbox = QCheckBox("部署加速版本")
        layout.addWidget(self.accel_checkbox)

        # 3. 转换按钮
        self.convert_btn = QPushButton("模型转换")
        self.convert_btn.setFixedHeight(36)
        self.convert_btn.clicked.connect(self._on_convert)
        layout.addWidget(self.convert_btn)

        layout.addStretch()

    # ---- 事件处理 ----

    def _on_browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "", MODEL_FILTER
        )
        if path:
            self.path_input.setText(path)

    def _on_convert(self) -> None:
        path = self.path_input.text().strip()
        if not path:
            QMessageBox.warning(self, "提示", "请先选择模型文件。")
            return
        if not Path(path).is_file():
            QMessageBox.warning(self, "提示", f"文件不存在:\n{path}")
            return

        suffix = Path(path).suffix.lower()
        if suffix not in (".onnx", ".h5", ".tflite"):
            QMessageBox.warning(self, "提示", f"不支持的格式: {suffix}")
            return

        accel = self.accel_checkbox.isChecked()
        # TODO: 调用编译器后端执行实际转换
        QMessageBox.information(
            self, "转换",
            f"模型: {path}\n加速: {'是' if accel else '否'}\n\n"
            "(编译器后端待接入)",
        )


def main() -> int:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
