"""基础冒烟测试——验证包能 import、CLI 能调用。"""

import subprocess
import sys


def test_import():
    """验证 tiny_ml_compiler 包能正常 import。"""
    import tiny_ml_compiler
    assert tiny_ml_compiler.__version__ == "0.1.0"


def test_ir_imports():
    """验证 IR 核心类可导入。"""
    from tiny_ml_compiler import Graph, Node, TensorType
    assert Graph is not None
    assert Node is not None
    assert TensorType is not None


def test_cli_help():
    """验证 CLI --help 能正常退出。"""
    result = subprocess.run(
        [sys.executable, "-m", "tiny_ml_compiler", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "TinyML" in result.stdout or "tiny-ml-compiler" in result.stdout
