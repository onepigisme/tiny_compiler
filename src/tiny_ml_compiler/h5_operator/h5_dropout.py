"""Dropout 层算子处理器。

Dropout 仅在训练时随机失活，推理阶段为恒等映射（无操作），
因此不写任何权重/超参数文件；forward() 中通过交换两个工作张量传递数据。
"""

from __future__ import annotations


def h5_dropout(info: dict) -> None:
    """推理时 Dropout 无操作，无需生成任何 C 常量。"""
    print(f"  Dropout 层 {info['name']}: 推理无操作（forward 中仅交换工作张量）")
