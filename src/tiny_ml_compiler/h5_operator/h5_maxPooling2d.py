"""MaxPooling2D 池化层算子处理器。

把 pool_size / strides 超参数写入 model.h / model.c。
无权重；推理调用由统一的 forward() 生成。
"""

from __future__ import annotations

from pathlib import Path

from .weight_codegen import write_pool_hyperparams

OUTPUT_DIR = Path(__file__).resolve().parents[3] / "model" / "weights"
MODEL_H = OUTPUT_DIR / "model.h"
MODEL_C = OUTPUT_DIR / "model.c"


def h5_maxPooling2d(info: dict) -> None:
    """把 MaxPooling2D 的 pool_size / strides 写入 model 文件。"""
    cfg = info["config"]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_pool_hyperparams(
        info["name"],
        tuple(cfg["pool_size"]),
        tuple(cfg["strides"]),
        cfg.get("padding", "valid"),
        MODEL_H,
        MODEL_C,
    )
