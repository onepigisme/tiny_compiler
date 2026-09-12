"""Dense 全连接层算子处理器。

把权重矩阵、偏置以 k2c_tensor 格式写入 weight.h / weight.c，
激活函数指针写入 model.h / model.c。推理调用由统一的 forward() 生成。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .weight_codegen import write_k2c_tensor, write_activation_hyperparam

OUTPUT_DIR = Path(__file__).resolve().parents[3] / "model" / "weights"
WEIGHT_H = OUTPUT_DIR / "weight.h"
WEIGHT_C = OUTPUT_DIR / "weight.c"
MODEL_H = OUTPUT_DIR / "model.h"
MODEL_C = OUTPUT_DIR / "model.c"


def h5_dense(info: dict) -> None:
    """把 Dense 层权重/偏置写入 weight 文件，激活函数写入 model 文件。"""
    layer_config = info["config"]
    weights = info["weights"]            # [kernel (in,out), bias (out,)]
    layer_name = info["name"]

    kernel_tensor = weights[0] if len(weights) > 0 else None
    bias_tensor = weights[1] if len(weights) > 1 else None
    if kernel_tensor is None:
        raise ValueError(f"Dense 层 {layer_name} 缺少权重矩阵")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"  Dense kernel shape: {np.asarray(kernel_tensor).shape}, "
          f"bias shape: {None if bias_tensor is None else np.asarray(bias_tensor).shape}")

    write_k2c_tensor(kernel_tensor, f"{layer_name}_kernel", WEIGHT_H, WEIGHT_C)
    if bias_tensor is not None:
        write_k2c_tensor(bias_tensor, f"{layer_name}_bias", WEIGHT_H, WEIGHT_C)

    activation = layer_config.get("activation", "linear")
    write_activation_hyperparam(layer_name, activation, MODEL_H, MODEL_C)
