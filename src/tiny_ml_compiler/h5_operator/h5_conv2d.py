"""Conv2D 层算子处理器。

从解析后的层信息中提取卷积核、偏置和超参数，
按 keras2c 的 k2c_tensor 格式把权重/偏置写入 weight.h / weight.c。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .weight_codegen import write_k2c_tensor, write_conv2d_hyperparams

# 统一输出到项目根/model/weights/
OUTPUT_DIR = Path(__file__).resolve().parents[3] / "model" / "weights"
WEIGHT_H = OUTPUT_DIR / "weight.h"
WEIGHT_C = OUTPUT_DIR / "weight.c"
MODEL_H = OUTPUT_DIR / "model.h"
MODEL_C = OUTPUT_DIR / "model.c"


def h5_conv2d(info: dict) -> None:
    """把 Conv2D 层的卷积核与偏置以 k2c_tensor 格式写入 weight.h / weight.c。"""
    layer_config = info["config"]
    weights = info["weights"]            # get_weights() 的结果

    # ---- 权重与偏置（来自 get_weights，而非 get_config）----
    kernel_tensor = weights[0] if len(weights) > 0 else None  # (H, W, in_c, out_c)
    bias_tensor = weights[1] if len(weights) > 1 else None    # (out_c,)

    if kernel_tensor is None:
        raise ValueError(f"Conv2D 层 {info['name']} 缺少卷积核权重")

    # 打印形状便于调试
    print(f"  Conv2D kernel shape: {np.asarray(kernel_tensor).shape}, "
          f"bias shape: {None if bias_tensor is None else np.asarray(bias_tensor).shape}")

    # ---- 写入 weight.h / weight.c（k2c_tensor 格式）----
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    layer_name = info["name"]                    # 如 conv2d / conv2d_1

    write_k2c_tensor(kernel_tensor, f"{layer_name}_kernel", WEIGHT_H, WEIGHT_C)
    if bias_tensor is not None:
        write_k2c_tensor(bias_tensor, f"{layer_name}_bias", WEIGHT_H, WEIGHT_C)

    # ---- 超参数（来自 get_config）----
    kernel_size = tuple(layer_config["kernel_size"])  # 如 (3, 3)
    strides = tuple(layer_config["strides"])          # 如 (1, 1)
    dilation_rate = tuple(layer_config["dilation_rate"])  # 如 (1, 1)
    padding = layer_config["padding"]                 # "same" / "valid"
    activation = layer_config.get("activation", "linear")

    # 将卷积核的 stride、padding、dilation_rate 超参数写入 model.h 与 model.c
    write_conv2d_hyperparams(
        layer_name, strides, dilation_rate, padding, activation, kernel_size,
        MODEL_H, MODEL_C,
    )
