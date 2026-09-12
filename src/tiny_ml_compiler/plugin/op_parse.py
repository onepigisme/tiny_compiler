"""H5 模型算子解析插件。

加载 Keras H5 模型，遍历各层并提取结构化算子信息，
再通过 op_registry 分发给对应的算子处理器。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 兼容两种运行方式：
#   python -m tiny_ml_compiler.plugin.op_parse   （推荐）
#   python src/tiny_ml_compiler/plugin/op_parse.py（IDE 直接运行）
# 直接运行脚本时没有父包上下文，需把 src 加入 sys.path 后再做绝对包导入。
_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from tensorflow.keras.models import load_model

from tiny_ml_compiler.plugin.op_registry import register_h5_ops
from tiny_ml_compiler.h5_operator.weight_codegen import (
    reset_weight_files,
    reset_model_files,
    reset_invoke_file,
    build_forward,
)

# 用脚本位置锚定路径，避免相对路径依赖运行时工作目录
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = _PROJECT_ROOT / "model" / "mnist_cnn.h5"

# 生成的 C 文件统一放在 model/weights/
OUTPUT_DIR = _PROJECT_ROOT / "model" / "weights"
WEIGHT_H = OUTPUT_DIR / "weight.h"
WEIGHT_C = OUTPUT_DIR / "weight.c"
MODEL_H = OUTPUT_DIR / "model.h"
MODEL_C = OUTPUT_DIR / "model.c"
INVOKE_C = OUTPUT_DIR / "model_invoke.c"


def h5_parse(model_path: Path = MODEL_PATH) -> list[dict]:
    """解析 H5 模型的每一层，分发给已注册的算子处理器。

    返回算子信息列表，每个元素包含:
    index / op_type / name / output_shape / params / config / weights
    """
    if not Path(model_path).is_file():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    model = load_model(str(model_path))
    model.summary()  # 人类可读摘要，仅作参考输出

    ops_registry = register_h5_ops()  # 只构建一次
    ops = []

    is_odd_layer = True
    for i, layer in enumerate(model.layers):
        op_type = type(layer).__name__   # 如 Conv2D / MaxPooling2D / Dense
        # get_weights() 返回 numpy 数组列表:
        #   Conv2D / Dense: [kernel, bias]
        #   MaxPooling2D / Flatten / Dropout: [] (无参数)
        weights = layer.get_weights()
        info = {
            "index": i,
            "op_type": op_type,
            "name": layer.name,
            "output_shape": layer.output_shape,   # 如 (None, 28, 28, 32)
            "input_shape": layer.input_shape,     # 如 (None, 28, 28, 1)
            "params": layer.count_params(),       # 可训练参数量
            "config": layer.get_config(),         # 层配置（kernel_size/strides 等）
            "weights": weights,                   # 层的权重和偏置（numpy 数组列表）
            "is_odd_layer": is_odd_layer,         # 记录当前层的奇偶性
        }
        is_odd_layer = not is_odd_layer

        handler = ops_registry.get(op_type)
        if handler is not None:
           handler(info)
           status = "已分发"
        else:
           status = "跳过(未注册)"

        ops.append(info)
        print(f"[{i}] {op_type:<14} 输出形状: {info['output_shape']}  参数: {info['params']}  -> {status}")

    return ops


if __name__ == "__main__":
    # 作为脚本运行时：先清空上次生成的 C 文件，再重新解析生成
    reset_weight_files(WEIGHT_H, WEIGHT_C)
    reset_model_files(MODEL_H, MODEL_C)
    reset_invoke_file(INVOKE_C)

    parsed = h5_parse()
    print(f"\n共解析出 {len(parsed)} 个算子")

    # 所有层解析完后，统一生成单一 forward()（双工作张量乒乓）
    build_forward(parsed, MODEL_H, INVOKE_C)
