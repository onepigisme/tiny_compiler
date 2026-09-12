"""手写数字识别模型（MNIST）—— TFLite 转换。

加载训练好的 H5 模型，转换为 TFLite 格式。
支持三种转换模式：
  - 默认（float32）：精度无损，体积略减
  - 动态范围量化：体积约减 4 倍
  - int8 全整数量化：体积约减 4 倍，适合 MCU 部署
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

# 路径配置
MODEL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "model"
H5_PATH = MODEL_DIR / "mnist_cnn.h5"
TFLITE_PATH = MODEL_DIR / "mnist_cnn.tflite"
TFLITE_QUANT_PATH = MODEL_DIR / "mnist_cnn_quant.tflite"
TFLITE_INT8_PATH = MODEL_DIR / "mnist_cnn_int8.tflite"


def convert_default() -> Path:
    """默认转换：float32 TFLite。"""
    model = tf.keras.models.load_model(str(H5_PATH))
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    TFLITE_PATH.write_bytes(tflite_model)
    print(f"已保存: {TFLITE_PATH} ({len(tflite_model) / 1024:.1f} KB)")
    return TFLITE_PATH


def convert_dynamic_range() -> Path:
    """动态范围量化：权重转 int8，体积约减 4 倍。"""
    model = tf.keras.models.load_model(str(H5_PATH))
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    TFLITE_QUANT_PATH.write_bytes(tflite_model)
    print(f"已保存: {TFLITE_QUANT_PATH} ({len(tflite_model) / 1024:.1f} KB)")
    return TFLITE_QUANT_PATH


def convert_int8(representative_data) -> Path:
    """int8 全整数量化：输入输出也量化，适合 MCU（TinyML）部署。"""
    model = tf.keras.models.load_model(str(H5_PATH))
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    TFLITE_INT8_PATH.write_bytes(tflite_model)
    print(f"已保存: {TFLITE_INT8_PATH} ({len(tflite_model) / 1024:.1f} KB)")
    return TFLITE_INT8_PATH


def representative_dataset():
    """代表性数据集：用于 int8 量化校准（取 100 张 MNIST 图片）。"""
    (x_train, _), _ = tf.keras.datasets.mnist.load_data()
    x = x_train[:100].astype("float32") / 255.0
    x = x[..., tf.newaxis]
    for img in x:
        yield [img[tf.newaxis, ...]]


def evaluate_tflite(tflite_path: Path) -> float:
    """用 TFLite 解释器在测试集上评估准确率。"""
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    inp_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]

    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.astype("float32") / 255.0
    x_test = x_test[..., tf.newaxis]

    is_int8 = inp_detail["dtype"] == np.int8
    scale, zero_point = inp_detail["quantization"]

    correct = 0
    n = len(x_test)
    for i in range(n):
        img = x_test[i]
        if is_int8:
            img = (img / scale + zero_point).round().clip(-128, 127).astype("int8")
        interpreter.set_tensor(inp_detail["index"], img[tf.newaxis, ...].astype(inp_detail["dtype"]))
        interpreter.invoke()
        output = interpreter.get_tensor(out_detail["index"])
        pred = output.argmax(axis=-1)[0]
        if pred == y_test[i]:
            correct += 1
    return correct / n


def main():
    if not H5_PATH.exists():
        print(f"未找到 H5 模型: {H5_PATH}")
        print("请先运行: python -m tiny_ml_compiler.model.h5_model")
        return

    print("=== 1/3 默认转换 (float32) ===")
    convert_default()

    print("\n=== 2/3 动态范围量化 ===")
    convert_dynamic_range()

    print("\n=== 3/3 int8 全整数量化 ===")
    convert_int8(representative_dataset)

    print("\n=== 体积对比 ===")
    h5_size = H5_PATH.stat().st_size / 1024
    print(f"H5 原始模型:        {h5_size:8.1f} KB")
    for p in (TFLITE_PATH, TFLITE_QUANT_PATH, TFLITE_INT8_PATH):
        if p.exists():
            print(f"{p.name}: {p.stat().st_size / 1024:8.1f} KB")


if __name__ == "__main__":
    main()
