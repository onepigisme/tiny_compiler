"""手写数字识别模型（MNIST）。

构建一个 CNN 模型，训练后保存为 H5 格式。
"""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 模型保存路径
MODEL_PATH = Path(__file__).resolve().parent.parent.parent.parent / "model" / "mnist_cnn.h5"

EPOCHS = 5
BATCH_SIZE = 128


def load_data():
    """加载 MNIST 数据集，返回预处理后的训练/测试数据。"""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # 归一化到 [0, 1]，并增加通道维度: (N, 28, 28) -> (N, 28, 28, 1)
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]
    return (x_train, y_train), (x_test, y_test)


def build_model() -> keras.Model:
    """构建 CNN 模型。"""
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train():
    """训练模型并保存为 H5。"""
    print("=== 加载 MNIST 数据 ===")
    (x_train, y_train), (x_test, y_test) = load_data()
    print(f"训练集: {x_train.shape}, 测试集: {x_test.shape}")

    print("\n=== 构建模型 ===")
    model = build_model()
    model.summary()

    print("\n=== 开始训练 ===")
    model.fit(
        x_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.1,
        verbose=1,
    )

    print("\n=== 评估 ===")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"测试集准确率: {test_acc:.4f}, 损失: {test_loss:.4f}")

    print("\n=== 保存模型 ===")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_PATH), save_format="h5")
    print(f"已保存到: {MODEL_PATH}")

    return model


if __name__ == "__main__":
    train()
