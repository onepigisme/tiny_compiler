"""加载 MNIST 手写数字数据集。

使用 tf.keras.datasets.mnist（TF 2.x API），
替代已废弃的 tensorflow.examples.tutorials.mnist。
"""

import tensorflow as tf
import numpy as np

# 加载 MNIST 数据集
# 首次运行会自动下载到 ~/.keras/datasets/，后续直接从本地读取
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

np.save("mnist_train_x.npy", x_train)
np.save("mnist_train_y.npy", y_train)
np.save("mnist_test_x.npy", x_test)
np.save("mnist_test_y.npy", y_test)



# 查看训练数据的大小
print(x_train.shape)  # (60000, 28, 28)
print(y_train.shape)  # (60000,)

# 查看测试数据的大小
print(x_test.shape)   # (10000, 28, 28)
print(y_test.shape)   # (10000,)

# 打印第 0 张训练图片
print(x_train[0])
print("label:", y_train[0])
