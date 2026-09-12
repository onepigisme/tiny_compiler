from tiny_ml_compiler.h5_operator.h5_dense import h5_dense
from tiny_ml_compiler.h5_operator.h5_conv2d import h5_conv2d
from tiny_ml_compiler.h5_operator.h5_maxPooling2d import h5_maxPooling2d
from tiny_ml_compiler.h5_operator.h5_dropout import h5_dropout


# 注册h5算子
# key 必须与 Keras 层的类名 (type(layer).__name__) 完全一致
def register_h5_ops():
    return {
        "Conv2D": h5_conv2d,
        "MaxPooling2D": h5_maxPooling2d,
        "Flatten": None,            # 无需算子转换，运行时跳过
        "Dense": h5_dense,
        "Dropout": h5_dropout,
    }
