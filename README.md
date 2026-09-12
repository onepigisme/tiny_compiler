# tiny_ml_compiler

轻量级机器学习模型编译器：将 Keras H5 模型编译为可在微控制器（MCU）上运行的纯 C 推理代码。基于 [keras2c](https://github.com/f0uriest/keras2c) 的 `k2c_tensor` 运行时实现，以 MNIST 手写数字识别 CNN 为 demo 模型。

## 特性

- **多格式模型支持**：训练并导出 H5 / ONNX / TFLite 三种格式的 MNIST 模型
- **H5 → C 编译**：解析 Keras H5 模型，按层提取权重与超参数，生成 `k2c_tensor` 格式的 C 数组
- **单函数推理**：生成单一 `forward()` 函数，采用**双工作张量乒乓交换**（`ta`/`tb`），按层奇偶性选择读写，内存占用最小化
- **Dropout 零拷贝传递**：推理阶段 Dropout 无操作，通过交换 `k2c_tensor` 结构体（指针+元数据）传递数据
- **same padding 支持**：`k2c_conv2d` 不接收 padding，自动插入 `k2c_pad2d` 预处理
- **PyQt5 图形界面**：可视化选择模型文件并触发转换
- **完整 keras2c 运行时**：内置 `networkop/` 下的 C 推理算子（Conv2D、Dense、MaxPooling、激活函数等）

## 编译流程

```
Keras H5 模型
      │
      ▼
 plugin/op_parse.py          加载模型，遍历各层
      │
      ▼
 op_registry.py              按算子类型分发到对应 handler
      │
      ├─ h5_conv2d.py        卷积核/偏置 → weight.h/c, 超参 → model.h/c
      ├─ h5_dense.py         权重矩阵/偏置 → weight.h/c, 激活 → model.h/c
      ├─ h5_maxPooling2d.py  pool_size/strides → model.h/c
      └─ h5_dropout.py       推理无操作
      │
      ▼
 weight_codegen.build_forward()  统一生成 forward() → model_invoke.c
      │
      ▼
 networkop/ (keras2c 运行时)  C 编译器 → MCU 可执行推理程序
```

## 目录结构

```
tiny_ml_compiler/
├── src/tiny_ml_compiler/
│   ├── plugin/
│   │   ├── op_parse.py        # 入口：加载 H5 模型，分层分发，生成 C 代码
│   │   └── op_registry.py     # 算子类型 → handler 注册表
│   ├── h5_operator/
│   │   ├── weight_codegen.py  # 核心：权重/超参/forward() 的 C 代码生成
│   │   ├── h5_conv2d.py       # Conv2D 层处理器
│   │   ├── h5_dense.py        # Dense 层处理器
│   │   ├── h5_maxPooling2d.py # MaxPooling2D 层处理器
│   │   └── h5_dropout.py      # Dropout 层处理器（推理无操作）
│   ├── networkop/             # keras2c C 推理运行时
│   │   ├── k2c_include.h      #   k2c_tensor 结构体与算子声明
│   │   ├── k2c_convolution_layers.c  # 卷积 + 填充
│   │   ├── k2c_core_layers.c  #   Dense / Flatten
│   │   ├── k2c_pooling_layers.c      # MaxPooling / AvgPooling
│   │   ├── k2c_activations.c  #   激活函数
│   │   └── ...
│   ├── model/
│   │   ├── h5_model.py        # 训练 MNIST CNN 并保存为 H5
│   │   ├── tflite_model.py    # H5 → TFLite 转换（含 int8 量化）
│   │   └── onnx_model.py      # ONNX 导出
│   ├── dataset/
│   │   └── load_minist.py     # MNIST 数据集加载
│   ├── ui/
│   │   └── main_ui.py         # PyQt5 转换界面
│   ├── ir/  frontend/  optimizer/  backend/   # 通用编译器骨架（待接入）
│   └── cli.py                 # 命令行入口
├── model/
│   ├── mnist_cnn.h5           # 训练好的 Keras 模型（测试集 99.04%）
│   ├── mnist_cnn.tflite       # float32 TFLite
│   ├── mnist_cnn_quant.tflite # 动态范围量化 TFLite
│   ├── mnist_cnn_int8.tflite  # int8 全整数量化 TFLite
│   └── weights/               # 编译生成的 C 文件（weight.* 可重新生成）
│       ├── model.h / model.c        # 各层超参数
│       └── model_invoke.c           # 统一 forward() 推理函数
├── tests/                  # pytest 测试
└── pyproject.toml
```

## 快速开始

### 环境准备

```bash
conda create -n tiny_ml python=3.10
conda activate tiny_ml
pip install -e ".[dev]"
# UI 功能需要 PyQt5：pip install PyQt5
```

### 1. 训练模型

```bash
python -m tiny_ml_compiler.model.h5_model
```

输出 `model/mnist_cnn.h5`（CNN: Conv2D→Pool→Conv2D→Pool→Flatten→Dense→Dropout→Dense）。

### 2. 编译 H5 → C

```bash
python -m tiny_ml_compiler.plugin.op_parse
```

在 `model/weights/` 下生成：
- `weight.h` / `weight.c` — 各层权重与偏置的 `k2c_tensor`
- `model.h` / `model.c` — 各层超参数（stride、padding、activation 等）
- `model_invoke.c` — 单一 `forward()` 推理函数

> `weight.c` / `weight.h` 体积较大（约 9.5MB），不纳入 git，每次编译重新生成。

### 3. 图形界面

```bash
python -m tiny_ml_compiler.ui.main_ui
```

支持选择 `.onnx` / `.h5` / `.tflite` 模型文件并触发转换。

### 4. 运行测试

```bash
pytest -v
```

## 生成的 forward() 示例

```c
static float buf_a[MAX_ACT], buf_b[MAX_ACT], buf_pad[MAX_PAD];
static k2c_tensor ta = {buf_a,...}, tb = {buf_b,...}, pad_work = {buf_pad,...};

int forward(const k2c_tensor* input, k2c_tensor** output) {
    ta.array = buf_a; tb.array = buf_b;          // 可重复调用

    k2c_pad2d(&pad_work, input, 0.0f, conv2d_padding);
    k2c_conv2d(&ta, &pad_work, &conv2d_kernel_tensor, &conv2d_bias_tensor,
               conv2d_stride, conv2d_dilation, conv2d_activation);  // 写 ta
    k2c_maxpool2d(&tb, &ta, ...);                 // 读 ta, 写 tb
    k2c_pad2d(...); k2c_conv2d(&ta, &tb, ...);   // 读 tb, 写 ta
    k2c_maxpool2d(&tb, &ta, ...);                 // 读 ta, 写 tb
    k2c_flatten(&ta, &tb);                        // 读 tb, 写 ta
    k2c_dense(&tb, &ta, ...);                     // 读 ta, 写 tb
    k2c_tensor_swap(&ta, &tb);                    // Dropout: 零拷贝交换
    k2c_dense(&tb, &ta, ...);                     // 读 ta, 写 tb

    *output = &tb;
    return 0;
}
```

## 支持的算子

| Keras 层 | 生成内容 | 运行时函数 |
|---|---|---|
| Conv2D | kernel/bias k2c_tensor + stride/padding/dilation/activation | `k2c_pad2d` + `k2c_conv2d` |
| MaxPooling2D | pool_size + strides | `k2c_maxpool2d` |
| Dense | kernel/bias k2c_tensor + activation | `k2c_dense` |
| Flatten | — | `k2c_flatten` |
| Dropout | —（推理无操作） | 结构体交换 |

## 开发环境

- Python >= 3.9（推荐 3.10）
- conda 环境名：`tiny_ml`
- C 编译器：gcc / MSVC / MinGW（用于编译生成的推理代码）
