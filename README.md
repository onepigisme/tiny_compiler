# tiny_ml_compiler

轻量级机器学习模型编译器：将 Keras H5 模型编译为可在微控制器（MCU）上运行的纯 C 推理代码。基于 [keras2c](https://github.com/f0uriest/keras2c) 的 `k2c_tensor` 运行时实现，以 MNIST 手写数字识别 CNN 为 demo 模型。

## 特性

- **多格式模型支持**：训练并导出 H5 / ONNX / TFLite 三种格式的 MNIST 模型
- **H5 → C 编译**：解析 Keras H5 模型，按层提取权重与超参数，生成 `k2c_tensor` 格式的 C 数组
- **单函数推理**：生成单一 `forward()` 函数，采用**双工作张量乒乓交换**（`ta`/`tb`），按层奇偶性选择读写，内存占用最小化
- **Dropout 零拷贝传递**：推理阶段 Dropout 无操作，通过交换 `k2c_tensor` 结构体（指针+元数据）传递数据
- **same padding 支持**：`k2c_conv2d` 不接收 padding，自动插入 `k2c_pad2d` 预处理
- **TFLite → Relax → C 编译**：以 TVM Relax 为前端解析 TFLite，跑图优化 Pass（常量折叠 / 死代码消除 / DPL 算子融合），做 Kahn 拓扑调度 + 生存期分析 + offset 级单池 arena 内存规划，导出可编译可运行的 C 推理代码，并与 TFLite golden 数值对拍
- **PyQt5 图形界面**：可视化选择模型文件并触发转换
- **C++17 推理框架参考实现**：以 NCNN/MNN 同款的工厂自注册模式实现算子注册表，含虚函数算子体系、双缓冲乒乓 Graph 与完整算子计算，可直接编译运行
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

## TFLite → Relax → C 编译流程（TVM 前端）

以 TVM Relax 为图 IR 的第二条编译管线（`tvm` conda 环境，TVM ≥ 0.27）：

```
TFLite 模型 (float32)
      │
      ▼
 tools/dump_relax_ir.py       前端解析 from_tflite → Relax IRModule
      │                       图优化 Pass：FoldConstant + DeadCodeElimination
      │                       + FuseOpsByPattern（conv2d/matmul + bias + relu 融合）
      │                       拓扑提取 → Kahn 执行序 → 生存期分析
      ▼
 tools/codegen_tflite_c.py    offset 级单池 arena 内存规划（区间 first-fit + 8B 对齐）
      │                       权重提取 → weights.h/c；算子映射 → model.h/c + main.c
      │                       （pad workspace 作为伪变量一并纳入 arena 规划）
      ▼
 model/<stem>_c/              生成的 C 推理工程 + networkop 运行时拷贝
      │
      ▼
 zig cc（或任意 C 编译器）     forward.exe：读 input.bin → forward() → 写 output.bin
      │
      ▼
 tools/gen_golden.py          tf.lite.Interpreter 生成 golden，与 C 输出对拍
```

```bash
# —— 第一步：导出 C 代码（只需要 tvm 环境，不需要任何 C 编译器）——
conda activate tvm
python tools/codegen_tflite_c.py model/mnist_cnn.tflite --plan-only   # 只看 arena 布局
python tools/codegen_tflite_c.py model/mnist_cnn.tflite               # 生成 model/*_c/ 下的 .c/.h

# —— 第二步（可选）：PC 上数值验证，任意 C 编译器均可，不属于交付产物 ——
pip install ziglang                                    # 本机无 gcc/cl 时用 zig cc
python -m ziglang cc -O2 -Imodel/mnist_cnn_c/networkop \
    model/mnist_cnn_c/*.c model/mnist_cnn_c/networkop/*.c \
    -o model/mnist_cnn_c/forward.exe -lm
conda activate tiny_ml
python tools/gen_golden.py model/mnist_cnn.tflite      # 生成 input.bin/golden.npy
model\mnist_cnn_c\forward.exe model\mnist_cnn_c\input.bin model\mnist_cnn_c\output.bin
python tools/gen_golden.py model/mnist_cnn.tflite      # 重跑完成对拍（PASS, max|diff|~1e-7）
```

**MCU 集成**：生成的 C 是纯 C99、无 malloc、无 OS 依赖的静态 arena 工程。把
`model.c`、`weights.c` 和 `networkop/` 下所需 `k2c_*.c` 加入目标芯片的工具链工程
（Cortex-M 用 arm-none-eabi-gcc / Keil / IAR；TC3xx 用 AURIX Studio / TASKING），
include 路径指向 `networkop/` 即可。不编译 PC 验证用的 `main.c`，固件侧自行实现入口：

```c
#include "model.h"
static float in[IN_SIZE];    /* 784 */
static float out[OUT_SIZE];  /* 10  */
forward(in, out);            /* 仅一个调用，结果在 out[] */
```

关键实现约定（详见 `tools/codegen_tflite_c.py` 文件头）：
- k2c 图像张量是 **HWC 三维**（无 batch 维），dense 是 `(batch, features)` 二维；
- Relax 前端已把 TFLite 的 OHWI 卷积权重 permute 成 **HWIO** 并被常量折叠，导出时直接使用；
- `k2c_pad2d` 的 pad 顺序是 `{top, bottom, left, right}`，与 Relax 的 `(top, left, bottom, right)` 需要重排；
- MNIST float 模型 arena 仅 **125440 B**（31360 floats），所有中间张量按生存期错位共享单池。

当前限制：仅支持 float32 图；int8 全量化模型的 QDQ 算子需要先做量化 legalization pass 才能映射到 k2c。

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
├── cpp_demo/
│   └── op_registry/           # C++17 推理框架参考实现（可独立编译）
│       ├── include/tml/       #   tensor / layer / op_registry / graph
│       ├── src/layers.cpp     #   5 个算子实现 + 静态自注册
│       ├── main.cpp           #   可运行 demo（组网络→前向推理）
│       └── CMakeLists.txt
├── model/
│   ├── mnist_cnn.h5           # 训练好的 Keras 模型（测试集 99.04%）
│   ├── mnist_cnn.tflite       # float32 TFLite
│   ├── mnist_cnn_quant.tflite # 动态范围量化 TFLite
│   ├── mnist_cnn_int8.tflite  # int8 全整数量化 TFLite
│   ├── mnist_cnn_c/           # TFLite→Relax→C 管线的导出产物（纯 C 源码，可重新生成）
│   │   ├── weights.h / weights.c     # 权重 k2c_tensor（只读段）
│   │   ├── model.h / model.c         # arena + forward()（offset 级内存规划）
│   │   ├── main.c                    # PC 验证入口（MCU 集成时不编译）
│   │   └── networkop/                # keras2c 纯 C99 运行时拷贝
│   └── weights/               # H5 编译生成的 C 文件（weight.* 可重新生成）
│       ├── model.h / model.c        # 各层超参数
│       └── model_invoke.c           # 统一 forward() 推理函数
├── tools/                  # TVM Relax 实验管线（tvm conda 环境）
│   ├── dump_relax_ir.py    #   TFLite→Relax 前端 / 图优化 Pass / 拓扑 / 调度
│   ├── codegen_tflite_c.py #   arena 内存规划 + k2c 算子映射 → C 代码导出
│   └── gen_golden.py       #   TFLite golden 生成与 C 输出对拍
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

## C++17 推理框架参考实现

`cpp_demo/op_registry/` 用现代 C++ 重写了编译器的算子注册与推理执行核心，演示工业界推理框架（NCNN / MNN / TFLite）的标准设计模式：

- **工厂 + 静态自注册**：`TML_REGISTER_LAYER("Conv2D", Conv2DLayer)` 一行完成注册，新增算子无需修改注册表（开闭原则）
- **虚函数算子体系**：`Layer` 抽象基类 + Conv2D / MaxPooling2D / Dense / Flatten / Dropout 子类
- **强类型层描述**：以 `LayerInfo` 结构体替代 Python 的 dict，编译期检查字段
- **双缓冲乒乓 Graph**：与生成的 C `forward()` 同构，两块工作缓冲按层交替读写，Dropout 零开销跳过
- **完整算子语义**：same/valid padding、空洞卷积、softmax 数值稳定、NHWC 布局

```bash
cd cpp_demo/op_registry
cmake -B build && cmake --build build --config Release
./build/tml_demo          # Windows: .\build\Release\tml_demo.exe
# 或：g++ -std=c++17 -Iinclude main.cpp src/layers.cpp -o tml_demo
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
