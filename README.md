# tiny_ml_compiler

轻量级机器学习模型编译器，将 ONNX 等格式的 ML 模型编译为可在微控制器上运行的 C 代码。

## 架构

```
前端解析 (Frontend)  →  中间表示 (IR)  →  优化器 (Optimizer)  →  后端代码生成 (Backend)
    OnnxParser          Graph              PassManager            CBackend
                        Node               ConstFoldPass
                        TensorType         DeadNodeElimPass
                                           FuseReluConvPass
```

## 目录结构

```
tiny_ml_compiler/
├── src/tiny_ml_compiler/
│   ├── __init__.py          # 包入口
│   ├── __main__.py          # python -m tiny_ml_compiler
│   ├── cli.py               # 命令行入口
│   ├── ir/                  # 中间表示层
│   │   ├── graph.py         #   计算图
│   │   ├── node.py          #   算子节点
│   │   └── tensor.py        #   张量类型
│   ├── frontend/            # 前端解析层
│   │   ├── base.py          #   解析器基类
│   │   └── onnx_parser.py   #   ONNX 解析器
│   ├── optimizer/           # 优化器层
│   │   ├── base.py          #   Pass 基类 + PassManager
│   │   └── passes.py        #   常量折叠 / 死节点消除 / 算子融合
│   └── backend/             # 后端代码生成层
│       ├── base.py          #   后端基类
│       └── c_backend.py     #   C 代码生成
├── tests/
│   ├── test_basic.py        # 冒烟测试
│   └── test_ir.py           # IR 单元测试
└── pyproject.toml
```

## 快速开始

```bash
# 安装（开发模式）
pip install -e ".[dev]"

# 编译模型
tiny-ml-compiler compile model.onnx -o output.c --optimize

# 查看模型信息
tiny-ml-compiler info model.onnx

# 跑测试
pytest -v
```

## 开发环境

- Python >= 3.9
- conda 环境名: `tiny_ml` (Python 3.10)
