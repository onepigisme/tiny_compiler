"""将 numpy 权重数组以 keras2c 的 k2c_tensor 格式写入 C 文件。

每个权重生成两部分：
  - 展平（行优先）的 const float[] 数值数组
  - 一个 k2c_tensor 结构体，包装该数组（含 ndim / numel / shape）

约定（单一真源，避免声明/定义不兼容）：
  - weight.h：#pragma once + 包含 k2c_tensor_include.h，所有 extern 声明
  - weight.c：#include "weight.h"，所有数组与结构体的实际定义
  两者的类型、维度、名字严格一致。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

K2C_MAX_NDIM = 5

# weight.h / weight.c 中的起始标记，用于判断是否需要写文件头
_H_HEADER = '#pragma once\n#include "k2c_tensor_include.h"\n\n'
_C_HEADER = '#include "weight.h"\n\n'


def _format_float(v: float) -> str:
    """单个 float 的 C 字面量，带 f 后缀。"""
    s = repr(float(v))
    return s + "f"


def _flatten_values(arr: np.ndarray) -> str:
    """把多维数组按 C 序（行优先）展平，用逗号分隔。"""
    flat = arr.reshape(-1)
    return ", ".join(_format_float(v) for v in flat)


def _shape_c_array(shape: tuple) -> str:
    """把 (3, 3, 1, 32) 转成 C 初始化列表，不足 K2C_MAX_NDIM 补 0。"""
    dims = list(shape)
    while len(dims) < K2C_MAX_NDIM:
        dims.append(0)
    return "{" + ", ".join(str(d) for d in dims) + "}"


def write_k2c_tensor(
    arr: np.ndarray,
    var_name: str,
    header_path: str | Path,
    source_path: str | Path,
) -> None:
    """把一个权重数组以 k2c_tensor 格式追加写入 weight.h / weight.c。

    生成内容（以 var_name="conv2d_0_kernel", shape=(3,3,1,32) 为例）：

    weight.h:
        extern const float conv2d_0_kernel[];
        extern k2c_tensor conv2d_0_kernel_tensor;

    weight.c:
        const float conv2d_0_kernel[288] = { 0.1f, -0.2f, ... };
        k2c_tensor conv2d_0_kernel_tensor = {
            (float*)conv2d_0_kernel, 4, 288, {3, 3, 1, 32, 0}
        };

    多次调用会追加到同一对文件中；首次调用会写入文件头（include guard / #include）。
    """
    arr = np.asarray(arr)
    header_path = Path(header_path)
    source_path = Path(source_path)

    numel = int(arr.size)
    ndim = int(arr.ndim)
    shape_list = list(arr.shape)
    values = _flatten_values(arr)
    shape_c = _shape_c_array(shape_list)

    # ---- weight.h：首次写文件头，随后追加 extern 声明 ----
    h_exists = header_path.exists() and header_path.stat().st_size > 0
    with header_path.open("a", encoding="utf-8") as f:
        if not h_exists:
            f.write(_H_HEADER)
        f.write(f"extern const float {var_name}[];\n")
        f.write(f"extern k2c_tensor {var_name}_tensor;\n")

    # ---- weight.c：首次写 #include，随后追加定义 ----
    c_exists = source_path.exists() and source_path.stat().st_size > 0
    with source_path.open("a", encoding="utf-8") as f:
        if not c_exists:
            f.write(_C_HEADER)
        f.write(f"const float {var_name}[{numel}] = {{\n    {values}\n}};\n\n")
        f.write(
            f"k2c_tensor {var_name}_tensor = {{\n"
            f"    (float*){var_name},\n"
            f"    {ndim},\n"
            f"    {numel},\n"
            f"    {shape_c}\n"
            f"}};\n\n"
        )

    print(f"  已写入 k2c_tensor: {var_name}  "
          f"(shape={arr.shape}, numel={numel}, dtype={arr.dtype})")


def reset_weight_files(header_path: str | Path, source_path: str | Path) -> None:
    """清空 weight.h / weight.c，便于重新生成。"""
    for p in (Path(header_path), Path(source_path)):
        if p.exists():
            p.unlink()


# ============================================================================
# 超参数（stride / dilation / padding / activation）写入 model.h / model.c
# ============================================================================

# model.h 需要包含 k2c_include.h 以获得 k2c_activationType 与 k2c_* 函数指针
_MODEL_H_HEADER = '#pragma once\n#include <stddef.h>\n#include "k2c_include.h"\n\n'
_MODEL_C_HEADER = '#include "model.h"\n\n'

# Keras 激活函数名 -> keras2c 函数指针
_ACTIVATION_MAP = {
    "relu": "k2c_relu",
    "linear": "k2c_linear",
    "sigmoid": "k2c_sigmoid",
    "softmax": "k2c_softmax",
    "tanh": "k2c_tanh",
    "softplus": "k2c_softplus",
    "softsign": "k2c_softsign",
    "exponential": "k2c_exponential",
    "hard_sigmoid": "k2c_hard_sigmoid",
}


def _compute_same_padding(kernel_size: tuple, strides: tuple) -> tuple:
    """计算 "same" 填充量，返回 (top, bottom, left, right)。

    Keras "same" 填充在 stride=1 时：pad_total = k - 1，均分到两边。
    """
    kH, kW = kernel_size
    sH, sW = strides
    # 通用 same 填充公式（以 output = ceil(input/stride) 推导）
    # 当 stride=1 时退化为 pad = k-1，均分给 top/bottom
    pad_h = kH - 1
    pad_w = kW - 1
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    return (top, bottom, left, right)


def write_conv2d_hyperparams(
    layer_name: str,
    stride: tuple,
    dilation_rate: tuple,
    padding: str,
    activation: str,
    kernel_size: tuple,
    header_path: str | Path,
    source_path: str | Path,
) -> None:
    """把 Conv2D 层的超参数写入 model.h（声明）与 model.c（定义）。

    - stride / dilation: const size_t[2]（k2c_conv2d 接口要求）
    - padding: const size_t[4] = {top, bottom, left, right}
               "valid" -> 全 0；"same" -> 按 kernel_size 计算
    - activation: k2c_activationType * 函数指针（如 &k2c_relu）

    生成内容（以 layer_name="conv2d", kernel=(3,3), padding="same" 为例）：
    model.h:
        extern const size_t conv2d_stride[2];
        extern const size_t conv2d_dilation[2];
        extern const size_t conv2d_padding[4];
        extern k2c_activationType * conv2d_activation;

    model.c:
        const size_t conv2d_stride[2] = {1, 1};
        const size_t conv2d_dilation[2] = {1, 1};
        const size_t conv2d_padding[4] = {1, 1, 1, 1};
        k2c_activationType * conv2d_activation = &k2c_relu;
    """
    header_path = Path(header_path)
    source_path = Path(source_path)

    stride_s = ", ".join(str(int(s)) for s in stride)
    dilation_s = ", ".join(str(int(d)) for d in dilation_rate)

    # padding 转成 size_t[4] 数组
    if padding == "same":
        pad = _compute_same_padding(kernel_size, stride)
    else:  # valid
        pad = (0, 0, 0, 0)
    padding_s = ", ".join(str(int(p)) for p in pad)

    # activation 映射到 k2c 函数指针
    k2c_act = _ACTIVATION_MAP.get(activation, "k2c_linear")

    # ---- model.h：首次写文件头，随后追加 extern 声明 ----
    h_exists = header_path.exists() and header_path.stat().st_size > 0
    with header_path.open("a", encoding="utf-8") as f:
        if not h_exists:
            f.write(_MODEL_H_HEADER)
        f.write(f"extern const size_t {layer_name}_stride[2];\n")
        f.write(f"extern const size_t {layer_name}_dilation[2];\n")
        f.write(f"extern const size_t {layer_name}_padding[4];\n")
        f.write(f"extern k2c_activationType * {layer_name}_activation;\n")

    # ---- model.c：首次写 #include，随后追加定义 ----
    c_exists = source_path.exists() and source_path.stat().st_size > 0
    with source_path.open("a", encoding="utf-8") as f:
        if not c_exists:
            f.write(_MODEL_C_HEADER)
        f.write(f"const size_t {layer_name}_stride[2] = {{{stride_s}}};\n")
        f.write(f"const size_t {layer_name}_dilation[2] = {{{dilation_s}}};\n")
        f.write(f"const size_t {layer_name}_padding[4] = {{{padding_s}}};\n")
        f.write(f"k2c_activationType * {layer_name}_activation = &{k2c_act};\n\n")

    print(f"  已写入超参数: {layer_name}  "
          f"(stride={stride}, dilation={dilation_rate}, "
          f"padding={padding}{pad}, activation={activation}->{k2c_act})")


def reset_model_files(header_path: str | Path, source_path: str | Path) -> None:
    """清空 model.h / model.c，便于重新生成。"""
    for p in (Path(header_path), Path(source_path)):
        if p.exists():
            p.unlink()


def write_activation_hyperparam(
    layer_name: str,
    activation: str,
    header_path: str | Path,
    source_path: str | Path,
) -> None:
    """只写入 activation 函数指针（Dense 等层使用，无 stride/padding）。"""
    header_path = Path(header_path)
    source_path = Path(source_path)
    k2c_act = _ACTIVATION_MAP.get(activation, "k2c_linear")

    h_exists = header_path.exists() and header_path.stat().st_size > 0
    with header_path.open("a", encoding="utf-8") as f:
        if not h_exists:
            f.write(_MODEL_H_HEADER)
        f.write(f"extern k2c_activationType * {layer_name}_activation;\n")

    c_exists = source_path.exists() and source_path.stat().st_size > 0
    with source_path.open("a", encoding="utf-8") as f:
        if not c_exists:
            f.write(_MODEL_C_HEADER)
        f.write(f"k2c_activationType * {layer_name}_activation = &{k2c_act};\n\n")

    print(f"  已写入激活函数: {layer_name}  (activation={activation}->{k2c_act})")


def write_pool_hyperparams(
    layer_name: str,
    pool_size: tuple,
    strides: tuple,
    padding: str,
    header_path: str | Path,
    source_path: str | Path,
) -> None:
    """写入 MaxPooling2D 的 pool_size / strides（size_t[2]）。

    注：k2c_maxpool2d 不接收 padding，仅支持 Keras padding=valid，
    因此不生成 padding 数组。
    """
    if padding != "valid":
        raise NotImplementedError(f"{layer_name}: k2c_maxpool2d 仅支持 padding=valid")

    header_path = Path(header_path)
    source_path = Path(source_path)
    pool_s = ", ".join(str(int(p)) for p in pool_size)
    stride_s = ", ".join(str(int(s)) for s in strides)

    h_exists = header_path.exists() and header_path.stat().st_size > 0
    with header_path.open("a", encoding="utf-8") as f:
        if not h_exists:
            f.write(_MODEL_H_HEADER)
        f.write(f"extern const size_t {layer_name}_pool_size[2];\n")
        f.write(f"extern const size_t {layer_name}_strides[2];\n")

    c_exists = source_path.exists() and source_path.stat().st_size > 0
    with source_path.open("a", encoding="utf-8") as f:
        if not c_exists:
            f.write(_MODEL_C_HEADER)
        f.write(f"const size_t {layer_name}_pool_size[2] = {{{pool_s}}};\n")
        f.write(f"const size_t {layer_name}_strides[2] = {{{stride_s}}};\n\n")

    print(f"  已写入池化超参数: {layer_name}  "
          f"(pool_size={pool_size}, strides={strides})")


# ============================================================================
# 推理代码（model_invoke.c）——单一 forward() + 双工作张量乒乓交换
# ============================================================================

# 计算型算子（会把结果写入新的工作张量）；Dropout 推理时无操作，仅交换结构体
_COMPUTE_OPS = {"Conv2D", "MaxPooling2D", "Flatten", "Dense"}


def _strip_batch(shape) -> tuple:
    """把含 batch 维的形状 (None, H, W, C) 规范化成 (H, W, C)。"""
    return tuple(int(d) for d in shape if d is not None)


def _prod(shape: tuple) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


def _set_tensor_meta(tensor: str, shape: tuple) -> list[str]:
    """生成设置工作张量 ndim/numel/shape 的 C 语句（计算前必须设置，
    因为 k2c_conv2d 等函数内部读取 output->shape）。"""
    ndim = len(shape)
    numel = _prod(shape)
    dims = list(shape) + [0] * (K2C_MAX_NDIM - ndim)
    lines = [
        f"{tensor}.ndim = {ndim}; {tensor}.numel = {numel};",
        f"{tensor}.shape[0]={dims[0]}; {tensor}.shape[1]={dims[1]}; "
        f"{tensor}.shape[2]={dims[2]}; {tensor}.shape[3]={dims[3]}; "
        f"{tensor}.shape[4]={dims[4]};",
    ]
    return lines


def _read_write(info: dict, first: bool) -> tuple[str, str]:
    """按 is_odd_layer 决定本层读/写哪个工作张量。

    奇数层: 写 ta；偶数层: 写 tb。第一层输入来自 forward 参数 input。
    返回 (read_expr, write_name)。
    """
    odd = bool(info["is_odd_layer"])
    write_name = "ta" if odd else "tb"
    if first:
        read_expr = "input"
    else:
        read_expr = "&tb" if odd else "&ta"
    return read_expr, write_name


def _snippet_conv2d(info: dict, read_expr: str, write: str) -> tuple[list[str], int]:
    """生成 Conv2D 调用片段，返回 (C 语句列表, padded_numel)。"""
    name = info["name"]
    out_shape = _strip_batch(info["output_shape"])
    in_shape = _strip_batch(info["input_shape"])
    cfg = info["config"]
    padding = cfg["padding"]
    kernel_size = tuple(cfg["kernel_size"])
    strides = tuple(cfg["strides"])

    lines = [f"/* [{info['index']}] {name}: Conv2D (padding={padding}) */"]
    lines += _set_tensor_meta(write, out_shape)

    padded_numel = 0
    if padding == "same":
        # k2c_conv2d 不接收 padding，先 k2c_pad2d 到复用的 pad_work 工作张量
        top, bottom, left, right = _compute_same_padding(kernel_size, strides)
        padded_shape = (in_shape[0] + top + bottom,
                        in_shape[1] + left + right, in_shape[2])
        padded_numel = _prod(padded_shape)
        lines += _set_tensor_meta("pad_work", padded_shape)
        lines.append(
            f"k2c_pad2d(&pad_work, {read_expr}, 0.0f, {name}_padding);"
        )
        conv_input = "&pad_work"
    else:
        conv_input = read_expr

    has_bias = len(info["weights"]) > 1
    bias_arg = f"&{name}_bias_tensor" if has_bias else "NULL"
    lines.append(
        f"k2c_conv2d(&{write}, {conv_input},\n"
        f"           &{name}_kernel_tensor, {bias_arg},\n"
        f"           {name}_stride, {name}_dilation, {name}_activation);"
    )
    return lines, padded_numel


def _snippet_maxpooling2d(info: dict, read_expr: str, write: str) -> tuple[list[str], int]:
    """生成 MaxPooling2D 调用片段。"""
    name = info["name"]
    out_shape = _strip_batch(info["output_shape"])
    cfg = info["config"]
    if cfg.get("padding", "valid") != "valid":
        raise NotImplementedError(
            f"MaxPooling2D 暂只支持 padding=valid，{name} 为 {cfg['padding']}")

    lines = [f"/* [{info['index']}] {name}: MaxPooling2D */"]
    lines += _set_tensor_meta(write, out_shape)
    lines.append(
        f"k2c_maxpool2d(&{write}, {read_expr}, "
        f"{name}_pool_size, {name}_strides);"
    )
    return lines, 0


def _snippet_flatten(info: dict, read_expr: str, write: str) -> tuple[list[str], int]:
    """生成 Flatten 调用片段（k2c_flatten 内部 memcpy 并设置 output 元数据）。"""
    lines = [f"/* [{info['index']}] {info['name']}: Flatten */"]
    lines.append(f"k2c_flatten(&{write}, {read_expr});")
    return lines, 0


def _snippet_dense(info: dict, read_expr: str, write: str) -> tuple[list[str], int]:
    """生成 Dense 调用片段；单样本 1D 输入时 fwork 传 NULL。"""
    name = info["name"]
    out_shape = _strip_batch(info["output_shape"])
    lines = [f"/* [{info['index']}] {name}: Dense */"]
    lines += _set_tensor_meta(write, out_shape)
    has_bias = len(info["weights"]) > 1
    bias_arg = f"&{name}_bias_tensor" if has_bias else "NULL"
    lines.append(
        f"k2c_dense(&{write}, {read_expr},\n"
        f"          &{name}_kernel_tensor, {bias_arg}, "
        f"{name}_activation, NULL);"
    )
    return lines, 0


_SNIPPETS = {
    "Conv2D": _snippet_conv2d,
    "MaxPooling2D": _snippet_maxpooling2d,
    "Flatten": _snippet_flatten,
    "Dense": _snippet_dense,
}


def build_forward(ops: list[dict], model_h_path: str | Path,
                  invoke_c_path: str | Path) -> None:
    """根据解析出的所有层，生成单一 forward() 到 model_invoke.c。

    两个静态工作张量 ta/tb 复用同尺寸内存（按网络最大激活分配），
    按 is_odd_layer 乒乓选择读写；Dropout 推理无操作，直接交换两个
    k2c_tensor 结构体（array 指针 + 元数据），零拷贝传递数据。
    """
    model_h_path = Path(model_h_path)
    invoke_c_path = Path(invoke_c_path)

    # ---- 第一遍：计算最大激活尺寸 / 最大 pad 尺寸，并确定最终输出句柄 ----
    max_act = 1
    max_pad = 1
    last_writer = "ta"
    seen_compute = False
    for info in ops:
        op = info["op_type"]
        if op in _COMPUTE_OPS:
            numel = _prod(_strip_batch(info["output_shape"]))
            max_act = max(max_act, numel)
        if op == "Conv2D" and info["config"]["padding"] == "same":
            in_shape = _strip_batch(info["input_shape"])
            cfg = info["config"]
            top, bottom, left, right = _compute_same_padding(
                tuple(cfg["kernel_size"]), tuple(cfg["strides"]))
            max_pad = max(max_pad, _prod(
                (in_shape[0] + top + bottom,
                 in_shape[1] + left + right, in_shape[2])))
        # 记录数据最终落在哪个句柄（计算层翻转，Dropout 交换也翻转）
        if op in _COMPUTE_OPS or op == "Dropout":
            last_writer = "ta" if info["is_odd_layer"] else "tb"

    # ---- 第二遍：生成逐层片段 ----
    body: list[str] = []
    first_compute = True
    for info in ops:
        op = info["op_type"]
        name = info["name"]

        if op == "Dropout":
            # 推理阶段 Dropout 无操作：交换两个工作张量，数据即传递到另一头
            body.append(
                f"/* [{info['index']}] {name}: Dropout 推理无操作，"
                f"交换工作张量传递数据 */"
            )
            body.append("k2c_tensor_swap(&ta, &tb);")
            body.append("")
            continue

        snippet_fn = _SNIPPETS.get(op)
        if snippet_fn is None:
            body.append(f"/* [{info['index']}] {name}: {op} 暂不支持，跳过 */")
            continue

        read_expr, write = _read_write(info, first_compute)
        lines, pad_numel = snippet_fn(info, read_expr, write)
        body.extend(lines)
        body.append("")
        first_compute = False

    # ---- 组装 model_invoke.c ----
    out: list[str] = [
        "/* Auto-generated by tiny_ml_compiler. Do not edit. */",
        '#include "model.h"',
        '#include "weight.h"',
        "",
        "/* 交换两个 k2c_tensor 的全部内容（array 指针 + ndim/numel/shape），",
        "   供 Dropout 等无操作层零成本传递数据。 */",
        "static inline void k2c_tensor_swap(k2c_tensor *a, k2c_tensor *b) {",
        "    k2c_tensor tmp = *a;",
        "    *a = *b;",
        "    *b = tmp;",
        "}",
        "",
        f"/* 工作张量内存：按网络最大激活 {max_act}、最大 same 填充 {max_pad} 分配 */",
        f"static float buf_a[{max_act}] = {{0}};",
        f"static float buf_b[{max_act}] = {{0}};",
        f"static float buf_pad[{max_pad}] = {{0}};",
        f"static k2c_tensor ta = {{buf_a, 0, 0, {{0,0,0,0,0}}}};",
        f"static k2c_tensor tb = {{buf_b, 0, 0, {{0,0,0,0,0}}}};",
        f"static k2c_tensor pad_work = {{buf_pad, 0, 0, {{0,0,0,0,0}}}};",
        "",
        "/* 单样本前向推理。input 为调用方提供的输入张量，",
        "   *output 指向最终结果所在的工作张量；返回 0 表示成功。 */",
        "int forward(const k2c_tensor* input, k2c_tensor** output) {",
        "    /* 归一化工作张量的 array 指针：层间 swap 会交换 ta/tb 的底层缓冲，",
        "       每次进入 forward 时恢复映射，保证可重复调用。 */",
        "    ta.array = buf_a;",
        "    tb.array = buf_b;",
    ]
    out.extend("    " + line if line else "" for line in body)
    out.append(f"    *output = &{last_writer};")
    out.append("    return 0;")
    out.append("}")
    out.append("")

    invoke_c_path.write_text("\n".join(out), encoding="utf-8")

    # 在 model.h 末尾追加 forward 声明
    with model_h_path.open("a", encoding="utf-8") as f:
        f.write("\n/* 模型前向推理入口（model_invoke.c） */\n")
        f.write("int forward(const k2c_tensor* input, k2c_tensor** output);\n")

    print(f"  已生成 forward()：{len(ops)} 层，"
          f"MAX_ACT={max_act}, MAX_PAD={max_pad}, 最终输出=&{last_writer}")


def reset_invoke_file(source_path: str | Path) -> None:
    """清空 model_invoke.c，便于重新生成。"""
    p = Path(source_path)
    if p.exists():
        p.unlink()
