"""TFLite -> k2c C 推理代码导出：offset 级 arena 规划 + k2c 算子映射。

管线（复用 dump_relax_ir 的前端/Pass/调度）：
    TFLite -> Relax IR -> 图优化 Pass（融合）-> Kahn 调度 + 生存期
    -> offset 级单池 arena 规划 -> C 代码（weights / model / main）

k2c 契约（src/tiny_ml_compiler/networkop/，已逐一核对源码）：
  - k2c_tensor{float* array, ndim, numel, shape[5]}，行主序；图像张量为 HWC 三维
    （无 batch 维），dense 为 (batch, features) 二维；
  - k2c_conv2d 的 kernel 排布为 HWIO；Relax 前端已把 TFLite 的 OHWI permute 成
    HWIO 并被 FoldConstant 折叠，导出时直接使用（勿再转置）；
  - k2c_dense 的 kernel 排布为 (in, out)，与 Relax matmul 的折叠权重方向一致；
  - k2c_pad2d 的 pad 顺序 {top, bottom, left, right}；
  - k2c_dense 需要 fwork 工作区（ndim<=2 分支不使用，传 arena 兜底）；
  - 激活是函数指针（k2c_relu / k2c_linear）。

用法（tvm conda 环境）：
    python tools/codegen_tflite_c.py model/mnist_cnn.tflite               # 生成 model/tflite_c/
    python tools/codegen_tflite_c.py model/mnist_cnn.tflite --plan-only   # 只打印内存布局
    编译（zig cc）：python -m ziglang cc -O2 -I<out>/networkop <out>/*.c <out>/networkop/*.c -o <out>/forward.exe -lm
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from math import prod
from pathlib import Path

import numpy as np

try:
    import tvm
    from tvm import relax
except ModuleNotFoundError as e:     # dump_relax_ir 顶部会做带提示的环境检查
    raise SystemExit(f"[环境错误] 缺少依赖 {e.name}，请用 tvm 环境运行") from e

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dump_relax_ir import (  # noqa: E402
    ROOT, load_tflite, optimize, extract_topology, kahn_schedule,
    analyze_lifetimes, plan_memory_offset, tensor_bytes, _shape_dims,
    _is_constant, short_op_name,
)
from tvm.relax.frontend.tflite import from_tflite  # noqa: E402

NETWORKOP_SRC = ROOT / "src" / "tiny_ml_compiler" / "networkop"
K2C_NAMES = [f"k2c_{n}.c" for n in
             ("activations", "convolution_layers", "core_layers",
              "helper_functions", "merge_layers", "normalization_layers",
              "pooling_layers", "tensor_include", "embedding_layers",
              "recurrent_layers")] + ["k2c_include.h", "k2c_tensor_include.h"]


# ---------------------------------------------------------------- 数据结构

@dataclass
class Weight:
    cname: str              # C 数组/张量名，如 wk0 / wb0
    data: np.ndarray
    role: str               # kernel / bias


@dataclass
class OpEmit:
    node: object            # 对应的 topo Node
    op: str                 # 语义算子：conv2d / matmul / max_pool2d / reshape / softmax / add
    activation: str = "linear"
    kernel: Weight | None = None
    bias: Weight | None = None
    strides: tuple = ()
    dilation: tuple = ()
    pool_size: tuple = ()
    axis: int = -1
    pad: tuple = ()         # same padding (top,bottom,left,right)；非空则发射 k2c_pad2d
    pad_var: str = ""       # pad workspace 的伪变量名（进 arena 统一规划）
    pad_shape: tuple = ()   # pad workspace 的形状


# ---------------------------------------------------------------- 常量提取

def _const_array(expr, exprs: dict) -> np.ndarray:
    """Var / Constant / Call(permute_dims|reshape) -> numpy 数组（解引用折叠常量）。

    循环不变量：每次递归要么命中 Constant 终止，要么沿 exprs 表回退一次定义；
    exprs 由 SSA binding 构成（后绑覆盖先绑），常量 var 名唯一，必然终止。
    """
    if _is_constant(expr):
        # 0.27 FFI 的 GenericConst 只有 .value（runtime.Tensor）；旧版用 .data
        nd = getattr(expr, "data", None)
        if nd is None:
            nd = expr.value
        return nd.numpy()
    names = {c.__name__ for c in type(expr).__mro__}
    if "Var" in names or "DataflowVar" in names:
        return _const_array(exprs[expr.name], exprs)
    if "Call" in names and isinstance(expr.op, tvm.ir.Op):
        op_name = short_op_name(expr.op.name)
        if op_name == "permute_dims":
            axes = [int(a) for a in expr.attrs.axes]
            return np.transpose(_const_array(expr.args[0], exprs), axes)
        if op_name == "reshape":
            shape = [int(v) for v in expr.args[1].fields]
            return np.reshape(_const_array(expr.args[0], exprs), shape)
        raise NotImplementedError(f"常量表达式中的算子 {op_name} 暂不支持")
    raise TypeError(f"无法解析常量: {type(expr).__name__}")


def _reg_weight(weights: list[Weight], index: dict, counters: dict,
                data: np.ndarray, role: str) -> Weight:
    """注册常量张量（按内容去重），命名 wk0/wb1...，统一转 float32 C 序。"""
    arr = np.ascontiguousarray(data, dtype=np.float32)
    key = arr.tobytes()
    if key in index:
        return index[key]
    prefix = "wk" if role == "kernel" else "wb"
    w = Weight(f"{prefix}{counters[prefix]}", arr, role)
    counters[prefix] += 1
    weights.append(w)
    index[key] = w
    return w


def _conv_pad_attrs(padding, out_hw, in_hw, strides, dilation, k_hw) -> tuple:
    """Relax conv2d 的 padding=(top,left,bottom,right) -> k2c (top,bottom,left,right)，
    并校验 padding 后的输出尺寸与 Relax 图一致（防 pad 顺序假设出错）。

    循环不变量：对每个空间维，in + pad_before + pad_after == (out-1)*s + d*(k-1) + 1。
    """
    def check(in_d: int, out_d: int, before: int, after: int, s: int, d: int, k: int) -> None:
        got = in_d + before + after
        want = (out_d - 1) * s + d * (k - 1) + 1
        if got != want:
            raise NotImplementedError(
                f"padding 校验失败: in={in_d} pad=({before},{after}) out={out_d} "
                f"(期望输入侧 {want})")
    p = tuple(int(x) for x in padding)
    if len(p) != 4:
        raise NotImplementedError(f"conv padding 应为 4 元组，实得 {p}")
    check(in_hw[0], out_hw[0], p[0], p[2], strides[0], dilation[0], k_hw[0])
    check(in_hw[1], out_hw[1], p[1], p[3], strides[1], dilation[1], k_hw[1])
    return (p[0], p[2], p[1], p[3])


def _emit_fused(opt_mod: tvm.IRModule, gvar, node, weights, index, counters) -> OpEmit:
    """解析融合子图函数体：主算子（conv2d/matmul）+ 常量权重 + 激活 + 超参。

    k2c 的层粒度自带 bias+activation，恰好与融合模式一对一：
      fused_conv2d_bias_relu -> k2c_pad2d?(同 padding) + k2c_conv2d(..., k2c_relu)
      fused_matmul_bias_relu -> k2c_dense(..., k2c_relu, fwork)
    """
    callee = opt_mod[gvar]
    exprs: dict = {}
    calls: list[tuple[str, object]] = []
    for blk in callee.body.blocks:
        for b in blk.bindings:
            if not isinstance(b, relax.VarBinding):
                continue
            exprs[b.var.name] = b.value
            if isinstance(b.value, relax.Call) and isinstance(b.value.op, tvm.ir.Op):
                calls.append((short_op_name(b.value.op.name), b.value))

    mains = [(n, c) for n, c in calls if n in ("conv2d", "matmul")]
    if len(mains) != 1:
        raise NotImplementedError(f"融合子图 {node.name} 主算子不唯一: {[n for n, _ in calls]}")
    op_name, call = mains[0]
    activation = "relu" if node.inner_ops and node.inner_ops[-1] == "relu" else "linear"

    emit = OpEmit(node, op_name, activation)
    if op_name == "conv2d":
        # Relax 前端已把 TFLite 的 OHWI 权重 permute 成 HWIO 并被 FoldConstant 折叠，
        # 这里直接使用，不再转置
        kernel = _const_array(call.args[1], exprs)
        if kernel.ndim != 4:
            raise NotImplementedError(f"conv kernel 应为 HWIO 4 维，实得 {kernel.shape}")
        emit.kernel = _reg_weight(weights, index, counters, kernel, "kernel")
        emit.strides = tuple(int(s) for s in call.attrs.strides)
        emit.dilation = tuple(int(s) for s in call.attrs.dilation)
        in_shape = [int(d) for d in call.args[0].ty.shape]
        out_shape = _shape_dims(node.shape)
        emit.pad = _conv_pad_attrs(call.attrs.padding, out_shape[1:3], in_shape[1:3],
                                   emit.strides, emit.dilation, kernel.shape[:2])
        if emit.pad != (0, 0, 0, 0):
            emit.pad_var = f"tp{node.idx}"
            emit.pad_shape = (in_shape[0], in_shape[1] + emit.pad[0] + emit.pad[1],
                              in_shape[2] + emit.pad[2] + emit.pad[3], in_shape[3])
    else:  # matmul：前端已把 TFLite (out,in) 权重 permute 成 (in,out) 后折叠
        kernel = _const_array(call.args[1], exprs)
        if kernel.ndim != 2:
            raise NotImplementedError(f"dense kernel 应为 (in,out) 2 维，实得 {kernel.shape}")
        emit.kernel = _reg_weight(weights, index, counters, kernel, "kernel")

    # bias：融合体内 add 的一个常量操作数
    for n, c in calls:
        if n != "add":
            continue
        for a in c.args:
            try:
                emit.bias = _reg_weight(weights, index, counters,
                                        _const_array(a, exprs), "bias")
                break
            except (TypeError, NotImplementedError, KeyError):
                continue
    return emit


def _emit_op(call, node, exprs, weights, index, counters) -> OpEmit:
    """main 内逐算子（未融合）节点的映射。"""
    op_name = short_op_name(call.op.name)
    if op_name == "max_pool2d":
        if any(int(s) for s in call.attrs.padding):
            raise NotImplementedError("max_pool2d 带 padding 暂不支持")
        return OpEmit(node, "max_pool2d",
                      strides=tuple(int(s) for s in call.attrs.strides),
                      pool_size=tuple(int(s) for s in call.attrs.pool_size))
    if op_name == "reshape":
        return OpEmit(node, "reshape")          # 新形状直接取 node.shape
    if op_name == "softmax":
        return OpEmit(node, "softmax", axis=int(call.attrs.axis))
    if op_name == "matmul":
        kernel = _const_array(call.args[1], exprs)
        if kernel.ndim != 2:
            raise NotImplementedError(f"matmul 权重应为 2 维，实得 {kernel.shape}")
        return OpEmit(node, "matmul",
                      kernel=_reg_weight(weights, index, counters, kernel, "kernel"))
    if op_name == "add":
        for a in call.args:                     # 一侧为常量 -> bias_add 语义
            try:
                arr = _const_array(a, exprs)
            except (TypeError, NotImplementedError, KeyError):
                continue
            return OpEmit(node, "add",
                          bias=_reg_weight(weights, index, counters, arr, "bias"))
        raise NotImplementedError("add 的两个操作数均非常量（通用二目加法暂不支持）")
    raise NotImplementedError(f"算子 {op_name} 暂不支持（{node.name}）")


def collect_ops(opt_mod: tvm.IRModule, topo) -> tuple[dict, list[Weight], set[str]]:
    """与 extract_topology 同序重扫 main bindings（循环不变量：第 i 个 VarBinding
    对应 topo.nodes[i+1]），收集每节点的发射信息、权重表与常量节点集合。"""
    func = opt_mod["main"]
    bindings = [b for blk in func.body.blocks for b in blk.bindings
                if isinstance(b, relax.VarBinding)]
    if len(bindings) != len(topo.nodes) - 1:
        raise RuntimeError("binding 与拓扑节点失去对齐（extract_topology 遍历逻辑变更？）")
    exprs = {b.var.name: b.value for b in bindings}

    emits: dict[str, OpEmit] = {}
    weights: list[Weight] = []
    index: dict = {}
    counters = {"wk": 0, "wb": 0}
    const_nodes: set[str] = set()

    for node, b in zip(topo.nodes[1:], bindings):
        val = b.value
        if _is_constant(val):                   # 权重常量节点 -> weights.c，不发射
            const_nodes.add(node.name)
            continue
        if node.kind == "FUSED":
            emits[node.name] = _emit_fused(opt_mod, val.op, node, weights, index, counters)
        elif node.kind == "OP":
            emits[node.name] = _emit_op(val, node, exprs, weights, index, counters)
        else:
            raise NotImplementedError(f"节点 {node.name}（{node.kind}/{node.op}）暂不支持")
    return emits, weights, const_nodes


# ---------------------------------------------------------------- C 代码发射

def _c_floats(arr: np.ndarray, per_line: int = 8) -> str:
    vals = arr.astype(np.float32).ravel()
    chunks = [", ".join(f"{v:.9g}f" for v in vals[i:i + per_line])
              for i in range(0, len(vals), per_line)]
    return ",\n    ".join(chunks)


def _shape5(shape) -> str:
    return ", ".join(str(d) for d in shape) + ", 0" * (5 - len(shape))


def gen_weights(weights: list[Weight]) -> tuple[str, str]:
    h = ["#ifndef WEIGHTS_H", "#define WEIGHTS_H", "#include \"k2c_tensor_include.h\"", ""]
    c = ["#include \"weights.h\"", ""]
    for w in weights:
        numel = w.data.size
        h.append(f"extern float {w.cname}_data[{numel}];")
        h.append(f"extern k2c_tensor {w.cname};")
        c.append(f"/* {w.role} shape={list(w.data.shape)} */")
        c.append(f"float {w.cname}_data[{numel}] = {{")
        c.append(f"    {_c_floats(w.data)}")
        c.append("};")
        c.append(f"k2c_tensor {w.cname} = {{{w.cname}_data, {w.data.ndim}, {numel}, "
                 f"{{{_shape5(w.data.shape)}}}}};")
        c.append("")
    h += ["#endif", ""]
    return "\n".join(h), "\n".join(c)


def _tensor_decl(name: str, ptr: str, shape, numel: int) -> str:
    return (f"    k2c_tensor {name} = {{{ptr}, {len(shape)}, {numel}, "
            f"{{{_shape5(shape)}}}}};")


def _k2c_shape(shape) -> list:
    """Relax NHWC -> k2c HWC：图像类 4 维张量去掉 batch 维。

    k2c 约定 shape[0]=行(H)、shape[1]=列(W)、shape[2]=通道(C)，无 batch 维；
    dense 类 2D (batch, features) 保留（k2c_dense 按 shape[0]=outrows 处理）。
    """
    s = list(shape)
    return s[1:] if len(s) == 4 else s


def gen_model(topo, order, life, emits, weights, offsets, arena_bytes) -> tuple[str, str]:
    out_set = set(topo.outputs)
    in_node = next(n for n in topo.nodes if n.kind == "INPUT")
    out_node = next(n for n in topo.nodes if n.name in out_set)
    in_shape = _shape_dims(in_node.shape)
    out_shape = _shape_dims(out_node.shape)

    def cname(n) -> str:
        if n.kind == "INPUT":
            return "tin"
        return "tout" if n.name in out_set else f"t{n.idx}"

    # ---- 超参数组（去重） -------------------------------------------------
    hypers: dict[tuple, tuple[str, str]] = {}   # (kind, values) -> (C名, 声明)

    def hyper(kind: str, values) -> str:
        key = (kind, tuple(int(v) for v in values))
        if key not in hypers:
            cname_ = f"{kind}{len(hypers)}"
            n = len(key[1])
            hypers[key] = (cname_,
                           f"static const size_t {cname_}[{n}] = "
                           f"{{{', '.join(str(v) for v in key[1])}}};")
        return hypers[key][0]

    hyper_decls: list[str] = []

    # ---- 张量定义 ---------------------------------------------------------
    decls = [
        _tensor_decl("tin", "(float*)in", _k2c_shape(in_shape), prod(in_shape)),
        _tensor_decl("tout", "out", _k2c_shape(out_shape), prod(out_shape)),
    ]
    for n in topo.nodes:                        # 中间张量 + pad workspace
        e = emits.get(n.name)
        if n.name in out_set or n.kind == "INPUT":
            continue
        if e is None:
            continue                            # 常量节点
        shape = _k2c_shape(_shape_dims(n.shape))
        decls.append(_tensor_decl(f"t{n.idx}", f"&arena[{offsets[n.name] // 4}]",
                                  shape, prod(shape)))
        if e.pad_var:
            decls.append(_tensor_decl(e.pad_var, f"&arena[{offsets[e.pad_var] // 4}]",
                                      _k2c_shape(e.pad_shape), prod(e.pad_shape)))

    # ---- 按调度序发射调用 ---------------------------------------------------
    calls: list[str] = []
    for n in order:
        e = emits.get(n.name)
        if e is None:
            continue
        iname = cname(topo.by_name[n.inputs[0]]) if n.inputs else ""
        oname = cname(n)
        if e.op == "conv2d":
            s_ = hyper("stride", e.strides)
            d_ = hyper("dil", e.dilation)
            act = "k2c_relu" if e.activation == "relu" else "k2c_linear"
            if e.pad_var:
                p_ = hyper("pad", e.pad)
                calls.append(f"    k2c_pad2d(&{e.pad_var}, &{iname}, 0.0f, {p_});")
            calls.append(f"    k2c_conv2d(&{oname}, &{e.pad_var or iname}, "
                         f"&{e.kernel.cname}, &{e.bias.cname}, {s_}, {d_}, {act});")
        elif e.op == "matmul" and n.kind == "FUSED":
            act = "k2c_relu" if e.activation == "relu" else "k2c_linear"
            calls.append(f"    k2c_dense(&{oname}, &{iname}, &{e.kernel.cname}, "
                         f"&{e.bias.cname}, {act}, arena);")
        elif e.op == "matmul":                  # 裸 matmul（输出层，无融合）
            ishape = _shape_dims(topo.by_name[n.inputs[0]].shape)
            oshape = _shape_dims(n.shape)
            calls.append(f"    k2c_matmul({oname}.array, {iname}.array, "
                         f"{e.kernel.cname}.array, {oshape[0]}, {oshape[1]}, {ishape[-1]});")
        elif e.op == "max_pool2d":
            p_ = hyper("pool", e.pool_size)
            s_ = hyper("pstride", e.strides)
            calls.append(f"    k2c_maxpool2d(&{oname}, &{iname}, {p_}, {s_});")
        elif e.op == "reshape":
            shp = hyper("shape", _shape_dims(n.shape))
            calls.append(f"    k2c_reshape(&{oname}, &{iname}, {shp}, "
                         f"{len(_shape_dims(n.shape))});")
        elif e.op == "add":
            numel = prod(_shape_dims(n.shape))
            calls.append(f"    memcpy({oname}.array, {iname}.array, {numel} * sizeof(float));")
            calls.append(f"    k2c_bias_add(&{oname}, &{e.bias.cname});")
        elif e.op == "softmax":
            numel = prod(_shape_dims(n.shape))
            calls.append(f"    memcpy({oname}.array, {iname}.array, {numel} * sizeof(float));")
            calls.append(f"    k2c_softmax({oname}.array, {oname}.numel);")
        else:
            raise NotImplementedError(f"发射失败：未知算子 {e.op}")

    hyper_decls = [decl for _, decl in hypers.values()]

    model_c = "\n".join(
        ["#include <string.h>", "#include \"k2c_include.h\"",
         "#include \"weights.h\"", "#include \"model.h\"", "",
         "/* offset 级单池 arena：所有中间张量按字节偏移共存于此 */",
         f"static float arena[ARENA_FLOATS];", ""]
        + hyper_decls + ["", "void forward(const float* in, float* out) {"] + decls + [""]
        + calls + ["}", ""])

    model_h = "\n".join([
        "#ifndef MODEL_H", "#define MODEL_H", "",
        f"#define ARENA_FLOATS {arena_bytes // 4}   /* {arena_bytes} B */",
        f"#define IN_SIZE {prod(in_shape)}",
        f"#define OUT_SIZE {prod(out_shape)}", "",
        "void forward(const float* in, float* out);", "", "#endif", ""])
    return model_h, model_c


MAIN_C = """\
/* 生成的推理入口：读 input.bin -> forward() -> 写 output.bin */
#include <stdio.h>
#include "model.h"

int main(int argc, char** argv) {
    const char* in_path = argc > 1 ? argv[1] : "input.bin";
    const char* out_path = argc > 2 ? argv[2] : "output.bin";

    static float in_buf[IN_SIZE];
    static float out_buf[OUT_SIZE];

    FILE* f = fopen(in_path, "rb");
    if (!f || fread(in_buf, sizeof(float), IN_SIZE, f) != IN_SIZE) {
        fprintf(stderr, "读取 %s 失败\\n", in_path);
        return 1;
    }
    fclose(f);

    forward(in_buf, out_buf);

    FILE* g = fopen(out_path, "wb");
    if (!g || fwrite(out_buf, sizeof(float), OUT_SIZE, g) != OUT_SIZE) {
        fprintf(stderr, "写入 %s 失败\\n", out_path);
        return 1;
    }
    fclose(g);
    for (int i = 0; i < OUT_SIZE; ++i) {
        printf("out[%d] = %.6f\\n", i, out_buf[i]);
    }
    return 0;
}
"""


# ---------------------------------------------------------------- 主流程

def render_layout(topo, order, life, emits, offsets, bytes_of, arena) -> str:
    L = ["# offset 级内存布局（单池 arena，8B 对齐）", ""]
    L.append(f"{'张量':<10} {'shape':<18} {'字节':>9} {'offset':>9}")
    for n in order:
        if n.name in offsets:
            e = emits.get(n.name)
            shape = e.pad_shape if (e and e.pad_var == n.name) else _shape_dims(n.shape)
            tag = " (pad ws)" if (e and e.pad_var == n.name) else ""
            L.append(f"{n.name:<10} {str(shape):<18} {bytes_of[n.name]:>9} "
                     f"{offsets[n.name]:>9}{tag}")
    L.append("")
    L.append(f"arena 总计: {arena} B（{arena // 4} floats）")
    return "\n".join(L)


def main() -> int:
    parser = argparse.ArgumentParser(description="TFLite -> k2c C 推理代码导出")
    parser.add_argument("model", nargs="?", default=str(ROOT / "model" / "mnist_cnn.tflite"))
    parser.add_argument("--out", default=None, help="输出目录（默认 model/<stem>_c/）")
    parser.add_argument("--plan-only", action="store_true", help="只打印内存布局")
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    if not model_path.exists():
        print(f"[错误] 模型不存在: {model_path}")
        return 1

    tflite_model, size = load_tflite(model_path)
    opt_mod = optimize(from_tflite(tflite_model))
    topo = extract_topology(opt_mod)
    order = kahn_schedule(topo)
    life = analyze_lifetimes(topo, order)
    emits, weights, const_nodes = collect_ops(opt_mod, topo)

    step_of = {n.name: i for i, n in enumerate(order)}
    bytes_of = {n.name: tensor_bytes(n) for n in topo.nodes
                if n.name not in const_nodes}
    # pad workspace 作为伪变量纳入统一规划（生命期 = conv 所在步）
    for e in emits.values():
        if e.pad_var:
            bytes_of[e.pad_var] = prod(e.pad_shape) * 4
            life[e.pad_var] = (step_of[e.node.name], step_of[e.node.name])

    skip = (const_nodes | set(topo.outputs)
            | {n.name for n in topo.nodes if n.kind == "INPUT"})
    arena_bytes_of = {k: v for k, v in bytes_of.items() if k not in skip}
    offsets, arena = plan_memory_offset(arena_bytes_of, life)

    print(render_layout(topo, order, life, emits, offsets, arena_bytes_of, arena))
    print(f"\n权重张量 {len(weights)} 个，共 {sum(w.data.size for w in weights) * 4} B（只读段）")
    if args.plan_only:
        return 0

    out_dir = Path(args.out) if args.out else model_path.parent / f"{model_path.stem}_c"
    out_dir.mkdir(parents=True, exist_ok=True)
    w_h, w_c = gen_weights(weights)
    m_h, m_c = gen_model(topo, order, life, emits, weights, offsets, arena)
    (out_dir / "weights.h").write_text(w_h, encoding="utf-8")
    (out_dir / "weights.c").write_text(w_c, encoding="utf-8")
    (out_dir / "model.h").write_text(m_h, encoding="utf-8")
    (out_dir / "model.c").write_text(m_c, encoding="utf-8")
    (out_dir / "main.c").write_text(MAIN_C, encoding="utf-8")

    net_dir = out_dir / "networkop"
    net_dir.mkdir(exist_ok=True)
    for name in K2C_NAMES:
        src = NETWORKOP_SRC / name
        if src.exists():
            shutil.copy2(src, net_dir / name)

    print(f"\n生成完成 -> {out_dir}")
    print(f"编译: python -m ziglang cc -O2 -I{net_dir.name} "
          f"{out_dir / '*.c'} {net_dir.name}/*.c -o {out_dir / 'forward.exe'} -lm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
