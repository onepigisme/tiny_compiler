"""TFLite -> TVM Relax：前端 IR、拓扑结构、图优化 Pass 一站式提取。

TVM >= 0.25 的官方 PyPI wheel（含 Windows）已移除 Relay，图级 IR 统一为 Relax，
TFLite 前端入口是 `tvm.relax.frontend.tflite.from_tflite`。

本脚本做四件事：
  1. 前端解析：TFLite -> Relax IRModule（忠实翻译，不做优化）；
  2. 拓扑提取：按 SSA binding 顺序列出节点（op / shape / dtype / 输入边）；
  3. 图优化：常量折叠 + 死代码消除 + DPL 模式融合（conv/matmul + bias + relu），
     保存优化后的 IR，并统计融合子图数量；
  4. 拓扑调度：对 Pass 后的可执行图做 Kahn 拓扑排序求执行序、变量生存期分析、
     贪心区间着色做 buffer 复用，输出调度表与内存对比。

用法（在 tvm conda 环境中）：
    python tools/dump_relax_ir.py                            # 默认 mnist_cnn.tflite
    python tools/dump_relax_ir.py model/mnist_cnn_int8.tflite
    python tools/dump_relax_ir.py model/mnist_cnn.tflite --no-save

产物（与模型同目录，<stem> 为模型文件名去后缀）：
    <stem>.relax_ir.txt      前端 IR（优化前）
    <stem>.topo.txt          拓扑节点表 + 依赖边
    <stem>.optimized.txt     优化后 IR（含 fused 子图）
    <stem>.schedule.txt      调度表（执行序 / 生存期 / buffer 复用 / 内存对比）

本脚本同时是 codegen_tflite_c.py 的前端库：后者复用其 load_tflite / optimize /
extract_topology / kahn_schedule / analyze_lifetimes / plan_memory_offset，
接续完成「offset 级 arena 规划 + k2c 算子映射」并导出可编译的 C 推理代码
（见 tools/codegen_tflite_c.py 文件头说明）。
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from math import prod
from pathlib import Path

try:
    import tflite
    import tvm
    from tvm import relax
    from tvm.relax.dpl import is_op, wildcard
    from tvm.relax.frontend.tflite import from_tflite
except ModuleNotFoundError as e:
    raise SystemExit(
        f"[环境错误] 缺少依赖 {e.name}。请改用 tvm 环境运行本脚本：\n"
        f"    conda activate tvm\n"
        f"    python {Path(__file__).name}\n"
        f"当前解释器：{sys.executable}"
    ) from e

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "model" / "mnist_cnn.tflite"


# ---------------------------------------------------------------- 数据结构

@dataclass
class Node:
    idx: int
    name: str
    kind: str           # INPUT / OP / FUSED / OTHER
    op: str             # 短算子名，INPUT 为 "-"
    shape: str
    dtype: str
    inputs: list[str] = field(default_factory=list)   # 依赖的前序变量名
    n_const: int = 0    # 常量参数个数（权重/偏置）
    inner_ops: list[str] = field(default_factory=list)  # FUSED 节点内部的原始算子序列


@dataclass
class TopoGraph:
    nodes: list[Node]
    outputs: list[str]
    by_name: dict[str, Node]


# ---------------------------------------------------------------- 基础工具

def load_tflite(path: Path):
    """读取 .tflite 字节并解析为 tflite.Model（FlatBuffers 根对象）。"""
    data = path.read_bytes()
    return tflite.Model.GetRootAsModel(data, 0), len(data)


def tensor_desc(expr) -> tuple[str, str]:
    """从 Var/Constant 的 ty(TensorType) 提取静态 shape 与 dtype。"""
    ty = getattr(expr, "ty", None)
    if ty is None:
        return "?", "?"
    dims: list[str] = []
    shape = getattr(ty, "shape", None)
    if shape is not None:
        for d in shape:
            try:
                dims.append(str(int(d)))
            except (TypeError, ValueError):
                dims.append(str(d))
    dtype = getattr(ty, "dtype", "?")
    if not isinstance(dtype, str):          # 0.27 中可能返回 PrimType 对象
        dtype = getattr(dtype, "dtype", str(dtype))
    return ("(" + ",".join(dims) + ")" if dims else "scalar"), dtype


def _mro_names(expr) -> set[str]:
    return {c.__name__ for c in type(expr).__mro__}


def _is_constant(expr) -> bool:
    # TVM 0.27 wheel 中常量在 Python 侧名为 GenericConst（MRO 含 Constant）
    return "Constant" in _mro_names(expr)


def _collect_var_args(expr, var_names: list[str], n_const: list[int]) -> None:
    """递归收集 Call 参数中引用的变量（产生数据依赖边）与常量个数。

    注意：0.27 FFI 运行时生成的 Var 类与 relax.Var 不是同一个 Python 类，
    isinstance 会失效，因此按 MRO 类名判别。
    """
    names = _mro_names(expr)
    if "DataflowVar" in names or "Var" in names:
        var_names.append(expr.name)
    elif "Constant" in names:
        n_const[0] += 1
    elif "Tuple" in names:
        for f in expr.fields:
            _collect_var_args(f, var_names, n_const)
    elif "Call" in names:
        for a in expr.args:
            _collect_var_args(a, var_names, n_const)


def _gv_name(gv) -> str:
    """跨版本取 GlobalVar 名字。"""
    for attr in ("name_hint", "name"):
        v = getattr(gv, attr, None)
        if isinstance(v, str):
            return v
    m = re.search(r'"([^"]+)"', str(gv))
    return m.group(1) if m else str(gv)


def short_op_name(name: str) -> str:
    return name.replace("relax.nn.", "").replace("relax.", "")


def resolve_fused(mod: tvm.IRModule, gvar) -> tuple[str | None, list[str]]:
    """解析融合子图的 op 类型：返回 (Composite 模式名, 内部原始算子序列)。

    Pass 之后调用点的 call.op 是 GlobalVar，op 类型信息转移到两处：
      1. 被调函数的 Composite 函数属性 —— 融合时注册的模式名，
         是后端 codegen / op_registry 派发的标准 key；
      2. 被调函数体 —— 内部仍是原始 op 调用序列，遍历即可精确还原。
    函数名本身（fused_relax_nn_conv2d_relax_add_relax_nn_relu）虽可读，
    但同名会加数字后缀（relu1），只能当提示，不能当 key。
    """
    try:
        callee = mod[gvar]
    except KeyError:
        return None, []

    comp = None
    try:
        if callee.attrs is not None:
            c = callee.attrs["Composite"]
            comp = str(c) if c is not None else None
    except (AttributeError, KeyError):
        comp = None

    inner: list[str] = []
    for block in callee.body.blocks:
        for binding in block.bindings:
            if not isinstance(binding, relax.VarBinding):
                continue
            v = binding.value
            if isinstance(v, relax.Call) and isinstance(v.op, tvm.ir.Op):
                inner.append(short_op_name(v.op.name))
    return comp, inner


# ---------------------------------------------------------------- 拓扑提取

def extract_topology(mod: tvm.IRModule, entry: str = "main") -> TopoGraph:
    """按 binding 顺序（SSA 下即拓扑序）提取 main 的节点与边。

    循环不变量：每登记一个节点，其 inputs 引用的变量一定已登记（前端 IR 的
    数据依赖必然先定义后使用），因此节点下标递增方向就是数据流方向。
    """
    func = mod[entry]
    nodes: list[Node] = []
    by_name: dict[str, Node] = {}
    latest: dict[str, str] = {}   # 原始变量名 -> 最近一次定义的唯一 key

    def add(node: Node) -> None:
        # Relax 的 Var 名字只是 hint，FuseOps 后可能出现同名重绑定；
        # 重名时加 "#2/#3..." 后缀生成唯一 key，保证按名字建边不串。
        node.idx = len(nodes)
        key, k = node.name, 2
        while key in by_name:
            key = f"{node.name}#{k}"
            k += 1
        node.name = key
        nodes.append(node)
        by_name[key] = node
        latest[node.name.split("#")[0]] = key

    # 1) 图输入（函数参数）
    for p in func.params:
        shape, dtype = tensor_desc(p)
        add(Node(0, p.name, "INPUT", "-", shape, dtype))

    # 2) 逐 block、逐 binding 登记
    for block in func.body.blocks:
        for binding in block.bindings:
            if not isinstance(binding, relax.VarBinding):
                continue   # MatchCast 等在本模型中不出现
            var = binding.var
            val = binding.value
            shape, dtype = tensor_desc(var)
            inputs: list[str] = []
            n_const = [0]
            inner_ops: list[str] = []

            if isinstance(val, relax.Call):
                for a in val.args:
                    _collect_var_args(a, inputs, n_const)
                op_obj = val.op
                if isinstance(op_obj, tvm.ir.Op):
                    kind, op_name = "OP", short_op_name(op_obj.name)
                elif isinstance(op_obj, relax.GlobalVar):
                    # 融合后调用的是子图函数：回头查 Composite 属性与函数体
                    comp, inner_ops = resolve_fused(mod, op_obj)
                    kind = "FUSED"
                    op_name = comp or _gv_name(op_obj)
                else:
                    kind, op_name = "OTHER", str(op_obj).split("(")[0]
            elif isinstance(val, relax.Var):
                _collect_var_args(val, inputs, n_const)
                kind, op_name = "OTHER", "alias"
            else:
                kind, op_name = "OTHER", type(val).__name__

            # consumer 引用的变量按“最近一次定义”解析成唯一 key（SSA 文本序即作用域）
            inputs = [latest.get(nm, nm) for nm in dict.fromkeys(inputs)]
            add(Node(0, var.name, kind, op_name, shape, dtype,
                     inputs, n_const[0], inner_ops))

    # 3) 图输出（SeqExpr.body，可能是 Var 或 Tuple）
    outputs: list[str] = []
    body = func.body.body
    if isinstance(body, relax.Var):
        outputs.append(latest.get(body.name, body.name))
    elif isinstance(body, relax.Tuple):
        outputs.extend(latest.get(v.name, v.name)
                       for v in body.fields if isinstance(v, relax.Var))

    return TopoGraph(nodes, outputs, by_name)


def render_topology(topo: TopoGraph, title: str) -> str:
    lines = [f"# {title}", ""]
    header = (f"{'idx':>3}  {'KIND':<6} {'NAME':<28} {'OP':<26} "
              f"{'SHAPE':<20} {'DTYPE':<8} {'INNER':<22} INPUTS")
    lines.append(header)
    lines.append("-" * len(header))
    for n in topo.nodes:
        const_txt = f"+{n.n_const} const"
        if n.inputs:
            ins = ", ".join(n.inputs) + (f"  ({const_txt})" if n.n_const else "")
        else:
            ins = const_txt if n.n_const else "-"
        inner = ",".join(n.inner_ops) if n.inner_ops else "-"
        lines.append(
            f"{n.idx:>3}  {n.kind:<6} {n.name:<28} {n.op:<26} {n.shape:<20} "
            f"{n.dtype:<8} {inner:<22} {ins}"
        )
    lines.append("")
    lines.append(f"OUTPUT: {', '.join(topo.outputs)}")
    lines.append("")
    lines.append("# 依赖边（producer -> consumer）")
    edge_cnt = 0
    for n in topo.nodes:
        for src in n.inputs:
            lines.append(f"{src} -> {n.name}")
            edge_cnt += 1
    lines.append("")
    n_op = sum(n.kind == "OP" for n in topo.nodes)
    n_fused = sum(n.kind == "FUSED" for n in topo.nodes)
    lines.append(f"# 统计: 节点 {len(topo.nodes)}（INPUT {sum(n.kind == 'INPUT' for n in topo.nodes)}, "
                 f"OP {n_op}, FUSED {n_fused}, OTHER {sum(n.kind == 'OTHER' for n in topo.nodes)}），"
                 f"边 {edge_cnt} 条")
    return "\n".join(lines)


# ---------------------------------------------------------------- 图优化 Pass

def _conv_bias_relu_pattern():
    x, w, b = wildcard(), wildcard(), wildcard()
    conv = is_op("relax.nn.conv2d")(x, w)
    bias = is_op("relax.add")(conv, b)
    return is_op("relax.nn.relu")(bias)


def _matmul_bias_relu_pattern():
    x, w, b = wildcard(), wildcard(), wildcard()
    mm = is_op("relax.matmul")(x, w)
    bias = is_op("relax.add")(mm, b)
    return is_op("relax.nn.relu")(bias)


def optimize(mod: tvm.IRModule) -> tvm.IRModule:
    """常量折叠 + 死代码消除 + 模式融合。

    注：TVM 0.27 的官方 wheel 中通用 FuseOps 的算子模式表为空（AnnotateTIROpPattern
    不生效），因此用 DPL 显式描述 conv/matmul + bias + relu 融合模式。
    """
    seq = tvm.transform.Sequential([
        relax.transform.FoldConstant(),        # 折叠常量表达式（如对权重的 permute_dims）
        relax.transform.DeadCodeElimination(), # 删除无用 binding
        relax.transform.FuseOpsByPattern([
            ("fused_conv2d_bias_relu", _conv_bias_relu_pattern()),
            ("fused_matmul_bias_relu", _matmul_bias_relu_pattern()),
        ]),
    ])
    return seq(mod)


def fused_subgraphs(mod: tvm.IRModule) -> list[tuple[str, str]]:
    """列出带 Composite 属性的融合子图（函数名 -> 复合模式名）。"""
    out: list[tuple[str, str]] = []
    for gv, func in mod.functions.items():
        comp = None
        try:
            if func.attrs is not None:
                comp = func.attrs["Composite"]
        except (AttributeError, KeyError):
            comp = None
        if comp is not None:
            out.append((_gv_name(gv), str(comp)))
    return out


# ---------------------------------------------------------------- 拓扑调度

DTYPE_SIZE = {
    "bool": 1, "int8": 1, "uint8": 1,
    "int16": 2, "uint16": 2, "float16": 2, "bfloat16": 2,
    "int32": 4, "uint32": 4, "float32": 4,
    "int64": 8, "uint64": 8, "float64": 8,
}


def _shape_dims(shape: str) -> list[int]:
    """"(1,28,28,8)" -> [1,28,28,8]；含 '?' 视为动态 shape，返回空表。"""
    body = shape.strip().strip("()").strip()
    if not body or "?" in body:
        return []
    try:
        return [int(d) for d in body.split(",")]
    except ValueError:
        return []


def tensor_bytes(node: Node) -> int:
    dims = _shape_dims(node.shape)
    return prod(dims) * DTYPE_SIZE.get(node.dtype, 4) if dims else 0


def kahn_schedule(topo: TopoGraph) -> list[Node]:
    """Kahn 算法求执行序：显式按依赖边排序，而非直接沿用 binding 顺序。

    循环不变量：ready 中节点的入度均为 0，即其全部前驱（生产者）都已进入
    order —— 因此弹出任意 ready 节点执行都是安全的；每弹出一个节点，将其
    消费者入度减 1，减到 0 者进入 ready，不变量保持。图无环时结束时
    len(order) == 节点总数，且 order 中每个节点都排在其所有生产者之后。
    """
    indeg = {n.name: 0 for n in topo.nodes}
    consumers: dict[str, list[str]] = {n.name: [] for n in topo.nodes}
    for n in topo.nodes:
        for src in dict.fromkeys(n.inputs):     # 同一输入去重（如 add(x,x)）
            if src != n.name and src in topo.by_name:
                indeg[n.name] += 1
                consumers[src].append(n.name)

    ready = [n.name for n in topo.nodes if indeg[n.name] == 0]
    order: list[Node] = []
    while ready:
        ready.sort(key=lambda nm: topo.by_name[nm].idx)   # 稳定：与 binding 序可比
        name = ready.pop(0)
        order.append(topo.by_name[name])
        for c in consumers[name]:
            indeg[c] -= 1
            if indeg[c] == 0:
                ready.append(c)
    if len(order) != len(topo.nodes):
        raise RuntimeError("调度失败：拓扑图中存在环（前端 SSA 图不应出现）")
    return order


def analyze_lifetimes(topo: TopoGraph, order: list[Node]) -> dict[str, tuple[int, int]]:
    """生存期分析：变量 v 的活跃区间 [def_step, last_use_step]（闭区间，步号）。

    循环不变量：扫到第 k 个消费者时 last_use[v] 已是前 k 个消费者步号的最大值，
    扫完全部消费者即真实最后使用步；无消费者的变量：图输出则存活到最后一步
    （调度结束前不得释放），否则定义即死（DCE 后不应出现）。
    """
    step_of = {n.name: i for i, n in enumerate(order)}
    last_use: dict[str, int] = {}
    for n in order:
        for src in dict.fromkeys(n.inputs):
            if src in step_of:
                last_use[src] = max(last_use.get(src, -1), step_of[n.name])
    last_step = len(order) - 1
    life: dict[str, tuple[int, int]] = {}
    for n in topo.nodes:
        d = step_of[n.name]
        u = last_use.get(n.name, last_step if n.name in topo.outputs else d)
        life[n.name] = (d, max(u, d))
    return life


def plan_memory(topo: TopoGraph, life: dict[str, tuple[int, int]],
                ) -> tuple[dict[str, int], list[dict], int, int]:
    """贪心区间着色做 buffer 复用（TFLite arena / StaticPlanBlockMemory 同思路）。

    循环不变量：每个内存块内成员的活跃区间两两不交，故块内张量可轮流占用
    同一块内存（块大小 = 成员字节数最大值）。新张量 t 放入块 b 当且仅当 t
    与 b 内所有成员区间不交，放入后不变量保持；全部张量分配完即得合法方案。

    返回 (assign: 变量->块号, blocks, peak: 按生存期分配但不复用的峰值字节,
          total: 每变量独占一块且全程持有的总字节)。
    """
    bytes_of = {n.name: tensor_bytes(n) for n in topo.nodes}
    # 定义早的先分配；同一步先大后小，first-fit 装箱效果更稳
    names = sorted(topo.by_name, key=lambda nm: (life[nm][0], -bytes_of[nm], nm))

    blocks: list[dict] = []   # {"size": 字节, "iv": [(def, last)...], "members": [变量名]}
    assign: dict[str, int] = {}
    for nm in names:
        d, u = life[nm]
        for bid, b in enumerate(blocks):
            if all(u < pd or d > pu for pd, pu in b["iv"]):
                b["iv"].append((d, u))
                b["size"] = max(b["size"], bytes_of[nm])
                b["members"].append(nm)
                assign[nm] = bid
                break
        else:
            blocks.append({"size": bytes_of[nm], "iv": [(d, u)], "members": [nm]})
            assign[nm] = len(blocks) - 1

    peak = 0
    n_step = max((u for _, u in life.values()), default=0) + 1
    for s in range(n_step):
        live = sum(nb for nm, nb in bytes_of.items()
                   if life[nm][0] <= s <= life[nm][1])
        peak = max(peak, live)
    return assign, blocks, peak, sum(bytes_of.values())


def plan_memory_offset(bytes_of: dict[str, int], life: dict[str, tuple[int, int]],
                       align: int = 8,
                       ) -> tuple[dict[str, int], int]:
    """offset 级单池 arena 规划（TFLite arena 同思路）：字节偏移 + 区间 first-fit。

    与 plan_memory 的"块编号"不同：单池内按字节偏移摆放，大小不同的张量可以
    错位共享空间，arena 更省。bytes_of/life 允许包含拓扑外的伪变量（如
    k2c_pad2d 的 pad workspace），便于调用方把额外工作区纳入统一规划。

    循环不变量：任意调度步 s，存活张量集合的 [offset, offset+size) 字节区间
    两两不交 —— 因此任一时刻所有存活张量可同时共存于同一池内互不踩踏。
    新张量 t 放在最小候选偏移 off 处：候选取 0 与所有生命期重叠区间的 end，
    off 与所有生命期与 t 重叠的已分配区间不相交（off+size<=a.start 或
    off>=a.end），放入后不变量保持。

    返回 (offsets: 变量->字节偏移, arena_bytes: 池总字节数)。
    """
    allocated: list[tuple[int, int, int, int]] = []   # (start, end, def, last)
    offsets: dict[str, int] = {}
    # 定义早的先放；同一步大张量优先（装箱更稳）
    names = sorted(bytes_of, key=lambda nm: (life[nm][0], -bytes_of[nm], nm))
    for nm in names:
        d, u = life[nm]
        size = bytes_of[nm]
        # 闭区间相交判定：[d,u] 与 [a.def,a.last] 共享任一步即为重叠
        overlapping = [a for a in allocated if not (u < a[2] or d > a[3])]
        candidates = sorted({0} | {a[1] for a in overlapping})
        for off in candidates:
            off = -(-off // align) * align            # 向上对齐
            if all(off + size <= a[0] or off >= a[1] for a in overlapping):
                offsets[nm] = off
                allocated.append((off, off + size, d, u))
                break
        else:
            raise RuntimeError(f"内存规划失败：{nm} 找不到可用偏移")
    arena = max((o + bytes_of[n] for n, o in offsets.items()), default=0)
    return offsets, arena


def schedule(topo: TopoGraph, title: str) -> str:
    """对 TopoGraph 做完整调度：Kahn 执行序 -> 生存期 -> buffer 复用 -> 调度表。"""
    order = kahn_schedule(topo)
    life = analyze_lifetimes(topo, order)
    assign, blocks, peak, total = plan_memory(topo, life)
    bytes_of = {n.name: tensor_bytes(n) for n in topo.nodes}
    arena = sum(b["size"] for b in blocks)
    const_bytes = sum(tensor_bytes(n) for n in topo.nodes
                      if "const" in n.op.lower())

    L = [f"# {title}", ""]
    same = [n.idx for n in order] == list(range(len(topo.nodes)))
    L.append("执行序(Kahn): " + " -> ".join(str(n.idx) for n in order))
    L.append("（与 IR binding 序" + ("一致：图无独立就绪分支，串行链）" if same
                                    else "不同：存在可提前就绪的分支，调度器已重排）"))
    L.append("")

    hdr = (f"{'step':>4}  {'KIND':<6} {'NODE':<24} {'OP':<24} "
           f"{'READ(buf)':<26} {'WRITE(buf)':<20} {'LIVE_BYTES'}")
    L.append(hdr)
    L.append("-" * len(hdr))
    for i, n in enumerate(order):
        reads = " ".join(f"{s}@b{assign[s]}" for s in n.inputs if s in assign) or "-"
        write = f"{n.name}@b{assign[n.name]}"
        live = sum(nb for nm, nb in bytes_of.items()
                   if life[nm][0] <= i <= life[nm][1])
        L.append(f"{i:>4}  {n.kind:<6} {n.name:<24} {n.op:<24} "
                 f"{reads:<26} {write:<20} {live:>9} B")

    L.append("")
    L.append("# 内存块复用表（同块内活跃区间两两不交 -> 轮流复用同一 buffer）")
    for bid, b in enumerate(blocks):
        mem = ", ".join(f"{nm}[{life[nm][0]}..{life[nm][1]}]" for nm in b["members"])
        L.append(f"  buf#{bid:<2} {b['size']:>8} B : {mem}")

    L.append("")
    if total:
        L.append(f"# 内存对比: 复用后 arena {arena} B vs 无复用 {total} B "
                 f"-> 节省 {100 * (1 - arena / total):.0f}%")
    L.append(f"# 峰值(按生存期分配、不复用) {peak} B；"
             f"常量张量共 {const_bytes} B（部署时可放只读段，不必占 workspace）")
    return "\n".join(L)


# ---------------------------------------------------------------- 主流程

def main() -> int:
    parser = argparse.ArgumentParser(description="TFLite -> TVM Relax IR / 拓扑 / Pass")
    parser.add_argument("model", nargs="?", default=str(DEFAULT_MODEL),
                        help="TFLite 模型路径（默认 model/mnist_cnn.tflite）")
    parser.add_argument("--no-save", action="store_true", help="只打印，不写文件")
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    if not model_path.exists():
        print(f"[错误] 模型不存在: {model_path}")
        return 1

    tflite_model, size = load_tflite(model_path)
    print(f"模型: {model_path}  ({size / 1024:.1f} KB)")

    # ---- 1. 前端解析 ----------------------------------------------------
    mod = from_tflite(tflite_model)
    ir_text = mod.script()

    # ---- 2. 拓扑结构（优化前，逐算子）-----------------------------------
    topo = extract_topology(mod)
    topo_text = render_topology(topo, "前端拓扑（逐算子，binding 顺序即拓扑序）")
    print(topo_text)
    print()

    # ---- 3. 图优化 Pass --------------------------------------------------
    opt_mod = optimize(mod)
    opt_text = opt_mod.script()
    opt_topo = extract_topology(opt_mod)
    fused = fused_subgraphs(opt_mod)

    print("=" * 78)
    print(f"Pass 后：main 中节点 {len(topo.nodes)} -> {len(opt_topo.nodes)}；"
          f"融合子图 {len(fused)} 个：")
    for fname, comp in fused:
        print(f"  [{comp}] {fname}")
    print("优化后 main 的调用链（FUSED 节点已解析出内部算子序列）：")
    for n in opt_topo.nodes:
        if n.kind in ("OP", "FUSED"):
            inner_txt = f"  <- {','.join(n.inner_ops)}" if n.inner_ops else ""
            print(f"  {n.idx:>2}. {n.kind:<5} {n.op:<24} {n.shape:<16} {n.dtype}{inner_txt}")
    print(f"  OUTPUT: {', '.join(opt_topo.outputs)}")
    print("=" * 78)

    # ---- 4. 拓扑调度（执行序 + 生存期 + buffer 复用）----------------------
    sched_text = schedule(opt_topo, "拓扑调度（Pass 后可执行图）")
    print(sched_text)
    print("=" * 78)

    # ---- 5. 存盘 ---------------------------------------------------------
    if not args.no_save:
        stem = model_path.stem
        base = model_path.parent
        ir_path = base / f"{stem}.relax_ir.txt"
        topo_path = base / f"{stem}.topo.txt"
        opt_path = base / f"{stem}.optimized.txt"
        sched_path = base / f"{stem}.schedule.txt"
        ir_path.write_text(ir_text, encoding="utf-8")
        topo_path.write_text(topo_text, encoding="utf-8")
        opt_path.write_text(opt_text, encoding="utf-8")
        sched_path.write_text(sched_text, encoding="utf-8")
        print(f"前端 IR:   {ir_path}")
        print(f"拓扑结构:  {topo_path}")
        print(f"优化后 IR: {opt_path}")
        print(f"调度表:    {sched_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
