"""IR 层单元测试。"""

from tiny_ml_compiler.ir import Graph, GraphInput, GraphOutput, Node, TensorType


def test_tensor_type_basic():
    t = TensorType(shape=[1, 3, 224, 224], dtype="float32")
    assert t.rank == 4
    assert t.is_static is True
    assert t.num_elements() == 1 * 3 * 224 * 224


def test_tensor_type_dynamic():
    t = TensorType(shape=[None, 3, 224, 224])
    assert t.is_static is False
    assert t.num_elements() is None


def test_graph_topo_order():
    """构建 Conv -> Relu -> Flatten 的小图，验证拓扑排序。"""
    g = Graph(name="test")
    g.inputs.append(GraphInput(name="x", dtype=TensorType([1, 3, 8, 8])))
    g.inputs.append(GraphInput(name="w", dtype=TensorType([16, 3, 3, 3])))

    g.add_node(Node(name="conv", op_type="Conv", inputs=["x", "w"], outputs=["conv_out"]))
    g.add_node(Node(name="relu", op_type="Relu", inputs=["conv_out"], outputs=["relu_out"]))
    g.add_node(Node(name="flatten", op_type="Flatten", inputs=["relu_out"], outputs=["y"]))

    g.outputs.append(GraphOutput(name="y", dtype=TensorType([1, 1024])))

    order = g.topo_order()
    names = [n.name for n in order]
    assert names == ["conv", "relu", "flatten"]
    assert len(g) == 3


def test_graph_producer():
    g = Graph(name="test")
    g.add_node(Node(name="conv", op_type="Conv", inputs=["x"], outputs=["conv_out"]))
    assert g.producer_of("conv_out").name == "conv"
    assert g.producer_of("x") is None  # 图输入


def test_dead_node_elim():
    """测试死节点消除 Pass。"""
    from tiny_ml_compiler.optimizer import DeadNodeElimPass

    g = Graph(name="test")
    g.add_node(Node(name="conv", op_type="Conv", inputs=["x"], outputs=["conv_out"]))
    g.add_node(Node(name="dead", op_type="Relu", inputs=["conv_out"], outputs=["dead_out"]))
    g.outputs.append(GraphOutput(name="conv_out", dtype=TensorType([1, 16, 8, 8])))

    # dead 节点的输出 dead_out 没人消费，应被删除
    pm = DeadNodeElimPass()
    g = pm.run(g)
    names = [n.name for n in g.nodes]
    assert "dead" not in names
    assert "conv" in names


def test_c_backend_generates_code(tmp_path):
    """测试 C 后端能生成代码文件。"""
    from tiny_ml_compiler.backend import CBackend

    g = Graph(name="test")
    g.inputs.append(GraphInput(name="x", dtype=TensorType([1, 3])))
    g.add_node(Node(name="relu", op_type="Relu", inputs=["x"], outputs=["y"]))
    g.outputs.append(GraphOutput(name="y", dtype=TensorType([1, 3])))

    out_file = tmp_path / "model.c"
    backend = CBackend()
    result = backend.compile(g, out_file)

    code = open(result, encoding="utf-8").read()
    assert "model_invoke" in code
    assert "Relu" in code
