// op_registry C++ 版演示：
//   1. 通过字符串名从注册表创建算子（对应 Keras 层类名分发）
//   2. 组装一个 MNIST CNN 的缩小版网络
//   3. 用双工作缓冲乒乓交换跑一次前向推理
//   4. 演示未注册算子的容错处理
//
// 网络：Conv2D(3x3,same,2ch,relu) -> MaxPool(2x2) -> Conv2D(3x3,same,4ch,relu)
//       -> MaxPool(2x2) -> Flatten -> Dense(4->3,relu) -> Dropout -> Dense(3->2,softmax)

#include "tml/graph.hpp"
#include "tml/layers.hpp"
#include "tml/op_registry.hpp"

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

using namespace tml;

namespace {

// 构造一份确定性的“伪权重”，让 demo 无需外部文件即可运行
std::vector<float> make_weights(size_t n, float seed) {
    std::vector<float> w(n);
    for (size_t i = 0; i < n; ++i) {
        w[i] = (static_cast<int>((i * 7 + static_cast<size_t>(seed * 10)) % 11) - 5) * 0.1f;
    }
    return w;
}

LayerInfo make_conv(const std::string& name, int cin, int cout, bool odd) {
    LayerInfo info;
    info.op_type = "Conv2D";
    info.name = name;
    info.is_odd_layer = odd;
    info.kernel_size = {3, 3};
    info.strides = {1, 1};
    info.dilation_rate = {1, 1};
    info.padding = "same";
    info.activation = "relu";
    info.kernel_shape = {3, 3, cin, cout};
    info.kernel = make_weights(3 * 3 * cin * cout, static_cast<float>(cout) * 0.3f);
    info.bias = make_weights(cout, 1.0f);
    return info;
}

void print_tensor(const std::string& tag, const Tensor& t) {
    std::cout << tag << " shape=[";
    for (size_t i = 0; i < t.shape.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << t.shape[i];
    }
    std::cout << "] 前 8 个值: ";
    for (size_t i = 0; i < std::min<size_t>(8, t.data.size()); ++i) {
        std::cout << t.data[i] << " ";
    }
    std::cout << "\n";
}

}  // namespace

int main() {
    OpRegistry& registry = OpRegistry::instance();

    std::cout << "已注册算子: ";
    for (const auto& op : registry.registered_ops()) std::cout << op << " ";
    std::cout << "\n\n";

    // ---- 通过注册表构建网络（实际场景中 op_type 来自模型文件解析）----
    Graph graph;

    auto add = [&](const LayerInfo& info) {
        auto layer = registry.create(info.op_type, info);
        if (!layer) {
            std::cerr << "构建失败，跳过: " << info.name << "\n";
            return;
        }
        graph.add_layer(std::move(layer));
    };

    add(make_conv("conv2d", /*cin*/ 1, /*cout*/ 2, /*odd*/ true));

    LayerInfo pool1;
    pool1.op_type = "MaxPooling2D";
    pool1.name = "max_pooling2d";
    pool1.pool_size = {2, 2};
    pool1.strides = {2, 2};
    pool1.padding = "valid";
    pool1.is_odd_layer = false;
    add(pool1);

    add(make_conv("conv2d_1", /*cin*/ 2, /*cout*/ 4, /*odd*/ true));

    LayerInfo pool2 = pool1;
    pool2.name = "max_pooling2d_1";
    pool2.is_odd_layer = true;
    add(pool2);

    LayerInfo flatten;
    flatten.op_type = "Flatten";
    flatten.name = "flatten";
    flatten.is_odd_layer = false;
    add(flatten);

    LayerInfo dense1;
    dense1.op_type = "Dense";
    dense1.name = "dense";
    dense1.is_odd_layer = true;
    dense1.units = 3;
    dense1.activation = "relu";
    dense1.kernel_shape = {4, 3};
    dense1.kernel = make_weights(12, 2.0f);
    dense1.bias = make_weights(3, 0.5f);
    add(dense1);

    LayerInfo dropout;
    dropout.op_type = "Dropout";
    dropout.name = "dropout";
    dropout.is_odd_layer = false;
    add(dropout);

    LayerInfo dense2;
    dense2.op_type = "Dense";
    dense2.name = "dense_1";
    dense2.is_odd_layer = true;
    dense2.units = 2;
    dense2.activation = "softmax";
    dense2.kernel_shape = {3, 2};
    dense2.kernel = make_weights(6, 3.0f);
    dense2.bias = make_weights(2, 0.2f);
    add(dense2);

    // ---- 构造输入 4x4x1 ----
    Tensor input;
    input.resize({4, 4, 1});
    for (size_t i = 0; i < input.data.size(); ++i) {
        input.data[i] = static_cast<float>(i) / 15.0f;  // 0 ~ 1
    }
    print_tensor("input       ", input);

    // ---- 前向 ----
    const Tensor& output = graph.forward(input);
    print_tensor("output      ", output);

    // softmax 概率和应为 1
    float prob_sum = 0.0f;
    for (float v : output.data) prob_sum += v;
    std::cout << "softmax 概率和: " << prob_sum << " (应≈1)\n\n";

    // ---- 容错：未注册算子 ----
    LayerInfo bad;
    bad.op_type = "Reshape";
    bad.name = "reshape_x";
    auto maybe = registry.create("Reshape", bad);
    std::cout << "创建未注册算子 Reshape 返回空指针: " << (maybe == nullptr ? "是 ✓" : "否 ✗")
              << "\n";

    return 0;
}
