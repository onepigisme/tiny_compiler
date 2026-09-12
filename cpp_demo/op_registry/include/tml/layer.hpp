#pragma once

#include "tensor.hpp"

#include <array>
#include <string>
#include <utility>
#include <vector>

namespace tml {

/// 层描述信息，对应 Python 版 op_parse.py 中分发给 handler 的 info 字典。
/// 用强类型字段代替 dict，编译期即可发现字段拼写错误。
struct LayerInfo {
    std::string op_type;   // 算子类型名，必须与注册表 key 一致，如 "Conv2D"
    std::string name;      // 层名，如 conv2d / dense_1
    bool is_odd_layer = true;  // 层奇偶性（生成代码中乒乓缓冲的读写依据）

    // ---- Conv2D / MaxPooling2D 超参数 ----
    std::array<int, 2> kernel_size{1, 1};    // {KH, KW}
    std::array<int, 2> strides{1, 1};        // {SH, SW}
    std::array<int, 2> dilation_rate{1, 1};  // {DH, DW}
    std::array<int, 2> pool_size{2, 2};      // MaxPooling 池化窗
    std::string padding{"valid"};            // "same" / "valid"

    // ---- Dense ----
    std::string activation{"linear"};  // linear / relu / softmax
    int units = 0;                     // Dense 输出维度

    // ---- 权重（展平的一维存储，配合 kernel_shape 还原维度）----
    std::vector<float> kernel;
    std::vector<float> bias;
    std::vector<int> kernel_shape;  // Conv2D: [KH,KW,Cin,Cout]，Dense: [In,Out]
};

/// 所有算子的抽象基类。
/// 约定：forward 完全覆盖 out，因此调用方可以用两块工作缓冲做乒乓交换，
/// 而无需为每层单独分配输出内存。
class Layer {
public:
    explicit Layer(LayerInfo info) : info_(std::move(info)) {}
    virtual ~Layer() = default;

    /// 读 in、写 out（out 的旧数据无意义，可被整体覆盖）
    virtual void forward(const Tensor& in, Tensor& out) = 0;

    /// 推理阶段的恒等层（如 Dropout）不产生任何数据搬运，
    /// Graph 直接跳过它并交换读写角色，等价于 C 代码里的 k2c_tensor_swap。
    virtual bool is_identity() const { return false; }

    const std::string& op_type() const { return info_.op_type; }
    const std::string& name() const { return info_.name; }

protected:
    LayerInfo info_;
};

}  // namespace tml
