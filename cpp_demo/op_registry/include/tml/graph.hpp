#pragma once

#include "tml/layer.hpp"

#include <memory>
#include <vector>

namespace tml {

/// 计算图：持有按拓扑顺序排列的算子，用两块静态工作缓冲做乒乓交换。
///
/// 这与项目中代码生成器产出的 forward() 完全同构：
///   - 每层只读“上一层输出”、整体覆盖“另一块缓冲”
///   - 恒等层（Dropout）不搬运数据，仅交换读写角色
///   - 峰值内存 = max(各层输出) × 2，与网络层数解耦
class Graph {
public:
    void add_layer(std::unique_ptr<Layer> layer) {
        layers_.push_back(std::move(layer));
    }

    bool empty() const { return layers_.empty(); }

    /// 执行整条网络，返回最终输出张量（引用指向内部工作缓冲，下次调用失效）
    const Tensor& forward(const Tensor& input) {
        const Tensor* read = &input;
        for (auto& layer : layers_) {
            if (layer->is_identity()) {
                continue;  // 零开销：read 保持不变，下一层自动写另一块缓冲
            }
            // 选一块与当前输入不别名的工作缓冲作为输出
            Tensor* write = (read == &work_[0]) ? &work_[1] : &work_[0];
            layer->forward(*read, *write);
            read = write;
        }
        return *read;
    }

private:
    std::vector<std::unique_ptr<Layer>> layers_;
    Tensor work_[2];  // 双工作缓冲（ta / tb）
};

}  // namespace tml
