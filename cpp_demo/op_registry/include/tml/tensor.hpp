#pragma once

#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace tml {

/// 最小张量实现，字段布局与 keras2c 的 k2c_tensor 对应：
/// array(数据) + shape(维度) + numel(元素总数)
struct Tensor {
    std::vector<float> data;
    std::vector<size_t> shape;

    size_t numel() const {
        size_t n = 1;
        for (size_t d : shape) n *= d;
        return n;
    }

    /// 按新形状分配并清零
    void resize(const std::vector<size_t>& new_shape) {
        shape = new_shape;
        data.assign(numel(), 0.0f);
    }

    /// 3D NHWC 布局 (H, W, C) 的安全访问，越界返回 0（same padding 补零）
    float at3d(int h, int w, int c) const {
        if (shape.size() != 3) {
            throw std::runtime_error("at3d 仅支持 3D NHWC 张量");
        }
        int H = static_cast<int>(shape[0]);
        int W = static_cast<int>(shape[1]);
        int C = static_cast<int>(shape[2]);
        if (h < 0 || h >= H || w < 0 || w >= W || c < 0 || c >= C) return 0.0f;
        return data[(static_cast<size_t>(h) * W + w) * C + c];
    }
};

}  // namespace tml
