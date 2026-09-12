#pragma once

#include "layer.hpp"

namespace tml {

/// Conv2D（NHWC，支持 same/valid、空洞卷积、relu/linear 激活、偏置）
class Conv2DLayer : public Layer {
public:
    explicit Conv2DLayer(LayerInfo info);
    void forward(const Tensor& in, Tensor& out) override;

private:
    int kh_, kw_, cin_, cout_;
    int sh_, sw_, dh_, dw_;
    bool same_;
};

/// MaxPooling2D（NHWC，支持 same/valid）
class MaxPooling2DLayer : public Layer {
public:
    explicit MaxPooling2DLayer(LayerInfo info);
    void forward(const Tensor& in, Tensor& out) override;

private:
    int ph_, pw_, sh_, sw_;
    bool same_;
};

/// Dense 全连接层（linear / relu / softmax）
class DenseLayer : public Layer {
public:
    explicit DenseLayer(LayerInfo info);
    void forward(const Tensor& in, Tensor& out) override;

private:
    int in_units_, out_units_;
};

/// Flatten：保持数据顺序不变，把任意形状展平为一维
class FlattenLayer : public Layer {
public:
    explicit FlattenLayer(LayerInfo info) : Layer(std::move(info)) {}
    void forward(const Tensor& in, Tensor& out) override;
};

/// Dropout：推理阶段无计算。is_identity 让 Graph 跳过它，
/// 仅交换两块工作缓冲的读写角色（零数据搬运）。
class DropoutLayer : public Layer {
public:
    explicit DropoutLayer(LayerInfo info) : Layer(std::move(info)) {}
    void forward(const Tensor& in, Tensor& out) override;
    bool is_identity() const override { return true; }
};

}  // namespace tml
