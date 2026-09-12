#include "tml/layers.hpp"
#include "tml/op_registry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace tml {
namespace {

/// 计算 same 填充量。keras 的 same 规则：输出尺寸 ceil(in/stride)，
/// 总填充 pad 在两端不对称分配（后面多 1）。
void same_padding(int in_size, int kernel, int stride, int dilation,
                  int& pad_before, int& pad_after) {
    int effective_k = dilation * (kernel - 1) + 1;
    int out_size = (in_size + stride - 1) / stride;  // ceil
    int pad = std::max(0, (out_size - 1) * stride + effective_k - in_size);
    pad_before = pad / 2;
    pad_after = pad - pad_before;
}

float apply_activation(float x, const std::string& act) {
    if (act == "relu") return x > 0.0f ? x : 0.0f;
    return x;  // linear / softmax(softmax 按整层另行处理)
}

}  // namespace

// ============================ Conv2D ============================

Conv2DLayer::Conv2DLayer(LayerInfo info) : Layer(std::move(info)) {
    if (info_.kernel_shape.size() != 4) {
        throw std::runtime_error("Conv2D kernel_shape 必须是 [KH, KW, Cin, Cout]");
    }
    kh_ = info_.kernel_shape[0];
    kw_ = info_.kernel_shape[1];
    cin_ = info_.kernel_shape[2];
    cout_ = info_.kernel_shape[3];
    sh_ = info_.strides[0];
    sw_ = info_.strides[1];
    dh_ = info_.dilation_rate[0];
    dw_ = info_.dilation_rate[1];
    same_ = (info_.padding == "same");

    if (static_cast<int>(info_.kernel.size()) != kh_ * kw_ * cin_ * cout_) {
        throw std::runtime_error("Conv2D 权重数量与 kernel_shape 不符: " + info_.name);
    }
    if (static_cast<int>(info_.bias.size()) != cout_) {
        throw std::runtime_error("Conv2D 偏置数量与输出通道不符: " + info_.name);
    }
}

void Conv2DLayer::forward(const Tensor& in, Tensor& out) {
    int H = static_cast<int>(in.shape[0]);
    int W = static_cast<int>(in.shape[1]);

    int pad_top = 0, pad_bottom = 0, pad_left = 0, pad_right = 0;
    if (same_) {
        same_padding(H, kh_, sh_, dh_, pad_top, pad_bottom);
        same_padding(W, kw_, sw_, dw_, pad_left, pad_right);
    }
    int out_h = (H + pad_top + pad_bottom - dh_ * (kh_ - 1) - 1) / sh_ + 1;
    int out_w = (W + pad_left + pad_right - dw_ * (kw_ - 1) - 1) / sw_ + 1;
    out.resize({static_cast<size_t>(out_h), static_cast<size_t>(out_w),
                static_cast<size_t>(cout_)});

    for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {
            for (int oc = 0; oc < cout_; ++oc) {
                float acc = info_.bias[oc];
                for (int ic = 0; ic < cin_; ++ic) {
                    for (int p = 0; p < kh_; ++p) {
                        for (int q = 0; q < kw_; ++q) {
                            int ih = oh * sh_ - pad_top + p * dh_;
                            int iw = ow * sw_ - pad_left + q * dw_;
                            float x = in.at3d(ih, iw, ic);  // 越界自动补 0
                            // 权重排布 [KH, KW, Cin, Cout] 行主序展平
                            float w = info_.kernel[((p * kw_ + q) * cin_ + ic) * cout_ + oc];
                            acc += x * w;
                        }
                    }
                }
                out.data[(oh * out_w + ow) * cout_ + oc] =
                    apply_activation(acc, info_.activation);
            }
        }
    }
}

// ======================== MaxPooling2D =========================

MaxPooling2DLayer::MaxPooling2DLayer(LayerInfo info) : Layer(std::move(info)) {
    ph_ = info_.pool_size[0];
    pw_ = info_.pool_size[1];
    sh_ = info_.strides[0];
    sw_ = info_.strides[1];
    same_ = (info_.padding == "same");
}

void MaxPooling2DLayer::forward(const Tensor& in, Tensor& out) {
    int H = static_cast<int>(in.shape[0]);
    int W = static_cast<int>(in.shape[1]);
    int C = static_cast<int>(in.shape[2]);

    int pad_top = 0, pad_bottom = 0, pad_left = 0, pad_right = 0;
    if (same_) {
        same_padding(H, ph_, sh_, 1, pad_top, pad_bottom);
        same_padding(W, pw_, sw_, 1, pad_left, pad_right);
    }
    int out_h = (H + pad_top + pad_bottom - ph_) / sh_ + 1;
    int out_w = (W + pad_left + pad_right - pw_) / sw_ + 1;
    out.resize({static_cast<size_t>(out_h), static_cast<size_t>(out_w),
                static_cast<size_t>(C)});

    for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {
            for (int c = 0; c < C; ++c) {
                float best = -std::numeric_limits<float>::infinity();
                for (int p = 0; p < ph_; ++p) {
                    for (int q = 0; q < pw_; ++q) {
                        int ih = oh * sh_ - pad_top + p;
                        int iw = ow * sw_ - pad_left + q;
                        best = std::max(best, in.at3d(ih, iw, c));
                    }
                }
                out.data[(oh * out_w + ow) * C + c] = best;
            }
        }
    }
}

// ============================= Dense ===========================

DenseLayer::DenseLayer(LayerInfo info) : Layer(std::move(info)) {
    if (info_.kernel_shape.size() != 2) {
        throw std::runtime_error("Dense kernel_shape 必须是 [In, Out]");
    }
    in_units_ = info_.kernel_shape[0];
    out_units_ = info_.kernel_shape[1];
    if (static_cast<int>(info_.kernel.size()) != in_units_ * out_units_) {
        throw std::runtime_error("Dense 权重数量与 kernel_shape 不符: " + info_.name);
    }
}

void DenseLayer::forward(const Tensor& in, Tensor& out) {
    if (static_cast<int>(in.data.size()) != in_units_) {
        throw std::runtime_error("Dense 输入维度不符: " + info_.name);
    }
    out.resize({static_cast<size_t>(out_units_)});

    for (int o = 0; o < out_units_; ++o) {
        float acc = o < static_cast<int>(info_.bias.size()) ? info_.bias[o] : 0.0f;
        for (int i = 0; i < in_units_; ++i) {
            acc += in.data[i] * info_.kernel[i * out_units_ + o];  // [In, Out]
        }
        out.data[o] = acc;
    }

    if (info_.activation == "relu") {
        for (float& v : out.data) v = apply_activation(v, "relu");
    } else if (info_.activation == "softmax") {
        float max_v = *std::max_element(out.data.begin(), out.data.end());
        float sum = 0.0f;
        for (float& v : out.data) {
            v = std::exp(v - max_v);  // 减最大值防溢出
            sum += v;
        }
        for (float& v : out.data) v /= sum;
    }
}

// ============================ Flatten ==========================

void FlattenLayer::forward(const Tensor& in, Tensor& out) {
    out.resize({in.data.size()});
    out.data = in.data;  // 数据顺序不变，仅改变形状语义
}

// ============================ Dropout ==========================

void DropoutLayer::forward(const Tensor& in, Tensor& out) {
    // 正常情况下不会被调用：Graph 通过 is_identity() 跳过本层。
    // 保留实现以保证 Layer 接口语义完整（如单独测试时）。
    (void)in;
    (void)out;
}

// ==================== 静态自注册（无需改动注册表）====================

TML_REGISTER_LAYER("Conv2D", Conv2DLayer)
TML_REGISTER_LAYER("MaxPooling2D", MaxPooling2DLayer)
TML_REGISTER_LAYER("Dense", DenseLayer)
TML_REGISTER_LAYER("Flatten", FlattenLayer)
TML_REGISTER_LAYER("Dropout", DropoutLayer)

}  // namespace tml
