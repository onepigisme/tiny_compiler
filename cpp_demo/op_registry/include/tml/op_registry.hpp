#pragma once

#include "layer.hpp"

#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace tml {

/// 工厂函数签名：由层描述信息构造一个算子实例
using LayerFactory = std::function<std::unique_ptr<Layer>(const LayerInfo&)>;

/// 算子注册表（单例）。
///
/// 对应 Python 版 op_registry.py 的 dict 分发，但采用工业界推理框架
/// （NCNN / MNN / TFLite）通用的“字符串 key → 工厂函数”自注册模式：
/// 新增算子只需实现 Layer 子类并在 .cpp 中写一行 TML_REGISTER_LAYER，
/// 无需修改注册表本体，符合开闭原则。
class OpRegistry {
public:
    static OpRegistry& instance() {
        static OpRegistry registry;
        return registry;
    }

    /// 注册一个算子，重名返回 false
    bool register_op(const std::string& op_type, LayerFactory factory) {
        return factories_.emplace(op_type, std::move(factory)).second;
    }

    /// 按类型名创建算子；未注册时打印错误并返回 nullptr（由调用方决定容错策略）
    std::unique_ptr<Layer> create(const std::string& op_type, const LayerInfo& info) const {
        auto it = factories_.find(op_type);
        if (it == factories_.end()) {
            std::cerr << "[OpRegistry] 未注册的算子类型: " << op_type << '\n';
            return nullptr;
        }
        return it->second(info);
    }

    bool contains(const std::string& op_type) const {
        return factories_.find(op_type) != factories_.end();
    }

    /// 列出所有已注册算子（调试/启动信息用）
    std::vector<std::string> registered_ops() const {
        std::vector<std::string> names;
        names.reserve(factories_.size());
        for (const auto& [name, _] : factories_) names.push_back(name);
        return names;
    }

private:
    OpRegistry() = default;
    std::unordered_map<std::string, LayerFactory> factories_;
};

}  // namespace tml

/// 静态自注册宏：在 .cpp 文件中使用，程序启动前自动把算子注册进表单。
///   TML_REGISTER_LAYER("Conv2D", Conv2DLayer)
#define TML_REGISTER_LAYER(OP_TYPE, CLASS)                                                   \
    namespace {                                                                              \
    const bool tml_reg_##CLASS = ::tml::OpRegistry::instance().register_op(                  \
        OP_TYPE, [](const ::tml::LayerInfo& info) {                                          \
            return std::unique_ptr<::tml::Layer>(new CLASS(info));                           \
        });                                                                                  \
    }
