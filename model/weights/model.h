#pragma once
#include <stddef.h>
#include "k2c_include.h"

extern const size_t conv2d_stride[2];
extern const size_t conv2d_dilation[2];
extern const size_t conv2d_padding[4];
extern k2c_activationType * conv2d_activation;
extern const size_t max_pooling2d_pool_size[2];
extern const size_t max_pooling2d_strides[2];
extern const size_t conv2d_1_stride[2];
extern const size_t conv2d_1_dilation[2];
extern const size_t conv2d_1_padding[4];
extern k2c_activationType * conv2d_1_activation;
extern const size_t max_pooling2d_1_pool_size[2];
extern const size_t max_pooling2d_1_strides[2];
extern k2c_activationType * dense_activation;
extern k2c_activationType * dense_1_activation;

/* 模型前向推理入口（model_invoke.c） */
int forward(const k2c_tensor* input, k2c_tensor** output);
