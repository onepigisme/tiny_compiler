#include "model.h"

const size_t conv2d_stride[2] = {1, 1};
const size_t conv2d_dilation[2] = {1, 1};
const size_t conv2d_padding[4] = {1, 1, 1, 1};
k2c_activationType * conv2d_activation = &k2c_relu;

const size_t max_pooling2d_pool_size[2] = {2, 2};
const size_t max_pooling2d_strides[2] = {2, 2};

const size_t conv2d_1_stride[2] = {1, 1};
const size_t conv2d_1_dilation[2] = {1, 1};
const size_t conv2d_1_padding[4] = {1, 1, 1, 1};
k2c_activationType * conv2d_1_activation = &k2c_relu;

const size_t max_pooling2d_1_pool_size[2] = {2, 2};
const size_t max_pooling2d_1_strides[2] = {2, 2};

k2c_activationType * dense_activation = &k2c_relu;

k2c_activationType * dense_1_activation = &k2c_softmax;

