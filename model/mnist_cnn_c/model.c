#include <string.h>
#include "k2c_include.h"
#include "weights.h"
#include "model.h"

/* offset 级单池 arena：所有中间张量按字节偏移共存于此 */
static float arena[ARENA_FLOATS];

static const size_t stride0[2] = {1, 1};
static const size_t dil1[2] = {1, 1};
static const size_t pad2[4] = {1, 1, 1, 1};
static const size_t pool3[2] = {2, 2};
static const size_t pstride4[2] = {2, 2};
static const size_t shape5[2] = {1, 3136};

void forward(const float* in, float* out) {
    k2c_tensor tin = {(float*)in, 3, 784, {28, 28, 1, 0, 0}};
    k2c_tensor tout = {out, 2, 10, {1, 10, 0, 0, 0}};
    k2c_tensor t1 = {&arena[0], 3, 25088, {28, 28, 32, 0, 0}};
    k2c_tensor tp1 = {&arena[25088], 3, 900, {30, 30, 1, 0, 0}};
    k2c_tensor t2 = {&arena[25088], 3, 6272, {14, 14, 32, 0, 0}};
    k2c_tensor t3 = {&arena[0], 3, 12544, {14, 14, 64, 0, 0}};
    k2c_tensor tp3 = {&arena[12544], 3, 8192, {16, 16, 32, 0, 0}};
    k2c_tensor t4 = {&arena[12544], 3, 3136, {7, 7, 64, 0, 0}};
    k2c_tensor t5 = {&arena[0], 2, 3136, {1, 3136, 0, 0, 0}};
    k2c_tensor t6 = {&arena[3136], 2, 128, {1, 128, 0, 0, 0}};
    k2c_tensor t7 = {&arena[0], 2, 10, {1, 10, 0, 0, 0}};
    k2c_tensor t8 = {&arena[10], 2, 10, {1, 10, 0, 0, 0}};

    k2c_pad2d(&tp1, &tin, 0.0f, pad2);
    k2c_conv2d(&t1, &tp1, &wk0, &wb0, stride0, dil1, k2c_relu);
    k2c_maxpool2d(&t2, &t1, pool3, pstride4);
    k2c_pad2d(&tp3, &t2, 0.0f, pad2);
    k2c_conv2d(&t3, &tp3, &wk1, &wb1, stride0, dil1, k2c_relu);
    k2c_maxpool2d(&t4, &t3, pool3, pstride4);
    k2c_reshape(&t5, &t4, shape5, 2);
    k2c_dense(&t6, &t5, &wk2, &wb2, k2c_relu, arena);
    k2c_matmul(t7.array, t6.array, wk3.array, 1, 10, 128);
    memcpy(t8.array, t7.array, 10 * sizeof(float));
    k2c_bias_add(&t8, &wb3);
    memcpy(tout.array, t8.array, 10 * sizeof(float));
    k2c_softmax(tout.array, tout.numel);
}
