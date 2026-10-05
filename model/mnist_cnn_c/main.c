/* 生成的推理入口：读 input.bin -> forward() -> 写 output.bin */
#include <stdio.h>
#include "model.h"

int main(int argc, char** argv) {
    const char* in_path = argc > 1 ? argv[1] : "input.bin";
    const char* out_path = argc > 2 ? argv[2] : "output.bin";

    static float in_buf[IN_SIZE];
    static float out_buf[OUT_SIZE];

    FILE* f = fopen(in_path, "rb");
    if (!f || fread(in_buf, sizeof(float), IN_SIZE, f) != IN_SIZE) {
        fprintf(stderr, "读取 %s 失败\n", in_path);
        return 1;
    }
    fclose(f);

    forward(in_buf, out_buf);

    FILE* g = fopen(out_path, "wb");
    if (!g || fwrite(out_buf, sizeof(float), OUT_SIZE, g) != OUT_SIZE) {
        fprintf(stderr, "写入 %s 失败\n", out_path);
        return 1;
    }
    fclose(g);
    for (int i = 0; i < OUT_SIZE; ++i) {
        printf("out[%d] = %.6f\n", i, out_buf[i]);
    }
    return 0;
}
