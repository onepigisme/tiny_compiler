#ifndef MODEL_H
#define MODEL_H

#define ARENA_FLOATS 31360   /* 125440 B */
#define IN_SIZE 784
#define OUT_SIZE 10

void forward(const float* in, float* out);

#endif
