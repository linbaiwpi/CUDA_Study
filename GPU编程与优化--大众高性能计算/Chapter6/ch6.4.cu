#include "ch6.h"
#include <stdio.h>

__global__ void vector_add_gpu_3(DATATYPE *a, DATATYPE *b, DATATYPE *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < n) {
        c[idx] = a[idx] + b[idx];
        printf("%d: %f + %f = %f\n", idx, a[idx], b[idx], c[idx]);
        idx += blockDim.x * gridDim.x;
    }
}