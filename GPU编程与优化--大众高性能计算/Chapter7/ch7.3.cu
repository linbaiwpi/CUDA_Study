#include "ch7.h"
#include <iostream>

__global__ void vector_dot_product_gpu_3(DATATYPE* a, DATATYPE* b, DATATYPE* c, int n) {
    const int threadnum = 128;
    __shared__ DATATYPE tmp[threadnum];
    int tidx = threadIdx.x;
    int t_n = gridDim.x * blockDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    double temp = 0.0;

    while (tid < n) {
        temp += a[tid] * b[tid];
        tid += t_n;
    }
    tmp[tidx] = temp;
    __syncthreads();

    int i = threadnum / 2;
    while (i != 0) {
        if (tidx < i) {
            tmp[tidx] += tmp[tidx+i];
        }
        __syncthreads();
        i /= 2;
    }

    if (tidx == 0) {
        c[blockIdx.x] = tmp[0];
    }
}

DATATYPE vector_dot_product_cpu_3(DATATYPE* c, int n) {
    DATATYPE result = 0;
    for (int i=0; i<n; ++i) {
        result += c[i];
    }
    return result;
}
