#include "ch7.h"
// #include <iostream>

__global__ void vector_dot_product_gpu_2(DATATYPE* a, DATATYPE* b, DATATYPE* c, int n) {
    const int threadnum = 128;
    __shared__ DATATYPE tmp[threadnum];
    int tidx = threadIdx.x;
    int t_n = blockDim.x;
    int tid = tidx;
    double temp = 0.0;
    while (tid < n) {
        temp += a[tid] * b[tid];
        tid += t_n;
    }
    tmp[tidx] = temp;
    __syncthreads();

    int i=threadnum/2;
    while (i != 0) {
        if (tidx < i) {
            tmp[tidx] += tmp[tidx+i];
        }
        __syncthreads();
        i /= 2;
    }

    if (tidx == 0) {
        c[0] = tmp[0];
    }
}
