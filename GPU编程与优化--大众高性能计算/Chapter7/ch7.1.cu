#include "ch7.h"
// #include <iostream>

__global__ void vector_dot_product_gpu_1(DATATYPE* a, DATATYPE* b, DATATYPE* c, int n) {
    const int threadnum = 128;
    __shared__ DATATYPE tmp[threadnum];
    const int tidx = threadIdx.x;
    const int t_n = blockDim.x;
    int tid = tidx;
    double temp = 0.0;
    while(tid < n) {
        temp += a[tidx] * b[tidx];
        tid += t_n;
    }
    tmp[tidx] = temp;
    __syncthreads();

    int i=2;
    int j=1;
    while (i<=threadnum) {
        if (tidx % i == 0) {
            tmp[tidx] += tmp[tidx+j];
        }
        __syncthreads();
        i *= 2;
        j *= 2;
    }

    if (tidx == 0) {
        c[0] = tmp[0];
    }
}
