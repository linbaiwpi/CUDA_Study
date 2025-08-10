#pragma once
#ifndef __CH7_H__
#define __CH7_H__

#include <cuda_runtime.h>

#define N           1000
#define DATATYPE    float

__global__ void vector_dot_product_gpu_1(DATATYPE *a, DATATYPE *b, DATATYPE *c, int n);
__global__ void vector_dot_product_gpu_2(DATATYPE *a, DATATYPE *b, DATATYPE *c, int n);

#endif // __CH7_H__
