#pragma once
#ifndef __CH6_H__
#define __CH6_H__

#include <cuda_runtime.h>

#define N           100
#define DATATYPE    float

void vector_add_serial(DATATYPE* a, DATATYPE* b, DATATYPE* c, int n);
__global__ void vector_add_gpu_1(DATATYPE *a, DATATYPE *b, DATATYPE *c, int n);
__global__ void vector_add_gpu_2(DATATYPE *a, DATATYPE *b, DATATYPE *c, int n);
__global__ void vector_add_gpu_3(DATATYPE *a, DATATYPE *b, DATATYPE *c, int n);

#endif // __CH6_H__
