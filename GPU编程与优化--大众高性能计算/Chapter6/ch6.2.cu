#include "ch6.h"
// #include <stdio.h>

__global__ void vector_add_gpu_1(DATATYPE *a, DATATYPE *b, DATATYPE *c, int n) {

    for (int i=0; i<n; i++) {
        c[i] = a[i] + b[i];
        // printf("%f + %f = %f\n", a[i], b[i], c[i]);
    }
}
