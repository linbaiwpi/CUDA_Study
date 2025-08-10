#include "ch7.h"

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <cublas_v2.h>

// #define DEBU

int main(int argc, char** argv) {
    // CPU memory alloc
    DATATYPE* a = (DATATYPE *)malloc(sizeof(DATATYPE) * N);
    DATATYPE* b = (DATATYPE *)malloc(sizeof(DATATYPE) * N);
    DATATYPE* c = (DATATYPE *)malloc(sizeof(DATATYPE) * N);

    // random data generation
    srand((unsigned int)time(NULL));
#ifdef DEBUG
    srand(0);
#endif
    for (int i=0; i<N; ++i) {
        a[i] = ((float)rand() / RAND_MAX) * 100.0f;
        b[i] = ((float)rand() / RAND_MAX) * 100.0f;
    }

    std::cout << "a = ";
    for (int i=0; i<5; ++i) {
        std::cout << a[i] << ", ";
    }
    std::cout << std::endl;
    std::cout << "b = ";
    for (int i=0; i<5; ++i) {
        std::cout << b[i] << ", ";
    }
    std::cout << std::endl;

    std::cout << argv[0]<< std::endl;
    std::cout << argv[1]<< std::endl;

    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0;

    double sum = 0;
    for (int i=0; i<N; ++i) {
        sum += a[i] * b[i];
    }
    std::cout << "Ref sum = " << sum << std::endl;

    std::cout << "GPU reference" << std::endl;
    // GPU memory alloc
    DATATYPE *d_a, *d_b, *d_c;
    cudaMalloc((void **) &d_a, sizeof(DATATYPE) * N);
    cudaMalloc((void **) &d_b, sizeof(DATATYPE) * N);
    cudaMalloc((void **) &d_c, sizeof(DATATYPE) * N);


    if (strcmp(argv[1], "cublas") == 0) {
        cublasSetVector(N, sizeof(DATATYPE), a, 1, d_a, 1);
        cublasSetVector(N, sizeof(DATATYPE), b, 1, d_b, 1);
    } else {
        // data a and b copy to GPU
        cudaMemcpy(d_a, a, sizeof(DATATYPE) * N, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, sizeof(DATATYPE) * N, cudaMemcpyHostToDevice);
    }

    // c = a + b
    int thereadnum = 128;
    if (strcmp(argv[1], "7.1") == 0) {        
        vector_dot_product_gpu_1<<<1,thereadnum>>>(d_a, d_b, d_c, N);
    } else if (strcmp(argv[1], "7.2") == 0) {
        vector_dot_product_gpu_2<<<1,thereadnum>>>(d_a, d_b, d_c, N);
    // } else if (strcmp(argv[1], "6.4") == 0) {
    //     int blocknum = 10;
    //     int threadnum = 10;
    //     vector_add_gpu_3<<<blocknum, threadnum>>>(d_a, d_b, d_c, N);
    // } else if (strcmp(argv[1], "cublas") == 0) {
    //     std::cout << "Calling cublas" << std::endl;
    //     cublasSaxpy_v2(handle, N, &alpha, d_a, 1, d_b, 1);
    } else {
        std::cout << "" << std::endl;
    }

    // result copy back to CPU
    if (strcmp(argv[1], "cublas") == 0) {
        cublasGetVector(N, sizeof(DATATYPE), d_b, 1, c, 1);
    } else {
        cudaMemcpy(c, d_c, sizeof(DATATYPE) * N, cudaMemcpyDeviceToHost);
    }

    std::cout << "c = ";
    for (int i=0; i<5; ++i) {
        std::cout << c[i] << ", ";
    }
    std::cout << std::endl;

    // GPU memory free
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    cublasDestroy(handle);
}
