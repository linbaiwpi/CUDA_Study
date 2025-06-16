#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void gemm_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

void my_gemm_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    const int THREAD_X = 16;
    const int THREAD_Y = 16;

    // THREAD_X == 16, time = 11.016152667999268

    dim3 threads(THREAD_Y, THREAD_X);
    dim3 blocks((N + THREAD_Y - 1) / THREAD_Y, (M + THREAD_X - 1) / THREAD_X);

    gemm_kernel<<<blocks, threads>>>(
        A_ptr, B_ptr, C_ptr, M, N, K
    );

    // 同步 CUDA 流
    cudaDeviceSynchronize();
}

