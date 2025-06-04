#include "my_gemm.h"

void my_gemm_cpu(torch::Tensor A, torch::Tensor B, torch::Tensor& C) {
    // 简单的矩阵乘法，使用 AT_DISPATCH_FLOATING_TYPES 支持 float/double 等
    TORCH_CHECK(A.device().is_cpu(), "A must be a CPU tensor");
    TORCH_CHECK(B.device().is_cpu(), "B must be a CPU tensor");

    C = torch::matmul(A, B);
}
