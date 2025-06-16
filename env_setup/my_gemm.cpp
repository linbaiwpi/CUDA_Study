#include <torch/extension.h>
#include "my_gemm.h"

torch::Tensor my_gemm(torch::Tensor A, torch::Tensor B) {
    // 输出张量
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());
    if (A.device().is_cuda()) {
        // 检查是否在 CUDA 上
        TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
        TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
        // 调用 CUDA 实现
        my_gemm_cuda(A, B, C);
    } else {
        my_gemm_cpu(A, B, C);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_gemm", &my_gemm, "Custom GEMM (CUDA)");
}

