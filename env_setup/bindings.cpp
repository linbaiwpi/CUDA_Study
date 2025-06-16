#include <torch/extension.h>

torch::Tensor my_gemm(torch::Tensor A, torch::Tensor B);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_gemm", &my_gemm, "My custom GEMM");
}

