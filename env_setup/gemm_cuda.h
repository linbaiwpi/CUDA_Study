#pragma once
#include <torch/extension.h>

void my_gemm_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor C);

