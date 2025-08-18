#include "conv.h"
#include <stdint.h>
#include <stdio.h>

// 模板化的3D卷积kernel实现
template<typename AT, typename WT, typename OT>
__global__ void conv3d_gpu(void* input, void* kernel, void* output,
    int img_height, int img_width, int img_channels,
    int kernel_height, int kernel_width, int kernel_in_channels, int kernel_out_channels,
    int tileH, int tileW, int tileC, int kN,
    bool use_2d_config) {

    // printf("conv3d_gpu...\n");

    AT* act = static_cast<AT*>(input);
    WT* wt = static_cast<WT*>(kernel);
    OT* out = static_cast<OT*>(output);

    // shared memory assignment
    int actH = img_height;
    int actW = img_width;
    int actC = img_channels;
    int kH = kernel_height;
    int kW = kernel_width;
    int sm_size_act = tileH * tileW * tileC;
    int sm_size_wt = kH * kW * tileC * kN;
    int sm_size_out = tileH * tileW * kN;
    int sm_size = sm_size_act + sm_size_wt + sm_size_out;
    extern __shared__ uint8_t sm[];  // 动态大小的共享内存
    AT *sm_act = (AT *)sm;
    WT *sm_wt = (WT *)(sm + sm_size_act * sizeof(AT));
    OT *sm_out = (OT *)(sm + sm_size_act * sizeof(AT) + sm_size_wt * sizeof(WT));

    for (int tc = 0; tc < actC; tc += tileC) {

        // shared memory loading data
        // load activation
        // thread Idx.x = tileH, Idx.y = tileW, Idx.z = kN
        if (threadIdx.z == 0) {
            int actH_idx = blockIdx.x * blockDim.x + threadIdx.x;
            int actW_idx = blockIdx.y * blockDim.y + threadIdx.y;
            int actC_idx = tc * tileC;
            int sm_idx = threadIdx.x * tileW * tileC + 
                         threadIdx.y * tileC;
            if (actH_idx >= actH || actW_idx >= actW || actC_idx >= actC) { // board of tensor H, W and C
                // TODO how about move this out? init sm to all 0?
                sm_act[sm_idx] = (AT)0; //  act tile in HWC format for storage
            } else {
                for (int i = 0; i < tileC; ++i) {
                    sm_act[sm_idx + i] = AT(act[actH_idx * actW * actC + actW_idx * actC + actC_idx + i]);

                    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0
                        &&
                        threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
                        printf("sm_idx = %d, actH_idx * actW * actC + actW_idx * actC + actC_idx = %d\n", 
                            sm_idx+i, actH_idx * actW * actC + actW_idx * actC + actC_idx+i);
                    }
                }
            }
        }
        __syncthreads();

        // Debug print for first block and thread
        if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0
            &&
            threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
            for (int i = 0; i < 16; ++i)
                printf("%f\n", (float)sm_act[i]);
        }

        return;
    }
}

// 显式实例化模板函数
template __global__ void conv3d_gpu<float, float, float>(void* input, void* kernel, void* output,
    int img_height, int img_width, int img_channels,
    int kernel_height, int kernel_width, int kernel_in_channels, int kernel_out_channels,
    int tileH, int tileW, int tileC, int kN,
    bool use_2d_config);


// GPU kernel：3D卷积（支持2D和3D配置）
__global__ void conv3d_gpu_ref(DATATYPE* input, DATATYPE* kernel, DATATYPE* output,
    int img_height, int img_width, int img_channels,
    int kernel_height, int kernel_width, int kernel_in_channels, int kernel_out_channels,
    bool use_2d_config) {
int output_height = img_height - kernel_height + 1;
int output_width = img_width - kernel_width + 1;

int out_ch, i, j;

if (use_2d_config) {
// 2D配置：将输出通道和高度维度合并
int combined_idx = blockIdx.x * blockDim.x + threadIdx.x;
out_ch = combined_idx / output_height;
i = combined_idx % output_height;
j = blockIdx.y * blockDim.y + threadIdx.y;
} else {
// 3D配置
out_ch = blockIdx.x * blockDim.x + threadIdx.x;
i = blockIdx.y * blockDim.y + threadIdx.y;
j = blockIdx.z * blockDim.z + threadIdx.z;
}

// 边界检查
if (out_ch >= kernel_out_channels || i >= output_height || j >= output_width) {
return;
}

DATATYPE sum = 0.0f;

// 执行卷积计算
for (int in_ch = 0; in_ch < kernel_in_channels; in_ch++) {
for (int ki = 0; ki < kernel_height; ki++) {
for (int kj = 0; kj < kernel_width; kj++) {
int input_i = i + ki;
int input_j = j + kj;

int input_idx = (in_ch * img_height + input_i) * img_width + input_j;
int kernel_idx = ((out_ch * kernel_in_channels + in_ch) * kernel_height + ki) * kernel_width + kj;

sum += input[input_idx] * kernel[kernel_idx];
}
}
}

int output_idx = (out_ch * output_height + i) * output_width + j;
output[output_idx] = sum;
}