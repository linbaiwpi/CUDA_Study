#pragma once
#ifndef __CONV_H__


// 定义数据类型和常量
#define DATATYPE float
#define IMAGE_HEIGHT 32
#define IMAGE_WIDTH 32
#define IMAGE_CHANNELS 8
#define KERNEL_HEIGHT 3
#define KERNEL_WIDTH 3
#define KERNEL_IN_CHANNELS 8
#define KERNEL_OUT_CHANNELS 16
#define OUTPUT_HEIGHT (IMAGE_HEIGHT - KERNEL_HEIGHT + 1)
#define OUTPUT_WIDTH (IMAGE_WIDTH - KERNEL_WIDTH + 1)


template<typename AT, typename WT, typename OT>
__global__ void conv3d_gpu(void* input, void* kernel, void* output,
    int img_height, int img_width, int img_channels,
    int kernel_height, int kernel_width, int kernel_in_channels, int kernel_out_channels,
    int tileH, int tileW, int tileC, int kN,
    bool use_2d_config = false);


__global__ void conv3d_gpu_ref(DATATYPE* input, DATATYPE* kernel, DATATYPE* output,
    int img_height, int img_width, int img_channels,
    int kernel_height, int kernel_width, int kernel_in_channels, int kernel_out_channels,
    bool use_2d_config = false);


#endif // __CONV_H__