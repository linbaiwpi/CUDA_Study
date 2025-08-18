#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <iomanip>

#include "conv.h"

// CPU参考实现：3D卷积
void conv3d_cpu(DATATYPE* input, DATATYPE* kernel, DATATYPE* output, 
                 int img_height, int img_width, int img_channels,
                 int kernel_height, int kernel_width, int kernel_in_channels, int kernel_out_channels) {
    int output_height = img_height - kernel_height + 1;
    int output_width = img_width - kernel_width + 1;
    
    for (int out_ch = 0; out_ch < kernel_out_channels; out_ch++) {
        for (int i = 0; i < output_height; i++) {
            for (int j = 0; j < output_width; j++) {
                DATATYPE sum = 0.0f;
                
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
        }
    }
}


// 打印矩阵的辅助函数（2D）
void print_matrix(DATATYPE* matrix, int rows, int cols, const char* name) {
    std::cout << name << " (" << rows << "x" << cols << "):" << std::endl;
    for (int i = 0; i < std::min(rows, 8); i++) {  // 只打印前8行
        for (int j = 0; j < std::min(cols, 8); j++) {  // 只打印前8列
            std::cout << std::setw(8) << std::fixed << std::setprecision(2) 
                      << matrix[i * cols + j] << " ";
        }
        if (cols > 8) std::cout << "...";
        std::cout << std::endl;
    }
    if (rows > 8) std::cout << "..." << std::endl;
    std::cout << std::endl;
}

// 打印3D tensor的辅助函数（前8x8x8 subtensor）
void print_tensor_3d(DATATYPE* tensor, int height, int width, int channels, const char* name) {
    std::cout << name << " (" << height << "x" << width << "x" << channels << "):" << std::endl;
    
    // 只打印前8x8x8的subtensor
    int max_h = std::min(height, 8);
    int max_w = std::min(width, 8);
    int max_c = std::min(channels, 8);
    
    for (int c = 0; c < max_c; c++) {
        std::cout << "Channel " << c << ":" << std::endl;
        for (int h = 0; h < max_h; h++) {
            for (int w = 0; w < max_w; w++) {
                int idx = (c * height + h) * width + w;
                std::cout << std::setw(8) << std::fixed << std::setprecision(2) 
                          << tensor[idx] << " ";
            }
            if (width > 8) std::cout << "...";
            std::cout << std::endl;
        }
        if (height > 8) std::cout << "..." << std::endl;
        std::cout << std::endl;
    }
    if (channels > 8) std::cout << "..." << std::endl;
    std::cout << std::endl;
}

// 打印NHWC格式tensor的辅助函数（前8x8x8 subtensor）
void print_tensor_nhwc(DATATYPE* tensor, int batch, int height, int width, int channels, const char* name) {
    std::cout << name << " (" << batch << "x" << height << "x" << width << "x" << channels << "):" << std::endl;
    
    // 只打印第一个batch的前8x8x8的subtensor
    int max_h = std::min(height, 8);
    int max_w = std::min(width, 8);
    int max_c = std::min(channels, 8);
    
    for (int c = 0; c < max_c; c++) {
        std::cout << "Channel " << c << ":" << std::endl;
        for (int h = 0; h < max_h; h++) {
            for (int w = 0; w < max_w; w++) {
                int idx = (0 * height * width * channels) + (h * width * channels) + (w * channels) + c;
                std::cout << std::setw(8) << std::fixed << std::setprecision(2) 
                          << tensor[idx] << " ";
            }
            if (width > 8) std::cout << "...";
            std::cout << std::endl;
        }
        if (height > 8) std::cout << "..." << std::endl;
        std::cout << std::endl;
    }
    if (channels > 8) std::cout << "..." << std::endl;
    std::cout << std::endl;
}

// 验证结果
bool verify_results(DATATYPE* cpu_result, DATATYPE* gpu_result, int size, float tolerance = 1e-4) {
    for (int i = 0; i < size; i++) {
        if (abs(cpu_result[i] - gpu_result[i]) > tolerance) {
            std::cout << std::setprecision(5) << std::endl;
            std::cout << "Mismatch at index " << i << ": CPU=" << cpu_result[i] 
                      << ", GPU=" << gpu_result[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    std::cout << "=== CUDA 3D卷积示例 ===" << std::endl;
    std::cout << "输入tensor大小: " << IMAGE_HEIGHT << "x" << IMAGE_WIDTH << "x" << IMAGE_CHANNELS << std::endl;
    std::cout << "卷积核大小: " << KERNEL_OUT_CHANNELS << "x" << KERNEL_HEIGHT << "x" << KERNEL_WIDTH << "x" << KERNEL_IN_CHANNELS << std::endl;
    std::cout << "输出tensor大小: " << OUTPUT_HEIGHT << "x" << OUTPUT_WIDTH << "x" << KERNEL_OUT_CHANNELS << std::endl;
    std::cout << std::endl;
    
    // 分配CPU内存
    DATATYPE* h_input = (DATATYPE*)malloc(1 * IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS * sizeof(DATATYPE));  // NHWC, N=1
    DATATYPE* h_kernel = (DATATYPE*)malloc(KERNEL_OUT_CHANNELS * KERNEL_HEIGHT * KERNEL_WIDTH * KERNEL_IN_CHANNELS * sizeof(DATATYPE));
    DATATYPE* h_output_cpu = (DATATYPE*)malloc(1 * OUTPUT_HEIGHT * OUTPUT_WIDTH * KERNEL_OUT_CHANNELS * sizeof(DATATYPE));  // NHWC, N=1
    DATATYPE* h_output_gpu = (DATATYPE*)malloc(1 * OUTPUT_HEIGHT * OUTPUT_WIDTH * KERNEL_OUT_CHANNELS * sizeof(DATATYPE));  // NHWC, N=1
    
    // 初始化输入tensor（NHWC格式，N=1）
    srand(42);  // 固定种子以获得可重复的结果
    for (int n = 0; n < 1; n++) {  // batch size = 1
        for (int h = 0; h < IMAGE_HEIGHT; h++) {
            for (int w = 0; w < IMAGE_WIDTH; w++) {
                for (int c = 0; c < IMAGE_CHANNELS; c++) {
                    int idx = (n * IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS) + 
                              (h * IMAGE_WIDTH * IMAGE_CHANNELS) + 
                              (w * IMAGE_CHANNELS) + c;
                    h_input[idx] = ((float)rand() / RAND_MAX) * 10.0f;
                }
            }
        }
    }

    std::cout << "===================" << std::endl;
    for (int idx = 0; idx < 16; ++idx) {
        std::cout << h_input[idx] << ", ";
    }
    std::cout << std::endl;
    
    // 初始化卷积核（OHWI格式，已经是正确的）
    for (int oc = 0; oc < KERNEL_OUT_CHANNELS; oc++) {
        for (int h = 0; h < KERNEL_HEIGHT; h++) {
            for (int w = 0; w < KERNEL_WIDTH; w++) {
                for (int ic = 0; ic < KERNEL_IN_CHANNELS; ic++) {
                    int idx = (oc * KERNEL_HEIGHT * KERNEL_WIDTH * KERNEL_IN_CHANNELS) + 
                              (h * KERNEL_WIDTH * KERNEL_IN_CHANNELS) + 
                              (w * KERNEL_IN_CHANNELS) + ic;
                    h_kernel[idx] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;  // [-1, 1]范围
                }
            }
        }
    }
    
    // 打印输入数据（部分）
    print_tensor_nhwc(h_input, 1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, "输入tensor (前8x8x8)");
    print_matrix(h_kernel, KERNEL_OUT_CHANNELS, KERNEL_HEIGHT * KERNEL_WIDTH * KERNEL_IN_CHANNELS, "卷积核 (前8x9)");
    
    // CPU计算参考结果
    std::cout << "执行CPU卷积计算..." << std::endl;
    clock_t cpu_start = clock();
    conv3d_cpu(h_input, h_kernel, h_output_cpu, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS,
                KERNEL_HEIGHT, KERNEL_WIDTH, KERNEL_IN_CHANNELS, KERNEL_OUT_CHANNELS);
    clock_t cpu_end = clock();
    double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000.0;
    
    print_tensor_nhwc(h_output_cpu, 1, OUTPUT_HEIGHT, OUTPUT_WIDTH, KERNEL_OUT_CHANNELS, "CPU输出结果 (前8x8x16)");
    std::cout << "CPU计算时间: " << cpu_time << " ms" << std::endl;
    
    // GPU计算
    std::cout << "执行GPU卷积计算..." << std::endl;
    
    // 分配GPU内存
    DATATYPE *d_input, *d_kernel, *d_output;
    cudaMalloc((void**)&d_input, 1 * IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS * sizeof(DATATYPE));  // NHWC, N=1
    cudaMalloc((void**)&d_kernel, KERNEL_OUT_CHANNELS * KERNEL_HEIGHT * KERNEL_WIDTH * KERNEL_IN_CHANNELS * sizeof(DATATYPE));
    cudaMalloc((void**)&d_output, 1 * OUTPUT_HEIGHT * OUTPUT_WIDTH * KERNEL_OUT_CHANNELS * sizeof(DATATYPE));  // NHWC, N=1
    
    // 复制数据到GPU
    cudaMemcpy(d_input, h_input, 1 * IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS * sizeof(DATATYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, KERNEL_OUT_CHANNELS * KERNEL_HEIGHT * KERNEL_WIDTH * KERNEL_IN_CHANNELS * sizeof(DATATYPE), cudaMemcpyHostToDevice);
    
    // 设置block和grid大小 - 根据新的conv3d_gpu设计
    int tileH = 8, tileW = 8, tileC = 16, kN = 8;  // 减小tile参数
    
    // 3D block配置
    dim3 block_size(tileH, tileW, kN);  // 对应 tileH, tileW, kN
    
    // 3D grid配置 - 根据tile大小和输出尺寸计算
    dim3 grid_size((OUTPUT_HEIGHT + tileH - 1) / tileH,
                   (OUTPUT_WIDTH + tileW - 1) / tileW,
                   (KERNEL_OUT_CHANNELS + kN - 1) / kN);
    
    // 确保grid大小合理
    if (grid_size.x == 0) grid_size.x = 1;
    if (grid_size.y == 0) grid_size.y = 1;
    if (grid_size.z == 0) grid_size.z = 1;
    
    bool use_2d_config = false;  // 新设计使用3D配置
    
    std::cout << "Grid大小: (" << grid_size.x << ", " << grid_size.y << ", " << grid_size.z << ")" << std::endl;
    std::cout << "Block大小: (" << block_size.x << ", " << block_size.y << ", " << block_size.z << ")" << std::endl;
    std::cout << "使用配置: " << (use_2d_config ? "2D" : "3D") << std::endl;
    std::cout << "Tile参数: H=" << tileH << ", W=" << tileW << ", C=" << tileC << ", N=" << kN << std::endl;
    
    // 创建CUDA事件来测量时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    // 调用GPU kernel - 使用2D配置
    // 保持之前计算的2D配置
    std::cout << "最终Grid大小: (" << grid_size.x << ", " << grid_size.y << ", " << grid_size.z << ")" << std::endl;
    std::cout << "最终Block大小: (" << block_size.x << ", " << block_size.y << ", " << block_size.z << ")" << std::endl;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);  // 0 = 第一个GPU

    printf("GPU Name: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Max Shared Memory per Block: %zu bytes (%zu KB)\n",
           prop.sharedMemPerBlock,
           prop.sharedMemPerBlock / 1024);
    // 计算需要的共享内存大小
    int sm_size_act = tileH * tileW * tileC * sizeof(DATATYPE);
    int sm_size_wt = KERNEL_HEIGHT * KERNEL_WIDTH * tileC * kN * sizeof(DATATYPE);
    int sm_size_out = tileH * tileW * kN * sizeof(DATATYPE);
    int shared_mem_size = sm_size_act + sm_size_wt + sm_size_out;
    
    std::cout << "共享内存需求: " << shared_mem_size << " bytes (" << shared_mem_size/1024 << " KB)" << std::endl;
    std::cout << "GPU共享内存限制: " << prop.sharedMemPerBlock << " bytes (" << prop.sharedMemPerBlock/1024 << " KB)" << std::endl;
    
    // 检查共享内存是否足够
    if (shared_mem_size > prop.sharedMemPerBlock) {
        std::cout << "警告：共享内存需求超过GPU限制，将使用GPU最大共享内存" << std::endl;
        shared_mem_size = prop.sharedMemPerBlock;
    }
    
    conv3d_gpu<DATATYPE, DATATYPE, DATATYPE><<<grid_size, block_size, shared_mem_size>>>(
        d_input, d_kernel, d_output, 
        IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS,
        KERNEL_HEIGHT, KERNEL_WIDTH, KERNEL_IN_CHANNELS, KERNEL_OUT_CHANNELS, 
        tileH, tileW, tileC, kN,  // 使用变量而不是硬编码
        use_2d_config
    );
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);
    
    // 检查kernel执行错误
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << "CUDA错误: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }
    
/*
    // 复制结果回CPU
    cudaMemcpy(h_output_gpu, d_output, 1 * OUTPUT_HEIGHT * OUTPUT_WIDTH * KERNEL_OUT_CHANNELS * sizeof(DATATYPE), cudaMemcpyDeviceToHost);
    
    print_tensor_nhwc(h_output_gpu, 1, OUTPUT_HEIGHT, OUTPUT_WIDTH, KERNEL_OUT_CHANNELS, "GPU输出结果 (前8x8x16)");
    std::cout << "GPU计算时间: " << gpu_time << " ms" << std::endl;
    
    // 验证结果
    std::cout << "验证结果..." << std::endl;
    if (verify_results(h_output_cpu, h_output_gpu, 1 * OUTPUT_HEIGHT * OUTPUT_WIDTH * KERNEL_OUT_CHANNELS)) {
        std::cout << "✓ 结果验证成功！CPU和GPU结果一致" << std::endl;
    } else {
        std::cout << "✗ 结果验证失败！CPU和GPU结果不一致" << std::endl;
    }
    
    // 性能比较
    std::cout << std::endl;
    std::cout << "=== 性能比较 ===" << std::endl;
    std::cout << "CPU时间: " << cpu_time << " ms" << std::endl;
    std::cout << "GPU时间: " << gpu_time << " ms" << std::endl;
    std::cout << "加速比: " << cpu_time / gpu_time << "x" << std::endl;
*/
 
    // 清理内存
    free(h_input);
    free(h_kernel);
    free(h_output_cpu);
    free(h_output_gpu);
    
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    std::cout << std::endl << "程序执行完成！" << std::endl;
    return 0;
} 