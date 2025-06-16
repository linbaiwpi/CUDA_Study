#include <stdio.h>

#define N 1024 // 3*1024*1024
#define THREAD_PER_BLOCK 256

__global__ void reduce_kernel(float *input, float *output) {
    /*
    // printf("input: %f\n", input[threadIdx.x]);
    float *input_begin = input + blockIdx.x * blockDim.x;
    for (int i = 1; i < blockDim.x; i *= 2) {
        if (threadIdx.x % (i * 2) == 0) {
            input_begin[threadIdx.x] += input_begin[threadIdx.x + i];
            // printf("threadIdx.x : %d = %f\n", threadIdx.x, input_begin[threadIdx.x]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        output[blockIdx.x] = input_begin[0];
        // printf("%f\n", output[blockIdx.x]);
    }
    */
    /*
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i=1; i<blockDim.x; i++) {
        if (tid % (2 * i) == 0)
            input[idx] += input[idx+i];
        __syncthreads();
    }
    if (tid==0)
        output[blockIdx.x] = input[tid];
    */

    float *input_begin = input + blockDim.x * blockIdx.x;
    for (int i = 1; i < blockDim.x; i *= 2) {
        if (threadIdx.x % (2 * i) == 0)
            input_begin[threadIdx.x] += input_begin[threadIdx.x + i];
        __syncthreads();
    }
    if (threadIdx.x==0)
        output[blockIdx.x] = *input_begin;
}

int main(int argc, char **argv) {
    float *input, *d_input;
    float *output, *d_output;
    float *result;

    // cpu malloc
    input = (float *)malloc(N * sizeof(float));
    int block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
    output = (float *)malloc(block_num * sizeof(float));
    result = (float *)malloc(block_num * sizeof(float));
    // cuda malloc
    cudaMalloc((void **)&d_input, N * sizeof(float));
    cudaMalloc((void **)&d_output, block_num * sizeof(float));


    // cpu assigment values
    for (int i = 0; i< N ; ++i) {
        input[i] = 2.0 * (float)drand48() - 1.0;
    }
    for (int i = 0; i< block_num; ++i) {
        float curr = 0;
        for (int j = 0; j < THREAD_PER_BLOCK; ++j) {
            if (i * THREAD_PER_BLOCK + j < N)
                curr += input[i * THREAD_PER_BLOCK + j];
        }
        result[i] = curr;
    }


    // cpoy value into cuda
    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid_dim(block_num);
    dim3 block_dim(THREAD_PER_BLOCK);
    printf("%d\n", block_num);
    printf("%d\n", THREAD_PER_BLOCK);
    reduce_kernel<<<grid_dim, block_dim>>>(d_input, d_output);

    cudaMemcpy(output, d_output, block_num*sizeof(float), cudaMemcpyDeviceToHost);

    printf("RESULT ===== \n");
    for (int i = 0; i < block_num; ++i) {
        printf("%2f, %2f, %2f\n", output[i], result[i], (output[i] - result[i]));
    }

    free(input);
    free(output);
    free(result);
    cudaFree(d_input);
    cudaFree(d_output);

    printf("Reduce\n");
    return 0;
}

