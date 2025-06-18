#include <cstdio>

/*
 * Kernel 3 tile
 * 
 */

__device__ void mul_add_8(float *subA, float *subB, float *sum) {
  *sum = 0.0;
  for (int i = 0; i < 8; ++i) {
    *sum += subA[i] * subB[i];
  }
}

__global__ void gemm_kernel_ref(
    const float* A, const float* B, float* C,
    const int M, const int K, const int N, 
    const int tileM, const int tileK, const int tileN // iTileW, iTileC, oTileC
) {
    int cCol = blockDim.x * blockIdx.x + threadIdx.x;
    int cRow = blockDim.y * blockIdx.y + threadIdx.y;
    if (cCol >= N || cRow >= M) {
        return;
    }

    float sum;
    float sum_odd = 0.0;
    float sum_even = 0.0;
    constexpr int calcK = 8;
    float subA[calcK];
    float subB[calcK];
    for (int k = 0; k < K; k += tileK) {
        for (int k8 = 0; k8 < tileK; k8 += calcK) {
            // calculate 8 elements
            for (int i = 0; i < calcK; ++i) {
                subA[i] = A[cRow * K + k + k8 + i];
                subB[i] = B[(k + k8 + i) * N + cCol];
                if (cRow == 0 && cCol == 0) {
                  // printf("cRow = %d, cCol = %d, subA[%d] = %f, subB[%d] = %f\n", cRow, cCol, i, subA[i], i, subB[i]);
                }
            }
            mul_add_8(subA, subB, &sum);
            // if (cRow == 0 && cCol == 0) {
            //   printf("sum = %f\n", sum);
            // }
            // sum up
            int k8_idx = k8 % calcK;
            if (k8_idx % 2 == 0) {
                sum_even += sum;
            } else {
                sum_odd += sum;
            }
        }
    }
    C[cRow * N + cCol] = sum_odd + sum_even;
}

__global__ void gemm_kernel(
    const float* A, const float* B, float* C,
    const int M, const int K, const int N, 
    const int tileM, const int tileK, const int tileN // iTileW, iTileC, oTileC
) {
  
    int cCol = blockIdx.x;
    int cRow = blockIdx.y;
    //指代一个block计算的分块矩阵在C矩阵中所处的位置

    // -------> threadIdx.x
    // |
    // |
    // |
    // threadIdx.y

    extern __shared__ float sm[];
    float *smA = sm;
    float *smB = sm + blockDim.y * tileK;

    float sum;
    float sum_odd = 0.0;
    float sum_even = 0.0;
    constexpr int calcK = 8;
    float subA[calcK];
    float subB[calcK];
    for (int k = 0; k < K; k += tileK) {

        // start position of each thread
        int aRow = blockDim.y * blockIdx.y + threadIdx.y;
        int aCol = k * tileK;
        int bCol = blockDim.x * blockIdx.x + threadIdx.x;
        int bRow = k * tileK;
        if (blockIdx.x==0 && blockIdx.y==0) {
          for (int i = 0; i < calcK; ++i) {
            if (aRow >= M || aCol + threadIdx.x * calcK + i >= K)
              continue;
            smA[threadIdx.y * tileK + threadIdx.x * calcK + i] = A[aRow * K + aCol + threadIdx.x * calcK + i];
            // printf("(%d, %d) = %f\n", threadIdx.x, threadIdx.y, A[aRow * K + aCol + threadIdx.x * calcK + i]);
            // printf("(%d, %d) = %f, %f\n", threadIdx.x, threadIdx.y, A[aRow * K + aCol + threadIdx.x * calcK + i], smA[threadIdx.y * tileK + threadIdx.x * calcK]);
            if (bRow + threadIdx.y * calcK + i >= M || bCol >= K)
              continue;
            smB[blockDim.x * (threadIdx.y * calcK + i) + threadIdx.x] = B[(bRow + threadIdx.y * calcK + i) * N + bCol];
            printf("(%d, %d) = %f, %f\n", threadIdx.x, threadIdx.y, B[(bRow + threadIdx.y * calcK + i) * N + bCol], smB[blockDim.x * (threadIdx.y * calcK + i) + threadIdx.x]);
          }
          __syncthreads();

          if (threadIdx.x == 0 && threadIdx.y == 0) {
            printf("================ smA\n");
            for (int i = 0; i < blockDim.y; ++i) {
              for (int j = 0; j < tileK; ++j) {
                printf("%f ", smA[i * tileK + j]);
              }
              printf("\n");
            }
            printf("\n");
            printf("================ smB\n");
            for (int i = 0; i < tileK; ++i) {
              for (int j = 0; j < blockDim.x; ++j) {
                printf("%f ", smB[i * blockDim.x + j]);
              }
              printf("\n");
            }
            printf("\n");
          }
          __syncthreads();
        }

/*
        for (int k8 = 0; k8 < tileK; k8 += calcK) {
            // calculate 8 elements
            for (int i = 0; i < calcK; ++i) {
                subA[i] = A[cRow * K + k + k8 + i];
                subB[i] = B[(k + k8 + i) * N + cCol];
                if (cRow == 0 && cCol == 0) {
                  printf("cRow = %d, cCol = %d, subA[%d] = %f, subB[%d] = %f\n", cRow, cCol, i, subA[i], i, subB[i]);
                }
            }
            mul_add_8(subA, subB, &sum);
            if (cRow == 0 && cCol == 0) {
              printf("sum = %f\n", sum);
            }
            // sum up
            int k8_idx = k8 % calcK;
            if (k8_idx % 2 == 0) {
                sum_even += sum;
            } else {
                sum_odd += sum;
            }
        }
*/
    }
    C[cRow * N + cCol] = sum_odd + sum_even;
}

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

int main() {
    constexpr int M = 128;
    constexpr int K = 128;
    constexpr int N = 128;
    
    float *A = (float *)malloc(sizeof(float) * M * K);
    float *B = (float *)malloc(sizeof(float) * K * N);
    float *C = (float *)malloc(sizeof(float) * M * N);

    float *d_A;
    float *d_B;
    float *d_C;
    cudaMalloc((void **)&d_A, sizeof(float) * M * K);
    cudaMalloc((void **)&d_B, sizeof(float) * K * N);
    cudaMalloc((void **)&d_C, sizeof(float) * M * N);

    float *Cref = (float *)malloc(sizeof(float) * M * N);
    float *d_Cref;
    cudaMalloc((void **)&d_Cref, sizeof(float) * M * N);

    for (int i = 0; i < M * K; ++i) {
        A[i] = 0.1 * ((i % K) + 1);
    }
    for (int i = 0; i < K * N; ++i) {
        B[i] = 0.1 * ((i % N) + 1);
    }
    printf("============================== A\n");
    for (int m = 0; m < K; ++m) {
        printf("%f ", A[m]);
    }
    printf("\n");
    printf("============================== B\n");
    for (int m = 0; m < K; ++m) {
        printf("%f ", B[m * N]);
    }
    printf("\n");

    cudaMemcpy(d_A, A, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(float) * K * N, cudaMemcpyHostToDevice);

    uint tileM = 1;
    uint tileN = 1;
    uint tileK = 32;

    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    dim3 blockDim(32/tileM, 32/tileN);

    gemm_kernel_ref<<<gridDim, blockDim>>>(d_A, d_B, d_Cref, M, K, N, tileM, tileK, tileN);
    cudaMemcpy(Cref, d_Cref, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    printf("Cref[0] : %f\n", Cref[0]);

    // -----------------------------------------------------------------------------
    // 每个block的threadIdx.x 计算8个数据乘加,在y轴上计算blockDim.y
    // 因此blockDim应该是（tileK * 8, 1024 / （tileK * 8））
    // 此处1024为一个block最多可以计算的thread的总量
    blockDim.x = tileK / 8;
    constexpr int MAX_THREAD_PER_BLOCK = 512; // TODO fit into shared memory
    blockDim.y = tileK / 8; // MAX_THREAD_PER_BLOCK / blockDim.x;
    // gridDim的计算跟普通分块矩阵乘法一致
    gridDim.x = CEIL_DIV(M, blockDim.x);
    gridDim.y = CEIL_DIV(N, blockDim.y);
    size_t sharedMemSize = (blockDim.x + blockDim.y) * tileK * sizeof(float);
    printf("blockDim = (%d, %d), gridDim = (%d, %d), sharedMemSize = %d\n", blockDim.x, blockDim.y, gridDim.x, gridDim.y, sharedMemSize);
    gemm_kernel<<<gridDim, blockDim, sharedMemSize>>>(d_A, d_B, d_C, M, K, N, tileM, tileK, tileN);
    cudaMemcpy(C, d_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    printf("C[0] : %f\n", C[0]);

    /*
    bool result = true;
    for (int m = 0; m < M*N; ++m) {
      if (Cref[m] - C[m] > 0.1) {
        result = false;
        printf("Cref[%d] = %f, C[%d] = %f, diff = %f\n", m, Cref[m], m, C[m], Cref[m]-C[m]);
      }
    }
    if (result) {
      printf("PASS\n");
    } else {
      printf("FALSE\n");
    }
    printf("============================== Cref\n");
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            printf("%f ", Cref[m * N + n]);
        }
        printf("\n");
    }
    printf("\n");
    */
    free(A);
    free(B);
    free(C);
    free(Cref);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_Cref);

    return 0;
}  

