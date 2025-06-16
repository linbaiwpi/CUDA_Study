cmake_minimum_required(VERSION 3.20.0)
project(cuda_reduce VERSION 0.1.0 LANGUAGES CUDA CXX C)

find_package(CUDAToolkit)
# find_package(BLAS REQUIRED)

add_subdirectory(reduce)
