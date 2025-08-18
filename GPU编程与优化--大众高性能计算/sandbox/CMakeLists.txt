cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(conv_cuda CUDA CXX)

# 设置CUDA架构（根据你的GPU调整）
set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置CUDA标准
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 设置编译选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# 创建可执行文件
add_executable(conv_main main.cu conv.cu)

# 打印信息
message(STATUS "CUDA版本: ${CMAKE_CUDA_VERSION}")
message(STATUS "CUDA架构: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "构建类型: ${CMAKE_BUILD_TYPE}") 