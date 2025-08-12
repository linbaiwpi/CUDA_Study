cmake_minimum_required(VERSION 3.18)  # 建议使用 3.18 及以上支持更好的 CUDA integration
project(MyCudaProject LANGUAGES C CXX CUDA)

# 设置 CUDA 架构（可选）
set(CMAKE_CUDA_ARCHITECTURES 86)  # 可根据目标 GPU 选择架构

# 设置 CUDA 编译选项（可选）
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 源文件
# set(SOURCES
#     main.cu
#     ch7.1.cu
#     ch7.2.cu
#     ch7.3.cu
# #    ch7.4.cu
# )
file(GLOB SOURCES "*.cu")

# 创建可执行文件
add_executable(cuda_ch7 ${SOURCES})

# 设置编译属性
set_target_properties(cuda_ch7 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
)

# 链接库（如有需要）
target_link_libraries(cuda_ch7 PRIVATE cublas)
