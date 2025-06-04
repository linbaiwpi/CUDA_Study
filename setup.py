from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='my_gemm',
    ext_modules=[
        CUDAExtension(
            name='my_gemm',
            sources=['my_gemm.cpp', 'gemm_cuda.cu', 'gemm_cpu.cpp'],
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)

