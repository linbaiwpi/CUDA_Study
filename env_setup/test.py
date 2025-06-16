import torch
import my_gemm
import time

A = torch.randn(1280*2, 2560*2, device='cuda')
B = torch.randn(2560*2, 1280*2, device='cuda')

sum = 0
for i in range(10):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    C = my_gemm.my_gemm(A, B)
    end.record()
    torch.cuda.synchronize()
    curr_time = start.elapsed_time(end)
    sum = sum + curr_time
    print(str(curr_time) + " ms") # Time in milliseconds
print("===== " + str(sum / 100))
print(C.shape)  # Should be [128, 512]

# 比较精度
C_ref = A @ B
print(torch.allclose(C, C_ref, atol=1e-4))

A = torch.randn(128, 256, device='cpu')
B = torch.randn(256, 128, device='cpu')

start_time = time.perf_counter()
C = my_gemm.my_gemm(A, B)
end_time = time.perf_counter()
print("===== " + str((end_time - start_time)*1_000_000) + " ms")
print(C.shape)  # Should be [128, 512]

# 比较精度
C_ref = A @ B
print(torch.allclose(C, C_ref, atol=1e-4))

