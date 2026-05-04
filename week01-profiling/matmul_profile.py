import torch
from torch.profiler import profile, record_function, ProfilerActivity

device = torch.device("cuda")
size = 2048

A = torch.randn(size, size, device=device, dtype=torch.float32)
B = torch.randn(size, size, device=device, dtype=torch.float32)

# Warmup
for _ in range(3):
    torch.matmul(A, B)

# Profile
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_flops=True
) as prof:
    with record_function("matmul"):
        for _ in range(10):
            torch.matmul(A, B)

# Print results
print(prof.key_averages().table(
    sort_by="cuda_time_total",
    row_limit=10
))

# Export trace for visualization
prof.export_chrome_trace("matmul_trace.json")
print("\nTrace saved to matmul_trace.json")