import torch
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn.functional as F

device = torch.device("cuda")

# Simulate attention with different sequence lengths
# batch=1, heads=8, seq_len, head_dim=64
configs = [
    (1, 8, 512, 64),
    (1, 8, 1024, 64),
    (1, 8, 2048, 64),
]

def standard_attention(Q, K, V):
    scale = Q.shape[-1] ** -0.5
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [B, H, S, S]
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, V)

for batch, heads, seq_len, head_dim in configs:
    Q = torch.randn(batch, heads, seq_len, head_dim, device=device)
    K = torch.randn(batch, heads, seq_len, head_dim, device=device)
    V = torch.randn(batch, heads, seq_len, head_dim, device=device)

    # Warmup
    for _ in range(3):
        standard_attention(Q, K, V)

    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_flops=True,
        profile_memory=True  # NEW: track memory usage
    ) as prof:
        with record_function(f"attention_seq{seq_len}"):
            for _ in range(10):
                out = standard_attention(Q, K, V)
            torch.cuda.synchronize()

    print(f"\n{'='*60}")
    print(f"Sequence length: {seq_len}")
    print(f"Attention matrix size: {seq_len}x{seq_len} = {seq_len*seq_len*4/1024/1024:.1f} MB")
    print(f"{'='*60}")
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=8
    ))

    prof.export_chrome_trace(f"attention_trace_seq{seq_len}.json")