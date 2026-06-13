# naive_softmax.py
# Goal: measure how many VRAM round trips naive softmax makes
# This is the baseline we will beat with the fused kernel

import torch
import triton
import triton.language as tl
import time

# ─────────────────────────────────────────────
# NAIVE SOFTMAX — PyTorch implementation
# We use PyTorch ops to simulate what a naive kernel does
# Each torch op is a separate kernel launch = separate VRAM round trip
# ─────────────────────────────────────────────

def naive_softmax_pytorch(X):
    # Step 1: read X, compute max, write max to VRAM
    row_max = X.max(dim=-1, keepdim=True).values      # VRAM read + write

    # Step 2: read X, read max, compute exp, write exp to VRAM
    X_stable = X - row_max                             # VRAM read x2 + write
    exp_x = torch.exp(X_stable)                        # VRAM read + write

    # Step 3: read exp, compute sum, write sum to VRAM
    row_sum = exp_x.sum(dim=-1, keepdim=True)          # VRAM read + write

    # Step 4: read exp, read sum, divide, write output to VRAM
    output = exp_x / row_sum                           # VRAM read x2 + write

    return output


# ─────────────────────────────────────────────
# TRITON NAIVE SOFTMAX
# One kernel but explicitly wasteful — reads x, writes intermediates,
# reads them back — to show what each VRAM trip costs
# ─────────────────────────────────────────────

@triton.jit
def naive_softmax_kernel(
    X_ptr,
    O_ptr,
    stride_row,
    N_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(axis=0)
    row_start_x = X_ptr + row_idx * stride_row
    row_start_o = O_ptr + row_idx * stride_row
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_cols

    # VRAM read #1 — load input row
    x = tl.load(row_start_x + offsets, mask=mask, other=-float('inf'))

    # Compute max — stays in register, no VRAM write needed
    row_max = tl.max(x, axis=0)

    # Subtract max for numerical stability
    x_stable = x - row_max

    # Compute exp — stays in register
    exp_x = tl.exp(x_stable)

    # Zero out the masked positions so they don't pollute the sum
    exp_x = tl.where(mask, exp_x, 0.0)

    # Compute sum — stays in register
    row_sum = tl.sum(exp_x, axis=0)

    # Normalize — stays in register
    output = exp_x / row_sum

    # VRAM write #1 — write final output
    tl.store(row_start_o + offsets, output, mask=mask)


def naive_softmax_triton(X, BLOCK_SIZE=None):
    n_rows, n_cols = X.shape
    O = torch.zeros_like(X)

    if BLOCK_SIZE is None:
        BLOCK_SIZE = 1
        while BLOCK_SIZE < n_cols:
            BLOCK_SIZE *= 2

    grid = (n_rows,)
    naive_softmax_kernel[grid](
        X, O,
        X.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return O


# ─────────────────────────────────────────────
# BENCHMARK HELPER
# ─────────────────────────────────────────────

def benchmark(fn, X, warmup=10, repeat=50):
    for _ in range(warmup):
        fn(X)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        fn(X)
    torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) / repeat * 1000


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # ── Correctness check ─────────────────────
    print("=== Correctness Check ===")
    torch.manual_seed(0)
    X = torch.randn(128, 512, device='cuda', dtype=torch.float32)

    O_triton  = naive_softmax_triton(X)
    O_pytorch_naive = naive_softmax_pytorch(X)
    O_pytorch_ref   = torch.softmax(X, dim=-1)

    print(f"Triton matches PyTorch ref:        {torch.allclose(O_triton, O_pytorch_ref, atol=1e-5)}")
    print(f"Naive PyTorch matches PyTorch ref: {torch.allclose(O_pytorch_naive, O_pytorch_ref, atol=1e-5)}")
    print()

    # ── Benchmark ─────────────────────────────
    print("=== Benchmark: Naive vs PyTorch ===")
    print(f"{'Seq Len':<12} {'Naive PyTorch (ms)':<22} {'Naive Triton (ms)':<20} {'PyTorch ref (ms)'}")
    print("-" * 75)

    for seq_len in [512, 1024, 2048]:
        X = torch.randn(seq_len, seq_len, device='cuda', dtype=torch.float32)

        ms_naive_pt  = benchmark(naive_softmax_pytorch, X)
        ms_naive_tri = benchmark(naive_softmax_triton, X)
        ms_ref       = benchmark(lambda x: torch.softmax(x, dim=-1), X)

        print(f"{seq_len:<12} {ms_naive_pt:<22.4f} {ms_naive_tri:<20.4f} {ms_ref:.4f}")

    print()
    print("=== Profiler (seq=2048, Naive Triton) ===")
    X = torch.randn(2048, 2048, device='cuda', dtype=torch.float32)
    for _ in range(5):
        naive_softmax_triton(X)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        naive_softmax_triton(X)
        torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=5))