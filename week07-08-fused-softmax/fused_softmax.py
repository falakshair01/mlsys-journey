# fused_softmax.py
# Goal: do max, exp, sum, normalize in ONE pass
# Keep all intermediates in registers — never touch VRAM between steps
# No zeros_like — write every position explicitly

import torch
import triton
import triton.language as tl
import time

# ─────────────────────────────────────────────
# FUSED SOFTMAX KERNEL
# One VRAM read, one VRAM write, everything else in registers
# ─────────────────────────────────────────────

@triton.jit
def fused_softmax_kernel(
    X_ptr,               # input in VRAM
    O_ptr,               # output in VRAM
    stride_row,          # elements to skip per row
    N_cols,              # actual number of columns
    BLOCK_SIZE: tl.constexpr,
):
    # One block handles one entire row
    row_idx = tl.program_id(axis=0)

    # Pointer to start of this row
    row_start_x = X_ptr + row_idx * stride_row
    row_start_o = O_ptr + row_idx * stride_row

    # Column offsets for this block
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_cols

    # ── VRAM READ #1 — only read in the entire kernel ────────────
    # Load the full row into registers
    # Masked positions get -inf so they never affect max or sum
    x = tl.load(row_start_x + offsets, mask=mask, other=-float('inf'))

    # ── ALL COMPUTATION HAPPENS IN REGISTERS FROM HERE ───────────

    # Step 1: find max across the row
    # This is a reduction — Triton handles cross-thread communication
    row_max = tl.max(x, axis=0)
    # row_max is now a single scalar sitting in every thread's register

    # Step 2: subtract max for numerical stability, compute exp
    # x is still in registers — no VRAM read needed
    x_stable = x - row_max
    exp_x = tl.exp(x_stable)
    # exp(-inf) = 0 for masked positions, so they contribute nothing to sum

    # Step 3: sum all exp values across the row
    # Again a reduction — Triton handles it
    # We zero out masked positions explicitly to be safe
    exp_x_clean = tl.where(mask, exp_x, 0.0)
    row_sum = tl.sum(exp_x_clean, axis=0)
    # row_sum is a scalar in registers

    # Step 4: normalize
    output = exp_x_clean / row_sum
    # output is the final softmax probabilities, still in registers

    # ── VRAM WRITE #1 — only write in the entire kernel ──────────
    tl.store(row_start_o + offsets, output, mask=mask)


# ─────────────────────────────────────────────
# LAUNCHER
# ─────────────────────────────────────────────

def fused_softmax(X, BLOCK_SIZE=None):
    n_rows, n_cols = X.shape

    # No zeros_like — we write every position explicitly in the kernel
    O = torch.empty_like(X)

    if BLOCK_SIZE is None:
        BLOCK_SIZE = 1
        while BLOCK_SIZE < n_cols:
            BLOCK_SIZE *= 2

    # One block per row — each block handles the full row in one pass
    grid = (n_rows,)

    fused_softmax_kernel[grid](
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
# NAIVE SOFTMAX — for comparison
# Same as naive_softmax.py but inline here for side by side numbers
# ─────────────────────────────────────────────

@triton.jit
def naive_softmax_kernel(
    X_ptr, O_ptr, stride_row, N_cols, BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(axis=0)
    row_start_x = X_ptr + row_idx * stride_row
    row_start_o = O_ptr + row_idx * stride_row
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_cols
    x = tl.load(row_start_x + offsets, mask=mask, other=-float('inf'))
    row_max = tl.max(x, axis=0)
    x_stable = x - row_max
    exp_x = tl.exp(x_stable)
    exp_x = tl.where(mask, exp_x, 0.0)
    row_sum = tl.sum(exp_x, axis=0)
    output = exp_x / row_sum
    tl.store(row_start_o + offsets, output, mask=mask)

def naive_softmax_triton(X, BLOCK_SIZE=None):
    n_rows, n_cols = X.shape
    O = torch.zeros_like(X)              # ← zeros_like, the overhead we measured
    if BLOCK_SIZE is None:
        BLOCK_SIZE = 1
        while BLOCK_SIZE < n_cols:
            BLOCK_SIZE *= 2
    naive_softmax_kernel[(n_rows,)](X, O, X.stride(0), n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return O


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # ── Correctness check ─────────────────────
    print("=== Correctness Check ===")
    torch.manual_seed(0)
    X = torch.randn(128, 512, device='cuda', dtype=torch.float32)

    O_fused   = fused_softmax(X)
    O_ref     = torch.softmax(X, dim=-1)
    O_naive   = naive_softmax_triton(X)

    print(f"Fused  matches PyTorch ref: {torch.allclose(O_fused, O_ref, atol=1e-5)}")
    print(f"Naive  matches PyTorch ref: {torch.allclose(O_naive, O_ref, atol=1e-5)}")
    print()

    # ── Side by side benchmark ─────────────────
    print("=== Side By Side Benchmark ===")
    print(f"{'Seq':<8} {'Naive Triton':>14} {'Fused Triton':>14} {'PyTorch ref':>14} {'Speedup':>10}")
    print("-" * 65)

    for seq_len in [512, 1024, 2048]:
        X = torch.randn(seq_len, seq_len, device='cuda', dtype=torch.float32)

        ms_naive  = benchmark(naive_softmax_triton, X)
        ms_fused  = benchmark(fused_softmax, X)
        ms_ref    = benchmark(lambda x: torch.softmax(x, dim=-1), X)
        speedup   = ms_naive / ms_fused

        print(f"{seq_len:<8} {ms_naive:>14.4f} {ms_fused:>14.4f} {ms_ref:>14.4f} {speedup:>9.2f}x")

    print()

    # ── Profiler on fused kernel at seq=2048 ──
    print("=== Profiler Output (Fused, seq=2048) ===")
    X = torch.randn(2048, 2048, device='cuda', dtype=torch.float32)
    for _ in range(5):
        fused_softmax(X)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        fused_softmax(X)
        torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=5))

    # ── Connect back to Week 1 ─────────────────
    print()
    print("=== Connection to Week 1 ===")
    print("Week 1: softmax was 21% of total GPU time at seq=2048")
    print("Week 1 total forward pass time at seq=2048: 75.5ms")
    week1_softmax_time = 75.5 * 0.21
    X = torch.randn(2048, 2048, device='cuda', dtype=torch.float32)
    ms_fused_2048 = benchmark(fused_softmax, X)
    ms_naive_2048 = benchmark(naive_softmax_triton, X)
    print(f"Week 1 softmax time (estimated): {week1_softmax_time:.2f}ms")
    print(f"Our naive Triton softmax:        {ms_naive_2048:.4f}ms")
    print(f"Our fused Triton softmax:        {ms_fused_2048:.4f}ms")
    print(f"Reduction from naive to fused:   {(1 - ms_fused_2048/ms_naive_2048)*100:.1f}%")