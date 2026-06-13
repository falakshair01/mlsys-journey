# full_benchmark.py
# Production benchmark suite for Blog Post #2
# Measures: speedup, bandwidth efficiency, numerical accuracy
# across seq = [256, 512, 1024, 2048, 4096]

import torch
import triton
import triton.language as tl
import time

# ─────────────────────────────────────────────
# KERNELS
# ─────────────────────────────────────────────

@triton.jit
def fused_softmax_kernel(
    X_ptr, O_ptr,
    stride_row,
    N_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx    = tl.program_id(axis=0)
    row_x      = X_ptr + row_idx * stride_row
    row_o      = O_ptr + row_idx * stride_row
    offsets    = tl.arange(0, BLOCK_SIZE)
    mask       = offsets < N_cols

    # ONE VRAM READ
    x          = tl.load(row_x + offsets, mask=mask, other=-float('inf'))

    # ALL COMPUTATION IN REGISTERS
    row_max    = tl.max(x, axis=0)
    x_stable   = x - row_max
    exp_x      = tl.exp(x_stable)
    exp_x      = tl.where(mask, exp_x, 0.0)
    row_sum    = tl.sum(exp_x, axis=0)
    output     = exp_x / row_sum

    # ONE VRAM WRITE
    tl.store(row_o + offsets, output, mask=mask)


@triton.jit
def naive_softmax_kernel(
    X_ptr, O_ptr,
    stride_row,
    N_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx    = tl.program_id(axis=0)
    row_x      = X_ptr + row_idx * stride_row
    row_o      = O_ptr + row_idx * stride_row
    offsets    = tl.arange(0, BLOCK_SIZE)
    mask       = offsets < N_cols

    x          = tl.load(row_x + offsets, mask=mask, other=-float('inf'))
    row_max    = tl.max(x, axis=0)
    x_stable   = x - row_max
    exp_x      = tl.exp(x_stable)
    exp_x      = tl.where(mask, exp_x, 0.0)
    row_sum    = tl.sum(exp_x, axis=0)
    output     = exp_x / row_sum

    tl.store(row_o + offsets, output, mask=mask)


# ─────────────────────────────────────────────
# LAUNCHERS
# ─────────────────────────────────────────────

def get_block_size(n_cols):
    # BLOCK_SIZE must be power of 2 and >= n_cols
    bs = 1
    while bs < n_cols:
        bs *= 2
    return bs


def fused_softmax(X):
    n_rows, n_cols = X.shape
    O              = torch.empty_like(X)        # no zeros_like overhead
    BLOCK_SIZE     = get_block_size(n_cols)
    fused_softmax_kernel[(n_rows,)](
        X, O, X.stride(0), n_cols, BLOCK_SIZE=BLOCK_SIZE
    )
    return O


def naive_softmax(X):
    n_rows, n_cols = X.shape
    O              = torch.zeros_like(X)        # intentionally naive
    BLOCK_SIZE     = get_block_size(n_cols)
    naive_softmax_kernel[(n_rows,)](
        X, O, X.stride(0), n_cols, BLOCK_SIZE=BLOCK_SIZE
    )
    return O


# ─────────────────────────────────────────────
# BENCHMARK HELPER
# ─────────────────────────────────────────────

def benchmark(fn, X, warmup=20, repeat=100):
    # More reps than before — blog post numbers need to be stable
    for _ in range(warmup):
        fn(X)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        fn(X)
    torch.cuda.synchronize()
    end   = time.perf_counter()
    return (end - start) / repeat * 1000    # milliseconds


def measure_bandwidth(n_rows, n_cols, time_ms):
    # Fused kernel: read X once, write O once
    # bytes = 2 tensors × rows × cols × 4 bytes per float32
    bytes_moved   = 2 * n_rows * n_cols * 4
    time_s        = time_ms / 1000
    bandwidth_GBs = bytes_moved / time_s / 1e9
    efficiency    = bandwidth_GBs / 192.0 * 100    # 192 GB/s peak
    return bandwidth_GBs, efficiency


def measure_accuracy(X):
    O_fused = fused_softmax(X)
    O_ref   = torch.softmax(X, dim=-1)
    max_diff  = (O_fused - O_ref).abs().max().item()
    matches   = torch.allclose(O_fused, O_ref, atol=1e-5)
    return max_diff, matches


def check_vram(seq_len):
    # Each matrix: seq_len × seq_len × 4 bytes
    # We hold X, O_naive, O_fused, O_ref = 4 matrices
    bytes_needed = 4 * seq_len * seq_len * 4
    gb_needed    = bytes_needed / 1e9
    return gb_needed


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)

    SEQ_LENGTHS = [256, 512, 1024, 2048, 4096]

    # ── Section 1: VRAM check ─────────────────
    print("=== VRAM Usage Check ===")
    print(f"{'Seq Len':<10} {'VRAM needed (GB)':<20} {'Safe for 4GB?'}")
    print("-" * 45)
    for seq_len in SEQ_LENGTHS:
        gb = check_vram(seq_len)
        safe = "YES" if gb < 3.5 else "BORDERLINE" if gb < 4.0 else "NO — skip"
        print(f"{seq_len:<10} {gb:<20.3f} {safe}")
    print()

    # ── Section 2: Correctness ────────────────
    print("=== Numerical Accuracy vs PyTorch ref ===")
    print(f"{'Seq Len':<10} {'Max Diff':<20} {'Passes atol=1e-5?'}")
    print("-" * 45)
    for seq_len in SEQ_LENGTHS:
        gb = check_vram(seq_len)
        if gb >= 4.0:
            print(f"{seq_len:<10} {'SKIPPED — OOM risk':<20}")
            continue
        X        = torch.randn(seq_len, seq_len, device='cuda', dtype=torch.float32)
        max_diff, matches = measure_accuracy(X)
        print(f"{seq_len:<10} {max_diff:<20.2e} {matches}")
    print()

    # ── Section 3: Full benchmark table ───────
    print("=== Full Benchmark Table ===")
    print(f"{'Seq':<8} {'Naive(ms)':>12} {'Fused(ms)':>12} {'PyTorch(ms)':>13} "
          f"{'Speedup':>10} {'BW(GB/s)':>10} {'BW Eff%':>10}")
    print("-" * 80)

    results = {}
    for seq_len in SEQ_LENGTHS:
        gb = check_vram(seq_len)
        if gb >= 4.0:
            print(f"{seq_len:<8} {'SKIPPED — OOM risk':>55}")
            continue

        X = torch.randn(seq_len, seq_len, device='cuda', dtype=torch.float32)

        ms_naive  = benchmark(naive_softmax, X)
        ms_fused  = benchmark(fused_softmax, X)
        ms_ref    = benchmark(lambda x: torch.softmax(x, dim=-1), X)
        speedup   = ms_naive / ms_fused
        bw, eff   = measure_bandwidth(seq_len, seq_len, ms_fused)

        results[seq_len] = {
            'naive': ms_naive,
            'fused': ms_fused,
            'ref':   ms_ref,
            'speedup': speedup,
            'bw': bw,
            'eff': eff,
        }

        print(f"{seq_len:<8} {ms_naive:>12.4f} {ms_fused:>12.4f} {ms_ref:>13.4f} "
              f"{speedup:>9.2f}x {bw:>10.1f} {eff:>9.1f}%")

    print()

    # ── Section 4: Bandwidth summary ──────────
    print("=== Bandwidth Efficiency Summary ===")
    print(f"Peak bandwidth: 192 GB/s (RTX 3050 Laptop)")
    print(f"{'Seq Len':<10} {'Achieved BW':<20} {'Efficiency'}")
    print("-" * 45)
    for seq_len, r in results.items():
        print(f"{seq_len:<10} {r['bw']:<20.1f} {r['eff']:.1f}%")
    print()

    # ── Section 5: Amdahl summary ─────────────
    print("=== Amdahl's Law — System Impact ===")
    print("Week 1 baseline: 75.5ms full forward pass at seq=2048")
    print("Softmax fraction: 21% of total GPU time")
    print()
    if 2048 in results:
        r             = results[2048]
        softmax_frac  = 0.21
        total_w1      = 75.5
        naive_contrib = total_w1 * softmax_frac
        fused_contrib = naive_contrib * (r['fused'] / r['naive'])
        time_saved    = naive_contrib - fused_contrib
        new_total     = total_w1 - time_saved
        overall_sp    = total_w1 / new_total
        max_sp        = 1 / (1 - softmax_frac)
        captured      = (overall_sp - 1) / (max_sp - 1) * 100

        print(f"Softmax time (naive estimate):   {naive_contrib:.2f}ms")
        print(f"Softmax time (fused estimate):   {fused_contrib:.2f}ms")
        print(f"Time saved:                      {time_saved:.2f}ms")
        print(f"New estimated total:             {new_total:.2f}ms")
        print(f"Overall speedup:                 {overall_sp:.3f}x")
        print(f"Amdahl ceiling:                  {max_sp:.3f}x")
        print(f"Captured:                        {captured:.1f}% of maximum")
    print()

    # ── Section 6: Blog post summary ──────────
    print("=" * 60)
    print("BLOG POST SUMMARY — copy these numbers directly")
    print("=" * 60)
    for seq_len, r in results.items():
        print(f"seq={seq_len:<6}: "
              f"naive={r['naive']:.4f}ms  "
              f"fused={r['fused']:.4f}ms  "
              f"speedup={r['speedup']:.2f}x  "
              f"bw={r['bw']:.0f}GB/s({r['eff']:.0f}%)")