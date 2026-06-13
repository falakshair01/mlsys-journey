# Week 07–08: Fused Softmax Kernel — Findings

## Goal
Replace naive softmax (4 VRAM round trips) with a fused Triton kernel
(1 VRAM round trip) and measure real impact on forward pass time.

## Hardware
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU
- SMs: 16, CUDA cores: 2,048, VRAM: 4.0 GB
- Peak bandwidth: 192 GB/s

---

## The Problem With Naive Softmax

Naive softmax makes 4 VRAM round trips per row:

```
Step 1: read X → compute max → write max to VRAM
Step 2: read X → compute exp(x - max) → write exp to VRAM
Step 3: read exp → compute sum → write sum to VRAM
Step 4: read exp + read sum → divide → write output to VRAM
```

At seq=2048 this is a 2048×2048 matrix.
Every round trip consumes bandwidth on the 192 GB/s conveyor belt.
Week 1 showed this cost: softmax = 21% of total GPU time.

## Why Max Subtraction Is Required

exp() overflows float32 above ~89. Raw attention scores can reach
100+ easily. Without max subtraction:

```
exp(100) = 2.7e43   ← near float32 limit
exp(800) = inf      ← overflow
inf / inf = NaN     ← broken output
```

With max subtraction (subtract row max before exp):
```
largest input to exp() = exp(0) = 1.0   ← always safe
all other values between 0 and 1        ← never overflow
```

Discovered during implementation: first kernel returned
Max difference: inf — fixed by adding tl.where(mask, exp_x, 0.0)
to zero out masked padding positions before sum.

---

## Experiment 1: Naive vs Fused vs PyTorch ref

```python
# Naive Triton — uses zeros_like (extra kernel launch)
O = torch.zeros_like(X)

# Fused Triton — uses empty_like (no initialization needed)
O = torch.empty_like(X)
# writes every position explicitly in one kernel pass
```

### Results

| Seq Len | Naive Triton | Fused Triton | PyTorch ref | Speedup |
|---------|-------------|-------------|-------------|---------|
| 512     | 0.0517ms    | 0.0378ms    | 0.0184ms    | 1.37x   |
| 1024    | 0.0717ms    | 0.0484ms    | 0.0481ms    | 1.48x   |
| 2048    | 0.2757ms    | 0.1856ms    | 0.1875ms    | 1.49x   |

Key result: fused kernel matched PyTorch ref at seq=1024 exactly
and beat it at seq=2048 (0.1856ms vs 0.1875ms).

---

## Experiment 2: Profiler — Where The Time Went

```
Naive kernel profiler (seq=2048):
  naive_softmax_kernel:           184us   67%
  vectorized_elementwise_kernel:   90us   33%  ← zeros_like overhead
  Total CUDA time:                274us

Fused kernel profiler (seq=2048):
  fused_softmax_kernel:           183us  100%
  vectorized_elementwise_kernel:    0us    0%  ← eliminated
  Total CUDA time:                183us
```

The zeros_like kernel was 33% of naive kernel total time.
Switching to empty_like eliminated it entirely.
Actual computation time was identical (183-184us) — only the
wasted initialization was removed.

---

## Experiment 3: Amdahl's Law — System Impact

```python
# Week 1 baseline
total_time       = 75.5ms     # full forward pass at seq=2048
softmax_fraction = 0.21       # softmax was 21% of total GPU time
softmax_time     = 15.85ms

# Our improvement
reduction        = 0.327      # 32.7% faster softmax
time_saved       = 5.18ms
new_total        = 70.32ms
overall_speedup  = 1.074x

# Theoretical ceiling (Amdahl's Law)
max_speedup      = 1 / (1 - 0.21) = 1.266x
captured         = 1.074 / 1.266  = 84.8% of maximum possible gain
```

Amdahl's Law: no matter how fast softmax becomes, the forward pass
cannot speed up more than 1.266x because softmax is only 21% of
total work. We captured 84.8% of that ceiling.

---

## What Fused Kernel Actually Does

```
Naive (4 VRAM round trips):
  VRAM → registers (load x)
  registers → VRAM (write max)
  VRAM → registers (load exp)
  registers → VRAM (write exp)
  VRAM → registers (load exp again)
  registers → VRAM (write sum)
  VRAM → registers (load exp + sum)
  registers → VRAM (write output)

Fused (1 VRAM round trip):
  VRAM → registers (load x)        ← only read
  registers: compute max            ← stays in registers
  registers: compute exp(x - max)   ← stays in registers
  registers: compute sum            ← stays in registers
  registers: compute output         ← stays in registers
  registers → VRAM (write output)  ← only write
```

---

## Connection To Previous Weeks

Week 1: softmax = 21% GPU time at seq=2048 due to S×S VRAM writes
Week 2: MLP bottleneck below S=6,144 — Amdahl explains why softmax
        optimization alone cannot fix the full model
Week 5-6: learned thread blocks, SM scheduling, bandwidth efficiency
Week 7-8: applied that knowledge — fused kernel beats PyTorch ref

## Key Insight
Amdahl's Law is why Week 2 crossover matters.
Below S=6,144: MLP is 46% of time → optimize MLP first
Above S=6,144: Attention dominates → FlashAttention gives real gains
Always optimize the largest bottleneck. Optimizing a 21% component
can never give more than 1.266x system speedup regardless of
how perfect the kernel is.

## Files
- naive_softmax.py  — baseline with zeros_like overhead measured
- fused_softmax.py  — fused kernel beating PyTorch ref at seq=2048
- findings.md       — this file
