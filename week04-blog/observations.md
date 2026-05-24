# Week 04 — Phase 1 Blog Post

**Published on Medium:**
[I Profiled LLM Inference from First Principles — Here's What I Found](https://medium.com/@falakshair563/i-profiled-llm-inference-from-first-principles-heres-what-i-found-823083e502dc)

---

## What This Post Covers

A three-week hands-on investigation into LLM inference bottlenecks on an RTX 3050,
synthesized into a single public article. Each week corrected a prior assumption
with real measurements.

---

## Week 1 — Attention's O(S²) Memory Problem

Standard attention creates quadratic memory complexity relative to sequence length.

**Measurements on RTX 3050:**

| Sequence Length | CUDA Memory | Execution Time |
|---|---|---|
| 512 tokens | 90 MB | 4.4 ms |
| 1,024 tokens | 340 MB | 16.9 ms |
| 2,048 tokens | 1.29 GB | 75.5 ms |

At 2,048 tokens, softmax consumed **21% of GPU execution time** — a memory-bandwidth
bottleneck, not a compute bottleneck. This is precisely why FlashAttention's tiling
approach (avoiding full matrix construction) provides meaningful gains.

---

## Week 2 — MLP is the Real Bottleneck at Production Lengths

**GPT-2 profiling at seq=512:**

| Component | GPU Time |
|---|---|
| MLP (feed-forward) | 46.2% |
| Attention | 8.8% |
| Other | 45.0% |

The crossover point where attention overtakes MLP is around **S ≈ 6,000 tokens**,
derivable from architecture dimensions alone.

**Production implication:** Most real deployments operate below 2K tokens. In that
regime, quantization (targeting MLP) delivers more gains than FlashAttention
(targeting attention). The research literature's emphasis on long-context attention
optimization does not map to typical serving workloads.

---

## Week 3 — Production Serving Overhead

GPU utilization stayed at **42%** — the remaining 58% was CPU-side waiting.

**Throughput comparison:**

| Method | Throughput |
|---|---|
| Sequential naive (HuggingFace) | 0.42 req/s |
| Batched naive (HuggingFace) | 1.73 req/s (4.2× over sequential) |
| vLLM (Colab T4) | 2.52 req/s (6× over sequential) |

vLLM's infrastructure — continuous batching, PagedAttention, kernel scheduling —
adds ~0.9GB overhead beyond model weights. On a 4GB GPU this causes OOM even when
the model fits. The 6× throughput gain is real, but so is the memory tax.

**Limitation noted in the post:** The vLLM comparison used different hardware
(RTX 3050 vs. Colab T4), confounding software and hardware effects. A controlled
apples-to-apples experiment would strengthen the claim.

---

## Key Conclusions

1. **Profile first, then optimize.** Every week corrected a prior assumption.
2. **Optimization is workload-dependent.** Sequence length determines whether
   quantization or FlashAttention is the right lever.
3. **Infrastructure is a first-class cost.** Serving overhead, not model size,
   is often what causes OOM in production.
4. **Hardware constraints are real.** 4GB VRAM is insufficient for production
   vLLM — not because the model doesn't fit, but because the serving layer needs room.

---

## What's Next

Phase 2: Writing Triton kernels to directly target the **21% softmax overhead**
identified in Week 1 profiling.
