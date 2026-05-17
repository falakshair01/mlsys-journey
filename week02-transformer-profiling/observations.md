# Week 02 — GPT-2 Full Forward Pass Profiling

## Goal of This Week
Profile a real transformer (GPT-2) end-to-end.
Identify which operations are memory-bound vs compute-bound inside a full model.
Challenge the assumption that attention is always the bottleneck.

---

## Hardware & Setup
- GPU: RTX 3050 Laptop, 4GB VRAM
- Model: GPT-2 small — 117M parameters, 12 layers, hidden dim 768
- CUDA: 13.0
- Code: `gpt2_profile.py`, `gpt2_scaling.py`

---

## GPT-2 Architecture — What Is Being Profiled

Each of the 12 transformer layers contains two blocks:

```
Input
  └── [Attention Block]   ← O(S²) — scores matrix grows quadratically
  └── [MLP Block]         ← O(S)  — linear growth, but massive weight matrices
Output
```

**MLP block internals (per layer):**
```
Linear: 768 → 3072   (4x expansion)
GELU activation
Linear: 3072 → 768   (compression back)
```

These two weight matrices are large. This matters.

---

## Key Results — Scaling By Sequence Length

| Seq Len | Total Time | Attention % | MLP % | Bottleneck |
|---------|------------|-------------|-------|------------|
| 16      | 4.26ms     | 3.8%        | 61.1% | MLP        |
| 32      | 4.99ms     | 3.3%        | 58.2% | MLP        |
| 64      | 6.75ms     | 3.0%        | 57.6% | MLP        |
| 128     | 12.36ms    | 4.9%        | 53.1% | MLP        |
| 256     | 23.87ms    | 6.1%        | 50.7% | MLP        |
| 512     | 54.21ms    | 8.8%        | 46.2% | MLP        |

**MLP dominates at every sequence length tested.**
Attention never exceeds 8.8% of total GPU time in this range.

---

## Scaling Behavior — Theory vs Measurement

### Attention Scaling (should be O(S²) → 4x per doubling)
```
128 → 256 tokens: 2.4x slowdown
256 → 512 tokens: 3.3x slowdown
```
Approaching quadratic. Will hit 4x at longer sequences.

### MLP Scaling (should be O(S) → 2x per doubling)
```
128 → 256 tokens: 1.8x slowdown
256 → 512 tokens: 2.1x slowdown
```
Approximately linear. Matches theory.

**Measurement confirms theory on real hardware.**

---

## Insight 1 — Why MLP Dominates At seq=512

At seq=512, MLP processes each token through two massive matrix multiplies across all 12 layers:

```
Per layer MLP work:
  512 tokens × (768×3072 + 3072×768) = ~4.7 million operations

Per layer Attention work:
  512×512 scores matrix = 262,144 operations
```

Even though attention is O(S²), at seq=512 its constant factor is still tiny compared to MLP's enormous weight matrices. The profiler confirms this: **MLP = 46.2% of GPU time, Attention = 8.8%.**

---

## Insight 2 — The Crossover Point

At what sequence length does attention finally become the bottleneck?

**Setting attention work = MLP work per layer:**
```
Attention: S² × 768
MLP:       S  × (768 × 3072 × 2) = S × 4,718,592

S² × 768 = S × 4,718,592
S = 4,718,592 / 768
S ≈ 6,144 tokens
```

```
Below ~6,000 tokens  →  MLP is the bottleneck
Above ~6,000 tokens  →  Attention becomes the bottleneck
```

This is not guesswork. This is derived from measured architecture dimensions.

---

## Insight 3 — Real World Optimization Implications

| Use Case | Typical Seq Length | Bottleneck | Right Optimization |
|---|---|---|---|
| Chatbot (short replies) | 200–500 tokens | MLP | Quantize MLP weights (4-bit, 8-bit) |
| Document Q&A | 2,000–4,000 tokens | MLP still | Quantization + batching |
| Long context (RAG, legal) | 32,000+ tokens | Attention | FlashAttention, PagedAttention |
| Code generation | 8,000–16,000 tokens | Transitioning | Both matter |

**Practical rule:** For a chatbot serving 500-token conversations, optimizing attention first is the wrong move. MLP quantization will give 2-4x more speedup per engineering hour.

---

## Insight 4 — The Wrong Assumption Corrected

**Before Week 2:** Assumed attention was always the primary bottleneck in transformers because of its O(S²) scaling proved in Week 1.

**After Week 2:** Attention is only the bottleneck at long sequences (>6K tokens). At typical real-world context lengths, MLP's massive weight matrices dominate GPU time.

**The lesson:** Never assume where the bottleneck is. Always profile the actual workload. The Week 1 isolated attention experiment was correct but incomplete — it did not represent a full model under real conditions.

This is the core skill of MLSys research: measuring before optimizing.

---

## Connection To Week 1

```
Week 1 proved:   Isolated attention scales O(S²)
                 At seq=2048: 1.29GB VRAM, 75ms, softmax = 21% of time

Week 2 proved:   Inside a full transformer, attention is NOT always dominant
                 MLP takes 46-61% of GPU time at sequences below 6K tokens
                 Optimizing attention alone would miss the real bottleneck
```

These two results together tell the complete story:
- For short-context inference → quantize MLP weights
- For long-context inference → FlashAttention + PagedAttention
- For serving systems → continuous batching to maximize GPU utilization (Week 3)

---

## What Comes Next — Week 3

Profile vLLM vs naive HuggingFace serving on a small model (Qwen 1.5B).

Questions Week 3 will answer:
- How much faster is vLLM than naive serving?
- What specifically does vLLM do differently? (PagedAttention, continuous batching)
- Where does time go in a real serving system vs a single forward pass?
- How does GPU utilization change under concurrent requests?

---

## Key Takeaway — One Paragraph

GPT-2 profiling revealed that transformers have two distinct bottlenecks depending on sequence length. MLP layers — with their 768→3072→768 weight matrices — dominate GPU time at sequences below approximately 6,144 tokens, consuming 46-61% of total compute. Attention only becomes the primary bottleneck above this crossover point due to its quadratic O(S²) scaling. This means that for typical chatbot workloads (200-500 tokens), quantizing MLP weights delivers more speedup than FlashAttention. For long-context workloads (32K+ tokens), attention optimization becomes critical. The core MLSys skill demonstrated here: profile first, identify the actual bottleneck, then optimize the right thing.