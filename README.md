# MLSys Journey

A documented, week-by-week research and engineering portfolio focused on
**Machine Learning Systems (MLSys)** — specifically LLM inference optimization
and hardware-level performance analysis.

---

## Why I Started This

A few months ago, a colleague came to me frustrated.

He was trying to run a local LLM on CPU for our company's knowledge base system.
The model ran — but inference was painfully slow, and accuracy degraded under
the hardware constraints. When they later moved to a GPU-based server, a
deep question through the knowledge base still took nearly a minute to answer.

He was not doing anything wrong. The hardware was the problem. But neither of
us could explain *why* precisely — or what to actually fix.

That question stayed with me. I started reading about how LLMs actually execute
on hardware. The early days were overwhelming — new terminology everywhere,
concepts that assumed background I did not have. But somewhere between reading
about memory bandwidth and compute throughput, something clicked. I was not
confused anymore. I was curious.

I realized the problem my colleague faced was not unique. Every company
deploying local LLMs hits the same wall: the model works, but the system
does not. Inference is slow. Memory runs out. GPU utilization is poor.
These are not ML problems. They are systems problems.

That is what this repository documents — my journey from "why is this slow"
to being able to measure, explain, and eventually fix it.

---

## What I Am Building Toward

I am working toward contributing to MLSys research — the field that sits at
the intersection of machine learning and systems engineering. The engineers
and researchers in this space are the ones who made LLMs fast enough to
actually deploy. FlashAttention, PagedAttention, continuous batching —
these are not model improvements. They are systems improvements. And they
are what make the difference between a model that works in a notebook
and a model that serves millions of users.

My background is in backend engineering — FastAPI, Docker, on-prem deployment,
RAG pipelines. I understand systems. I understand deployment constraints.
I understand what it means when something works in theory but fails in
production. This repository is where I am learning to apply that intuition
at the hardware level.

---

## Research Focus

**Machine Learning Systems — LLM Inference Optimization**

> Identifying and resolving hardware-level memory bottlenecks during LLM
> inference using GPU profiling, custom Triton kernels, and systems-level
> analysis of inference frameworks like vLLM.

### What MLSys Is

MLSys is not about training new models or designing new architectures.
It is about making existing models run faster, cheaper, and more efficiently
on real hardware. The problems are concrete:

- Why does attention get slow at long sequences?
- Why is GPU utilization only 42% during naive serving?
- Why does vLLM need 0.9GB of infrastructure overhead beyond the model itself?
- Where exactly is time being wasted, and how do we stop wasting it?

These are the questions this repository works through — with real measurements,
on real hardware, one week at a time.

---

## Hardware

- NVIDIA RTX 3050 Laptop GPU — 4GB VRAM, 192 GB/s bandwidth
- Google Colab T4 — for experiments exceeding local VRAM

---

## Published Writing

| Post | Where |
|---|---|
| [I Profiled LLM Inference from First Principles — Here's What I Found](https://medium.com/@falakshair563/i-profiled-llm-inference-from-first-principles-heres-what-i-found-823083e502dc) | Medium — Phase 1 synthesis (Weeks 1–3) |

---

## What I Have Found So Far

Four weeks of profiling and a published blog post have already challenged assumptions:

**Attention is not always the bottleneck.**
At sequence lengths below ~6,144 tokens, MLP layers consume more GPU time
than attention — 46% vs 8.8% in GPT-2 at seq=512. The crossover point is
derivable from architecture dimensions. Optimizing attention for a chatbot
serving 500-token conversations is solving the wrong problem.

**Serving overhead is a first-class cost.**
vLLM's infrastructure — CUDA graphs, PagedAttention block tables, the
continuous batching scheduler — requires ~0.9GB beyond model weights.
On a 4GB GPU, this causes OOM even when the model fits. Production serving
systems need larger GPUs not because the models are bigger, but because
the serving layer itself needs room to operate.

**The bottleneck is never where you assume.**
Every week so far has corrected a prior assumption with real measurements.
This is the core skill of MLSys research: profile first, then optimize.

---

## Repository Structure

```
mlsys-journey/
├── README.md                            ← You are here
├── MASTER_OBSERVATIONS.md               ← Top-level findings, updated weekly
│
├── week01-profiling/
│   ├── matmul_profile.py                ← Matmul benchmark across sizes
│   ├── attention_profile.py             ← Attention profiling vs sequence length
│   └── observations.md                  ← O(S²) memory scaling proved on real GPU
│
├── week02-transformer-profiling/
│   ├── gpt2_profile.py                  ← GPT-2 full forward pass profiling
│   ├── gpt2_scaling.py                  ← Scaling analysis seq=16 to 512
│   └── observations.md                  ← MLP/attention crossover at S≈6,144
│
├── week03-vllm-vs-hf/
│   ├── naive_serving.py                 ← Sequential + batched HuggingFace serving
│   ├── vllm_serving.py                  ← vLLM on Colab T4
│   └── observations.md                  ← 6x speedup, 0.9GB serving overhead
│
├── week04-blog/                         ← Phase 1 blog post (published on Medium)
├── week05-triton/                       ← Coming: First Triton kernel
├── week11-paper-teardown/               ← Coming: MLSys 2024 paper implementation
└── blogs/                               ← Published technical writing
```

---

## Roadmap

### Phase 1 — GPU Mental Model + Profiling (Weeks 1–4)
- [x] Week 1: Proved attention's O(S²) memory scaling on real hardware
- [x] Week 2: Derived MLP/attention crossover point from architecture dimensions
- [x] Week 3: Measured vLLM serving overhead and continuous batching speedup
- [x] Week 4: Phase 1 blog post — [published on Medium](https://medium.com/@falakshair563/i-profiled-llm-inference-from-first-principles-heres-what-i-found-823083e502dc)

### Phase 2 — Triton Kernels (Weeks 5–10)
- [ ] Week 5–6: GPU thread model, memory coalescing, Triton fundamentals
- [ ] Week 7–8: Fused softmax Triton kernel — target the 21% softmax overhead
- [ ] Week 9–10: Benchmark and publish Blog Post #1

### Phase 3 — Paper Teardown (Weeks 11–20)
- [ ] Week 11–12: Select and fully read one MLSys 2024 / OSDI 2024 paper
- [ ] Week 13–16: Implement the core mechanism from scratch
- [ ] Week 17–18: Profile and document bottlenecks
- [ ] Week 19–20: Publish Blog Post #2 — full technical teardown

### Phase 4 — Professor Outreach (Weeks 21–28)
- [ ] Identify 20 target professors in US, Canada, UAE, KSA
- [ ] Read their recent papers, find specific connection points
- [ ] Cold email with GitHub and blog attached — technical proof first

---

## Key Resources

| Resource | What It Taught Me |
|---|---|
| [Making Deep Learning Go Brrrr — Horace He](https://horace.io/brrr_intro.html) | Memory-bound vs compute-bound from first principles |
| [FlashAttention — Dao et al., 2022](https://arxiv.org/abs/2205.14135) | Why tiling eliminates O(S²) memory |
| [vLLM / PagedAttention — Kwon et al., 2023](https://arxiv.org/abs/2309.06180) | KV cache as virtual memory |
| [Orca — Yu et al., 2022](https://www.usenix.org/conference/osdi22/presentation/yu) | Continuous batching and iteration-level scheduling |
