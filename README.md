# MLSys Journey

A documented, week-by-week learning and research portfolio focused on **Machine Learning Systems (MLSys)** — specifically LLM inference optimization and hardware-level performance analysis.

## Goal

Secure an **RA (Research Assistant) position** in an MLSys lab for the 2026/2027 intake across US, Canada, UAE (MBZUAI), or KSA (KAUST).

## My Background

- Computer Engineering graduate, 2024
- Python backend engineer — FastAPI, Django, Docker, RAGs, DevOps
- Hardware: RTX 3050 Laptop GPU, 4GB VRAM

## Research Focus

> **Machine Learning Systems — LLM Inference Optimization**
>
> Specifically: identifying and resolving hardware-level memory bottlenecks during LLM inference using profiling, custom Triton kernels, and systems-level analysis of inference frameworks like vLLM.

---

## Repository Structure

```
mlsys-journey/
├── README.md                        ← You are here
│
├── week01-profiling/
│   ├── matmul_profile.py            ← Benchmark matmul across matrix sizes
│   ├── attention_profile.py         ← Profile standard attention vs sequence length
│   └── observations.md              ← Written analysis of profiling results
│
├── week02-transformer-profiling/    ← GPT-2 forward pass profiling
│
├── week03-vllm-vs-hf/              ← Coming: vLLM vs HuggingFace serving comparison
│
├── week05-triton/                   ← Coming: First Triton kernel (fused softmax)
│
├── week11-paper-teardown/           ← Coming: MLSys 2024 paper implementation
│
└── blogs/                           ← Drafts for published technical blog posts
```

---

## Roadmap

### Phase 1 — GPU Mental Model + Profiling (Weeks 1–4)
Learn to see where time is wasted before attempting to fix anything.

- [x] Week 1: Profile matmul and attention. Learn memory-bound vs compute-bound.
- [X] Week 2: Profile GPT-2 full forward pass
- [ ] Week 3: Profile vLLM vs naive HuggingFace serving
- [ ] Week 4: Write Phase 1 summary blog draft

### Phase 2 — Triton Kernels (Weeks 5–10)
Write real GPU kernels. Measure the difference.

- [ ] Week 5–6: Triton mental model — thread blocks, memory coalescing
- [ ] Week 7–8: Write fused softmax Triton kernel
- [ ] Week 9–10: Benchmark and publish Blog Post #1

### Phase 3 — Paper Teardown (Weeks 11–20)
Pick one MLSys 2024 paper. Implement its core mechanism. Profile it. Write about it.

- [ ] Week 11–12: Select and fully read target paper
- [ ] Week 13–16: Implement naive version of core mechanism
- [ ] Week 17–18: Profile and document bottlenecks
- [ ] Week 19–20: Publish Blog Post #2 — full technical teardown

### Phase 4 — Professor Outreach (Weeks 21–28)
Cold email 20 professors with concrete technical proof attached.

- [ ] Week 21–23: Build target professor spreadsheet
- [ ] Week 24–25: Read 2 papers per professor
- [ ] Week 26–28: Send emails in batches of 5

### Phase 5 — Polish + Applications (Weeks 29–32)
- [ ] GitHub cleanup and README polish
- [ ] Follow-up emails
- [ ] Portal applications where professor has responded

---

## Key Concepts Learned

| Concept | Plain English Summary |
|---|---|
| Memory-bound operation | GPU compute cores sit idle waiting for data from memory |
| Compute-bound operation | Memory feeds data fast enough, GPU math units are the bottleneck |
| GPU Memory Bandwidth | Speed of conveyor belt from VRAM to compute cores (192 GB/s on RTX 3050) |
| GPU Compute (TFLOPS) | Speed of workers once they have data (9 TFLOPS on RTX 3050) |
| VRAM | Warehouse size — how much data fits on GPU (4GB on RTX 3050) |
| cudaDeviceSynchronize | CPU pausing and waiting for GPU to finish |
| PyTorch Profiler | Stopwatch on every operation — shows exactly where time is spent |
| FlashAttention insight | Avoid repeated trips to VRAM by keeping data on compute cores |

---

## Resources

| Resource | Purpose |
|---|---|
| [Making Deep Learning Go Brrrr — Horace He](https://horace.io/brrr_intro.html) | Memory-bound vs compute-bound foundation |
| [Let's Build GPT — Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY) | Transformer intuition from code |
| [Triton Official Tutorial](https://triton-lang.org/main/getting-started/tutorials/) | Kernel writing |
| [vLLM Paper — PagedAttention](https://arxiv.org/abs/2309.06180) | Core inference optimization paper |
| [Orca Paper — Continuous Batching](https://www.usenix.org/conference/osdi22/presentation/yu) | LLM serving efficiency |
| [MLSys 2024 Accepted Papers](https://mlsys.org/virtual/2024/papers.html) | Paper teardown selection |

---

## Contact

Building this portfolio for MLSys RA applications — 2026/2027 intake.
Focused on inference optimization, GPU profiling, and Triton kernel development.
