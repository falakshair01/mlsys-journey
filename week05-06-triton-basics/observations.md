# Week 05–06: Triton Basics — Findings

## Hardware
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU
- SMs: 16
- CUDA cores: 2,048 (16 SMs × 128 cores each)
- VRAM: 4.0 GB
- Peak bandwidth: 192 GB/s

## Experiment 1: BLOCK_SIZE vs Time
N = 1,000,000 float32 elements

| BLOCK_SIZE | Workers Launched | Time (ms) |
|------------|-----------------|-----------|
| 64         | 15,625          | 0.0790    |
| 128        | 7,813           | 0.0782    |
| 256        | 3,907           | 0.0764    |
| 512        | 1,954           | 0.0779    |
| 1024       | 977             | 0.0757    |
| 2048       | 489             | 0.0759    |
| 4096       | 245             | 0.0779    |
| PyTorch    | —               | 0.0742    |

What happened: Time barely changed across a 64x range of BLOCK_SIZE.
Why: Total work is fixed at 1,000,000 additions. BLOCK_SIZE only
changes how that work is sliced across blocks — not how much work
exists. More blocks means smaller slices and more rounds. Fewer blocks
means larger slices and fewer rounds. Total time stays roughly equal.

## Experiment 2: Bandwidth Utilization

```python
    bytes_moved = 3 * 1_000_000 * 4   # read A, read B, write C
    time_s = 71.839e-6                 # from profiler
    bandwidth_GBs = bytes_moved / time_s / 1e9
    print(f'Achieved bandwidth: {bandwidth_GBs:.1f} GB/s')
    print(f'Efficiency: {bandwidth_GBs/192*100:.1f}%')
```

- Data moved: 12.0 MB (read A, read B, write C)
- Kernel time: 71.8 us
- Achieved bandwidth: 167.0 GB/s
- Peak bandwidth: 192 GB/s
- Efficiency: 87%

What memory-bound means: The math (A + B) finishes so fast that
CUDA cores sit idle waiting for the next data to arrive from VRAM.
The bottleneck is the VRAM highway, not the workers doing math.

Why 87% is good: The VRAM highway was running at 87% of its physical
speed limit. The remaining 13% is unavoidable hardware overhead —
memory controller setup, last wave gaps, launch latency. It cannot
be recovered through better code. First kernel matching
production-quality PyTorch performance.

## Experiment 3: Last Wave Problem

```python
N = 1_000_000
BLOCK_SIZE = 1024
SMs = 16

blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
full_rounds = blocks // SMs
last_wave = blocks % SMs

print(f'Total blocks: {blocks}')
print(f'Full rounds: {full_rounds}')
print(f'Last wave blocks: {last_wave} out of {SMs} SMs')
print(f'Last wave SM utilization: {last_wave/SMs*100:.1f}%')
```

- Total blocks: 977
- Full rounds: 61 (16 SMs fully busy each round)
- Last wave: 1 block out of 16 SMs active
- Last wave SM utilization: 6.2%

What happened: In round 62, only 1 SM had work. The other 15 sat
idle waiting for it to finish. 977 does not divide evenly into 16.

Why this matters: For a fast vector add kernel this costs almost
nothing. For an expensive kernel like softmax at seq=2048, one
wasted round costs real milliseconds. Production kernels choose
N and BLOCK_SIZE so that block count divides evenly into SM count.

## Mental Model

A GPU kernel is: a function that runs millions of threads in parallel
on the GPU, each thread handling a small chunk of the total work.

An SM is: one independent assembly line inside the GPU that receives
one block at a time, runs all its threads in parallel, then
immediately picks up the next block from the queue.

A block is: a group of threads that land on one SM together and share
that SM's fast local memory (SRAM).

BLOCK_SIZE controls: how many threads are in each block, which
determines how the total work is sliced — but not how much
total work exists.

## Connection to Previous Weeks

Week 1 softmax was 21% GPU time because it wrote the full S×S scores
matrix to VRAM, then read it back multiple times for exp() and
normalization. Every read and write consumed bandwidth on the
192 GB/s conveyor belt.

Week 5 vector add hits 87% bandwidth because it is doing the simplest
possible memory operation — read two arrays, write one — with no
unnecessary round trips to VRAM.

Week 7–8 fused softmax will be faster than Week 1 because it will
keep intermediate values (scores, exp values, running sum) inside
SM shared memory instead of writing them to VRAM. Fewer VRAM round
trips = less conveyor belt time = faster kernel.

## What Changed This Week

Before Week 5: knew that memory-bound meant the conveyor belt was
the bottleneck, but did not know what was happening inside the GPU
to cause it.

After Week 5: can trace exactly what happens — kernel launches a
grid of blocks, 16 SMs pick up blocks from the queue, each SM runs
1024 threads in parallel loading data from VRAM, math finishes
instantly, result writes back to VRAM, SM picks up next block.
The bottleneck is always the VRAM load/store, not the math.

## Files
- vector_add.py — Triton vector add kernel with BLOCK_SIZE experiment
- findings.md — this file