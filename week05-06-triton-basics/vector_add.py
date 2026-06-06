# vector_add.py
# Goal: add two vectors A + B = C using a Triton kernel
# We'll measure how BLOCK_SIZE affects performance

import torch
import triton
import triton.language as tl
import time

# ─────────────────────────────────────────────
# THE KERNEL — this is what runs ON the GPU
# Think of this as the job description for one worker
# ─────────────────────────────────────────────

@triton.jit                          # @triton.jit = "compile this for the GPU"
def vector_add_kernel(
    A_ptr,                           # pointer = address of where A lives in VRAM
    B_ptr,                           # pointer to B in VRAM
    C_ptr,                           # pointer to C in VRAM (output)
    N,                               # total number of elements (e.g. 1,000,000)
    BLOCK_SIZE: tl.constexpr,        # how many elements ONE worker handles
                                     # constexpr = must be known at compile time
):
    # Each worker gets a unique ID — like an employee badge number
    # If we launch 1000 workers, they get IDs 0, 1, 2, ... 999
    pid = tl.program_id(axis=0)      # pid = "program ID" = this worker's ID number

    # Each worker calculates WHICH elements it is responsible for
    # Worker 0 handles elements 0..BLOCK_SIZE-1
    # Worker 1 handles elements BLOCK_SIZE..2*BLOCK_SIZE-1
    # Worker 2 handles elements 2*BLOCK_SIZE..3*BLOCK_SIZE-1
    block_start = pid * BLOCK_SIZE

    # offsets = the exact indices this worker will touch
    # e.g. if pid=2, BLOCK_SIZE=4: offsets = [8, 9, 10, 11]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Boundary check — the last worker might go out of bounds
    # e.g. N=10, BLOCK_SIZE=4: last worker gets [8,9,10,11] but 10,11 don't exist
    mask = offsets < N

    # Load data FROM VRAM into the worker's local registers (fast!)
    # mask=mask means: skip loading if out of bounds
    a = tl.load(A_ptr + offsets, mask=mask)
    b = tl.load(B_ptr + offsets, mask=mask)

    # Do the math — this happens in registers, NOT VRAM
    c = a + b

    # Write result BACK to VRAM
    tl.store(C_ptr + offsets, c, mask=mask)


# ─────────────────────────────────────────────
# THE LAUNCHER — this runs on CPU, controls the GPU
# ─────────────────────────────────────────────

def vector_add(A, B, BLOCK_SIZE=1024):
    # A and B must already be on GPU (torch tensors with .cuda())
    N = A.shape[0]                   # how many elements total

    # Allocate output tensor on GPU
    C = torch.empty_like(A)

    # How many workers do we need?
    # If N=1,000,000 and BLOCK_SIZE=1024: we need ceil(1M/1024) = 977 workers
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    #       ↑ this is a function that returns the number of workers (the "grid")
    #       triton.cdiv = ceiling division (rounds UP to avoid missing elements)

    # Launch the kernel — this sends work to the GPU
    vector_add_kernel[grid](
        A, B, C,         # the data
        N,               # total size
        BLOCK_SIZE=BLOCK_SIZE
    )

    return C


# ─────────────────────────────────────────────
# EXPERIMENT — run with different BLOCK_SIZEs
# ─────────────────────────────────────────────

def benchmark(N, BLOCK_SIZE, warmup=5, repeat=20):
    # Create two random vectors on GPU
    A = torch.randn(N, device='cuda', dtype=torch.float32)
    B = torch.randn(N, device='cuda', dtype=torch.float32)

    # Warmup — GPU needs a few runs to reach stable speed
    for _ in range(warmup):
        vector_add(A, B, BLOCK_SIZE=BLOCK_SIZE)

    # Actual timing
    torch.cuda.synchronize()         # wait for GPU to finish before starting timer
    start = time.perf_counter()

    for _ in range(repeat):
        vector_add(A, B, BLOCK_SIZE=BLOCK_SIZE)

    torch.cuda.synchronize()         # wait for GPU to finish before stopping timer
    end = time.perf_counter()

    avg_ms = (end - start) / repeat * 1000   # convert seconds → milliseconds
    return avg_ms


# ─────────────────────────────────────────────
# MAIN — run the experiment
# ─────────────────────────────────────────────

if __name__ == "__main__":
    N = 1_000_000    # 1 million elements — small enough for 4GB VRAM

    # First: verify correctness
    print("=== Correctness Check ===")
    A = torch.randn(N, device='cuda', dtype=torch.float32)
    B = torch.randn(N, device='cuda', dtype=torch.float32)
    C_triton = vector_add(A, B, BLOCK_SIZE=1024)
    C_torch   = A + B                            # PyTorch's own vector add (reference)
    match = torch.allclose(C_triton, C_torch)    # are all values close enough?
    print(f"Triton matches PyTorch: {match}")
    print()

    # Now: benchmark different BLOCK_SIZEs
    print("=== BLOCK_SIZE Experiment ===")
    print(f"{'BLOCK_SIZE':<15} {'Workers Launched':<20} {'Time (ms)':<12}")
    print("-" * 47)

    for bs in [64, 128, 256, 512, 1024, 2048, 4096]:
        num_workers = triton.cdiv(N, bs)         # how many blocks launched
        ms = benchmark(N, bs)
        print(f"{bs:<15} {num_workers:<20} {ms:.4f}")

    print()
    print("=== PyTorch Baseline ===")
    A = torch.randn(N, device='cuda', dtype=torch.float32)
    B = torch.randn(N, device='cuda', dtype=torch.float32)

    for _ in range(5):                           # warmup
        _ = A + B

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(20):
        C = A + B
    torch.cuda.synchronize()
    end = time.perf_counter()

    torch_ms = (end - start) / 20 * 1000
    print(f"PyTorch built-in vector add: {torch_ms:.4f} ms")