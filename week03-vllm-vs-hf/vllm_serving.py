"""
vLLM serving with continuous batching + PagedAttention.
Compare against naive baseline to see GPU utilization improvements.
"""

import time
import subprocess
from vllm import LLM, SamplingParams

# Using GPT-2 medium for 4GB VRAM constraint
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

def get_gpu_utilization():
    """Query nvidia-smi for current GPU utilization percentage."""
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        return int(result.strip())
    except:
        return -1

def test_vllm_serving(num_requests=5):
    """
    Test vLLM with continuous batching.
    vLLM automatically handles:
    - PagedAttention (efficient KV cache)
    - Continuous batching (remove finished requests, add new ones)
    - Optimized CUDA kernels
    """
    print("="*60)
    print("vLLM Serving Test")
    print("="*60)
    
    # Initialize vLLM engine
    print(f"\nLoading {MODEL_NAME} into vLLM engine...")
    llm = LLM(
        model=MODEL_NAME,
        dtype="float16",  # FP16 for 4GB VRAM
        gpu_memory_utilization=0.7,  # Conservative 70% to leave room for vLLM overhead
        max_model_len=512,  # Shorter sequences to save memory
    )
    print("vLLM engine loaded.")
    
    # Same prompts as naive baseline
    prompts = [
        "Explain machine learning in one sentence:",
        "What is the capital of France?",
        "Write a haiku about coding:",
        "Translate 'hello' to Spanish:",
        "What is 2+2?"
    ][:num_requests]
    
    # Sampling parameters (same as naive baseline)
    sampling_params = SamplingParams(
        temperature=0.0,  # Greedy decoding for consistency
        max_tokens=50,    # Same as naive baseline
    )
    
    print(f"\nSending {num_requests} requests to vLLM...")
    print("(vLLM will automatically batch and schedule these)")
    
    gpu_before = get_gpu_utilization()
    start_time = time.time()
    
    # vLLM handles batching + continuous scheduling automatically
    outputs = llm.generate(prompts, sampling_params)
    
    end_time = time.time()
    gpu_after = get_gpu_utilization()
    
    total_time = end_time - start_time
    throughput = num_requests / total_time
    
    print(f"\n--- Results ---")
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {throughput:.2f} requests/sec")
    print(f"GPU util during generation: ~{gpu_after}%")
    
    # Show a sample output
    print(f"\nSample output (Request 1):")
    print(f"Prompt: {prompts[0]}")
    print(f"Response: {outputs[0].outputs[0].text[:100]}...")
    
    return total_time, throughput

if __name__ == "__main__":
    print("Week 3 — vLLM Serving Comparison")
    print("Testing continuous batching + PagedAttention\n")
    
    vllm_time, vllm_throughput = test_vllm_serving(num_requests=5)
    
    # Compare against baseline (from previous run)
    # Note: These are for Qwen 1.5B, not GPT-2, but still shows concept
    naive_sequential_time = 12.05
    naive_batch_time = 2.89
    naive_throughput = 1.73
    
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    print(f"Naive sequential:  {naive_sequential_time:.2f}s  (0.42 req/s)")
    print(f"Naive batching:    {naive_batch_time:.2f}s     ({naive_throughput:.2f} req/s)")
    print(f"vLLM:              {vllm_time:.2f}s     ({vllm_throughput:.2f} req/s)")
    print(f"\nvLLM speedup over naive batch: {naive_batch_time/vllm_time:.2f}x")
    print(f"vLLM speedup over sequential:  {naive_sequential_time/vllm_time:.2f}x")
    
    print("\n" + "="*60)
    print("KEY INSIGHT")
    print("="*60)
    print("vLLM's throughput gains come from:")
    print("1. Continuous batching → GPU never waits for slow requests")
    print("2. PagedAttention → fit more requests in same VRAM")
    print("3. Optimized kernels → less overhead per operation")