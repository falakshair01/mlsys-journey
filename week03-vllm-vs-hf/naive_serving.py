"""
Naive HuggingFace LLM serving baseline.
This is the "obvious" way to serve requests — no fancy batching, no memory tricks.
We'll measure GPU utilization and throughput to see why this is inefficient.
"""

import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import subprocess


# Use Qwen 1.5B — small enough for 4GB VRAM
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

def load_model():
    """Load model onto GPU."""
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,  # Half precision to fit in 4GB
        device_map="auto"
    )
    model.eval()
    print(f"Model loaded. VRAM usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_new_tokens=50):
    """
    Generate response for a single prompt.
    This is the naive approach — no batching, no KV cache reuse.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding for consistency
            pad_token_id=tokenizer.eos_token_id
        )
    end_time = time.time()
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    latency = end_time - start_time
    
    return response, latency

def test_sequential_requests(model, tokenizer, num_requests=5):
    """
    Send requests one at a time, waiting for each to complete.
    This is how a naive server would work — process queue sequentially.
    """
    print("\n" + "="*60)
    print(f"TEST 1: {num_requests} Sequential Requests (no batching)")
    print("="*60)
    
    prompts = [
        "Explain machine learning in one sentence:",
        "What is the capital of France?",
        "Write a haiku about coding:",
        "Translate 'hello' to Spanish:",
        "What is 2+2?"
    ]
    
    total_start = time.time()
    latencies = []
    
    for i, prompt in enumerate(prompts[:num_requests]):
        print(f"\nRequest {i+1}: {prompt[:50]}...")
        gpu_before = get_gpu_utilization()
        
        response, latency = generate_response(model, tokenizer, prompt, max_new_tokens=50)
        latencies.append(latency)
        
        gpu_after = get_gpu_utilization()
        print(f"  Latency: {latency:.2f}s")
        print(f"  GPU util during generation: ~{gpu_after}%")
    
    total_time = time.time() - total_start
    avg_latency = sum(latencies) / len(latencies)
    
    print(f"\n--- Results ---")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average latency per request: {avg_latency:.2f}s")
    print(f"Throughput: {num_requests/total_time:.2f} requests/sec")
    
    return total_time, avg_latency

def test_naive_batch(model, tokenizer, num_requests=5):
    """
    Send all requests as a single batch.
    Naive batching: pad all sequences to same length, wait for slowest to finish.
    """
    print("\n" + "="*60)
    print(f"TEST 2: {num_requests} Requests in Single Batch (naive batching)")
    print("="*60)
    
    prompts = [
        "Explain machine learning in one sentence:",
        "What is the capital of France?",
        "Write a haiku about coding:",
        "Translate 'hello' to Spanish:",
        "What is 2+2?"
    ][:num_requests]
    
    # Tokenize all prompts together — this will pad to longest sequence
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    
    print(f"Batch size: {inputs['input_ids'].shape[0]}")
    print(f"Max sequence length in batch: {inputs['input_ids'].shape[1]}")
    
    gpu_before = get_gpu_utilization()
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    end_time = time.time()
    gpu_after = get_gpu_utilization()
    
    total_time = end_time - start_time
    
    print(f"\n--- Results ---")
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {num_requests/total_time:.2f} requests/sec")
    print(f"GPU util during batch: ~{gpu_after}%")
    print(f"VRAM used: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    
    return total_time

if __name__ == "__main__":
    print("Week 3 — Naive HuggingFace Serving Baseline")
    print("This demonstrates WHY we need vLLM / continuous batching.\n")
    
    # Load model once
    model, tokenizer = load_model()
    
    # Test 1: Sequential requests (real-world naive server)
    seq_time, seq_latency = test_sequential_requests(model, tokenizer, num_requests=5)
    
    # Test 2: Naive batching (wait for slowest request)
    batch_time = test_naive_batch(model, tokenizer, num_requests=5)
    
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"Sequential:   {seq_time:.2f}s total, {seq_latency:.2f}s avg per request")
    print(f"Naive batch:  {batch_time:.2f}s total")
    print(f"Speedup from batching: {seq_time/batch_time:.2f}x")
    print("\nKey observation: GPU is idle between sequential requests.")
    print("Next: vLLM continuous batching eliminates this idle time.")