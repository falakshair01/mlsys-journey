import torch
from transformers import GPT2LMHeadModel
from torch.profiler import profile, ProfilerActivity

model = GPT2LMHeadModel.from_pretrained('gpt2').cuda()

# Test different sequence lengths
seq_lengths = [16, 32, 64, 128, 256, 512]

print("Seq Len | Total Time | Attention Time | MLP Time | Attn % | MLP %")
print("-" * 75)

for seq_len in seq_lengths:
    # Create input of exact length
    input_ids = torch.randint(0, 50257, (1, seq_len)).cuda()
    
    # Warmup
    with torch.no_grad():
        _ = model(input_ids)
    
    # Profile
    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            _ = model(input_ids)
    
    # Extract timings
    events = prof.key_averages()
    
    # FIX: Use device_time_total for CUDA timings
    total_cuda = sum([e.device_time_total for e in events]) / 1000  # Convert to ms
    
    # Find attention time
    attn_time = sum([e.device_time_total for e in events 
                     if 'attention' in e.key.lower()]) / 1000
    
    # Find MLP time - look for the actual CUDA kernels
    mlp_time = 0
    for e in events:
        # Look for sgemm kernels (matrix multiply)
        if 'sgemm' in e.key.lower():
            # MLP layers have multiple calls (12 or 24)
            if e.count >= 12:  # MLP operations are called many times
                mlp_time += e.device_time_total / 1000
    
    attn_pct = (attn_time / total_cuda) * 100 if total_cuda > 0 else 0
    mlp_pct = (mlp_time / total_cuda) * 100 if total_cuda > 0 else 0
    
    print(f"{seq_len:7d} | {total_cuda:10.2f} | {attn_time:14.2f} | {mlp_time:8.2f} | {attn_pct:6.1f} | {mlp_pct:5.1f}")

print("\nLook for the crossover point where Attention % overtakes MLP %")