import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.profiler import profile, ProfilerActivity
import os

# Load GPT-2 small (117M parameters)
model = GPT2LMHeadModel.from_pretrained('gpt2').cuda()
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Create input — single forward pass, no generation
text = "The quick brown fox jumps over the lazy dog"
inputs = tokenizer(text, return_tensors='pt').to('cuda')
input_ids = inputs['input_ids']

print(f"Input shape: {input_ids.shape}")  # Will show [1, sequence_length]
print(f"Model has {model.num_parameters()} parameters")

# Warmup — first run loads things into GPU
with torch.no_grad():
    _ = model(input_ids)

# Profile the forward pass
with torch.no_grad():  # No gradients needed — inference only
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True
    ) as prof:
        outputs = model(input_ids)

# Print table sorted by CUDA time
print("\n=== Top Operations by CUDA Time ===")
print(prof.key_averages().table(
    sort_by="cuda_time_total", 
    row_limit=20
))

# Export trace for Chrome viewer
output_dir = "./week02-transformer-profiling/traces"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
trace_path = os.path.join(output_dir, "gpt2_trace.json")
prof.export_chrome_trace(trace_path)

print("\nTrace exported to gpt2_trace.json")