# CUDA_VISIBLE_DEVICES=2 proxychains4 python compare_cache.py
import time
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def measure_inference_throughput(model, tokenizer, input_text, max_length, use_cache):
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(model.device)
    attention_mask = torch.ones_like(input_ids).to(model.device)

    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            do_sample=False,
            use_cache=use_cache
        )
    end_time = time.time()

    num_tokens_generated = outputs.size(1) - input_ids.size(1)
    inference_time = end_time - start_time
    throughput = num_tokens_generated / inference_time

    return throughput

def measure_gpu_memory():
    return torch.cuda.max_memory_allocated() / (1024 ** 2)

def print_memory_usage(message=""):
    memory_usage = torch.cuda.memory_allocated() / (1024 ** 2)
    print(f"{message} Memory Usage: {memory_usage:.2f} MB")

def main():
    model_name = 'gpt2'
    prompts = [
        "Hello",
        "The cat sat on the mat",
        "A quick brown fox",
        "In a distant land",
        "The sun rises"
    ]
    max_lengths = [100, 300, 500, 700, 900]
    MAX_ATTEMPTS = 5
    WARMUP_STEPS = 3

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name).to('cuda')
    model.eval()

    # Warmup
    for _ in range(WARMUP_STEPS):
        measure_inference_throughput(model, tokenizer, prompts[0], max_lengths[-1], use_cache=False)

    for max_length in max_lengths:

        print(f"Max Length: {max_length}")

        no_cache_throughputs = []
        cached_throughputs = []
        no_cache_memories = []
        cached_memories = []

        for i in range(MAX_ATTEMPTS):
            print_memory_usage(f"Before Inference Attempt {i+1}")
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            no_cache_throughput = measure_inference_throughput(model, tokenizer, prompts[i], max_length, use_cache=False)
            no_cache_memory = measure_gpu_memory()
            print_memory_usage(f"After Inference Attempt {i+1}")

            print_memory_usage(f"Before Inference Attempt {i+1} (Cached)")
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            cached_throughput = measure_inference_throughput(model, tokenizer, prompts[i], max_length, use_cache=True)
            cached_memory = measure_gpu_memory()
            print_memory_usage(f"After Inference Attempt {i+1} (Cached)")

            no_cache_throughputs.append(no_cache_throughput)
            cached_throughputs.append(cached_throughput)
            no_cache_memories.append(no_cache_memory)
            cached_memories.append(cached_memory)
        
        avg_no_cache_throughput = np.mean(no_cache_throughputs)
        avg_cached_throughput = np.mean(cached_throughputs)
        var_no_cache_throughput = np.var(no_cache_throughputs)
        var_cached_throughput = np.var(cached_throughputs)

        avg_no_cache_memory = np.mean(no_cache_memories)
        avg_cached_memory = np.mean(cached_memories)
        var_no_cache_memory = np.var(no_cache_memories)
        var_cached_memory = np.var(cached_memories)

        print(f"Throughputs (No Cache): {no_cache_throughputs}")
        print(f"Throughputs (Cache): {cached_throughputs}")
        print(f"Average Throughput (No Cache): {avg_no_cache_throughput:.2f} tokens/s")
        print(f"Average Throughput (Cache): {avg_cached_throughput:.2f} tokens/s")
        print(f"Throughput Variance (No Cache): {var_no_cache_throughput:.2f}")
        print(f"Throughput Variance (Cache): {var_cached_throughput:.2f}")
        print(f"Peak Memory Usage (No Cache): {no_cache_memories}")
        print(f"Peak Memory Usage (Cache): {cached_memories}")
        print(f"Average Peak Memory Usage (No Cache): {avg_no_cache_memory:.2f} MB")
        print(f"Average Peak Memory Usage (Cache): {avg_cached_memory:.2f} MB")
        print(f"Memory Usage Variance (No Cache): {var_no_cache_memory:.2f}")
        print(f"Memory Usage Variance (Cache): {var_cached_memory:.2f}")

if __name__ == "__main__":
    main()