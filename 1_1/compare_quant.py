# CUDA_VISIBLE_DEVICES=2 proxychains4 python compare_quant.py

import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, QuantoConfig
from optimum.quanto import QuantizedModelForCausalLM, qint2, qint4, qint8
import numpy as np

def load_model(model_name, quantization=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to('cuda')
    
    if quantization is not None:
        model = QuantizedModelForCausalLM.quantize(model, weights=quantization, exclude='lm_head')

    model.eval()
    
    return model, tokenizer

def measure_inference(model, tokenizer, input_text, max_length, use_cache):
    inputs = tokenizer(input_text, return_tensors='pt').to(model.device)

    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length, do_sample=False, use_cache=use_cache)
    end_time = time.time()
    total_time = end_time - start_time

    generated_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
    throughput = generated_tokens / total_time

    del inputs
    del outputs

    return throughput

def measure_gpu_memory():
    return torch.cuda.max_memory_allocated() / (1024 ** 2)

def print_memory_usage(message=""):
    memory_usage = torch.cuda.memory_allocated() / (1024 ** 2)
    print(f"{message} Memory Usage: {memory_usage:.2f} MB")

def main():
    model_name = 'bigscience/bloom-560m'
    input_text = 'The quick brown'
    max_length = 500
    quantization = None
    max_rounds = 5
    warmup_steps = 3

    print(f"> Quantization Level: {quantization or 'None'}")

    print_memory_usage("> Before Model Loading")
    model, tokenizer = load_model(model_name, quantization)
    print_memory_usage("> After Model Loading")

    throughputs = []
    memories = []

    print("> Start warmup")
    for i in range(warmup_steps):
        measure_inference(model, tokenizer, input_text, max_length, use_cache=True)
    print("> Warmup finished")
        
    for i in range(max_rounds):
        print_memory_usage(f"> Before Inference Round {i+1}")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        throughput_cached = measure_inference(model, tokenizer, input_text, max_length, use_cache=True)
        memory_cached = measure_gpu_memory()

        throughputs.append(throughput_cached)
        memories.append(memory_cached)
        print_memory_usage(f"> After Inference Round {i+1}")
        
    avg_throughput = np.mean(throughputs)
    var_throughput = np.var(throughputs)
    avg_memory = np.mean(memories)
    var_memory = np.var(memories)

    print(f"Throughputs: {throughputs}")
    print(f"Memories: {memories}")
    print(f"Average Throughput: {avg_throughput:.2f} tokens/s")
    print(f"Throughput Variance: {var_throughput:.2f}")
    print(f"Average Memory: {avg_memory:.2f} MB")
    print(f"Memory Variance: {var_memory:.2f}")

if __name__ == "__main__":
    main()