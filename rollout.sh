CUDA_VISIBLE_DEVICES=0 \
swift rollout \
    --model Qwen/Qwen2.5-3B-Instruct \
    --vllm_gpu_memory_utilization 0.9