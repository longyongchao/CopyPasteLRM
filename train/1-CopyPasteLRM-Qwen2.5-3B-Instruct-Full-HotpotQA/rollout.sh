CUDA_VISIBLE_DEVICES=0 \
swift rollout \
    --model Qwen/Qwen2.5-3B-Instruct \
    --vllm_gpu_memory_utilization 0.85 \
    --vllm_tensor_parallel_size 1 \
    --vllm_max_model_len 4096 \
    --use_hf true \
