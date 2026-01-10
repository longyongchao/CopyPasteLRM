#!/bin/bash
# rollout.sh

# 如果外部没有设置 CURRENT_ROLLOUT_MODEL，则使用默认值
model_path=${CURRENT_ROLLOUT_MODEL:-MODEL_NAME}

echo "[Rollout] Starting vLLM with model: ${model_path}"

CUDA_VISIBLE_DEVICES=${ROLLOUT_CUDA_VISIBLE_DEVICES_LIST} \
swift rollout \
    --model "${model_path}" \
    --vllm_gpu_memory_utilization 0.88 \
    --vllm_tensor_parallel_size 1 \
    --vllm_max_model_len 32768 \
    --use_hf true \
    --enforce-eager \
    --port 8000
