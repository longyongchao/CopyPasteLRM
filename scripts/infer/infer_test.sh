#!/bin/bash

# 使用 ("$@") 将接收到的所有提示类型重新打包成一个新数组
prompt_types=("$@")

# 定义要循环的 prompt_type 列表
# prompt_types=(
#     "direct_inference"
#     "rag"
#     "cot"
#     "ircot"
#     "deepseek"
#     "copypaste"
# )

required_vars=(
    VLLM_PORT
    VLLM_MAX_S
    VLLM_SERVED_MODEL_NAME
    VLLM_TEMPERATURE
    DATASET_MAX_SAMPLES
    DATASET_SPLIT
    DATASET_NAME
    NOISE_DOCS
)

for v in "${required_vars[@]}"; do
  if [ -z "${!v}" ]; then
    echo "[ERROR] Environment variable $v is not set"
    exit 1
  fi
done

server_url="http://localhost:${VLLM_PORT}/v1"

for prompt_type in "${prompt_types[@]}"; do
    echo "开始使用 prompt_type: $prompt_type 评估数据集: $DATASET_NAME"
    echo "================================"
        
    python copypastelrm/inference/infer.py \
        --server-url "$server_url" \
        --model-name ${VLLM_SERVED_MODEL_NAME} \
        --num-threads ${VLLM_MAX_S} \
        --prompt-type "$prompt_type" \
        --dataset ${DATASET_NAME} \
        --max-samples ${DATASET_MAX_SAMPLES} \
        --split ${DATASET_SPLIT} \
        --temperature ${VLLM_TEMPERATURE} \
        --distractor-docs ${NOISE_DOCS} \
        --enable-thinking

done
