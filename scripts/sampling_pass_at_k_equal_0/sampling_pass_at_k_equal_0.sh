#!/bin/bash

# 使用 ("$@") 将接收到的所有提示类型重新打包成一个新数组
target_dataset=("$@")

required_vars=(
    VLLM_PORT
    VLLM_MAX_S
    VLLM_SERVED_MODEL_NAME
    PASS_TEMPERATURE
    DATASET_MAX_SAMPLES
    DATASET_SPLIT
)

for v in "${required_vars[@]}"; do
  if [ -z "${!v}" ]; then
    echo "[ERROR] Environment variable $v is not set"
    exit 1
  fi
done

SERVER_URL="http://localhost:${VLLM_PORT}/v1"

for dataset_name in "${target_dataset[@]}"; do
    
    # 执行 Python 脚本，使用当前循环中的 $DATASET_NAME 变量
    python copypastelrm/inference/inferPass@K.py \
        --server-url "${SERVER_URL}" \
        --model-name "${VLLM_SERVED_MODEL_NAME}" \
        --dataset "${dataset_name}" \
        --split "${DATASET_SPLIT}" \
        --num-threads "${VLLM_MAX_S}" \
        --max-samples "${DATASET_MAX_SAMPLES}" \
        --k "${PASS_K_VALUE}" \
        --temperature "${PASS_TEMPERATURE}" \
        --prior-threshold "${PASS_PRIOR_THRESHOLD}" \
        # --enable-thinking
        
    # 检查上一个命令的退出状态
    if [ $? -eq 0 ]; then
        echo "✅ 数据集 ${DATASET_NAME} 推理任务成功完成。"
    else
        echo "❌ 数据集 ${DATASET_NAME} 推理任务执行失败！"
        # 如果你希望在任何一个数据集失败后就停止整个脚本，可以取消下一行的注释：
        # exit 1 
    fi

done
