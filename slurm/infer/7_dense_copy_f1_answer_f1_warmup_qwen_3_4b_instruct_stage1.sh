#!/bin/bash
#SBATCH --job-name=qwen3_rlhf_pipeline
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --partition=gpu-a800
#SBATCH --nodelist=gpunode3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:00:00
#SBATCH --mem=64G
#SBATCH --requeue

set -e
set -o pipefail

# 直接引入
source scripts/utils/gpu_port.sh
source scripts/utils/feishu.sh
source scripts/utils/vllm.sh

# 记录开始时间戳
start_time=$(date +%s)


# 0. 设置变量
VLLM_CONDA_ENV="vllm"
# VLLM_CONDA_ENV="ms-swift"
VLLM_DEVICES="0,1,2,3"
# VLLM_DEVICES="0,1"
VLLM_SERVED_MODEL_NAME="7_dense_copy_f1_answer_f1_warmup_qwen_3_4b_instruct_stage1" # vLLM服务模型名称
VLLM_SERVED_MODEL_PATH="/mnt/lustre/DATA/longyongchao/CopyPasteLRM/checkpoint/Qwen/Qwen3-4B-Instruct-2507-dense_copypaste_warmup-20251224152332/stage1/last" # vLLM服务模型路径
VLLM_MAX_L=32768
VLLM_MAX_S=128
VLLM_PORT=8124
VLLM_TEMPERATURE=0.0

# 数据集相关变量
DATASET_NAME='copypaste'
DATASET_MAX_SAMPLES=-1
DATASET_SPLIT="test"

prompt_types=(
    # "direct_inference"
    # "rag"
    # "cot"
    # "ircot"
    # "deepseek"
    "copypaste"
)

# 1. 拉起vLLM服务

## 1.1 清理端口和GPU进程
kill_process_on_port $VLLM_PORT
kill_processes_on_gpus $VLLM_DEVICES

# 1.2 启动vLLM服务 (后台运行并重定向输出)
launch_vllm_service \
    "$VLLM_CONDA_ENV" \
    "$VLLM_DEVICES" \
    "$VLLM_SERVED_MODEL_NAME" \
    "$VLLM_SERVED_MODEL_PATH" \
    $VLLM_MAX_L \
    $VLLM_MAX_S \
    $VLLM_PORT > logs/vllm_server.log 2>&1 &

VLLM_PID=$!


# 2. 等待vLLM服务部署完成

wait_for_vllm $VLLM_PORT


# 3. 启动推理

source scripts/infer/infer_test.sh "${prompt_types[@]}"


# 4. 杀死vLLM服务
kill_process_on_port $VLLM_PORT
kill_processes_on_gpus $VLLM_DEVICES

# 5. 记录结束时间戳
end_time=$(date +%s)

# 5. 发送飞书通知，将总耗时，VLLM_SERVED_MODEL_NAME, DATASET_NAME, DATASET_SPLIT, prompt_types的信息囊括
send_feishu_msg "✅ 测试集推理完成\n总耗时: $(($end_time - $start_time)/60)分钟\nVLLM_SERVED_MODEL_NAME: $VLLM_SERVED_MODEL_NAME\nDATASET_NAME: $DATASET_NAME\nDATASET_SPLIT: $DATASET_SPLIT \n prompt_types: ${prompt_types[@]}"
