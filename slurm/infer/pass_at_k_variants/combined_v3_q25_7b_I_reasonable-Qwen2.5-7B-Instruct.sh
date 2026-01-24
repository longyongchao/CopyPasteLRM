#!/bin/bash
#SBATCH --job-name=qwen3
#SBATCH --output=/tmp/output_%j.txt
#SBATCH --error=/tmp/error_%j.txt
#SBATCH --partition=gpu-4090
#SBATCH --nodelist=gpunode1
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
VLLM_SERVED_MODEL_NAME="0110224421-combined_v3_q25_7b_I_reasonable-Qwen2.5-7B-Instruct-stage2" # vLLM服务模型名称
VLLM_SERVED_MODEL_PATH="/mnt/lustre/DATA/longyongchao/CopyPasteLRM/checkpoint/0110224421-combined_v3_q25_7b_I_reasonable-Qwen/Qwen2.5-7B-Instruct/stage2/best"
VLLM_MAX_L=32768
VLLM_MAX_S=32
VLLM_PORT=8125
VLLM_TEMPERATURE=0.6
NOISE_DOCS=8

# 数据集相关变量
DATASET_NAME='MuSiQue'
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
    $VLLM_PORT > /tmp/vllm_server.log 2>&1 &

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
duration=$(( (end_time - start_time) / 60 ))

# 5. 发送飞书通知
# 使用 $'' 字符串格式以支持 \n 换行，或者依赖 feishu 脚本的处理逻辑
msg_content="✅ 测试集推理完成
总耗时: ${duration} 分钟
VLLM_SERVED_MODEL_NAME: $VLLM_SERVED_MODEL_NAME
DATASET_NAME: $DATASET_NAME
DATASET_SPLIT: $DATASET_SPLIT
prompt_types: ${prompt_types[*]}"

send_feishu_msg "$msg_content"