#!/bin/bash
#SBATCH --job-name=qwen3   # 作业名称
#SBATCH --output=output.txt     # 输出文件名称（%j会被替换为作业ID）
#SBATCH --error=error.txt       # 错误文件名称
#SBATCH --partition=gpu-a800        # 指定分区
#SBATCH --nodelist=gpunode3       # 指定节点
#SBATCH --nodes=1                  # 需要的节点数
#SBATCH --ntasks-per-node=1        # 每个节点的任务数
#SBATCH --time=00:00:00            # 作业最大运行时间（小时：分钟：秒）
#SBATCH --mem=64G                   # 每个节点所需的内存
#SBATCH --requeue                  # 运行失败时重新排队

# 运行Python脚本


set -e
set -o pipefail


# export MODEL_NAME="Qwen/Qwen3-8B"
export MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"
# export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"

EXP_ROOT=/mnt/lustre/DATA/longyongchao/CopyPasteLRM/checkpoint
# EXP_ROOT=/data/lyc/CopyPasteLRM/checkpoint

export RLHF_CUDA_VISIBLE_DEVICES_LIST="0,1,2,3"
export RLHF_NPROC_PER_NODE=4

export BATCH_SIZE=8
export NUM_GENERATIONS=8

###################################################################

# %Y%m%d%H%M%S 生成前14位
timestamp=$(date +%Y%m%d%H%M%S)

EXP_NAME=${MODEL_NAME}-passAtK_0-wo_copying-${timestamp}

export STAGE1_OUTPUT_DIR=${EXP_ROOT}/${EXP_NAME}/stage1
export STAGE2_OUTPUT_DIR=${EXP_ROOT}/${EXP_NAME}/stage2
export SPLIT_DATASET_RATIO=0.03007 # 93/3093=0.0300678952，3000样本用于训练，93样本用于评估

export SAVE_STEPS=50
# export DATASET_SAMPLE=5
export DATASET_SAMPLE=3093

export SWANLAB_PROJECT="CopyPasteLRM"
export SWANLAB_TOKEN="eD9F8nh3oF5zAeyopbN8f"
export SWANLAB_LARK_WEBHOOK_URL="https://open.feishu.cn/open-apis/bot/v2/hook/880e2480-71ed-4f29-8495-b7fa75c8cbd7"
export SWANLAB_LARK_SECRET="IzE5LR2O7ojQkRUO9g96Qe"


echo "========== Stage 1 =========="

# Stage1: Copy-dominant
export REWARD_FORMAT=0.1
export REWARD_LENGTH=0.1
export REWARD_COPY=0.0
export REWARD_ANSWER=0.8

export SWANLAB_EXP_NAME="[stage1]-${EXP_NAME}"

bash rlhf_stage1.sh

STAGE1_LAST=${STAGE1_OUTPUT_DIR}/last

if [ ! -L "${STAGE1_LAST}" ]; then
  echo "[ERROR] Stage1 last checkpoint not found"
  exit 1
fi

echo "========== Stage 2 =========="

# Stage2: Answer-dominant
export REWARD_FORMAT=0.1
export REWARD_LENGTH=0.1
export REWARD_COPY=0.0
export REWARD_ANSWER=0.8

export SWANLAB_EXP_NAME="[stage2]${EXP_NAME}"

bash rlhf_stage2.sh
