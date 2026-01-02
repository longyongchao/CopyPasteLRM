#!/bin/bash
#SBATCH --job-name=qwen3_c
#SBATCH --output=/tmp/output_%j.txt
#SBATCH --error=/tmp/error_%j.txt
#SBATCH --partition=gpu-a800
#SBATCH --nodelist=gpunode3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:00:00
#SBATCH --mem=64G
#SBATCH --requeue

set -e
set -o pipefail

source scripts/utils/vllm.sh
source scripts/utils/gpu_port.sh

# 定义要判断的核心目录
TARGET_DIR="/mnt/lustre/DATA/longyongchao"

# 判断目录是否存在，并给EXP_ROOT赋值
if [[ -d "$TARGET_DIR" ]]; then
    # 目录存在时，赋值为第一个路径
    EXP_ROOT="/mnt/lustre/DATA/longyongchao/CopyPasteLRM/checkpoint"
else
    # 目录不存在时，赋值为第二个路径
    EXP_ROOT="/data/lyc/CopyPasteLRM/checkpoint"
fi

# ================= 环境变量配置 =================
export MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"
# export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"

export ROLLOUT_CUDA_VISIBLE_DEVICES_LIST="0"

export RLHF_CUDA_VISIBLE_DEVICES_LIST="1,2,3"
export RLHF_NPROC_PER_NODE=3
export BATCH_SIZE=3
export NUM_GENERATIONS=9 # 要求是 RLHF_NPROC_PER_NODE * BATCH_SIZE 的整数倍
export RLHF_DATASET="Qwen3-4B-I_MusiQue_128_without_2hop_reasonable_copypaste"

# 生成时间戳和实验名称
timestamp=$(date +%m%d%H%M%S)
EXP_NAME=${timestamp}-combined_q3_4b_I_reasonable-${MODEL_NAME}

export STAGE1_OUTPUT_DIR=${EXP_ROOT}/${EXP_NAME}/stage1
export STAGE2_OUTPUT_DIR=${EXP_ROOT}/${EXP_NAME}/stage2
export SPLIT_DATASET_RATIO=0.05
export GRPO_TEMPERATURE=0.8
export GRPO_BETA=0.01
export GRPO_TOP_P=0.95
export GRPO_MAX_NEW_TOKENS=2048
export GRPO_WARMUP_RATIO=0.05
export GRPO_GRADIENT_ACCUMULATION_STEPS=1
export GRPO_LEARNING_RATE=1e-6
export GRPO_SAVE_TOTAL_LIMIT=5
export GRPO_EPSILON_HIGH=0.25

export SAVE_STEPS=100
export EVAL_STEPS=100
export NUM_TRAIN_EPOCHS=3

# SwanLab 配置
export SWANLAB_EXP_NAME="将事实复制和答案挂钩-并加入复制错误惩罚(Q3-4B-I)"
export SWANLAB_MODEL="local"
export SWANLAB_PROJECT="CopyPasteLRM"
export SWANLAB_TOKEN="eD9F8nh3oF5zAeyopbN8f"
export SWANLAB_LARK_WEBHOOK_URL="https://open.feishu.cn/open-apis/bot/v2/hook/880e2480-71ed-4f29-8495-b7fa75c8cbd7"
export SWANLAB_LARK_SECRET="IzE5LR2O7ojQkRUO9g96Qe"

# --- 2. 注册 Trap ---
# EXIT: 脚本正常或异常退出时触发
# SIGINT: 用户按下 Ctrl+C 时触发
# SIGTERM: 被 kill 命令杀掉时触发
trap cleanup_rollout EXIT SIGINT SIGTERM


# ================= Stage 1 =================
echo "========== Starting Stage 1 =========="

# Stage1 Rewards
export COPY_REWARD_MODE="dense"
export REWARD_FUNCS="cplrm_format cplrm_length cplrm_combined"
export REWARD_FORMAT=0.1
export REWARD_LENGTH=0.1
export REWARD_COPY=0.8
export REWARD_WEIGHTS="${REWARD_FORMAT} ${REWARD_LENGTH} ${REWARD_COPY}"
export MAX_STEPS=-1

# 1. 设置 Stage 1 的 Rollout 模型为基座模型
export CURRENT_ROLLOUT_MODEL=${MODEL_NAME}

# 2. 后台启动 Rollout 服务
echo "[Stage 1] Launching Rollout Service..."
bash train/4-CopyPasteLRM-MusiQue/rollout.sh > /tmp/rollout_stage1.log 2>&1 &
ROLLOUT_PID=$!

echo "[Stage 1] Rollout Service PID: $ROLLOUT_PID"

# 3. 等待服务就绪
wait_for_vllm_old

# 4. 启动训练
echo "[Stage 1] Starting RLHF Training..."
bash train/4-CopyPasteLRM-MusiQue/rlhf_stage1.sh


# 检查产物
STAGE1_LAST=${STAGE1_OUTPUT_DIR}/last
if [ ! -d "${STAGE1_LAST}" ]; then
  echo "[ERROR] Stage1 last checkpoint directory not found at ${STAGE1_LAST}"
  exit 1
fi

echo "[Stage 1] Completed Successfully."
echo "-------------------------------------"

cleanup_rollout

