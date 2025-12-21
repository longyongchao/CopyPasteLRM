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

# ================= 环境变量配置 =================
export MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"
EXP_ROOT=/mnt/lustre/DATA/longyongchao/CopyPasteLRM/checkpoint
# EXP_ROOT=/data/lyc/CopyPasteLRM/checkpoint

export ROLLOUT_CUDA_VISIBLE_DEVICES_LIST="0"
export VLLM_PORT=8866

export RLHF_CUDA_VISIBLE_DEVICES_LIST="1,2,3"
export RLHF_NPROC_PER_NODE=3
export BATCH_SIZE=4
export NUM_GENERATIONS=12 # 要求是 RLHF_NPROC_PER_NODE * BATCH_SIZE 的整数倍

# 生成时间戳和实验名称
timestamp=$(date +%Y%m%d%H%M%S)
EXP_NAME=${MODEL_NAME}-passAtK_0-wo_copying-${timestamp}

export STAGE1_OUTPUT_DIR=${EXP_ROOT}/${EXP_NAME}/stage1
export STAGE2_OUTPUT_DIR=${EXP_ROOT}/${EXP_NAME}/stage2
export SPLIT_DATASET_RATIO=0.03007 

export SAVE_STEPS=300
export EVAL_STEPS=500
export DATASET_SAMPLE=3093
export NUM_TRAIN_EPOCHS=1

# SwanLab 配置
export SWANLAB_PROJECT="CopyPasteLRM"
export SWANLAB_TOKEN="eD9F8nh3oF5zAeyopbN8f"
export SWANLAB_LARK_WEBHOOK_URL="https://open.feishu.cn/open-apis/bot/v2/hook/880e2480-71ed-4f29-8495-b7fa75c8cbd7"
export SWANLAB_LARK_SECRET="IzE5LR2O7ojQkRUO9g96Qe"

# ================= 工具函数 =================

# 函数：等待 vLLM 服务启动
function wait_for_vllm() {
    echo "[Pipeline] Waiting for vLLM to be ready on port ${VLLM_PORT}..."
    local max_retries=60  # 等待 60 * 10s = 10分钟
    local count=0
    
    # 循环检查 /health 接口状态码是否为 200
    while ! curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:${VLLM_PORT}/health | grep -q "200"; do
        if [ $count -ge $max_retries ]; then
            echo "[Error] vLLM failed to start within timeout."
            return 1
        fi
        echo "   ... waiting for vLLM (attempt $((count+1))/$max_retries)"
        sleep 10
        count=$((count+1))
    done
    echo "[Pipeline] vLLM is ready!"
}

# 函数：清理后台进程
function cleanup_rollout() {
    if [ -n "$ROLLOUT_PID" ]; then
        echo "[Pipeline] Stopping vLLM (PID: $ROLLOUT_PID)..."
        kill $ROLLOUT_PID || true
        wait $ROLLOUT_PID || true
        echo "[Pipeline] vLLM stopped."
        # 稍微等待端口释放
        sleep 5
    fi
}

# 确保脚本退出时清理进程（防止意外退出导致僵尸进程）
trap cleanup_rollout EXIT

# ================= Stage 1 =================
echo "========== Starting Stage 1 =========="

# Stage1 Rewards
export REWARD_FUNCS="cplrm_format cplrm_length cplrm_copy cplrm_answer"
export REWARD_FORMAT=0.1
export REWARD_LENGTH=0.1
export REWARD_COPY=0.6
export REWARD_ANSWER=0.2
export REWARD_WEIGHTS="${REWARD_FORMAT} ${REWARD_LENGTH} ${REWARD_COPY} ${REWARD_ANSWER}"
export SWANLAB_EXP_NAME="[stage1]-${EXP_NAME}"

# 1. 设置 Stage 1 的 Rollout 模型为基座模型
export CURRENT_ROLLOUT_MODEL=${MODEL_NAME}

# 2. 后台启动 Rollout 服务
echo "[Stage 1] Launching Rollout Service..."
bash rollout.sh > rollout_stage1.log 2>&1 &
ROLLOUT_PID=$!

# 3. 等待服务就绪
wait_for_vllm

# 4. 启动训练
echo "[Stage 1] Starting RLHF Training..."
bash rlhf_stage1.sh

# 5. 训练完成，清理服务
cleanup_rollout

# 检查产物
STAGE1_LAST=${STAGE1_OUTPUT_DIR}/last
if [ ! -d "${STAGE1_LAST}" ]; then
  echo "[ERROR] Stage1 last checkpoint directory not found at ${STAGE1_LAST}"
  exit 1
fi

echo "[Stage 1] Completed Successfully."
echo "-------------------------------------"

# ================= Stage 2 =================
echo "========== Starting Stage 2 =========="

# Stage2 Rewards
export REWARD_FUNCS="cplrm_format cplrm_length cplrm_copy cplrm_answer"
export REWARD_FORMAT=0.1
export REWARD_LENGTH=0.1
export REWARD_COPY=0.2
export REWARD_ANSWER=0.6
export REWARD_WEIGHTS="${REWARD_FORMAT} ${REWARD_LENGTH} ${REWARD_COPY} ${REWARD_ANSWER}"
export SWANLAB_EXP_NAME="[stage2]${EXP_NAME}"

# 1. 设置 Stage 2 的 Rollout 模型为 Stage 1 的产出
# 注意：通常 GRPO 需要加载上一轮训练后的模型来采样
export CURRENT_ROLLOUT_MODEL=${MODEL_NAME}

# 2. 后台启动 Rollout 服务 (重新拉起)
echo "[Stage 2] Launching Rollout Service with Stage 1 Checkpoint..."
bash rollout.sh > rollout_stage2.log 2>&1 &
ROLLOUT_PID=$!

# 3. 等待服务就绪
wait_for_vllm

# 4. 启动 Stage 2 训练
echo "[Stage 2] Starting RLHF Training..."
bash rlhf_stage2.sh

# 5. 训练完成，清理服务 (trap 会自动再次处理，但显式调用更安全)
cleanup_rollout
# 重置PID防止trap重复kill
ROLLOUT_PID=""

echo "========== Pipeline Completed Successfully =========="