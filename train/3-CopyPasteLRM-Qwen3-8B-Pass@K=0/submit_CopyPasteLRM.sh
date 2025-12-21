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
export NUM_GENERATIONS=3 # 要求是 RLHF_NPROC_PER_NODE * BATCH_SIZE 的整数倍

# 生成时间戳和实验名称
timestamp=$(date +%Y%m%d%H%M%S)
EXP_NAME=${MODEL_NAME}-passAtK_0-${timestamp}

export STAGE1_OUTPUT_DIR=${EXP_ROOT}/${EXP_NAME}/stage1
export STAGE2_OUTPUT_DIR=${EXP_ROOT}/${EXP_NAME}/stage2
export SPLIT_DATASET_RATIO=0.01

export SAVE_STEPS=200
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
    echo "[Pipeline] Waiting for vLLM to be ready on port 8000..."
    local max_retries=60  # 等待 60 * 10s = 10分钟
    local count=0
    
    # 循环检查 /health 接口状态码是否为 200
    while ! curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health/ | grep -q "200"; do
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

function cleanup_rollout() {
    echo "[Cleanup] Starting safety cleanup..."

    # 1. 根据端口号 (8000) 杀死进程
    # 使用 lsof 或 ss 找到监听 8000 端口的 PID
    local port_pid=$(ss -tlnp | grep ':8000' | awk -F'pid=' '{print $2}' | cut -d',' -f1)
    if [ -n "$port_pid" ]; then
        echo "[Cleanup] Killing process on port 8000 (PID: $port_pid)"
        kill -9 $port_pid 2>/dev/null || true
    fi

    # 2. 杀死 GPU 0 上正在运行的所有进程
    # nvidia-smi --query-compute-apps 能够精确列出 GPU 上的进程 PID
    local gpu_pids=$(nvidia-smi --gpu-id=0 --query-compute-apps=pid --format=csv,noheader,nounits)
    if [ -n "$gpu_pids" ]; then
        for pid in $gpu_pids; do
            echo "[Cleanup] Killing process on GPU 0 (PID: $pid)"
            kill -9 $pid 2>/dev/null || true
        done
    fi

    # 3. 杀掉后台记录的 PID (预防万一)
    if [ -n "$ROLLOUT_PID" ]; then
        kill -9 $ROLLOUT_PID 2>/dev/null || true
    fi
    
    sleep 2
    echo "[Cleanup] GPU 0 and Port 8000 are now clear."
}

# --- 2. 注册 Trap ---
# EXIT: 脚本正常或异常退出时触发
# SIGINT: 用户按下 Ctrl+C 时触发
# SIGTERM: 被 kill 命令杀掉时触发
trap cleanup_rollout EXIT SIGINT SIGTERM


# ================= Stage 1 =================
echo "========== Starting Stage 1 =========="

# Stage1 Rewards
export REWARD_FUNCS="cplrm_format cplrm_length cplrm_copy cplrm_loose_answer"
export REWARD_FORMAT=0.1
export REWARD_LENGTH=0.1
export REWARD_COPY=0.7
export REWARD_ANSWER=0.1
export REWARD_WEIGHTS="${REWARD_FORMAT} ${REWARD_LENGTH} ${REWARD_COPY} ${REWARD_ANSWER}"
export SWANLAB_EXP_NAME="[stage1]-${EXP_NAME}"
export MAX_STEPS=200

# 1. 设置 Stage 1 的 Rollout 模型为基座模型
export CURRENT_ROLLOUT_MODEL=${MODEL_NAME}

# 2. 后台启动 Rollout 服务
echo "[Stage 1] Launching Rollout Service..."
bash rollout.sh > rollout_stage1.log 2>&1 &
ROLLOUT_PID=$!

echo "[Stage 1] Rollout Service PID: $ROLLOUT_PID"

# 3. 等待服务就绪
wait_for_vllm

# 4. 启动训练
echo "[Stage 1] Starting RLHF Training..."
bash rlhf_stage1.sh


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
export REWARD_FUNCS="cplrm_format cplrm_length cplrm_copy cplrm_strict_answer"
export REWARD_FORMAT=0.1
export REWARD_LENGTH=0.1
export REWARD_COPY=0.1
export REWARD_ANSWER=0.7
export REWARD_WEIGHTS="${REWARD_FORMAT} ${REWARD_LENGTH} ${REWARD_COPY} ${REWARD_ANSWER}"
export SWANLAB_EXP_NAME="[stage2]-${EXP_NAME}"
export MAX_STEPS=1000

# 1. 设置 Stage 2 的 Rollout 模型为 Stage 1 的产出
# 注意：通常 GRPO 需要加载上一轮训练后的模型来采样
export CURRENT_ROLLOUT_MODEL=${MODEL_NAME}

# 2. 后台启动 Rollout 服务 (重新拉起)
echo "[Stage 2] Launching Rollout Service with Stage 1 Checkpoint..."

# 3. 等待服务就绪
wait_for_vllm

# 4. 启动 Stage 2 训练
echo "[Stage 2] Starting RLHF Training..."
bash rlhf_stage2.sh

cleanup_rollout

echo "========== Pipeline Completed Successfully =========="