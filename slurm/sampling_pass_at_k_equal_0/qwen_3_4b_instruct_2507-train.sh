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
# VLLM_CONDA_ENV="vllm"
VLLM_CONDA_ENV="ms-swift"
# VLLM_DEVICES="0,1,2,3"
VLLM_DEVICES="0,1"
VLLM_SERVED_MODEL_NAME="Qwen3-4B-Instruct-2507" # vLLM服务模型名称
VLLM_SERVED_MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507" # vLLM服务模型路径
VLLM_MAX_L=32768
VLLM_MAX_S=16
VLLM_PORT=8124

# 这里需要 export 变量，或者确保 source 的脚本能访问到。
# 因为 sampling_pass_at_k_equal_0.sh 是 source 调用的，所以本地变量即可。
PASS_K_VALUE=128
PASS_PRIOR_THRESHOLD=120
PASS_TEMPERATURE=1.0

# 数据集相关变量
DATASET_MAX_SAMPLES=-1
DATASET_SPLIT="train"

# 格式说明：
# 1. 纯数据集名: "musique" -> 使用自动生成的路径
# 2. 指定重启路径: "dataset=/path/to/file.jsonl" -> 使用指定文件断点续传
target_datasets=(
    "hotpotqa=/data/lyc/CopyPasteLRM/pass_at_128/Qwen3-4B-Instruct-2507/resamples_-1/train/hotpotqa-tpr_1.0-tpp_0.95-enable_thinking_False-tips_threshold_120-1766626204.jsonl"
    "2wikimultihopqa=/data/lyc/CopyPasteLRM/pass_at_128/Qwen3-4B-Instruct-2507/resamples_-1/train/2wikimultihopqa-tpr_1.0-tpp_0.95-enable_thinking_False-tips_threshold_120-1766706218.jsonl"
    "popqa=/data/lyc/CopyPasteLRM/pass_at_128/Qwen3-4B-Instruct-2507/resamples_-1/train/popqa-tpr_1.0-tpp_0.95-enable_thinking_False-tips_threshold_120-1766718146.jsonl" 
    "musique"
    "multirc=/data/lyc/CopyPasteLRM/pass_at_128/Qwen3-4B-Instruct-2507/resamples_-1/train/multirc-tpr_1.0-tpp_0.95-enable_thinking_False-tips_threshold_120-1766715492.jsonl"
    # "qasper"
    # "pubmedqa"
    # "faitheval"
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
# 将 target_datasets 数组的所有元素传给脚本
source scripts/sampling_pass_at_k_equal_0/sampling_pass_at_k_equal_0.sh "${target_datasets[@]}"

# 4. 杀死vLLM服务
kill_process_on_port $VLLM_PORT
kill_processes_on_gpus $VLLM_DEVICES

# 5. 记录结束时间戳
end_time=$(date +%s)

# 格式化 target_datasets 以便在通知中显示 (将数组转换为字符串)
datasets_str="${target_datasets[*]}"

# 5. 发送飞书通知
# 注意：prompt_types 变量在当前脚本未定义，如果它是上游变量请确保已设置。
# 这里将 DATASET_NAME 替换为 datasets_str 以展示所有处理的任务。
send_feishu_msg "✅ Pass@K=${PASS_K_VALUE} Subset采样完成 \n总耗时: $(($end_time - $start_time)/60)分钟 \n Model: $VLLM_SERVED_MODEL_NAME \n Datasets: $datasets_str \n Split: $DATASET_SPLIT"