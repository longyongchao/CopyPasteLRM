#!/bin/bash

# ============= vLLM 服务启动函数 =================
# 参数说明:
# 1: env_name, 2: cuda_devices, 3: served_name, 4: model_path, 5: max_len, 6: max_seqs
function launch_vllm_service() {
    local env_name=$1
    local cuda_devices=$2
    local served_name=$3
    local model_path=$4
    local max_len=$5
    local max_seqs=$6
    local port=${7:-8124} # 增加可选参数端口，默认 8124

    echo "[vLLM] Activating environment: $env_name"
    eval "$(conda shell.bash hook)"
    conda activate "$env_name"

    # 自动计算 Tensor Parallel Size
    # 将 "0,1,2,3" 转换为数组并计算长度
    IFS=',' read -ra GPUS <<< "$cuda_devices"
    local tp_size=${#GPUS[@]}

    echo "[vLLM] Starting server on port $port with TP=$tp_size..."
    
    export CUDA_VISIBLE_DEVICES=$cuda_devices

    # 使用 nohup 或后台运行，建议在主脚本中控制日志重定向
    python -m vllm.entrypoints.openai.api_server \
        --served-model-name "$served_name" \
        --model "$model_path" \
        --host 0.0.0.0 \
        --port "$port" \
        --gpu-memory-utilization 0.88 \
        --max-model-len "$max_len" \
        --max-num-seqs "$max_seqs" \
        --tensor-parallel-size "$tp_size" \
        --disable-log-requests \
        --disable-log-stats
}


# ============= 等待 vLLM 服务就绪 ===============
function wait_for_vllm() {
    local port=${1:-8000} # 默认检查 8000 端口
    echo "[Pipeline] Waiting for vLLM to be ready on port ${port}..."
    local max_retries=60
    local count=0
    
    while ! (curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${port}/health/" | grep -q "200" || curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${port}/health" | grep -q "200"); do
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

# 函数：等待 vLLM 服务启动
function wait_for_vllm_old() {
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