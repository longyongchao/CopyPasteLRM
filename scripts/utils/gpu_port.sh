#!/bin/bash

# 1. 等待 vLLM 服务就绪
function wait_for_vllm() {
    local port=${1:-8000} # 默认检查 8000 端口
    echo "[Pipeline] Waiting for vLLM to be ready on port ${port}..."
    local max_retries=60
    local count=0
    
    while ! curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${port}/health/" | grep -q "200"; do
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


# 2. 杀死指定端口的进程
# 调用方法: kill_process_on_port 8000
function kill_process_on_port() {
    local port=$1
    if [ -z "$port" ]; then
        echo "[Warn] No port specified for killing."
        return
    fi

    # 使用 ss 查找进程 PID
    local port_pid=$(ss -tlnp | grep ":${port}" | awk -F'pid=' '{print $2}' | cut -d',' -f1)
    if [ -n "$port_pid" ]; then
        echo "[Cleanup] Killing process on port ${port} (PID: ${port_pid})"
        kill -9 $port_pid 2>/dev/null || true
    else
        echo "[Cleanup] No process found on port ${port}."
    fi
}


# 3. 杀死指定 GPU 列表上的所有进程
# 调用方法: kill_processes_on_gpus "0"  或  kill_processes_on_gpus "1,2,3"
function kill_processes_on_gpus() {
    local gpu_ids=$1
    if [ -z "$gpu_ids" ]; then
        echo "[Warn] No GPU IDs specified for killing."
        return
    fi

    echo "[Cleanup] Cleaning up processes on GPUs: ${gpu_ids}"
    
    # 将逗号分隔的 ID 转换为循环处理
    IFS=',' read -ra ADDR <<< "$gpu_ids"
    for id in "${ADDR[@]}"; do
        local gpu_pids=$(nvidia-smi --id=$id --query-compute-apps=pid --format=csv,noheader,nounits)
        if [ -n "$gpu_pids" ]; then
            for pid in $gpu_pids; do
                echo "[Cleanup] Killing process on GPU ${id} (PID: ${pid})"
                kill -9 $pid 2>/dev/null || true
            done
        fi
    done
}


# 4. 汇总清理函数（供主脚本调用）
function global_cleanup() {
    echo "[Cleanup] Starting safety cleanup sequence..."
    
    # 这里的端口和 GPU ID 建议根据主脚本的环境变量传入
    kill_process_on_port 8000
    kill_processes_on_gpus "${ROLLOUT_CUDA_VISIBLE_DEVICES_LIST}"
    
    # 额外清理后台记录的 PID
    if [ -n "$ROLLOUT_PID" ]; then
        kill -9 $ROLLOUT_PID 2>/dev/null || true
    fi
}
