#!/bin/bash

# ===============================================
# Pass@K 推理配置
# ===============================================

# --- [ 新增配置: 数据集列表 ] ---
# 定义需要遍历执行的数据集名称列表
# 确保这些名称与 inferPass@K.py 中支持的 --dataset 参数值一致
DATASETS=(
    "multirc"
    "popqa" 
    "qasper"
    "2wikimultihopqa"
    "musique"
    "hotpotqa"
)

# 模型和服务器配置
# MODEL_NAME="Qwen3-8B"
MODEL_NAME="Qwen2.5-3B-Instruct"
SERVER_URL="http://localhost:8124/v1"
API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# 实验参数
K_VALUE=1024                       # Pass@K 的 K 值，即最大采样次数
PRIOR_THRESHOLD=1000
TEMPERATURE=1.0                   # 模型生成温度
TOP_P=0.95                        # 模型生成 top-p
NUM_THREADS=128                   # 并行推理的线程数量
MAX_SAMPLES=3000                  # 最大处理样本数 (设置为 None 则处理全部)

# 检查 Python 文件是否存在
if [ ! -f "copypastelrm/inference/inferPass@K.py" ]; then
    echo "错误: 找不到 inferPass@K.py 文件。"
    exit 1
fi

echo "--- 启动 Pass@${K_VALUE} 推理任务 (共 ${#DATASETS[@]} 个数据集) ---"
echo "模型: ${MODEL_NAME}"
echo "========================================"

# --- [ 核心修改: 遍历循环 ] ---
for DATASET_NAME in "${DATASETS[@]}"; do
    
    echo ""
    echo "========================================"
    echo "➡️ 正在处理数据集: ${DATASET_NAME} ⬅️"
    echo "========================================"
    
    # 执行 Python 脚本，使用当前循环中的 $DATASET_NAME 变量
    python copypastelrm/inference/inferPass@K.py \
        --server-url "${SERVER_URL}" \
        --model-name "${MODEL_NAME}" \
        --dataset "${DATASET_NAME}" \
        --api-key "${API_KEY}" \
        --num-threads "${NUM_THREADS}" \
        --max-samples "${MAX_SAMPLES}" \
        --k "${K_VALUE}" \
        --temperature "${TEMPERATURE}" \
        --prior-threshold "${PRIOR_THRESHOLD}" \
        --top-p "${TOP_P}" \
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

echo ""
echo "--- 所有数据集的 Pass@${K_VALUE} 推理任务已完成！ ---"