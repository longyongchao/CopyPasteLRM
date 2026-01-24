#!/bin/bash

# 定义要推理的数据集列表
datasets=(
    "MultiRC"
    "ConFiQA-MR-Original"
    "ConFiQA-MC-Original"
    "ConFiQA-QA-Original"
    "ConFiQA-MR"
    "ConFiQA-MC"
    "ConFiQA-QA"
    "PubMedQA"
    # "Qasper"
    "PopQA"
    "2WikiMultiHopQA"
    "MuSiQue"
    "HotpotQA"
    "FaithEval"
)
# datasets=("copypaste")

# 定义要循环的 prompt_type 列表
# prompt_types=("reasoning_with_copypaste")
prompt_types=(
    # "rag"
    # "rag_rep_2"
    "rag_rep_q"
    # "direct_inference"
    # "cot"
    # "ircot"
    # "deepseek"
    # "copypaste"
)

server_url="http://localhost:8124/v1"
# server_url="https://api.siliconflow.cn/v1"
num_threads=4
# model_name="Qwen3-4B-Instruct-2507"
model_name="Qwen2.5-7B-Instruct"
max_samples=1000
split="test"
temperature=0.0
batch_size=16
distractor_docs=8  # Number of distractor documents to add

# 计算总任务数
total_tasks=$((${#prompt_types[@]} * ${#datasets[@]}))
current_task=0

echo "================================"
echo "开始推理任务"
echo "总任务数: $total_tasks (prompt_types: ${#prompt_types[@]}, 数据集: ${#datasets[@]})"
echo "================================"
echo ""

# 外层循环：遍历每个 prompt_type
for prompt_type in "${prompt_types[@]}"; do
    echo "================================"
    echo "开始使用 prompt_type: $prompt_type"
    echo "================================"
    echo ""

    # 内层循环：遍历每个数据集
    for dataset in "${datasets[@]}"; do
        # 更新进度计数器
        current_task=$((current_task + 1))
        echo "[$current_task/$total_tasks] 推理数据集: $dataset (prompt_type: $prompt_type)"

        python -m copypastelrm.inference.infer \
            --server-url "$server_url" \
            --model-name "$model_name" \
            --num-threads "$num_threads" \
            --prompt-type "$prompt_type" \
            --dataset "$dataset" \
            --max-samples $max_samples \
            --split $split \
            --temperature $temperature \
            --batch-size $batch_size \
            --distractor-docs $distractor_docs \
            --enable-thinking

    done
done

echo ""
echo "================================"
echo "所有推理任务完成！"
echo "================================"
