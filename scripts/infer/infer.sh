#!/bin/bash

# 定义要推理的数据集列表
datasets=(
    "copypaste"
    # "hotpotqa" 
    # "multirc" 
    # "musique" 
    # "2wikimultihopqa" 
    # "popqa" 
    # "qasper"
    # "faitheval" 
    # "pubmedqa" 
)
# datasets=("copypaste")

# 定义要循环的 prompt_type 列表
# prompt_types=("reasoning_with_copypaste")
prompt_types=(
    "direct_inference"
    "rag"
    # "cot"
    # "ircot"
    # "deepseek"
    # "copypaste"
)

# 定义重复次数，默认为3次
repeat_times=1

server_url="http://localhost:8124/v1"
# server_url="https://api.siliconflow.cn/v1"
num_threads=16
# model_name="Qwen3-4B-Instruct-2507"
model_name="Qwen3-4B"
max_samples=-1
split="test"
temperature=0.0

# 计算总任务数
total_tasks=$((repeat_times * ${#prompt_types[@]} * ${#datasets[@]}))
current_task=0

echo "================================"
echo "开始推理任务"
echo "总任务数: $total_tasks (重复次数: $repeat_times, prompt_types: ${#prompt_types[@]}, 数据集: ${#datasets[@]})"
echo "================================"
echo ""

# 最外层循环：重复整个推理过程
for ((i=1; i<=repeat_times; i++)); do
    echo "================================"
    echo "开始第 $i 次重复推理"
    echo "================================"

    # 中层循环：遍历每个 prompt_type
    for prompt_type in "${prompt_types[@]}"; do
        echo "开始使用 prompt_type: $prompt_type (第 $i 次重复)"
        echo "================================"

        # 内层循环：遍历每个数据集
        for dataset in "${datasets[@]}"; do
            # 更新进度计数器
            current_task=$((current_task + 1))
            
            echo "开始推理数据集: $dataset (使用 $prompt_type, 第 $i 次重复) [$current_task/$total_tasks]"
            echo "--------------------------------"

            python copypastelrm/inference/infer.py \
                --server-url "$server_url" \
                --model-name "$model_name" \
                --num-threads "$num_threads" \
                --prompt-type "$prompt_type" \
                --dataset "$dataset" \
                --max-samples $max_samples \
                --split $split \
                --temperature $temperature \
                --enable-thinking

            echo "数据集 $dataset 推理完成 [$current_task/$total_tasks]"
            echo "--------------------------------"
            echo ""
        done

        echo "prompt_type $prompt_type 的所有数据集推理完成 (第 $i 次重复)"
        echo "================================"
        echo ""
    done

    echo "第 $i 次重复推理完成"
    echo "================================"
    echo ""
done

echo "================================"
echo "所有 $repeat_times 次重复推理完成！"
echo "总进度: [$current_task/$total_tasks] (100%)"
echo "================================"
