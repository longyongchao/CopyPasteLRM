#!/bin/bash

# 定义要推理的数据集列表
datasets=("hotpotqa" "multirc" "pubmedqa" "musique" "2wikimultihopqa" "popqa" "faitheval" "qasper")

# 定义要循环的 prompt_type 列表
prompt_types=("direct" "reasoning" "reasoning_with_copypaste")

# 定义重复次数，默认为3次
repeat_times=2

model_name="DeepSeek-R1-Distill-Qwen-7B"

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
            echo "开始推理数据集: $dataset (使用 $prompt_type, 第 $i 次重复)"
            echo "--------------------------------"

            python copypastelrm/inference/infer.py \
                --server-url http://localhost:8124/v1 \
                --model-name "$model_name" \
                --num-threads 32 \
                --prompt-type "$prompt_type" \
                --dataset "$dataset" \
                --max-samples 1000

            echo "数据集 $dataset 推理完成"
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

echo "所有 $repeat_times 次重复推理完成！"
