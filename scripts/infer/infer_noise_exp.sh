#!/bin/bash

export HF_HUB_OFFLINE=0

# 定义要推理的数据集列表
datasets=(
    "FaithEval"
    "HotpotQA"
    "MultiRC"
    "MuSiQue"
    "PopQA"
    "PubMedQA"
    "Qasper"
    "2WikiMultiHopQA"
    "ConFiQA-QA"
    "ConFiQA-MC"
    "ConFiQA-MR"
    "ConFiQA-QA-Original"
    "ConFiQA-MC-Original"
    "ConFiQA-MR-Original"
)

distractor_docs=(
    0
    1
    3
    5
    7
    9
    16
    32
    64
)

# 定义要循环的 prompt_type 列表
# prompt_types=("reasoning_with_copypaste")
prompt_types=(
    # "direct_inference"
    # "cot"
    "rag"
    # "ircot"
    # "find_facts"
    # "deepseek"
    # "copypaste"
)

# 定义重复次数，默认为3次

server_url="http://localhost:8124/v1"
# server_url="https://api.siliconflow.cn/v1"
num_threads=36
model_name="Qwen2.5-7B-Instruct"
max_samples=2000
split="test"
temperature=0.6

# 计算总任务数
total_tasks=$(( ${#datasets[@]} * ${#prompt_types[@]} * ${#distractor_docs[@]} ))
current_task=0

echo "================================"
echo "开始推理任务"
echo "总任务数: $total_tasks (噪音文档数量: ${#distractor_docs[@]}, prompt_types: ${#prompt_types[@]}, 数据集: ${#datasets[@]})"
echo "================================"
echo ""

for prompt_type in "${prompt_types[@]}"; do

    for d in "${distractor_docs[@]}"; do

        # 内层循环：遍历每个数据集
        for dataset in "${datasets[@]}"; do
            # 更新进度计数器
            current_task=$((current_task + 1))

            # 均是噪音文档
            python copypastelrm/inference/infer.py \
                --server-url "$server_url" \
                --model-name "$model_name" \
                --num-threads "$num_threads" \
                --prompt-type "$prompt_type" \
                --dataset "$dataset" \
                --max-samples $max_samples \
                --split $split \
                --temperature $temperature \
                --distractor-docs $d \
                --unanswerable 

            # 存在噪音文档
            python copypastelrm/inference/infer.py \
                --server-url "$server_url" \
                --model-name "$model_name" \
                --num-threads "$num_threads" \
                --prompt-type "$prompt_type" \
                --dataset "$dataset" \
                --max-samples $max_samples \
                --split $split \
                --temperature $temperature \
                --distractor-docs $d \

            echo "进度[$current_task/$total_tasks]"
            echo "--------------------------------"
            echo ""
        done

    done
done

echo "================================"
echo "所有 $repeat_times 次重复推理完成！"
echo "总进度: [$current_task/$total_tasks] (100%)"
echo "================================"
