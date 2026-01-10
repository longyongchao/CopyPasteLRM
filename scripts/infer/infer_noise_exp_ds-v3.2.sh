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
    "2WikiMultiHopQA"
    "ConFiQA-MC"
    "ConFiQA-MR"
    "ConFiQA-QA"
    # "Qasper"
    # "ConFiQA-QA-Original"
    # "ConFiQA-MC-Original"
    # "ConFiQA-MR-Original"
)

distractor_docs=(
    0
    2
    4
    8
    16
    32
    64
)

# 定义要循环的 prompt_type 列表
# prompt_types=("reasoning_with_copypaste")
prompt_types=(
    # "direct_inference"
    # "cot"
    "find_facts"
    # "deepseek"
    "rag"
    "copypaste"
    # "ircot"
)

# 定义重复次数，默认为3次

# server_url="http://localhost:8124/v1"
server_url="https://api.siliconflow.cn/v1"
num_threads=2
model_name="deepseek-ai/DeepSeek-V3.2"
max_samples=200
split="test"
temperature=0.6

# 计算总任务数
total_tasks=$(( ${#datasets[@]} * ${#prompt_types[@]} * ${#distractor_docs[@]} ))
current_task=0

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

            echo "当前进度[$current_task/$total_tasks]"
            echo "--------------------------------"
            echo ""
        done

    done
done

echo "================================"
echo "所有 $repeat_times 次重复推理完成！"
echo "总进度: [$current_task/$total_tasks] (100%)"
echo "================================"
