#!/bin/bash

# 定义要推理的数据集列表
datasets=(
    "MultiRC"
    "PubMedQA"
    "PopQA"
    "2WikiMultiHopQA"
    "MuSiQue"
    "HotpotQA"
    "FaithEval"
    "ConFiQA-MR"
    "ConFiQA-MC"
    "ConFiQA-QA"
    "ConFiQA-MR-Original"
    "ConFiQA-MC-Original"
    "ConFiQA-QA-Original"
    # "Qasper"
)

# 定义要循环的 prompt_type 列表
# 可用类型：direct_inference, cot, rag, rag_rep_2, rag_rep_q, rag_qcq, rag_qcq2, rag_q_int_q, rag_q_int2_q, rag_decompressed, rag_decompressed_rep_q, ircot, deepseek, copypaste, find_facts
prompt_types=(
    # "direct_inference"        # 无上下文，直接回答
    # "cot"                     # 无上下文，CoT推理
    # "rag"                       # 基础RAG
    # "rag_rep_2"                 # 上下文+问题 重复2次
    # "rag_rep_q"                 # 上下文+问题，再重复问题
    # "rag_qcq"                   # 问题-上下文-问题
    # "rag_qcq2"                  # 问题-上下文-问题-问题
    # "rag_q_int_q"               # 问题-上下文(带内部问题)-问题
    # "rag_q_int2_q"              # 问题-上下文(带内部问题)-问题-问题
    # "rag_decompressed"          # 稀疏化问题（字符间加空格）
    # "rag_decompressed_rep_q"    # 上下文+稀疏化问题+稀疏化问题
    "rag_q_int_docs_q"          # 每一个文档后面都插入一个Question，另外最后也多添加一个Q，QDocQDocQDocQ...Q
    # "ircot"                   # 基于上下文的CoT
    # "deepseek"                # DeepSeek风格 <|think|>/<|answer|>
    # "copypaste"               # CopyPaste风格（带证据提取）
    # "find_facts"              # 提取事实任务
)

# 定义 distractor_docs 列表
# distractor_docs_list=(0 1 2 4 8 16 32 64)
distractor_docs_list=(8)

server_url="http://localhost:8124/v1"
# server_url="https://api.siliconflow.cn/v1"
num_threads=4
# model_name="Qwen3-4B-Instruct-2507"
model_name="Qwen2.5-7B-Instruct"
max_samples=1500
split="test"
temperature=0.0
batch_size=16

# 计算总任务数
total_tasks=$((${#prompt_types[@]} * ${#datasets[@]} * ${#distractor_docs_list[@]}))
current_task=0

echo "================================"
echo "开始推理任务"
echo "总任务数: $total_tasks (prompt_types: ${#prompt_types[@]}, 数据集: ${#datasets[@]}, distractor_docs: ${#distractor_docs_list[@]})"
echo "================================"
echo ""

# 最外层循环：遍历每个 distractor_docs 值
for distractor_docs in "${distractor_docs_list[@]}"; do
    echo "================================"
    echo "开始使用 distractor_docs: $distractor_docs"
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
            echo "[$current_task/$total_tasks] 推理数据集: $dataset (prompt_type: $prompt_type, distractor_docs: $distractor_docs)"

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
done

echo ""
echo "================================"
echo "所有推理任务完成！"
echo "================================"
