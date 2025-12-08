#!/bin/bash

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# 获取 results/infer 文件夹中所有需要处理的 json 文件
# 过滤条件：

echo "开始处理 evaluation 任务..."

# 遍历 results/infer 文件夹中的所有 json 文件
for json_file in $(find results/infer -name "*.json" ! -name "*_done.json" | sort); do
    # 构造对应的 _done.json 文件路径
    done_file="${json_file%.json}_done.json"

    # 检查是否存在对应的 _done.json 文件
    if [ ! -f "$done_file" ]; then
        echo "处理文件: $json_file"
        python copypastelrm/eval/eval.py "$json_file"

        # 评估完成后，立即为该文件生成对应的 Markdown 文件
        echo "为 $done_file 生成对应的 Markdown 文件..."
        python copypastelrm/eval/generate_obsidian_record.py --results-dir results/eval
    else
        echo "跳过已处理的文件: $json_file (存在 $done_file)"
    fi
done

echo "所有 evaluation 任务完成！"
echo "所有任务已完成！"
