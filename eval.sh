#!/bin/bash

# 获取 results 文件夹中所有需要处理的 json 文件
# 过滤条件：
# 1. 文件名不以 _done.json 结尾
# 2. 不存在对应的 _done.json 文件

echo "开始处理 evaluation 任务..."

# 遍历 results 文件夹中的所有 json 文件
for json_file in $(find results -name "*.json" ! -name "*_done.json" | sort); do
    # 构造对应的 _done.json 文件路径
    done_file="${json_file%.json}_done.json"

    # 检查是否存在对应的 _done.json 文件
    if [ ! -f "$done_file" ]; then
        echo "处理文件: $json_file"
        python eval/eval.py "$json_file"
    else
        echo "跳过已处理的文件: $json_file (存在 $done_file)"
    fi
done

echo "所有 evaluation 任务完成！"
