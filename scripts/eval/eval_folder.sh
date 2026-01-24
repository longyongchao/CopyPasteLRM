#!/bin/bash

# 批量评估脚本 - 适配新的文件夹格式结果

# 定义要评估的结果文件夹列表
folders=(
    # 示例：添加你的结果文件夹路径
    # "results/infer/test/Qwen2.5-7B-Instruct/resamples_1000/seed_42/tpr_0.0/prompt_rag/MultiRC-noise_8-answerable/20250124120000"
    # "results/infer/test/Qwen2.5-7B-Instruct/resamples_1000/seed_42/tpr_0.0/prompt_rag/HotpotQA-noise_8-answerable/20250124120000"
)

# 如果没有提供参数，使用上面的folders列表
# 如果提供了参数，使用命令行参数
if [ $# -gt 0 ]; then
    folders=("$@")
fi

echo "================================"
echo "开始评估任务"
echo "总文件夹数: ${#folders[@]}"
echo "================================"
echo ""

# 遍历所有文件夹
for i in "${!folders[@]}"; do
    folder="${folders[$i]}"
    echo "[$((i+1))/${#folders[@]}] 评估: $folder"

    if [ ! -d "$folder" ]; then
        echo "  [跳过] 文件夹不存在"
        echo ""
        continue
    fi

    # 运行评估
    python -m copypastelrm.eval.eval_folder "$folder"

    echo ""
done

echo "================================"
echo "所有评估任务完成！"
echo "================================"
