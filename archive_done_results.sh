#!/bin/bash

# 将 results 文件夹中的所有 *_done.json 文件打包压缩到一个 zip 文件

echo "开始压缩已完成的 evaluation 结果..."

# 生成带时间戳的压缩文件名
timestamp=$(date +"%Y%m%d_%H%M%S")
zip_filename="done_results_${timestamp}.zip"

# 查找所有 *_done.json 文件
done_files=$(find results -name "*_done.json" | sort)

if [ -z "$done_files" ]; then
    echo "没有找到任何 *_done.json 文件"
    exit 0
fi

# 统计文件数量
file_count=$(echo "$done_files" | wc -l)
echo "找到 $file_count 个 *_done.json 文件"

# 创建压缩文件
echo "正在创建压缩文件: $zip_filename"
echo "$done_files" | xargs zip "$zip_filename"

if [ $? -eq 0 ]; then
    echo "压缩完成！"
    echo "压缩文件: $zip_filename"
    echo "包含的文件:"
    echo "$done_files"
else
    echo "压缩失败！"
    exit 1
fi
