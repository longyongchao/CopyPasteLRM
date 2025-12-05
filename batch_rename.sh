TARGET_DIR="results"
OLD_STR="qwen2.5-3b-instruct"
NEW_STR="Qwen2.5-3B-Instruct"

# 预览
find "$TARGET_DIR" -type f -name "*.json" -print0 | while read -d '' -r file; do
  dir=$(dirname "$file")
  filename=$(basename "$file")
  new_filename="${filename//$OLD_STR/$NEW_STR}"
  echo "原文件：$file"
  echo "新文件：$dir/$new_filename"
  echo "-------------------------"
done

# 执行重命名（确认后取消注释）
find "$TARGET_DIR" -type f -name "*.json" -print0 | while read -d '' -r file; do
  dir=$(dirname "$file")
  filename=$(basename "$file")
  new_filename="${filename//$OLD_STR/$NEW_STR}"
  if [ "$filename" != "$new_filename" ]; then
    mv "$file" "$dir/$new_filename"
    echo "已重命名：$file -> $dir/$new_filename"
  fi
done
