#!/bin/bash

# 使用 ("$@") 将接收到的所有提示类型重新打包成一个新数组
target_dataset=("$@")

required_vars=(
    VLLM_PORT
    VLLM_MAX_S
    VLLM_SERVED_MODEL_NAME
    PASS_TEMPERATURE
    DATASET_MAX_SAMPLES
    DATASET_SPLIT
    PASS_K_VALUE
    PASS_PRIOR_THRESHOLD
)

for v in "${required_vars[@]}"; do
  if [ -z "${!v}" ]; then
    echo "[ERROR] Environment variable $v is not set"
    exit 1
  fi
done

SERVER_URL="http://localhost:${VLLM_PORT}/v1"

for entry in "${target_dataset[@]}"; do
    
    # 初始化变量
    output_arg=""
    current_dataset_name=""

    # 检查 entry 是否包含 '='
    if [[ "$entry" == *"="* ]]; then
        # 如果包含 =，则切割字符串
        # %%t=* 删除从第一个 = 开始往后的所有内容，保留前半部分 (dataset name)
        current_dataset_name="${entry%%=*}"
        # #*= 删除从第一个 = 开始往前（包含=）的所有内容，保留后半部分 (path)
        reload_path="${entry#*=}"
        
        # 构造参数
        output_arg="--output-file ${reload_path}"
        
        echo "🔄 检测到断点重启路径，数据集: [${current_dataset_name}]"
        echo "📂 指定输出文件: ${reload_path}"
    else
        # 如果不包含 =，则直接作为 dataset name
        current_dataset_name="$entry"
        echo "✨ 新任务（或自动路径），数据集: [${current_dataset_name}]"
    fi

    # 执行 Python 脚本
    # 注意：这里引用了 $output_arg，如果不为空，它会展开为 --output-file /path/...
    # 如果为空，则 Python 脚本会走默认路径生成逻辑
    python copypastelrm/inference/inferPass@K.py \
        --server-url "${SERVER_URL}" \
        --model-name "${VLLM_SERVED_MODEL_NAME}" \
        --dataset "${current_dataset_name}" \
        --split "${DATASET_SPLIT}" \
        --num-threads "${VLLM_MAX_S}" \
        --max-samples "${DATASET_MAX_SAMPLES}" \
        --k "${PASS_K_VALUE}" \
        --temperature "${PASS_TEMPERATURE}" \
        --prior-threshold "${PASS_PRIOR_THRESHOLD}" \
        $output_arg \
        # --enable-thinking 
        
    # 检查上一个命令的退出状态
    if [ $? -eq 0 ]; then
        echo "✅ 数据集 ${current_dataset_name} 推理任务成功完成。"
    else
        echo "❌ 数据集 ${current_dataset_name} 推理任务执行失败！"
        # 如果你希望在任何一个数据集失败后就停止整个脚本，可以取消下一行的注释：
        # exit 1 
    fi

done