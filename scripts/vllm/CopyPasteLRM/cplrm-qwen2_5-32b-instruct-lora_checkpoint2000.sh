#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate swift

export CUDA_VISIBLE_DEVICES=0,1,2,3

base_model="Qwen/Qwen2.5-32B-Instruct"
lora_modules_path="/mnt/lustre/DATA/longyongchao/ms-swift/v3-20251109-211436/backup/checkpoint-2000"

# 启动VLLM服务
python -m vllm.entrypoints.openai.api_server \
	--model $base_model \
	--enable-lora \
	--lora-modules cplrm-32b-lora=$lora_modules_path \
	--host 0.0.0.0 \
	--port 8124 \
	--gpu-memory-utilization 0.85 \
	--max-model-len 8192 \
	--max-num-seqs 128 \
	--tensor-parallel-size 4 \
	# --max-lora-rank 8 \
	# --disable-log-requests \
	# --disable-log-stats
