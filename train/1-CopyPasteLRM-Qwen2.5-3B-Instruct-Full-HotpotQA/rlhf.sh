#!/bin/bash

. ../.env 

CUDA_VISIBLE_DEVICES=1,2,3
NPROC_PER_NODE=3

swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-3B-Instruct \
    --model_type qwen2_5 \
    --external_plugins reward.py \
    --reward_funcs copypaste_uni \
    --custom_register_path dataset.py \
    --train_type full \
    --use_hf true \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8000 \
    --torch_dtype bfloat16 \
    --dataset hotpot_qa \
    --split_dataset_ratio 0.005 \
    --load_from_cache_file true \
    --max_length 4096 \
    --max_completion_length 2048 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --learning_rate 5e-7 \
    --gradient_accumulation_steps 4 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 10 \
    --logging_steps 1 \
    --output_dir /mnt/lustre/DATA/longyongchao/ms-swift/output/CopyPasteLRM \
    --warmup_ratio 0.01 \
    --dataloader_num_workers 4 \
    --num_generations 8 \
    --temperature 1.0 \
    --system You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer. \
    --deepspeed zero3 \
    --log_completions true \
    --log_entropy true \
    --report_to swanlab \
    --swanlab_token $SWANLEB_TOKEN \
    --swanlab_project CopyPasteLRM \
    --swanlab_exp_name v0_0_3-qwen2_5-3B-Instruct \
    --swanlab_mode cloud \
    --beta 0.001 \
    --num_iterations 1\