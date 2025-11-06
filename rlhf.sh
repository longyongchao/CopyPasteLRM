CUDA_VISIBLE_DEVICES=0,1 \
WANDB_API_KEY=480c15fed2c86a166517dcea0e82bcc11e19b513 \
NPROC_PER_NODE=2 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-3B-Instruct \
    --external_plugins reward.py \
    --reward_funcs copypaste_uni \
    --custom_register_path dataset.py \
    --train_type full \
    --use_hf true \
    --torch_dtype bfloat16 \
    --dataset 'hotpot_qa' \
    --split_dataset_ratio 0.001 \
    --load_from_cache_file true \
    --max_length 2048 \
    --max_completion_length 1024 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --learning_rate 5e-7 \
    --gradient_accumulation_steps 4 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 10 \
    --logging_steps 1 \
    --output_dir /data/lyc/ms-swift/output/CopyPasteLRM \
    --warmup_ratio 0.01 \
    --dataloader_num_workers 4 \
    --num_generations 8 \
    --temperature 1.0 \
    --system 'You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.' \
    --deepspeed zero3 \
    --log_completions true \
    --report_to wandb \
    --beta 0.001 \
    --num_iterations 1 \
    # --resume_from_checkpoint /data/lyc/ms-swift/output/CopyPasteLRM/v51-20251101-184710/checkpoint-600/ \
