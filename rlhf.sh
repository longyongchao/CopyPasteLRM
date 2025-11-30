CUDA_VISIBLE_DEVICES=1,2,3 \
WANDB_API_KEY=480c15fed2c86a166517dcea0e82bcc11e19b513 \
NPROC_PER_NODE=3 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3-8B \
    --external_plugins reward.py \
    --reward_funcs copypaste_uni \
    --custom_register_path dataset.py \
    --train_type lora \
    --use_hf true \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8000 \
    --use_hf true \
    --torch_dtype bfloat16 \
    --dataset 'hotpot_qa' \
    --split_dataset_ratio 0.001 \
    --load_from_cache_file true \
    --max_length 4096 \
    --max_completion_length 2048 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 3 \
    --learning_rate 5e-6 \
    --dynamic_sample true \
    --epsilon_high 0.25 \
    --overlong_filter true \
    --reward_funcs soft_overlong \
    --soft_cache_length 512 \
    --gradient_accumulation_steps 4 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 10 \
    --logging_steps 1 \
    --output_dir /mnt/lustre/DATA/longyongchao/ms-swift \
    --warmup_ratio 0.01 \
    --dataloader_num_workers 4 \
    --num_generations 9 \
    --temperature 0.7 \
    --system 'You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.' \
    --deepspeed zero2 \
    --log_completions true \
    --report_to swanlab \
    --swanlab_token eD9F8nh3oF5zAeyopbN8f \
    --swanlab_project CopyPasteLRM \
    --swanlab_exp_name v0_0_3-qwen-2_5-32b-instruct-lora \
    --swanlab_lark_webhook_url https://open.feishu.cn/open-apis/bot/v2/hook/880e2480-71ed-4f29-8495-b7fa75c8cbd7 \
    --swanlab_lark_secret IzE5LR2O7ojQkRUO9g96Qe \
    --swanlab_mode cloud \
    --beta 0.001 \
    --num_iterations 1 \
    # --resume_from_checkpoint /data/lyc/ms-swift/output/CopyPasteLRM/v51-20251101-184710/checkpoint-600/ \

