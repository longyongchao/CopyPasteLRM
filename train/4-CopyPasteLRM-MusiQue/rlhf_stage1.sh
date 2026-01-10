#!/bin/bash
set -e


CUDA_VISIBLE_DEVICES=${RLHF_CUDA_VISIBLE_DEVICES_LIST} \
NPROC_PER_NODE=${RLHF_NPROC_PER_NODE} \
swift rlhf \
    --rlhf_type grpo \
    --model ${MODEL_NAME} \
    --train_type full \
    --custom_register_path train/4-CopyPasteLRM-MusiQue/dataset.py \
    --dataset ${RLHF_DATASET} \
    --data_seed 42 \
    --split_dataset_ratio ${SPLIT_DATASET_RATIO} \
    --load_from_cache_file true \
    --torch_dtype bfloat16 \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --download_mode force_redownload \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --learning_rate ${GRPO_LEARNING_RATE} \
    --save_total_limit ${GRPO_SAVE_TOTAL_LIMIT} \
    --logging_steps 1 \
    --output_dir ${STAGE1_OUTPUT_DIR} \
    --add_version false \
    --create_checkpoint_symlink true \
    --gradient_accumulation_steps ${GRPO_GRADIENT_ACCUMULATION_STEPS} \
    --warmup_ratio ${GRPO_WARMUP_RATIO} \
    --dataloader_num_workers 1 \
    --max_length 32768 \
    --max_completion_length ${GRPO_MAX_NEW_TOKENS} \
    --external_plugins train/4-CopyPasteLRM-MusiQue/reward.py \
    --reward_funcs ${REWARD_FUNCS} \
    --reward_weights ${REWARD_WEIGHTS} \
    --num_generations ${NUM_GENERATIONS} \
    --deepspeed zero3 \
    --temperature ${GRPO_TEMPERATURE} \
    --top_p ${GRPO_TOP_P} \
    --log_completions true \
    --log_entropy true \
    --overlong_filter true \
    --save_steps ${SAVE_STEPS} \
    --eval_steps ${EVAL_STEPS} \
    --max_steps ${MAX_STEPS} \
    --report_to swanlab \
    --swanlab_token ${SWANLAB_TOKEN} \
    --swanlab_project ${SWANLAB_PROJECT} \
    --swanlab_exp_name ${SWANLAB_EXP_NAME} \
    --swanlab_lark_webhook_url ${SWANLAB_LARK_WEBHOOK_URL} \
    --swanlab_lark_secret ${SWANLAB_LARK_SECRET} \
    --swanlab_mode ${SWANLAB_MODEL} \
    --beta ${GRPO_BETA} \
    --dynamic_sample true \
    --epsilon_high ${GRPO_EPSILON_HIGH} \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8000 \

