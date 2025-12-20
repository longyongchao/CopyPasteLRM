#!/bin/bash
set -e

required_vars=(
    STAGE1_OUTPUT_DIR
    STAGE2_OUTPUT_DIR
    SWANLAB_TOKEN
    SWANLAB_PROJECT
    SWANLAB_EXP_NAME
    SWANLAB_LARK_WEBHOOK_URL
    SWANLAB_LARK_SECRET
    REWARD_FORMAT
    REWARD_LENGTH
    REWARD_COPY
    REWARD_ANSWER
    MODEL_NAME
    DATASET_SAMPLE
    SAVE_STEPS
    RLHF_NPROC_PER_NODE
    RLHF_CUDA_VISIBLE_DEVICES_LIST
    NUM_GENERATIONS
    BATCH_SIZE
    SPLIT_DATASET_RATIO
)

for v in "${required_vars[@]}"; do
  if [ -z "${!v}" ]; then
    echo "[ERROR] Environment variable $v is not set"
    exit 1
  fi
done

CUDA_VISIBLE_DEVICES=${RLHF_CUDA_VISIBLE_DEVICES_LIST} \
NPROC_PER_NODE=${RLHF_NPROC_PER_NODE} \
swift rlhf \
    --rlhf_type grpo \
    --model ${MODEL_NAME} \
    --train_type full \
    --custom_register_path dataset.py \
    --dataset "copypaste_qa#${DATASET_SAMPLE}" \
    --data_seed 42 \
    --split_dataset_ratio ${SPLIT_DATASET_RATIO} \
    --load_from_cache_file true \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --download_mode force_redownload \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --learning_rate 5e-7 \
    --save_total_limit 5 \
    --logging_steps 1 \
    --output_dir ${STAGE2_OUTPUT_DIR} \
    --add_version false \
    --gradient_accumulation_steps 1 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 1 \
    --max_length 32768 \
    --max_completion_length 2048 \
    --external_plugins reward.py \
    --reward_funcs cplrm_format cplrm_length cplrm_copy cplrm_answer \
    --reward_weights ${REWARD_FORMAT} ${REWARD_LENGTH} ${REWARD_COPY} ${REWARD_ANSWER} \
    --num_generations ${NUM_GENERATIONS} \
    --deepspeed zero3 \
    --temperature 1.0 \
    --top_p 0.95 \
    --log_completions true \
    --log_entropy true \
    --overlong_filter true \
    --save_steps ${SAVE_STEPS} \
    --report_to swanlab \
    --swanlab_token ${SWANLAB_TOKEN} \
    --swanlab_project ${SWANLAB_PROJECT} \
    --swanlab_exp_name ${SWANLAB_EXP_NAME} \
    --swanlab_lark_webhook_url ${SWANLAB_LARK_WEBHOOK_URL} \
    --swanlab_lark_secret ${SWANLAB_LARK_SECRET} \
    --swanlab_mode cloud \
    --beta 0.1 \
    --dynamic_sample true \
    --epsilon_high 0.25 \
    --resume_from_checkpoint ${STAGE1_OUTPUT_DIR}/last \
    --resume_only_model true \
    --ignore_data_skip true \
    # --use_vllm true \
    # --vllm_mode colocate \
    # --vllm_gpu_memory_utilization 0.4 \
    # --sleep_level 1 \
    # --offload_model true \
    # --offload_optimizer true \
    # --vllm_tensor_parallel_size 1 \
    # --vllm_max_model_len 32768 \
