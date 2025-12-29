#!/bin/bash

paths=(
    /mnt/lustre/home/longyongchao/projects/CopyPasteLRM/results/infer/test/Qwen2.5-7B-Instruct/resamples_-1/seed_42/tpr_0.0/copypaste-prompt_direct_inference-1766736772.json
    /mnt/lustre/home/longyongchao/projects/CopyPasteLRM/results/infer/test/Qwen2.5-7B-Instruct/resamples_-1/seed_42/tpr_0.0/copypaste-prompt_cot-1766740685.json
    /mnt/lustre/home/longyongchao/projects/CopyPasteLRM/results/infer/test/Qwen2.5-7B-Instruct/resamples_-1/seed_42/tpr_0.0/copypaste-prompt_rag-1766737306.json
    /mnt/lustre/home/longyongchao/projects/CopyPasteLRM/results/infer/test/Qwen2.5-7B-Instruct/resamples_-1/seed_42/tpr_0.0/copypaste-prompt_ircot-1766741143.json
    /mnt/lustre/home/longyongchao/projects/CopyPasteLRM/results/infer/test/Qwen2.5-7B-Instruct/resamples_-1/seed_42/tpr_0.0/copypaste-prompt_deepseek-1766744509.json
    /mnt/lustre/home/longyongchao/projects/CopyPasteLRM/results/infer/test/Qwen2.5-7B-Instruct/resamples_-1/seed_42/tpr_0.0/copypaste-prompt_copypaste-1766747920.json
)

# 遍历paths
for path in "${paths[@]}"
do
    python copypastelrm/eval/eval_1.py $path
done