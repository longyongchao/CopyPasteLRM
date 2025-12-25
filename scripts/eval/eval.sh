#!/bin/bash

paths=(
    "/home/lyc/projects/CopyPasteLRM/results/infer/test/Qwen3-4B-Instruct-2507/resamples_-1/seed_42/tpr_0.0/copypaste-prompt_direct_inference-1766410944.json"
    "/home/lyc/projects/CopyPasteLRM/results/infer/test/Qwen3-4B/resamples_-1/seed_42/tpr_0.0/copypaste-prompt_direct_inference-1766544226.json"
    "/home/lyc/projects/CopyPasteLRM/results/infer/test/Qwen3-4B-Instruct-2507/resamples_-1/seed_42/tpr_0.0/copypaste-prompt_cot-1766466358.json"
    "/home/lyc/projects/CopyPasteLRM/results/infer/test/Qwen3-4B-Instruct-2507/resamples_-1/seed_42/tpr_0.0/copypaste-prompt_rag-1766424839.json"
    "/home/lyc/projects/CopyPasteLRM/results/infer/test/Qwen3-4B/resamples_-1/seed_42/tpr_0.0/copypaste-prompt_rag-1766561772.json"
    "/home/lyc/projects/CopyPasteLRM/results/infer/test/Qwen3-4B-Instruct-2507/resamples_-1/seed_42/tpr_0.0/copypaste-prompt_ircot-1766471827.json"
    "/home/lyc/projects/CopyPasteLRM/results/infer/test/Qwen3-4B-Instruct-2507/resamples_-1/seed_42/tpr_0.0/copypaste-prompt_deepseek-1766478707.json"
    "/home/lyc/projects/CopyPasteLRM/results/infer/test/Qwen3-4B-Instruct-2507/resamples_-1/seed_42/tpr_0.0/copypaste-prompt_deepseek-1766583151.json"
    "/home/lyc/projects/CopyPasteLRM/results/infer/test/Qwen3-4B-Instruct-2507/resamples_-1/seed_42/tpr_0.0/copypaste-prompt_copypaste-1766487293.json"
    "/home/lyc/projects/CopyPasteLRM/results/infer/test/Qwen3-4B-Instruct-2507/resamples_-1/seed_42/tpr_0.0/copypaste-prompt_copypaste-1766590282.json"
)

# 遍历paths
for path in "${paths[@]}"
do
    python copypastelrm/eval/eval_1.py $path
done