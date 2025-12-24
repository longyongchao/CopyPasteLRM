#!/bin/bash

paths=(
    "results/infer/test/Qwen3-4B/resamples_-1/seed_42/tpr_0.0/copypaste-prompt_direct_inference-1766544226.json"
)

# 遍历paths
for path in "${paths[@]}"
do
    python copypastelrm/eval/eval_1.py $path
done