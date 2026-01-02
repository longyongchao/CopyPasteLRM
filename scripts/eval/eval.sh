#!/bin/bash

paths=(
    results/infer/test/12302157-0102002204-only_copy_format_q3_4b_I_reasonable/resamples_-1/seed_42/tpr_0.0/copypaste-prompt_copypaste-1767330421.json
)

# 遍历paths
for path in "${paths[@]}"
do
    python copypastelrm/eval/eval_1.py $path
done