#!/bin/bash

paths=(
    results/infer/test/V4-0112145512-answer_copy_v3_q25_7b_I_reasonable-Qwen2.5-7B-Instruct-stage2/resamples_-1/seed_42/tpr_0.6/prompt_copypaste/MuSiQue-noise_8-answerable.json
)

# 遍历paths
for path in "${paths[@]}"
do
    python copypastelrm/eval/eval_1.py $path
done