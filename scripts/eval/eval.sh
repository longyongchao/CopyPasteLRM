#!/bin/bash

paths=(
    results/infer/test/8_dense_copy_answer_f1_warmup_musique_reasonable_qwen_3_4b_instruct/resamples_-1/seed_42/tpr_0.0/copypaste-prompt_copypaste-1767114930.json
)

# 遍历paths
for path in "${paths[@]}"
do
    python copypastelrm/eval/eval_1.py $path
done