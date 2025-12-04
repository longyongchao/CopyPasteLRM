python inference/infer.py \
    --server-url http://localhost:8124/v1 \
    --model-name cplrm-qwen2.5-3b-instruct-step500 \
    --num-threads 32 \
    --prompt-type reasoning with copy-paste \
    --dataset faitheval
