# 激活conda环境
unset http_proxy
unset https_proxy

export CUDA_VISIBLE_DEVICES=1

python -m vllm.entrypoints.openai.api_server \
  --served-model-name llama-3.1-8b-instruct \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --port 8124 \
  --gpu-memory-utilization 0.88 \
  --max-model-len 8192 \
  --max-num-seqs 96 \
  --tensor-parallel-size 1 \
  --disable-log-requests \
  # --disable-log-stats

python inference/hotpotqa.py --model-url http://localhost:8124/v1 --model-name cplrm-32b-lora --output-file results/hotpotqa/cplrm-32b-lora-direct-prompt.json --num-threads 64 --prompt-type direct
