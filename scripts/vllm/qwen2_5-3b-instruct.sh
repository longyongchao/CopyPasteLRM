# 激活conda环境
unset http_proxy
unset https_proxy

export CUDA_VISIBLE_DEVICES=1

python -m vllm.entrypoints.openai.api_server \
  --served-model-name qwen2.5-3b-instruct \
  --model Qwen/Qwen2.5-3B-Instruct \
  --host 0.0.0.0 \
  --port 8124 \
  --gpu-memory-utilization 0.88 \
  --max-model-len 8192 \
  --max-num-seqs 96 \
  --tensor-parallel-size 1 \
  --disable-log-requests \
  # --disable-log-stats
