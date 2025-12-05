# 激活conda环境
unset http_proxy
unset https_proxy

export CUDA_VISIBLE_DEVICES=0,1

python -m vllm.entrypoints.openai.api_server \
  --served-model-name Qwen3-8B \
  --model Qwen/Qwen3-8B \
  --host 0.0.0.0 \
  --port 8124 \
  --gpu-memory-utilization 0.88 \
  --max-model-len 8192 \
  --max-num-seqs 64 \
  --tensor-parallel-size 2 \
  --disable-log-requests \
  # --disable-log-stats
