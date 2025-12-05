# 激活conda环境
unset http_proxy
unset https_proxy

export CUDA_VISIBLE_DEVICES=1

python -m vllm.entrypoints.openai.api_server \
  --served-model-name DeepSeek-R1-Distill-Qwen-7B \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --host 0.0.0.0 \
  --port 8124 \
  --gpu-memory-utilization 0.88 \
  --max-model-len 8192 \
  --max-num-seqs 32 \
  --tensor-parallel-size 1 \
  --disable-log-requests \
  --disable-log-stats
