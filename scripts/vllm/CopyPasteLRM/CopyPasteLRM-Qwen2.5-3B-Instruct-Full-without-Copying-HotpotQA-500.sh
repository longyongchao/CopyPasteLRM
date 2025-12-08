# 激活conda环境
eval "$(conda shell.bash hook)"
conda activate vllm

export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m vllm.entrypoints.openai.api_server \
  --served-model-name CopyPasteLRM-Qwen2.5-3B-Instruct-Full-without-Copying-HotpotQA-500 \
  --model  /mnt/lustre/DATA/longyongchao/CopyPasteLRM/v1-20251206-231329/best_checkpoint/checkpoint-500 \
  --host 0.0.0.0 \
  --port 8124 \
  --gpu-memory-utilization 0.88 \
  --max-model-len 8192 \
  --max-num-seqs 128 \
  --tensor-parallel-size 4 \
  --disable-log-requests \
  --disable-log-stats
