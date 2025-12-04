# 激活conda环境
eval "$(conda shell.bash hook)"
conda activate lyc

export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m vllm.entrypoints.openai.api_server \
  --served-model-name cplrm-qwen2.5-3b-instruct \
  --model  /mnt/lustre/DATA/longyongchao/ms-swift/output/CopyPasteLRM/v34-20251108-085932/checkpoint-1750/ \
  --host 0.0.0.0 \
  --port 8124 \
  --gpu-memory-utilization 0.88 \
  --max-model-len 8192 \
  --max-num-seqs 96 \
  --tensor-parallel-size 4 \
  --disable-log-requests \
  # --disable-log-stats
