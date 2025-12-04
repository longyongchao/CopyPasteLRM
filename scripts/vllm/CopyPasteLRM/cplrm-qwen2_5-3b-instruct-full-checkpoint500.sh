# 激活conda环境
eval "$(conda shell.bash hook)"
conda activate lyc

export CUDA_VISIBLE_DEVICES=2,3

python -m vllm.entrypoints.openai.api_server \
  --served-model-name cplrm-qwen2.5-3b-instruct-step500 \
  --model  /mnt/lustre/DATA/longyongchao/ms-swift/output/CopyPasteLRM/v34-20251108-085932/best_ck_backup/checkpoint-500 \
  --host 0.0.0.0 \
  --port 8124 \
  --gpu-memory-utilization 0.88 \
  --max-model-len 8192 \
  --max-num-seqs 64 \
  --tensor-parallel-size 2 \
  --disable-log-requests \
  # --disable-log-stats
