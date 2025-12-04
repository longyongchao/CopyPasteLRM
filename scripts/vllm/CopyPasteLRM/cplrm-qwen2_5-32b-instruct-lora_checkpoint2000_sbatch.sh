#!/bin/bash
#SBATCH --job-name=deploy_cplrm_32b_lora	# 作业名称
#SBATCH --output=log/output_%j.txt     		# 输出文件名称（%j会被替换为作业ID）
#SBATCH --error=log/error_%j.txt       		# 错误文件名称
#SBATCH --partition=gpu-a800			# 指定分区
#SBATCH --nodelist=gpunode3			# 指定节点
#SBATCH --nodes=1                 		# 需要的节点数
#SBATCH --ntasks-per-node=1    			# 每个节点的任务数
#SBATCH --time=72:00:00				# 作业最大运行时间（小时：分钟：秒）
#SBATCH --mem=64G                   		# 每个节点所需的内存
#SBATCH --requeue                  		# 运行失败时重新排队

# 运行Python脚本
bash scripts/vllm/CopyPasteLRM/cplrm-qwen2_5-32b-instruct-lora_checkpoint2000.sh
