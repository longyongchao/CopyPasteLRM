# 分析swanlab返回的训练数据，观察reward的数据集分布
import os
import glob

from copypastelrm.utils.json_tools import read_json

log_dir = "train/3-CopyPasteLRM-Qwen3-8B-Pass@K=0/swanlog/run-20251223_025749-aji0y6dpbhsahxthv3a3z"

# 遍历log_dir下的所有json文件

data = []

for json_file in glob.glob(os.path.join(log_dir + '/media', "**/*.json"), recursive=True):
    rowData = read_json(json_file)['rowData']
    data.extend(rowData)

passAtKEqual0_distribution_by_dataset = {}

for rollout in data:
    step = str(rollout['step'])
    dataset = rollout['solution']['dataset']
    answer_em_reward = rollout['AnswerEMReward']
    if dataset not in passAtKEqual0_distribution_by_dataset:
        passAtKEqual0_distribution_by_dataset[dataset] = {}
    if step not in passAtKEqual0_distribution_by_dataset[dataset]:
        passAtKEqual0_distribution_by_dataset[dataset][step] = []
    passAtKEqual0_distribution_by_dataset[dataset][step].append(answer_em_reward)


# 计算总数量、平均値、等于0的数量
for dataset, prompt_dict in passAtKEqual0_distribution_by_dataset.items():
    count_0 = 0
    count_1 = 1
    all_rewards = []
    for prompt, rewards in prompt_dict.items():
        all_rewards.extend(rewards)
        reward_sum = sum(rewards)
        if reward_sum == 0:
            count_0 += 1
        if reward_sum / len(rewards) == 1:
            count_1 += 1

    print("="*10, f"Dataset: {dataset}", "="*10)
    print(f"Total: {len(prompt_dict)}")
    print(f"Count of 0: {count_0}")
    print(f"Count of 1: {count_1}")
    print(f"AVG rewards: {sum(all_rewards) / len(all_rewards):.4f}")

