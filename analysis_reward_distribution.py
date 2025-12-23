# 分析swanlab返回的训练数据，观察reward的数据集分布
import os
import glob

from copypastelrm.utils.json_tools import read_json
from copypastelrm.metrics.utils import extract_answer_and_facts_old
from copypastelrm.metrics.HotpotQA import hit_score, f1_score

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
    if dataset not in passAtKEqual0_distribution_by_dataset:
        passAtKEqual0_distribution_by_dataset[dataset] = {}
    if step not in passAtKEqual0_distribution_by_dataset[dataset]:
        passAtKEqual0_distribution_by_dataset[dataset][step] = []
    passAtKEqual0_distribution_by_dataset[dataset][step].append(rollout)


# 计算总数量、平均値、等于0的数量
for dataset, step_dict in passAtKEqual0_distribution_by_dataset.items():
    count_0 = 0
    count_1 = 1
    all_answer_em_rewards = []
    all_hit = []
    all_f1 = []
    for step, rollouts in step_dict.items():
        batch_answer_em_rewards = []
        for rollout in rollouts:
            completion = rollout['completion']
            answers = rollout['solution']['answers']
            answer_em_reward = rollout['AnswerEMReward']

            predict_answer, _ = extract_answer_and_facts_old(completion)

            all_answer_em_rewards.append(answer_em_reward)
            batch_answer_em_rewards.append(answer_em_reward)
            hit = hit_score(predict_answer, answers)
            f1 = 0
            for ans in answers:
                current_f1, _, _ = f1_score(predict_answer, ans)
                f1 = max(f1, current_f1)
            all_hit.append(hit)
            all_f1.append(f1)

        batch_answer_em_reward_sum = sum(batch_answer_em_rewards)
        if batch_answer_em_reward_sum == 0:
            count_0 += 1
        if batch_answer_em_reward_sum / len(rollouts) == 1:
            count_1 += 1
    

    print("="*10, f"Dataset: {dataset}", "="*10)
    print(f"Total: {len(step_dict)}")
    print(f"Count of 0: {count_0}")
    print(f"Count of 1: {count_1}")
    print(f"AVG EM rewards: {sum(all_answer_em_rewards) / len(all_answer_em_rewards):.4f}")
    print(f"AVG hit: {sum(all_hit) / len(all_hit):.4f}")
    print(f"AVG f1: {sum(all_f1) / len(all_f1):.4f}")

