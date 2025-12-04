"""
HotpotQA 评估脚本

该脚本用于评估模型在 HotpotQA 数据集上的性能。
HotpotQA 是一个多跳问答数据集，需要模型回答问题并提供支持事实。

评估指标包括：
1. 答案准确性：精确匹配(EM)、F1分数、精确率、召回率
2. 支持事实准确性：SP_EM、SP_F1、SP_Precision、SP_Recall
3. 联合评估：同时考虑答案和支持事实的综合表现

使用方法：
python eval/hotpotqa.py prediction.json

注意：标准答案数据会自动从 HuggingFace 的 hotpotqa/hotpot_qa 数据集的 distractor 子集 validation split 中加载
"""

import argparse
import os
import sys

import ujson as json  # 使用更快的 ujson 库处理 JSON

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from eval.hotpotqa_metric import update_answer, update_sp
from eval.utils import extract_answer_and_facts
from load_datasets.load import data_loader
from utils.git import get_git_commit_id


def log_result(result: dict):
    """
    格式化输出结果为Obsidian Yaml格式，方便记录
    """
    print("metrics:")
    for key, value in result.items():
        print(f"\t- {key}={value}")


def eval(path: str):
    """
    主评估函数，执行完整的 HotpotQA 评估流程

    评估流程：
    1. 加载预测文件
    2. 从 HuggingFace 加载标准答案数据集
    3. 遍历每个样本，计算答案和支持事实指标
    4. 计算联合指标（答案和支持事实的乘积）
    5. 输出平均指标

    Args:
        prediction_file (str): 预测结果文件路径
    """
    # 加载预测结果文件
    with open(path) as f:
        prediction = json.load(f)

    info = prediction.get("info")
    dataset_name = info.get("dataset")

    dataset = data_loader(dataset_name, mode="dict")

    data = prediction.get("data")

    # 初始化所有评估指标
    metrics = {
        # 答案相关指标
        "hit": 0,
        "em": 0,  # 精确匹配
        "f1": 0,  # F1 分数
        "prec": 0,  # 精确率
        "recall": 0,  # 召回率
        # 支持事实相关指标
        "sp_em": 0,  # 支持事实精确匹配
        "sp_f1": 0,  # 支持事实 F1 分数
        "sp_prec": 0,  # 支持事实精确率
        "sp_recall": 0,  # 支持事实召回率
        # 联合指标
        "joint_em": 0,  # 联合精确匹配
        "joint_f1": 0,  # 联合 F1 分数
        "joint_prec": 0,  # 联合精确率
        "joint_recall": 0,  # 联合召回率
    }

    no_answer_count = 0
    no_facts_count = 0

    # 遍历每个样本进行评估
    for id, item in data.items():
        can_eval_joint = True  # 标记是否可以计算联合指标

        predicted_answer, predicted_facts = extract_answer_and_facts(item["predict"])

        # 评估答案部分
        if predicted_answer is None:
            no_answer_count += 1
            can_eval_joint = False
        else:
            # 计算答案指标并更新
            em, prec, recall = update_answer(metrics, predicted_answer, item["answer"])

        # 评估支持事实部分
        if len(predicted_facts) == 0 or id not in dataset:
            no_facts_count += 1
            can_eval_joint = False
        else:
            gold_supporting_facts = dataset[id].get("sfs", [])
            # 计算支持事实指标并更新
            sp_em, sp_prec, sp_recall = update_sp(metrics, predicted_facts, gold_supporting_facts)

        # 计算联合指标（只有当答案和支持事实都存在时才计算）
        if can_eval_joint:
            # 联合精确率 = 答案精确率 * 支持事实精确率
            joint_prec = prec * sp_prec
            # 联合召回率 = 答案召回率 * 支持事实召回率
            joint_recall = recall * sp_recall
            # 计算联合 F1 分数
            if joint_prec + joint_recall > 0:
                joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
            else:
                joint_f1 = 0.0
            # 联合精确匹配 = 答案精确匹配 * 支持事实精确匹配
            joint_em = em * sp_em

            # 累加联合指标
            metrics["joint_em"] += joint_em
            metrics["joint_f1"] += joint_f1
            metrics["joint_prec"] += joint_prec
            metrics["joint_recall"] += joint_recall

    prediction_count = len(data)  # 使用预测数据的样本数量作为分母

    # 使用预测数量作为分母计算平均指标
    for k in metrics.keys():
        metrics[k] /= prediction_count

    print("no answer count:", no_answer_count)
    print("no sfs count", no_facts_count)
    obsidian_card = {
        "project": "CopyPasteLRM",
        "type": "Experiment",
        "method": info["model_name"],
        "dataset": dataset_name,
        "eval git commit id": get_git_commit_id(),
        "infer git commit id": info.get("infer_git_commit_id"),
        "infer start time": info.get("start_time"),
        "infer end time": info.get("end_time"),
        "server url": info.get("server_url"),
        "prompt type": info.get("prompt_type"),
        "prompt snapshot": info.get("prompt_snapshot"),
        "temperature": info.get("temperature"),
        "top p": info.get("top_p"),
        "metrics": [f"{metric}={value}" for metric, value in metrics.items()],
        "samples count": [f"total={prediction_count}", f"no_answer={no_answer_count}", f"no_facts={no_facts_count}"],
        "output file": path,
    }

    prediction["eval_info"] = obsidian_card

    with open(path.replace(".json", "_done.json"), "w", encoding="utf-8") as f:
        json.dump(obsidian_card, f, ensure_ascii=False, indent=2)

    # 输出最终评估结果
    log_result(metrics)


def main():
    parser = argparse.ArgumentParser(description="Evaluation Script")
    parser.add_argument("path", default="/home/lyc/projects/CopyPasteLRM/results/hotpotqa/localhost:8124_v1-qwen2.5-3b-instruct-temp=0.7-topp=0.95-prompt=reasoning_with_copy-paste-maxsamples=None-1133230.json", type=str, help="Path to the prediction JSON file")
    args = parser.parse_args()

    eval(args.path)


if __name__ == "__main__":
    main()
