import argparse
import os
import sys
import copy

import ujson as json  # 使用更快的 ujson 库处理 JSON

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from copypastelrm.metrics.HotpotQA import update_answer, update_sp

from copypastelrm.metrics.utils import extract_answer_and_facts, extract_answer_and_facts_old

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
    
    if 'info' not in prediction:
        print('😬 尚未推理完成，跳过:', path)
        return None

    info = prediction.get("info")
    dataset_name = info.get("dataset")

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
        # count
        "count": 0,
        "without_answer_ids": [],
        "without_facts_ids": [],
        "context_length": 0,
        "context_length_word": 0,
        "predict_length": 0,
        "predict_length_word": 0,
        "answer_length": 0,
        "answer_length_word": 0
    }

    metrics_by_subset = {}

    # 遍历每个样本进行评估
    for id, item in data.items():
        can_eval_joint = True  # 标记是否可以计算联合指标
        prompt_type = info.get("prompt_type")

        subset = item.get("dataset")

        if subset not in metrics_by_subset:
            metrics_by_subset[subset] = copy.deepcopy(metrics)

        metrics_by_subset[subset]["count"] += 1

        if item['context']:
            metrics_by_subset[subset]['context_length'] += len(item['context'])
            metrics_by_subset[subset]['context_length_word'] += len(item['context'].split())

        if item['predict']:
            metrics_by_subset[subset]['predict_length'] += len(item['predict'])
            metrics_by_subset[subset]['predict_length_word'] += len(item['predict'].split())

        if "old" in prompt_type:
            predicted_answer, predicted_facts = extract_answer_and_facts_old(item["predict"])
        else:
            predicted_answer, predicted_facts = extract_answer_and_facts(item["predict"])

        if predicted_answer:
            metrics_by_subset[subset]['answer_length'] += len(predicted_answer)
            metrics_by_subset[subset]['answer_length_word'] += len(predicted_answer.split())

        gold_answers = None
        if isinstance(item["answer"], list):
            if len(item["answer"]) == 0:
                predicted_answer = None
            else:
                gold_answers = item["answer"]
        elif isinstance(item["answer"], str):
            gold_answers = [item["answer"]]

        # 评估答案部分
        if predicted_answer is None:
            metrics_by_subset[subset]["without_answer_ids"].append(id)
            can_eval_joint = False
        else:
            # 计算答案指标并更新
            em, prec, recall = update_answer(
                metrics_by_subset[subset], predicted_answer, gold_answers
            )

        # 评估支持事实部分
        if len(predicted_facts) == 0:
            metrics_by_subset[subset]["without_facts_ids"].append(id)
            can_eval_joint = False
        else:
            gold_supporting_facts = item["sfs"]
            # 计算支持事实指标并更新
            sp_em, sp_prec, sp_recall, _ = update_sp(
                metrics_by_subset[subset], predicted_facts, gold_supporting_facts
            )

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
            metrics_by_subset[subset]["joint_em"] += joint_em
            metrics_by_subset[subset]["joint_f1"] += joint_f1
            metrics_by_subset[subset]["joint_prec"] += joint_prec
            metrics_by_subset[subset]["joint_recall"] += joint_recall

    # 使用预测数量作为分母计算平均指标
    for subset, subset_metrics in metrics_by_subset.items():
        prediction_count = subset_metrics["count"]
        for k in subset_metrics.keys():
            if k != "without_answer_ids" and k != "without_facts_ids" and k != "count":
                subset_metrics[k] /= prediction_count
        
        without_answer_ids_set = set(subset_metrics["without_answer_ids"])
        without_facts_ids_set = set(subset_metrics["without_facts_ids"])
        subset_metrics["without answer only samples"] = len(
            without_answer_ids_set - without_facts_ids_set
        )
        subset_metrics["without facts only samples"] = len(
            without_facts_ids_set - without_answer_ids_set
        )
        subset_metrics["without answer and facts samples"] = len(
            without_answer_ids_set & without_facts_ids_set
        )
        subset_metrics["with answer and facts samples"] = prediction_count - len(
            without_answer_ids_set | without_facts_ids_set
        )

        subset_metrics["without_answer_ids"] = len(subset_metrics["without_answer_ids"])
        subset_metrics["without_facts_ids"] = len(subset_metrics["without_facts_ids"])

    res = {
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
        "metrics": metrics_by_subset,
        "total samples": len(data),
        "output file": path,
    }


    output_file = path.replace("results/infer", "results/eval")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"✅ Eval results saved to {output_file}")

    # 输出最终评估结果
    # log_result(metrics)
    print(json.dumps(res, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Evaluation Script")
    parser.add_argument("path")
    args = parser.parse_args()

    eval(args.path)
    # eval('results/infer/resamples_-1/seed_42/tpr_0.7-tpp_0.95/Qwen2.5-3B-Instruct/copypaste/enable_thinking_False-prompt_reasoning_with_copypaste_old-1765207378.json')

if __name__ == "__main__":
    main()
