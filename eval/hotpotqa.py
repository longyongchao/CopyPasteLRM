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

import pickle  # 导入但当前未使用，可能用于后续扩展
import re
import string
import sys
from collections import Counter

import ujson as json  # 使用更快的 ujson 库处理 JSON
from datasets import load_dataset


def normalize_answer(s):
    """
    对答案进行标准化处理，消除格式差异以便公平比较

    标准化步骤：
    1. 转换为小写
    2. 移除标点符号
    3. 移除冠词（a, an, the）
    4. 标准化空白字符

    Args:
        s (str): 原始答案字符串

    Returns:
        str: 标准化后的答案字符串
    """

    def remove_articles(text):
        """移除冠词 a, an, the"""
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        """标准化空白字符：去除多余空格，合并为单个空格"""
        return " ".join(text.split())

    def remove_punc(text):
        """移除所有标点符号"""
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        """转换为小写"""
        return text.lower()

    # 按顺序执行标准化步骤
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    """
    计算预测答案与标准答案之间的 F1 分数

    F1 分数是精确率和召回率的调和平均数，用于衡量答案的词级别匹配度

    Args:
        prediction (str): 模型预测的答案
        ground_truth (str): 标准答案

    Returns:
        tuple: (f1, precision, recall) - F1分数、精确率、召回率
    """
    # 标准化预测答案和标准答案
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)  # 零分指标常量

    # 特殊处理：对于 yes/no/noanswer 类型的答案，必须完全匹配
    if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    # 分词处理
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()

    # 计算共同词汇数量（使用 Counter 的交集操作）
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    # 如果没有共同词汇，返回零分
    if num_same == 0:
        return ZERO_METRIC

    # 计算精确率：正确词汇数 / 预测总词汇数
    precision = 1.0 * num_same / len(prediction_tokens)
    # 计算召回率：正确词汇数 / 标准答案总词汇数
    recall = 1.0 * num_same / len(ground_truth_tokens)
    # 计算 F1 分数：2 * precision * recall / (precision + recall)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    """
    计算精确匹配分数

    精确匹配要求预测答案与标准答案在标准化后完全相同

    Args:
        prediction (str): 模型预测的答案
        ground_truth (str): 标准答案

    Returns:
        bool: 如果完全匹配返回 True，否则返回 False
    """
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def update_answer(metrics, prediction, gold):
    """
    更新答案相关的评估指标

    计算单个样本的答案评估指标并累加到总指标中

    Args:
        metrics (dict): 存储累积指标的字典
        prediction (str): 模型预测的答案
        gold (str): 标准答案

    Returns:
        tuple: (em, prec, recall) - 精确匹配、精确率、召回率
    """
    # 计算精确匹配分数
    em = exact_match_score(prediction, gold)
    # 计算 F1 分数、精确率、召回率
    f1, prec, recall = f1_score(prediction, gold)

    # 累加到总指标中
    metrics["em"] += float(em)
    metrics["f1"] += f1
    metrics["prec"] += prec
    metrics["recall"] += recall

    return em, prec, recall


def update_sp(metrics, prediction, gold):
    """
    更新支持事实相关的评估指标

    支持事实是模型用来回答问题的依据，通常表示为 [title, sentence_id] 的元组

    Args:
        metrics (dict): 存储累积指标的字典
        prediction (list): 模型预测的支持事实列表
        gold (list): 标准支持事实列表

    Returns:
        tuple: (em, prec, recall) - 支持事实的精确匹配、精确率、召回率
    """
    # 将列表转换为集合以便比较，使用 tuple 确保可哈希
    cur_sp_pred = set(map(tuple, prediction))
    gold_sp_pred = set(map(tuple, gold))

    # 初始化混淆矩阵元素
    tp, fp, fn = 0, 0, 0

    # 计算真阳性（TP）和假阳性（FP）
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1  # 预测正确
        else:
            fp += 1  # 预测错误

    # 计算假阴性（FN）
    for e in gold_sp_pred:
        if e not in cur_sp_pred:
            fn += 1  # 遗漏的标准答案

    # 计算精确率：TP / (TP + FP)
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    # 计算召回率：TP / (TP + FN)
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    # 计算 F1 分数
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    # 计算精确匹配：只有当没有假阳性和假阴性时才为 1
    em = 1.0 if fp + fn == 0 else 0.0

    # 累加到总指标中
    metrics["sp_em"] += em
    metrics["sp_f1"] += f1
    metrics["sp_prec"] += prec
    metrics["sp_recall"] += recall

    return em, prec, recall


def load_hotpotqa_dataset():
    """
    从 HuggingFace 加载 HotpotQA 数据集的 distractor 子集 validation split

    Returns:
        List[Dict]: 包含数据样本的列表
    """
    print("正在加载 HotpotQA 数据集...")
    try:
        dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
        print(f"成功加载数据集，共 {len(dataset)} 个样本")
        return list(dataset)
    except Exception as e:
        print(f"加载数据集失败: {e}")
        sys.exit(1)


def eval(prediction_file):
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
    with open(prediction_file) as f:
        prediction = json.load(f)

    # 从 HuggingFace 加载数据集作为标准答案
    gold = load_hotpotqa_dataset()

    # 初始化所有评估指标
    metrics = {
        # 答案相关指标
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

    # 遍历每个样本进行评估
    for dp in gold:
        cur_id = dp["id"]  # 获取样本 ID
        can_eval_joint = True  # 标记是否可以计算联合指标

        # 评估答案部分
        if cur_id not in prediction["answer"]:
            print("missing answer {}".format(cur_id))  # 缺少答案预测
            can_eval_joint = False
        else:
            # 计算答案指标并更新
            em, prec, recall = update_answer(metrics, prediction["answer"][cur_id], dp["answer"])

        # 评估支持事实部分
        if cur_id not in prediction["sp"]:
            print("missing sp fact {}".format(cur_id))  # 缺少支持事实预测
            can_eval_joint = False
        else:
            # 转换支持事实格式：从字典格式转换为列表格式
            # 数据集中的格式：{'title': ['title1', 'title2'], 'sent_id': [0, 1]}
            # 需要转换为：[['title1', 0], ['title2', 1]]
            gold_supporting_facts = []
            if "supporting_facts" in dp and dp["supporting_facts"]:
                titles = dp["supporting_facts"]["title"]
                sent_ids = dp["supporting_facts"]["sent_id"]
                gold_supporting_facts = [[title, sent_id] for title, sent_id in zip(titles, sent_ids)]

            # 计算支持事实指标并更新
            sp_em, sp_prec, sp_recall = update_sp(metrics, prediction["sp"][cur_id], gold_supporting_facts)

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

    # # 计算平均指标（除以预测总数）
    # # 统计 prediction 中实际有预测的样本数量
    # answer_ids = set(prediction["answer"].keys()) if "answer" in prediction else set()
    # sp_ids = set(prediction["sp"].keys()) if "sp" in prediction else set()

    # # 取并集，只要答案或支持事实有一个有预测，就计入总数
    # prediction_count = len(answer_ids.union(sp_ids))

    # # 如果没有预测数据，避免除零错误
    # if prediction_count == 0:
    #     print("警告：没有找到任何预测数据")
    #     prediction_count = 1

    # print(f"预测样本数量: {prediction_count} (答案: {len(answer_ids)}, 支持事实: {len(sp_ids)})")
    prediction_count = len(gold)  # 使用 gold 数据集的样本数量作为分母

    # 使用预测数量作为分母计算平均指标
    for k in metrics.keys():
        metrics[k] /= prediction_count

    # 输出最终评估结果
    print(metrics)


if __name__ == "__main__":
    """
    主程序入口

    使用方法：
    python eval/hotpotqa.py prediction.json

    ### prediction.json 结构示例：
    ```json
    {
        "answer": {
            "问题ID": "答案字符串"
        },
        "sp": {
            "问题ID": [
                ["标题", 句子ID],
                ["标题", 句子ID]
            ]
        }
    }
    ```

    注意：标准答案数据会自动从 HuggingFace 的 hotpotqa/hotpot_qa 数据集的 distractor 子集 validation split 中加载

    参数：
    sys.argv[1]: 预测结果文件路径
    """
    if len(sys.argv) != 2:
        print("使用方法: python eval/hotpotqa.py prediction.json")
        sys.exit(1)

    eval(sys.argv[1])
