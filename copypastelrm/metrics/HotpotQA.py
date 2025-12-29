import re
import string
from collections import Counter
from typing import List, Optional


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


def f1_score(prediction, gold_answer):
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
    normalized_ground_truth = normalize_answer(gold_answer)

    ZERO_METRIC = (0, 0, 0)  # 零分指标常量

    # 特殊处理：对于 yes/no/noanswer 类型的答案，必须完全匹配
    if normalized_prediction in ["yes", "no", "noanswer", "maybe"] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ["yes", "no", "noanswer", "maybe"] and normalized_prediction != normalized_ground_truth:
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


def exact_match_score(prediction, gold_answers):
    """
    计算精确匹配分数

    精确匹配要求预测答案与标准答案在标准化后完全相同

    Args:
        prediction (str): 模型预测的答案
        ground_truth (str): 标准答案

    Returns:
        bool: 如果完全匹配返回 True，否则返回 False
    """
    if not prediction:
        return False
    for gold_answer in gold_answers:
        if normalize_answer(prediction) == normalize_answer(gold_answer):
            return True
    return False

def hit_score(prediction, gold_answers):
    """
    检查预测答案是否在标准答案中

    适用于多选题或包含多个可能答案的情况

    Args:
        prediction (str): 模型预测的答案
        ground_truth (str): 标准答案

    Returns:
        bool: 如果预测答案在标准答案中返回 True，否则返回 False
    """
    if not prediction:
        return False

    if isinstance(gold_answers, str):
        gold_answers = [gold_answers]

    for gold_answer in gold_answers:
        if normalize_answer(gold_answer) in normalize_answer(prediction):
            return True
    return False

def hit_answer(prediction, gold_answers):
    """
    检查预测答案是否在标准答案中

    适用于多选题或包含多个可能答案的情况

    Args:
        prediction (str): 模型预测的答案
        ground_truth (str): 标准答案

    Returns:
        bool: 如果预测答案在标准答案中返回 True，否则返回 False
    """
    if isinstance(gold_answers, str):
        gold_answers = [gold_answers]

    for gold_answer in gold_answers:
        if normalize_answer(gold_answer) in normalize_answer(prediction):
            return gold_answer
    return None


def update_answer(metrics: dict, prediction: str, gold_answers: list[str]) -> tuple[float, float, float]:
    """
    更新答案相关的评估指标。

    计算单个样本的答案评估指标（精确匹配、F1、精确率、召回率）并累加到总指标中。

    Args:
        metrics (dict): 存储累积指标的字典。
        prediction (str): 模型预测的答案。
        gold_answers (list[str]): 标准答案列表。

    Returns:
        tuple[float, float, float]: (em, precision, recall) - 精确匹配、精确率、召回率。
    """
    # 1. 计算精确匹配 (Exact Match, EM) 和命中分数 (Hit Score)
    # 假设 exact_match_score 和 hit_score 内部已处理 gold_answers
    em = exact_match_score(prediction, gold_answers)
    hit = hit_score(prediction, gold_answers)

    # 2. 计算 F1 分数、精确率、召回率
    # 存储所有 (f1, precision, recall) 候选项
    results_candidates = []
    for gold in gold_answers:
        # 假设 f1_score 函数返回 (f1, precision, recall)
        f1_val, prec_val, recall_val = f1_score(prediction, gold)
        results_candidates.append((f1_val, prec_val, recall_val))

    # 3. 找到 F1 分数最高的组合作为最终结果
    # 使用 max 函数和 key 参数，可以直接找到 F1 值最高的元组
    # 元组的第一个元素是 F1 (索引 0)
    if results_candidates:
        best_f1, best_precision, best_recall = max(results_candidates, key=lambda x: x[0])
    else:
        # 如果 gold_answers 为空，设置指标为 0.0
        best_f1, best_precision, best_recall = 0.0, 0.0, 0.0


    # 4. 累加到总指标中
    # 转换为 float 是好的做法，确保累加操作的类型一致性
    metrics["hit"] += float(hit)
    metrics["em"] += float(em)
    metrics["f1"] += best_f1
    metrics["prec"] += best_precision
    metrics["recall"] += best_recall

    # 5. 返回最终结果
    return float(em), best_precision, best_recall


def update_sp(predict_sfs: List[str], gold_sfs: List[str], metrics: Optional[dict] = None):
    """
    更新支持事实相关的评估指标

    支持事实是模型用来回答问题的依据

    Args:
        predict_sfs (List[str]): 模型预测的支持事实列表
        gold_sfs (List[str]): 标准支持事实列表
        metrics (dict): 存储累积指标的字典

    Returns:
        tuple: (em, prec, recall) - 支持事实的精确匹配、精确率、召回率
    """
    if predict_sfs is None or len(predict_sfs) == 0 or predict_sfs[0].strip() == "":
        return 0.0, 0.0, 0.0, 0.0
    if gold_sfs is None or len(gold_sfs) == 0 or gold_sfs[0].strip() == "":
        return 0.0, 0.0, 0.0, 0.0

    # 将列表转换为集合以便比较，使用 tuple 确保可哈希
    predict_sfs = [sf.lower().strip() for sf in predict_sfs]
    gold_sfs = [sf.lower().strip() for sf in gold_sfs]

    predict_sfs_cat = " ".join(predict_sfs)
    gold_sfs_cat = " ".join(gold_sfs)

    # 初始化混淆矩阵元素
    tp, fp, fn = 0, 0, 0

    # 计算真阳性（TP）和假阳性（FP）
    for predict_sf in predict_sfs:
        if predict_sf in gold_sfs_cat:
            tp += 1  # 预测正确
        else:
            fp += 1  # 预测错误

    # 计算假阴性（FN）
    for gold_sf in gold_sfs:
        if gold_sf not in predict_sfs_cat:
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
    if metrics is not None:
        metrics["sp_em"] += em
        metrics["sp_f1"] += f1
        metrics["sp_prec"] += prec
        metrics["sp_recall"] += recall

    return em, f1, prec, recall


def compute_answer_em_hit_f1(predict_answer: str, gold_answers: List[str]) -> tuple[float, float, float]:
    if not predict_answer:
        return 0.0, 0.0, 0.0
    hit = hit_score(predict_answer, gold_answers)
    em = exact_match_score(predict_answer, gold_answers)

    f1 = 0
    for gold_ans in gold_answers:
        current_f1, _, _ = f1_score(predict_answer, gold_ans)
        f1 = max(f1, current_f1)

    return em, hit, f1