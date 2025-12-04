import re
import string
from collections import Counter


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


def judge_hit(prediction, ground_truth):
    """
    检查预测答案是否在标准答案中

    适用于多选题或包含多个可能答案的情况

    Args:
        prediction (str): 模型预测的答案
        ground_truth (str): 标准答案

    Returns:
        bool: 如果预测答案在标准答案中返回 True，否则返回 False
    """
    return normalize_answer(ground_truth) in normalize_answer(prediction)


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
    hit = judge_hit(prediction, gold)
    # 计算 F1 分数、精确率、召回率
    f1, prec, recall = f1_score(prediction, gold)

    # 累加到总指标中
    metrics["hit"] += float(hit)
    metrics["em"] += float(em)
    metrics["f1"] += f1
    metrics["prec"] += prec
    metrics["recall"] += recall

    return em, prec, recall


def update_sp(metrics, predict_sfs, gold_sfs):
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
    metrics["sp_em"] += em
    metrics["sp_f1"] += f1
    metrics["sp_prec"] += prec
    metrics["sp_recall"] += recall

    return em, prec, recall
