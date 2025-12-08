import re
from typing import Any, Dict, List, Tuple

def remove_evidence_tags(text):
    """
    删除字符串中被<EVIDENCE></EVIDENCE>包裹的内容（包括标签本身）
    
    参数:
        text (str): 包含<EVIDENCE>标签的原始字符串
    
    返回:
        str: 移除标签及包裹内容后的字符串
    """
    # 正则表达式匹配<EVIDENCE>和</EVIDENCE>之间的所有内容（包括标签）
    pattern = r'<EVIDENCE>.*?</EVIDENCE>'
    # 使用re.sub替换匹配到的内容为空字符串，re.DOTALL让.匹配换行符
    cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
    # 去除替换后可能产生的多余空格（可选，根据需求调整）
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def extract_answer_and_facts(predict: str) -> Tuple[str, List[List[str]]]:
    """
    从模型响应中提取答案和支持事实

    Args:
        response: 模型生成的响应
    Returns:
        Tuple[str, List[List[str]]]: (答案, 支持事实列表)
    """
    if not isinstance(predict, str):
        return None, []
    # 提取答案
    pattern = r"Answer:\s*([^\n]+)(?:\n|$)"
    answer_match = re.findall(pattern, predict)
    if answer_match and len(answer_match) > 0:
        predict_answer = answer_match[-1]
        predict_answer = predict_answer.replace("<answer>", "").replace("</answer>", "").replace("<Answer>", "").replace("</Answer>", "")
        predict_answer = remove_evidence_tags(predict_answer)
        predict_answer = predict_answer.replace("<", "").replace(">", "")

    else:
        predict_answer = None

    # 提取所有 <copy> 标签中的内容
    evidence_matches = re.findall(r"<EVIDENCE>(.*?)</EVIDENCE>", predict, re.DOTALL)
    predict_sfs = [match.strip() for match in evidence_matches]

    return predict_answer, predict_sfs

def extract_answer_and_facts_old(predict: str) -> Tuple[str, List[List[str]]]:
    """
    从模型响应中提取答案和支持事实

    Args:
        response: 模型生成的响应
    Returns:
        Tuple[str, List[List[str]]]: (答案, 支持事实列表)
    """
    if not isinstance(predict, str):
        return None, []
    # 提取答案
    pattern = r'<answer>(.*?)</answer>'
    # 执行匹配（非贪婪模式，确保只匹配最近的闭合标签）
    match = re.search(pattern, predict, re.DOTALL)
    # 返回匹配结果，无匹配则返回空字符串
    predict_answer = match.group(1).strip() if match else ''

    # 提取所有 <copy> 标签中的内容
    evidence_matches = re.findall(r"<copy>(.*?)</copy>", predict, re.DOTALL)
    predict_sfs = [match.strip() for match in evidence_matches]

    return predict_answer, predict_sfs

def find_text_in_context(text: str, context: Dict[str, Any]) -> Tuple[str, int]:
    """
    在上下文中查找文本，返回标题和句子ID

    Args:
        text: 要查找的文本
        context: 上下文信息

    Returns:
        Tuple[str, int]: (标题, 句子ID)，如果找不到返回 ("", None)
    """
    for title, sentences in zip(context["title"], context["sentences"]):
        for sent_id, sentence in enumerate(sentences):
            if text.strip() in sentence.strip():
                return title, sent_id

    return "", None


if __name__ == "__main__":
    import json
    import random

    path = "results/infer/resamples_-1/seed_42/tpr_0.7-tpp_0.95/Qwen2.5-3B-Instruct/copypaste/enable_thinking_False-prompt_reasoning_with_copypaste_old-1765207378.json"
    with open(path) as f:
        data = json.load(f)
    info = data["info"]
    data = data["data"]

    ids = data.keys()
    # 随机选一个id
    id = random.choice(list(ids))

    predict = data[id]["predict"]
    print(predict)
    answer, sfs = extract_answer_and_facts_old(predict)
    print("----" * 10)
    print(answer)
    print(sfs)
