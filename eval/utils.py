import re
from typing import Any, Dict, List, Tuple


def extract_answer_and_facts(response: str) -> Tuple[str, List[List[str]]]:
    """
    从模型响应中提取答案和支持事实

    Args:
        response: 模型生成的响应
        gold_sfs: 原始上下文信息

    Returns:
        Tuple[str, List[List[str]]]: (答案, 支持事实列表)
    """
    if not isinstance(response, str):
        return None, []
    # 提取答案
    answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if answer_match:
        predict_answer = answer_match.group(1).strip()
    else:
        predict_answer = None

    # 提取所有 <copy> 标签中的内容
    copy_matches = re.findall(r"<copy>(.*?)</copy>", response, re.DOTALL)
    predict_sfs = [match.strip() for match in copy_matches]

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
