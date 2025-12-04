import re
from typing import Any, Dict, List, Tuple


def extract_answer_and_facts(response: str, context: Dict[str, Any]) -> Tuple[str, List[List[str]]]:
    """
    从模型响应中提取答案和支持事实

    Args:
        response: 模型生成的响应
        context: 原始上下文信息

    Returns:
        Tuple[str, List[List[str]]]: (答案, 支持事实列表)
    """
    # 提取答案
    answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()
    else:
        print("未找到答案标签，默认答案为空字符串\n", response)
        answer = ""

    # 提取所有 <copy> 标签中的内容
    copy_matches = re.findall(r"<copy>(.*?)</copy>", response, re.DOTALL)
    copied_texts = [match.strip() for match in copy_matches]

    # 将复制的文本映射回支持事实
    supporting_facts = []

    for copied_text in copied_texts:
        # 在上下文中查找复制的文本
        title, sent_id = find_text_in_context(copied_text, context)
        if title and sent_id is not None:
            if [title, sent_id] not in supporting_facts:
                supporting_facts.append([title, sent_id])

    return answer, supporting_facts


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
