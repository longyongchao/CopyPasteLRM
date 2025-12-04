import sys
from typing import Any, Dict, List

from datasets import load_dataset


def format_context(context: Dict[str, Any]) -> str:
    """
    格式化上下文信息

    Args:
        context: 包含 title 和 sentences 的字典

    Returns:
        str: 格式化后的上下文文本
    """
    context_text = ""
    for title, sentences in zip(context["title"], context["sentences"]):
        context_text += f"{title}. " + "".join(sentences) + "\n"
    return context_text


def load_hotpotqa(origin: bool = False) -> List[Dict[str, Any]]:
    """
    从 HuggingFace 加载 HotpotQA 数据集的 distractor 子集 validation split

    Returns:
        List[Dict]: 包含数据样本的列表
    """
    print("正在加载 HotpotQA 数据集...")
    dataset = load_dataset(path="hotpotqa/hotpot_qa", name="distractor", split="validation")
    dataset = list(dataset)

    if origin:
        return dataset

    formated_dataset = []

    for sample in dataset:
        formated_sample = {
            "id": sample["id"],
            "query": sample["question"],
            "context": format_context(sample["context"]),
            "answer": sample["answer"],
        }
        formated_dataset.append(formated_sample)

    return formated_dataset


if __name__ == "__main__":
    data = load_hotpotqa()
    print(f"数据集样本数: {len(data)}")
    print("第一个样本示例:")
    print(data[0])
