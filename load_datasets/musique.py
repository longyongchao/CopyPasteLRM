import sys
from typing import Any, Dict, List

from datasets import load_dataset


def format_context(context: list) -> str:
    context_text = ""
    for paragraph in context:
        title = paragraph["title"]
        content = paragraph["paragraph_text"]
        context_text += f"{title}. {content}\n"
    return context_text


def load_musique(origin: bool = False) -> List[Dict[str, Any]]:
    """
    从 HuggingFace 加载数据集

    Returns:
        List[Dict]: 包含数据样本的列表
    """
    path = "dgslibisey/MuSiQue"
    split = "validation"
    print(f"正在加载 {path}/{split} 数据集...")
    dataset = load_dataset(path=path, split=split)
    dataset = list(dataset)

    if origin:
        return dataset

    formated_dataset = []

    for sample in dataset:
        formated_sample = {
            "id": sample["id"],
            "query": sample["question"],
            "context": format_context(sample["paragraphs"]),
            "answer": sample["answer"],
        }
        formated_dataset.append(formated_sample)

    return formated_dataset


if __name__ == "__main__":
    data = load_musique(origin=False)
    print(f"数据集样本数: {len(data)}")
    print("第一个样本示例:")
    print(data[0])
