import sys
from typing import Any, Dict, List

from datasets import load_dataset


def format_context(context: Dict[str, Any]) -> str:
    context_text = "\n".join(context["contexts"])
    return context_text


def load_pubmedqa(origin: bool = False) -> List[Dict[str, Any]]:
    """
    从 HuggingFace 加载数据集

    Returns:
        List[Dict]: 包含数据样本的列表
    """
    path = "qiaojin/PubMedQA"
    subset = "pqa_labeled"
    split = "train"
    print(f"正在加载 {path}/{subset}/{split} 数据集...")
    dataset = load_dataset(path=path, name=subset, split=split)
    dataset = list(dataset)

    if origin:
        return dataset

    formated_dataset = []

    for sample in dataset:
        formated_sample = {
            "id": sample["pubid"],
            "query": sample["question"],
            "context": format_context(sample["context"]),
            "answer": sample["final_decision"],
        }
        formated_dataset.append(formated_sample)

    return formated_dataset


if __name__ == "__main__":
    data = load_pubmedqa(origin=False)
    print(f"数据集样本数: {len(data)}")
    print("第一个样本示例:")
    print(data[0])
