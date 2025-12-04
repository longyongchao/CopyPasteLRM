import sys
from typing import Any, Dict, List

from datasets import load_dataset


def load_faitheval(origin: bool = False) -> List[Dict[str, Any]]:
    """
    从 HuggingFace 加载 FaithEval-counterfactual 数据集的 test split

    Returns:
        List[Dict]: 包含数据样本的列表
    """
    path = "Salesforce/FaithEval-counterfactual-v1.0"
    split = "test"
    print(f"正在加载 {path} 数据集的 {split}...")
    dataset = load_dataset(path=path, split=split)
    dataset = list(dataset)

    if origin:
        return dataset

    formated_dataset = []

    for sample in dataset:
        formated_sample = {
            "id": sample["id"],
            "query": sample["question"],
            "context": sample["context"],
            "answer": sample["answer"],
        }
        formated_dataset.append(formated_sample)

    return formated_dataset


if __name__ == "__main__":
    data = load_faitheval()
    print(f"数据集样本数: {len(data)}")
    print("第一个样本示例:")
    print(data[0])
