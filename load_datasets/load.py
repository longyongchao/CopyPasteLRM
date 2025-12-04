import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from load_datasets.faitheval import load_faitheval
from load_datasets.hotpotqa import load_hotpotqa
from load_datasets.musique import load_musique
from load_datasets.pubmedqa import load_pubmedqa


def list2dict(dataset: list) -> dict:
    """
    将数据集列表转换为字典，方便按 ID 访问

    Args:
        dataset (list): 数据集列表

    Returns:
        dict: 以样本 ID 为键，样本数据为值的字典
    """
    dataset_dict = {}
    for sample in dataset:
        sample_id = sample["id"]
        dataset_dict[sample_id] = sample
    return dataset_dict


def data_loader(dataset_name: str, mode: str = "list"):
    """
    根据数据集名称加载对应的数据集
    """
    dataset = None
    if dataset_name == "hotpotqa":
        dataset = load_hotpotqa()
    elif dataset_name == "faitheval":
        dataset = load_faitheval()
    elif dataset_name == "musique":
        dataset = load_musique()
    elif dataset_name == "pubmedqa":
        dataset = load_pubmedqa()
    else:
        raise ValueError(f"未知的数据集名称: {dataset_name}")

    if mode == "dict":
        dataset = list2dict(dataset)

    return dataset


if __name__ == "__main__":
    data = data_loader("hotpotqa", mode="dict")
    print(f"数据集样本数: {len(data)}")
    print("第一个样本示例:")
    print(data.keys())
