import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from load_datasets.faitheval import load_faitheval
from load_datasets.hotpotqa import load_hotpotqa
from load_datasets.musique import load_musique
from load_datasets.pubmedqa import load_pubmedqa


def data_loader(dataset_name: str):
    """
    根据数据集名称加载对应的数据集
    """
    if dataset_name == "hotpotqa":
        return load_hotpotqa()
    elif dataset_name == "faitheval":
        return load_faitheval()
    elif dataset_name == "musique":
        return load_musique()
    elif dataset_name == "pubmedqa":
        return load_pubmedqa()
    else:
        raise ValueError(f"未知的数据集名称: {dataset_name}")
