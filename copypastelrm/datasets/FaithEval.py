import json
from typing import Any, Dict, List
from copypastelrm.datasets.BaseDatasetLoader import BaseDatasetLoader
import re


class FaithEval(BaseDatasetLoader):
    def __init__(self, reload: bool = False):
        super().__init__(
            dataset_path="Salesforce/FaithEval-counterfactual-v1.0",
            split="test",
            cache_path="cache/faitheval_counterfactual_test.jsonl",
            offline=True,
            reload=reload,
        )
    

    def format_supporting_facts(self, sample: Dict[str, Any]) -> List[str]:

        return []
        

if __name__ == "__main__":
    loader = FaithEval(reload=True)
    dataset = loader.dataset
    print(f"数据集样本数: {len(loader.dataset)}")
