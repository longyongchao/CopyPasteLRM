from typing import Any, Dict, List
import random

from copypastelrm.datasets.BaseDatasetLoader import BaseDatasetLoader
from copypastelrm.datasets.FaithEval import FaithEval
from copypastelrm.datasets.HotpotQA import HotpotQA
from copypastelrm.datasets.Qasper import Qasper
from copypastelrm.datasets.MultiRC import MultiRC
from copypastelrm.datasets.PopQA import PopQA
from copypastelrm.datasets.PubMedQA import PubMedQA
from copypastelrm.datasets.TwoWikiMultiHopQA import TwoWikiMultihopQA
from copypastelrm.datasets.MuSiQue import MuSiQue

random.seed(42)

class CopyPaste(BaseDatasetLoader):
    def __init__(self, max_samples: int = -1):

        # 初始化各种数据集评估对象，每个评估对象都接收max_samples参数
        self.faitheval = FaithEval(max_samples=max_samples)  # 信仰评估数据集
        self.hotpotqa = HotpotQA(max_samples=max_samples)    # HotpotQA问答数据集
        self.qasper = Qasper(max_samples=max_samples)        # Qasper科学问答数据集
        self.multirc = MultiRC(max_samples=max_samples)      # MultiRC阅读理解数据集
        self.popqa = PopQA(max_samples=max_samples)
        self.pubmedqa = PubMedQA(max_samples=max_samples)
        self.twoWikiqa = TwoWikiMultihopQA(max_samples=max_samples)
        self.musique = MuSiQue(max_samples=max_samples)

        self.max_samples = max_samples

        super().__init__(
            dataset_path="copypaste",
            split="test",
            name='copypaste_test',
            rename=False,
            offline=True,
            reload=True,
            format=False,
            max_samples=-1
        )
    
    def download_dataset(self):
        dataset = []
        dataset.extend(self.faitheval.dataset_list)
        dataset.extend(self.hotpotqa.dataset_list)
        dataset.extend(self.qasper.dataset_list)
        dataset.extend(self.multirc.dataset_list)
        dataset.extend(self.popqa.dataset_list)
        dataset.extend(self.pubmedqa.dataset_list)
        dataset.extend(self.twoWikiqa.dataset_list)
        dataset.extend(self.musique.dataset_list)
        print(f"FaithEval样本数: {len(self.faitheval.dataset_list)}")
        print(f"HotpotQA样本数: {len(self.hotpotqa.dataset_list)}")
        print(f"Qasper样本数: {len(self.qasper.dataset_list)}")
        print(f"MultiRC样本数: {len(self.multirc.dataset_list)}")
        print(f"PopQA样本数: {len(self.popqa.dataset_list)}")
        print(f"PubMedQA样本数: {len(self.pubmedqa.dataset_list)}")
        print(f"TwoWikiQA样本数: {len(self.twoWikiqa.dataset_list)}")
        print(f"MuSiQue样本数: {len(self.musique.dataset_list)}")
        return dataset


if __name__ == "__main__":
    loader = CopyPaste(max_samples=1)
    dataset = loader.dataset
    print(f"数据集样本数: {len(loader.dataset)}")
