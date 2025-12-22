from typing import Any, Dict, List, Literal
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
    def __init__(self, max_samples: int = -1, split: Literal['train', 'test'] = 'test', reload: bool = False):

        self.split = split


        if split == 'train':
            self.hotpotqa = HotpotQA(max_samples=max_samples, split='train', reload=reload)
            self.qasper = Qasper(max_samples=max_samples, split='train', reload=reload)
            self.multirc = MultiRC(max_samples=max_samples, split='train', reload=reload)
            self.popqa = PopQA(max_samples=max_samples, split='train', reload=reload)
            self.pubmedqa = PubMedQA(max_samples=max_samples, dataset_name='pqa_artificial', reload=reload)
            self.twoWikiqa = TwoWikiMultihopQA(max_samples=max_samples, split='dev', reload=reload)
            self.musique = MuSiQue(max_samples=max_samples, split='train', reload=reload)
        else:
            self.faitheval = FaithEval(max_samples=max_samples, reload=reload) # FaithEval只作为测试集
            self.hotpotqa = HotpotQA(max_samples=max_samples, split='validation', reload=reload)    
            self.qasper = Qasper(max_samples=max_samples, split='test', reload=reload)        
            self.multirc = MultiRC(max_samples=max_samples, split='dev', reload=reload)     
            self.popqa = PopQA(max_samples=max_samples, split='test', reload=reload)
            self.pubmedqa = PubMedQA(max_samples=max_samples, dataset_name='pqa_labeled', reload=reload)
            self.twoWikiqa = TwoWikiMultihopQA(max_samples=max_samples, split='test', reload=reload)
            self.musique = MuSiQue(max_samples=max_samples, split='validation', reload=reload)
            

        self.max_samples = max_samples

        super().__init__(
            dataset_path="copypaste",
            split=split,
            offline=True,
            reload=reload,
            format=False,
            max_samples=-1
        )
    
    def download_dataset(self):
        dataset = []
        if self.split != 'train':
            dataset.extend(self.faitheval.dataset_list)

        dataset.extend(self.hotpotqa.dataset_list)
        dataset.extend(self.qasper.dataset_list)
        dataset.extend(self.multirc.dataset_list)
        dataset.extend(self.popqa.dataset_list)
        dataset.extend(self.pubmedqa.dataset_list)
        dataset.extend(self.twoWikiqa.dataset_list)
        dataset.extend(self.musique.dataset_list)

        if self.split != 'train':
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
    loader = CopyPaste(max_samples=-1, reload=True, split='test')
    dataset = loader.dataset_list
    print(f"数据集样本数: {len(loader.dataset)}")
    print(dataset[0])
