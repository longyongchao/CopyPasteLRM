import json
from typing import Any, Dict, List
from copypastelrm.datasets.BaseDatasetLoader import BaseDatasetLoader
from typing import Literal


class TwoWikiMultihopQA(BaseDatasetLoader):
    def __init__(self, split: Literal['dev', 'test'] = 'dev', reload: str = False, max_samples: int = -1):
        super().__init__(
            dataset_path="data/2WikiMultihopQA" + '/' + split + '.json',
            split=split,
            offline=True,
            reload=reload,
            max_samples=max_samples
        )
    
    def download_dataset(self):

        with open(self.dataset_path, 'r') as f:
            return json.load(f)
    
    def format_id(self, sample: Dict[str, Any]) -> str:
        return sample["_id"]

    def format_context(self, sample: Dict[str, Any]) -> str:
        context_text = ""
        context = sample['context']

        for paragraph in context:
            title = paragraph[0]
            paragraph_text = " ".join(paragraph[1])
            context_text += f"\n{title}. {paragraph_text}"
        return context_text.strip()
    
    def format_supporting_facts(self, sample: Dict[str, Any]) -> List[str]:

        context = sample['context']
        supporting_facts = sample['supporting_facts']

        context_dict = {}
        for paragraph in context:
            title = paragraph[0]
            context_dict[title] = paragraph[1]
        
        sfs = []

        for sp in supporting_facts:
            title = sp[0]
            sent_idx = sp[1]
            sfs.append(context_dict[title][sent_idx])

        return sfs
        

if __name__ == "__main__":
    loader = TwoWikiMultihopQA(reload=True, split='dev')
    dataset = loader.dataset
    print(f"数据集样本数: {len(loader.dataset)}")
