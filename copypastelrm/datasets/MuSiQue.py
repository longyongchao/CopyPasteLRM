from typing import Any, Dict, List
from copypastelrm.datasets.BaseDatasetLoader import BaseDatasetLoader
from typing import Literal


class MuSiQue(BaseDatasetLoader):
    """https://huggingface.co/datasets/dgslibisey/MuSiQue"""
        
    def __init__(self, split: Literal["train", "validation"]="validation", reload: bool = False, max_samples: int = -1):
        super().__init__(
            dataset_path="dgslibisey/MuSiQue",
            split=split,
            offline=True,
            reload=reload,
            max_samples=max_samples
        )
    
    def format_answer(self, sample: Dict[str, Any]) -> List[str]:
        answer = sample.get("answer", "")
        answer_aliases = sample.get("answer_aliases", [])
        return [answer] + answer_aliases
    
    def format_context(self, sample: Dict[str, Any]) -> str:
        context = sample["paragraphs"]
        context_text = ""
        for paragraph in context:
            title = paragraph["title"]
            paragraph_text = paragraph["paragraph_text"]
            context_text += f"\n{title}. {paragraph_text}"
        return context_text
    
    def format_supporting_facts(self, sample: Dict[str, Any]) -> List[str]:
        context = sample["paragraphs"]
        sfs = []
        for paragraph in context:
            is_supporting = paragraph["is_supporting"]
            if is_supporting:
                paragraph_text = paragraph["paragraph_text"]
                sfs.append(paragraph_text.strip())
        return sfs
        

if __name__ == "__main__":
    loader = MuSiQue(reload=True)
    dataset = loader.dataset
    print(f"数据集样本数: {len(loader.dataset)}")
    loader.random_sample()


