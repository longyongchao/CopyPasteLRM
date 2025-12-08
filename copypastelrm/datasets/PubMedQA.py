from typing import Any, Dict, List
from copypastelrm.datasets.BaseDatasetLoader import BaseDatasetLoader


class PubMedQA(BaseDatasetLoader):
    """https://huggingface.co/datasets/qiaojin/PubMedQA"""
        
    def __init__(self, reload: bool = False, max_samples: int = -1):
        super().__init__(
            dataset_path="qiaojin/PubMedQA",
            dataset_name="pqa_labeled",
            split="train",
            name='pubmedqa_pqa_labeled_train',
            offline=True,
            reload=reload,
            max_samples=max_samples
        )
    
    def format_id(self, sample):
        return str(sample['pubid'])
    
    def format_answer(self, sample: Dict[str, Any]) -> str:
        return [sample['final_decision']]
    
    def format_context(self, sample: Dict[str, Any]) -> str:
        """
        格式化上下文信息
        
        Args:
            sample: 包含 title 和 sentences 的字典
            
        Returns:
            str: 格式化后的上下文文本
        """
        context = sample["context"]
        contexts = context['contexts']
        labels = context['labels']
        context_text = ""
        for label, ctx in zip(labels, contexts):
            context_text += f"\n{label}. " + ctx
        return context_text
    
    def format_supporting_facts(self, sample: Dict[str, Any]) -> List[str]:
        return []

if __name__ == "__main__":
    loader = PubMedQA(reload=True)
    dataset = loader.dataset
    print(f"数据集样本数: {len(loader.dataset)}")
    loader.random_sample()


