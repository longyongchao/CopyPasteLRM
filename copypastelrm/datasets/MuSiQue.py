from typing import Any, Dict, List
from copypastelrm.datasets.BaseDatasetLoader import BaseDatasetLoader
from typing import Literal


class MuSiQue(BaseDatasetLoader):
    """https://huggingface.co/datasets/dgslibisey/MuSiQue"""
        
    def __init__(
        self, 
        split: Literal["train", "validation"]="validation", 
        reload: bool = False, 
        max_samples: int = -1,        
        distractor_docs: int = 8,
        unanswerable: bool = False
    ):

        super().__init__(
            dataset_path="dgslibisey/MuSiQue",
            split=split,
            offline=True,
            reload=reload,
            max_samples=max_samples,
            distractor_docs=distractor_docs,
            unanswerable=unanswerable, # 是否不包含gold context
        )
    
    def format_item(self, sample: Dict[str, Any]):
        _id = sample['id']
        query = sample['question']
        answers = [sample['answer']] + sample['answer_aliases']
        paragraphs = sample["paragraphs"]
        corpus = []

        for paragraph in paragraphs:
            title = paragraph["title"]
            paragraph_sents = self.nlp.split_sentences_spacy(paragraph['paragraph_text'])
            is_supporting = paragraph['is_supporting']
            sentences = paragraph_sents
            facts = paragraph_sents if is_supporting else None
            corpus.append({
                "title": title,
                "sentences": sentences,
                "facts": facts
            })
        
        return {
            "id": _id,
            "query": query,
            "answers": answers,
            "corpus": corpus,
            "extra": {
                "question_decomposition": sample['question_decomposition']
            }
        }

    
if __name__ == "__main__":
    import json
    loader = MuSiQue(reload=True, split='validation')
    dataset = loader.dataset
    dataset_list = loader.dataset_list
    print(json.dumps(dataset_list[0], indent=4))


