from typing import Any, Dict
from copypastelrm.datasets.BaseDatasetLoader import BaseDatasetLoader
from typing import Literal


class PopQA(BaseDatasetLoader):
    """非官方来源：https://huggingface.co/datasets/Atipico1/popQA"""

    def __init__(
        self, 
        split: Literal['train', 'test'] = 'train', 
        reload: bool = False, 
        max_samples: int = -1,
        distractor_docs: int = 8,
        unanswerable: bool = False
    ):
        super().__init__(
            dataset_path="Atipico1/popQA",
            split=split,
            offline=True,
            reload=reload,
            max_samples=max_samples,
            distractor_docs=distractor_docs,
            unanswerable=unanswerable, # 是否不包含gold context
        )
    
    def format_item(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        _id = str(sample["id"])
        query = sample['question']
        answers = sample['answers']

        ctxs = sample['ctxs']
        corpus = []

        for ctx in ctxs:
           title = ctx['title'] 
           text = ctx['text']
           sentences = self.nlp.split_sentences_spacy(text)
           hasanswer = ctx['hasanswer']
           corpus.append({
               "title": title,
               "sentences": sentences,
               "facts": sentences if hasanswer else None
           })
        
        return {
            "id": _id,
            "query": query,
            "answers": answers,
            "corpus": corpus,
            "extra": {
                "s_pop": sample['s_pop'],
                "o_pop": sample['o_pop'],
            }
        }



if __name__ == "__main__":
    import json
    loader = PopQA(reload=True)
    dataset = loader.dataset
    dataset_list = loader.dataset_list
    print(json.dumps(dataset_list[0], indent=4))

