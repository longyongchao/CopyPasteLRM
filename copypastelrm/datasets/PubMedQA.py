from typing import Any, Dict, List
from copypastelrm.datasets.BaseDatasetLoader import BaseDatasetLoader
from typing import Literal


class PubMedQA(BaseDatasetLoader):
    """https://huggingface.co/datasets/qiaojin/PubMedQA"""

    def __init__(
        self,
        dataset_name: Literal[
            "pqa_labeled", "pqa_unlabeled", "pqa_artificial"
        ] = "pqa_labeled",
        reload: bool = False,
        max_samples: int = -1,
        distractor_docs: int = 8,
        unanswerable: bool = False,
    ):
        super().__init__(
            dataset_path="qiaojin/PubMedQA",
            dataset_name=dataset_name,
            split="train",
            offline=True,
            reload=reload,
            max_samples=max_samples,
            distractor_docs=distractor_docs,
            unanswerable=unanswerable,  # 是否不包含gold context
        )

    def format_item(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        _id = str(sample["pubid"])
        query = sample["question"]
        answers = [sample["final_decision"]]

        context = sample["context"]

        sentences = []

        for ctx in context["contexts"]:
            sentences.extend(self.nlp.split_sentences_spacy(ctx))

        corpus = [{"title": "Abstract", "sentences": sentences, "facts": sentences}]

        return {
            "id": _id,
            "query": query,
            "answers": answers,
            "corpus": corpus,
            "extra": {
                "long_answer_sents": self.nlp.split_sentences_spacy(
                    sample["long_answer"]
                ),
                "options": ["yes", "no", "maybe", "unknown"],
            },
        }

if __name__ == "__main__":
    import json

    loader = PubMedQA(reload=True, dataset_name="pqa_labeled")
    dataset = loader.dataset
    dataset_list = loader.dataset_list
    print(json.dumps(dataset_list[0], indent=4))
