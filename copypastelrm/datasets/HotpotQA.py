from typing import Any, Dict, List, Literal
from copypastelrm.datasets.BaseDatasetLoader import BaseDatasetLoader
import json


class HotpotQA(BaseDatasetLoader):
    """https://huggingface.co/datasets/hotpotqa/hotpot_qa"""

    def __init__(
        self,
        dataset_path="hotpotqa/hotpot_qa",
        dataset_name="distractor",
        split: Literal["train", "validation"] = "validation",
        reload: str = False,
        max_samples: int = -1,
        distractor_docs: int = 8,
        unanswerable: bool = False,  # 是否不包含gold context
    ):
        super().__init__(
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            split=split,
            offline=True,
            reload=reload,
            max_samples=max_samples,
            distractor_docs=distractor_docs,
            unanswerable=unanswerable,
        )

    def format_item(self, sample: Dict[str, Any]) -> str:
        """
        格式化上下文信息

        Args:
            sample: 包含 title 和 sentences 的字典

        Returns:
            str: 格式化后的上下文文本
        """
        context = sample["context"]
        supporting_facts = sample["supporting_facts"]
        corpus = []

        _id = sample["id"]
        query = sample["question"]
        answers = [sample["answer"]]
        type = sample["type"]
        level = sample["level"]

        for title, sentences in zip(context["title"], context["sentences"]):
            facts = []
            for fact_title, fact_sent_id in zip(
                supporting_facts["title"], supporting_facts["sent_id"]
            ):
                if title == fact_title:
                    if fact_sent_id < len(sentences):
                        facts.append(sentences[fact_sent_id])
            corpus.append(
                {
                    "title": title if title else "unknown title",
                    "sentences": sentences,
                    "facts": facts if len(facts) > 0 else None,
                }
            )

        return {
            "id": _id,
            "query": query,
            "answers": answers,
            "corpus": corpus,
            "extra": {"type": type, "level": level},
        }


if __name__ == "__main__":
    loader = HotpotQA(reload=True, split="validation", unanswerable=True)
    dataset = loader.dataset
    dataset_list = loader.dataset_list
    print(json.dumps(dataset_list[0], indent=4))
