import json
from typing import Any, Dict, List
from copypastelrm.datasets.BaseDatasetLoader import BaseDatasetLoader


class PopQA(BaseDatasetLoader):
    """非官方来源：https://huggingface.co/datasets/Atipico1/popQA"""

    def __init__(self, reload: bool = False, max_samples: int = -1):
        super().__init__(
            dataset_path="Atipico1/popQA",
            split="test",
            name="popqa_test",
            offline=True,
            reload=reload,
            max_samples=max_samples
        )

    def format_answer(self, sample) -> List[str]:
        answers = sample['answers']
        return answers

    def format_context(self, sample: Dict[str, Any]) -> str:
        context_text = ""
        ctxs = sample["ctxs"]

        for ctx in ctxs:
            title = ctx["title"]
            text = ctx["text"]
            context_text += f"\n{title}. {text}"
        return context_text.strip()

    def format_supporting_facts(self, sample: Dict[str, Any]) -> List[str]:

        ctxs = sample["ctxs"]

        sfs = []

        for ctx in ctxs:
            if ctx["hasanswer"]:
                title = ctx["title"]
                text = ctx["text"]
                sfs.append(f"{title}. {text}")

        return sfs


if __name__ == "__main__":
    loader = PopQA(reload=True)
    dataset = loader.dataset
    print(f"数据集样本数: {len(loader.dataset)}")
