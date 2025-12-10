from typing import Any, Dict, List, Literal
from copypastelrm.datasets.BaseDatasetLoader import BaseDatasetLoader


class HotpotQA(BaseDatasetLoader):
    """https://huggingface.co/datasets/hotpotqa/hotpot_qa"""

    def __init__(self, dataset_path="hotpotqa/hotpot_qa", dataset_name="distractor", split: Literal["train", "validation"] = "validation", reload: str = False, max_samples: int = -1):
        super().__init__(
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            split=split,
            offline=True,
            reload=reload,
            max_samples=max_samples,
        )

    def format_context(self, sample: Dict[str, Any]) -> str:
        """
        格式化上下文信息

        Args:
            sample: 包含 title 和 sentences 的字典

        Returns:
            str: 格式化后的上下文文本
        """
        context = sample["context"]
        context_text = ""
        for title, sentences in zip(context["title"], context["sentences"]):
            context_text += f"{title}. " + "".join(sentences) + "\n"
        return context_text

    def format_supporting_facts(self, sample: Dict[str, Any]) -> List[str]:
        """格式化支持事实"""
        supporting_facts = sample["supporting_facts"]
        context = sample["context"]
        sfs = []
        for title, sent_id in zip(
            supporting_facts["title"], supporting_facts["sent_id"]
        ):
            try:
                sentences_idx = context["title"].index(title)
                sf = context["sentences"][sentences_idx][sent_id]
                sfs.append(sf.strip())
            except:
                print(
                    f"Warning in formatting supporting facts: title={title}, sent_id={sent_id}"
                )
        return sfs


if __name__ == "__main__":
    loader = HotpotQA(reload=True)
    dataset = loader.dataset
    dataset_list = loader.dataset_list
    print(dataset_list[:5])
