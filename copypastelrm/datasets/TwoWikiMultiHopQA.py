import json
from typing import Any, Dict, List
from copypastelrm.datasets.BaseDatasetLoader import BaseDatasetLoader
from typing import Literal


class TwoWikiMultihopQA(BaseDatasetLoader):
    """
    TwoWikiMultihopQA数据集加载器类，继承自BaseDatasetLoader。
    用于处理和加载TwoWikiMultihopQA多跳问答数据集。
    """

    def __init__(
        self,
        split: Literal["dev", "test"] = "dev",
        reload: str = False,
        max_samples: int = -1,
        distractor_docs: int = 8,
        unanswerable: bool = False,
        filter_empty_answer=True,
    ):
        super().__init__(
            dataset_path="data/2WikiMultihopQA" + "/" + split + ".json",
            split=split,
            offline=True,
            reload=reload,
            max_samples=max_samples,
            distractor_docs=distractor_docs,
            unanswerable=unanswerable,  # 是否不包含gold context
            filter_empty_answer=filter_empty_answer
        )

    def download_dataset(self):

        with open(self.dataset_path, "r") as f:
            return json.load(f)

    def format_item(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        将 2WikiMultihopQA 的原始样本转换为 BaseDatasetLoader 要求的格式。
        """
        _id = sample["_id"]
        query = sample["question"]
        # 2Wiki 的 answer 通常是一个字符串，转为 list
        answers = [sample["answer"]] if "answer" in sample else []

        # 构建 supporting facts 的查找集合 {(title, sent_idx)} 用于快速匹配
        # 原始数据中 supporting_facts 格式为 [[title, sent_idx], ...]
        sp_set = set()
        if "supporting_facts" in sample:
            for sp in sample["supporting_facts"]:
                # sp[0] 是 title, sp[1] 是句子在段落中的索引
                sp_set.add((sp[0], sp[1]))

        corpus = []
        # 原始数据中 context 格式为 [[title, [sent1, sent2, ...]], ...]
        raw_context = sample.get("context", [])

        for paragraph in raw_context:
            title = paragraph[0]
            sentences = paragraph[1]  # 2Wiki 的句子已经是分好句的列表

            # 提取当前段落中的 supporting facts
            current_facts = []
            for idx, sent in enumerate(sentences):
                if (title, idx) in sp_set:
                    current_facts.append(sent)

            corpus.append(
                {
                    "title": title,
                    "sentences": sentences,
                    "facts": current_facts if len(current_facts) > 0 else None,
                }
            )

        return {
            "id": _id,
            "query": query,
            "answers": answers,
            "corpus": corpus,
            "extra": {
                "type": sample.get("type", None),
                "level": sample.get("level", None),
            },
        }


if __name__ == "__main__":
    import json
    loader = TwoWikiMultihopQA(reload=True, split="dev", filter_empty_answer=False)
    dataset = loader.dataset
    print(dataset['cd665d0c0bdd11eba7f7acde48001122'])
    dataset_list = loader.dataset_list
    print(len(dataset_list))
    # print(json.dumps(dataset_list[0], indent=4))
