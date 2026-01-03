import json
from typing import Any, Dict, Literal
from copypastelrm.datasets.BaseDatasetLoader import BaseDatasetLoader


class ConFiQA(BaseDatasetLoader):
    def __init__(
        self,
        subset: Literal["MC", "MR", "QA"] = "QA",
        split: Literal["original", "counterfactual"] = "original",
        reload: bool = False,
        max_samples: int = -1,
        distractor_docs: int = 8,
        unanswerable: bool = False
    ):
        """
        Args:
            subset: 数据集子集 ('MC', 'MR', 'QA')
            split: 数据分割模式 ('original' 或 'counterfactual')
        """
        # 映射子集到文件名
        file_map = {
            "MC": "ConFiQA-MC.json",
            "MR": "ConFiQA-MR.json",
            "QA": "ConFiQA-QA.json",
        }

        if subset not in file_map:
            raise ValueError(
                f"Unknown subset: {subset}. Must be one of {list(file_map.keys())}"
            )

        file_name = file_map[subset]

        # 记录当前的模式，供 format_item 使用
        self.mode = split
        self.subset = subset

        super().__init__(
            dataset_path=f"data/ConFiQA/{file_name}",
            split=split,  # 这里 split 仅作为标识，实际加载逻辑在 download_dataset 处理
            offline=True,
            reload=reload,
            max_samples=max_samples,
            dataset_name=subset,  # 传给父类用于缓存文件名区分
            distractor_docs=distractor_docs,
            unanswerable=unanswerable,  # 是否不包含gold context
        )

    def download_dataset(self):
        """直接加载本地 JSON 文件"""
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 原始数据没有 ID，我们需要为每个样本生成一个唯一 ID
        # 格式: {subset}_{mode}_{index}
        dataset_with_ids = []
        for idx, item in enumerate(data):
            # 将索引注入 item，方便 format_item 使用
            item["_generated_id"] = f"{self.subset}_{self.mode}_{idx}"
            dataset_with_ids.append(item)

        return dataset_with_ids

    def format_item(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据 self.mode ('original' 或 'counterfactual') 提取对应字段
        """
        prefix = "orig_" if self.mode == "original" else "cf_"

        # 1. 提取基础信息
        _id = sample["_generated_id"]
        query = sample["question"]

        # 2. 构建答案列表 (主答案 + 别名)
        main_answer = sample.get(f"{prefix}answer")
        aliases = sample.get(f"{prefix}alias", [])
        answers = [main_answer] + aliases if main_answer else []
        # 去重并去除空值
        answers = list(set([a for a in answers if a]))

        # 3. 处理 Context (Corpus)
        raw_context = sample.get(f"{prefix}context", "")

        # 使用 NLP 工具分句 (如果没有 self.nlp.split_sentences，可暂时用 split('. '))
        if hasattr(self, "nlp") and hasattr(self.nlp, "split_sentences"):
            sentences = self.nlp.split_sentences(raw_context)
        else:
            # 简单的备用分句逻辑
            sentences = [s.strip() + "." for s in raw_context.split(". ") if s.strip()]

        # 4. 构建 Facts
        # 策略：如果句子中包含了主答案，则认为该句是 Supporting Fact
        facts = []
        if main_answer:
            for sent in sentences:
                if main_answer.lower() in sent.lower():
                    facts.append(sent)

        # 5. 构建 Corpus 结构
        # ConFiQA 每个样本只有一个段落
        corpus = [
            {
                "title": f"Context_{_id}",  # 这里的 title 没有实际意义，生成一个
                "sentences": sentences,
                "facts": sentences,
            }
        ]

        return {
            "id": _id,
            "query": query,
            "answers": answers,
            "corpus": corpus,
            "extra": {
                "triple": sample.get(f"{prefix}triple"),
                "path": sample.get(f"{prefix}path_labeled"),
            },
        }


if __name__ == "__main__":
    # 测试代码
    try:
        # 测试加载 Original 分割
        loader = ConFiQA(subset="QA", split="original", reload=True)
        print(f"Original subset loaded. Samples: {len(loader.dataset)}")
        loader.random_sample()

        print("-" * 50)

        # 测试加载 Counterfactual 分割
        loader_cf = ConFiQA(subset="QA", split="counterfactual", reload=True)
        print(f"Counterfactual subset loaded. Samples: {len(loader_cf.dataset)}")
        loader_cf.random_sample()

    except Exception as e:
        print(f"Error: {e}")
