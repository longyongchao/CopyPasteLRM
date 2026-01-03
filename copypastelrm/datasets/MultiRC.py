import json
from typing import Any, Dict, List
from copypastelrm.datasets.BaseDatasetLoader import BaseDatasetLoader
import re
from typing import Literal


class MultiRC(BaseDatasetLoader):
    def __init__(
        self,
        dataset_path: Literal['data/multirc'] = "data/multirc",
        split: Literal["train", "dev"] = "dev",
        reload: bool = False,
        max_samples: int = -1,
        distractor_docs: int = 8,
        unanswerable: bool = False
    ):
        super().__init__(
            dataset_path=dataset_path + '/' + split + '.json',
            split=split,
            offline=True,
            reload=reload,
            max_samples=max_samples,
            distractor_docs=distractor_docs,
            unanswerable=unanswerable,  # 是否不包含gold context
        )

    def extract_sentences_with_regex(self, input_text: str) -> dict:
        """
        MultiRC 特有的清洗逻辑：
        根据 <br> 分句，并去除 <b>Sent N:</b> 标签
        返回字典: {'1': '句子内容', '2': '句子内容'}
        """
        # 1. 定义用于匹配和分割句子的模式
        sentences = re.split(r"<br\s*/?>", input_text, flags=re.IGNORECASE)

        # 2. 定义用于清理每个句子的模式
        clean_pattern = re.compile(
            r"^.*?<b>Sent\s+\d+:\s*<\/b>\s*", flags=re.IGNORECASE | re.DOTALL
        )

        output_dict = {}
        sentence_number = 1

        for sentence in sentences:
            stripped_sentence = sentence.strip()

            if not stripped_sentence:
                continue

            cleaned_content = re.sub(clean_pattern, "", stripped_sentence)
            final_content = cleaned_content.strip()

            if final_content:
                output_dict[str(sentence_number)] = final_content
                sentence_number += 1

        return output_dict

    def download_dataset(self):
        """
        读取原始 JSON，并将 MultiRC 的结构（1个段落 -> N个问题）
        扁平化为 List[Sample]（1个问题 -> 1个段落）
        """
        dataset = []

        with open(self.dataset_path, "r") as f:
            data = json.load(f)["data"]

        for item in data:
            # 清洗段落文本
            raw_context_text = item["paragraph"]["text"]
            context_dict = self.extract_sentences_with_regex(raw_context_text)
            
            questions = item["paragraph"]["questions"]
            doc_id = item["id"]  # 通常是文件名

            for q_item in questions:
                q_idx = q_item["idx"]
                # 构造唯一 ID
                sub_id = f"{doc_id}_{q_idx}"
                question = q_item["question"]
                supporting_facts_indices = q_item["sentences_used"]
                
                # 提取答案
                answer_item = q_item["answers"]
                answers = []
                for a_item in answer_item:
                    if a_item["isAnswer"]:
                        answers.append(a_item["text"])

                # 注意：这里 context 还是字典，supporting_facts 还是索引列表('1', '2'...)
                # 具体的文本转换将在 format_item 中进行
                dataset.append(
                    {
                        "id": sub_id,
                        "question": question,
                        "answers": answers,
                        "context": context_dict,
                        "supporting_facts": [str(sf + 1) for sf in supporting_facts_indices],
                        "doc_id": doc_id # 保留用于 Title
                    }
                )

        return dataset

    def format_item(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        将 download_dataset 返回的中间格式转换为 BaseDatasetLoader 标准格式
        """
        _id = sample['id']
        query = sample['question']
        answers = sample['answers']
        
        # 原始 context 是 {'1': 'sent1', '2': 'sent2'}
        # 转换为 BaseDatasetLoader 要求的 sentences 列表
        context_dict = sample['context']
        
        # 确保按 key 顺序（1, 2, 3...）排列句子
        # 虽然 Python 3.7+ 字典有序，但为了保险起见，这里可以显式排序或者直接利用 extraction 时的插入顺序
        # 这里直接取 values，因为 extraction 是按顺序插入的
        sentences = list(context_dict.values())
        
        # 提取 Supporting Facts 的文本
        # sample['supporting_facts'] 包含的是 key (如 "1", "5")
        facts = []
        for sf_key in sample['supporting_facts']:
            if sf_key in context_dict:
                facts.append(context_dict[sf_key])
        
        # 构建 corpus
        # MultiRC 是单文档阅读理解，所以 corpus 只有一个条目
        corpus_item = {
            "title": sample.get('doc_id', str(_id)), # 使用 doc_id 作为标题
            "sentences": sentences,
            "facts": facts if len(facts) > 0 else None
        }

        return {
            "id": _id,
            "query": query,
            "answers": answers,
            "corpus": [corpus_item],
            "extra": {} 
        }

if __name__ == "__main__":
    # 强制 reload 以更新缓存格式
    loader = MultiRC(reload=True)
    dataset = loader.dataset
    print(f"数据集样本数: {len(loader.dataset)}")
    
    # 随机采样打印
    if len(loader.dataset_list) > 0:
        loader.random_sample()