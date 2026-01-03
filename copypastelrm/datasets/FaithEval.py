from typing import Any, Dict, List
from copypastelrm.datasets.BaseDatasetLoader import BaseDatasetLoader
import re


class FaithEval(BaseDatasetLoader):
    def __init__(
        self, 
        reload: bool = False, 
        max_samples: int = -1,
        distractor_docs: int = 8,
        unanswerable: bool = False, # 是否不包含gold context
    ):
        super().__init__(
            dataset_path="Salesforce/FaithEval-counterfactual-v1.0",
            split="test",
            offline=True,
            reload=reload,
            max_samples=max_samples,
            distractor_docs=distractor_docs,
            unanswerable=unanswerable, # 是否不包含gold context
        )
    
    @staticmethod
    def extract_quoted_sentence(text):
        """
        提取字符串中被单引号包裹的句子。
        如果找到匹配项，返回引号内的内容；否则返回 None。
        """
        # 正则表达式解释：
        # '      : 匹配开始的单引号
        # (.*?)  : 非贪婪匹配任意字符（捕获组1，即我们需要的内容）
        # '      : 匹配结束的单引号
        # re.DOTALL : 允许 . 匹配换行符，防止句子跨行时失效
        pattern = r"'(.*?)'"
        
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            return match.group(1)
        else:
            return text
    
    def format_item(self, sample: Dict[str, Any]):
        _id = sample['id']
        query = sample['question']
        answers = [sample['answer']]
        justification = sample['justification']
        fact = self.extract_quoted_sentence(justification)
        context = sample['context']

        return {
            "id": _id,
            "query": query,
            "answers": answers,
            "corpus": [
                {
                    "title": "Document",
                    "sentences": self.nlp.split_sentences_spacy(context),
                    "facts": [fact]
                }
            ],
            "extra": {
                "choices": sample['choices'],
                "answerKey": sample['answerKey']
            }
        }

        

if __name__ == "__main__":
    import json

    loader = FaithEval(reload=True)
    dataset = loader.dataset
    dataset_list = loader.dataset_list
    print(json.dumps(dataset_list[0], indent=4))
