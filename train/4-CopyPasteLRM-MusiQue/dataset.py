from swift.llm import DatasetMeta, ResponsePreprocessor, load_dataset, register_dataset
from copypastelrm.prompt.passAtKEqual0 import system_prompt, user_prompt_template 
from copypastelrm.datasets.MuSiQue import MuSiQue
from string import Template
from typing import Literal
from copypastelrm.utils.dataset import StringContainmentFilter
import json

import spacy

# 加载英文模型 (建议在全局加载一次，避免函数重复调用导致性能下降)
# disable 参数禁用了不需要的组件（如命名实体识别），可以提高分句速度
try:
    nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
except OSError:
    print("正在下载 spacy 模型...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])

def split_sentences_spacy(text):
    """
    使用 spaCy 对英文文本进行分句。
    
    Args:
        text (str): 输入的大段文本。
        
    Returns:
        list: 包含分句后字符串的列表。
    """
    # 处理文本
    # nlp() 会自动进行分词、词性标注和依存句法分析，从而确定句子边界
    doc = nlp(text)
    
    # doc.sents 是一个生成器，生成 Span 对象
    # 我们将其转换为文本并去除首尾空白
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    
    return sentences

class DeepSeekR1Preprocessor(ResponsePreprocessor):

    def __init__(self, system_prompt: Literal["deepseek", "copypaste"], columns: dict):
        super().__init__(columns=columns)

        musique_dataloader = MuSiQue(split='train')
        self.musique = musique_dataloader.dataset
        self.user_template = Template(user_prompt_template)
        self.system_prompt = system_prompt

    def preprocess(self, row: dict):
        _id = str(row["id"])
        item = self.musique[_id]

        query = item["query"]
        context = item["context"]
        answer = item["answer"]

        sfs = item["sfs"]
        sfs_processor = StringContainmentFilter(sfs)
        filtered_sfs = sfs_processor.filter_maximal_superstrings()

        row.update(
            {
                "system": system_prompt[self.system_prompt],
                "query": self.user_template.substitute(
                    context=context,
                    question=query,
                ),
                "supporting_facts": filtered_sfs,
                "answer_candidates": answer,
                "context": context,
                "dataset": 'MuSiQue',
                "solution": {
                    "context": context,
                    "supporting_facts": [split_sentences_spacy(sfs) for sfs in filtered_sfs],
                    "answers": answer,
                    "dataset": 'MuSiQue',
                    "id": _id,
                }
            }
        )
        return super().preprocess(row)  # type: ignore[return-value]


register_dataset(
    DatasetMeta(
        dataset_path="train/4-CopyPasteLRM-MusiQue/trainset/Qwen3_4B_I-musique_128_without_2hop.jsonl",
        dataset_name="Qwen3-4B-I_MusiQue_128_without_2hop_copypaste",
        preprocess_func=DeepSeekR1Preprocessor(
            system_prompt="copypaste",
            columns={
                "solution": "solution",
            }
        ),
    )
)

register_dataset(
    DatasetMeta(
        dataset_path="train/4-CopyPasteLRM-MusiQue/trainset/Qwen3_4B_I-musique_128_without_2hop.jsonl",
        dataset_name="Qwen3-4B-I_MusiQue_128_without_2hop_deepseek",
        preprocess_func=DeepSeekR1Preprocessor(
            system_prompt="deepseek",
            columns={
                "solution": "solution",
            }
        ),
    )
)

register_dataset(
    DatasetMeta(
        dataset_path="train/4-CopyPasteLRM-MusiQue/trainset/Qwen3_4B_I-musique_128_without_2hop_reasonable.jsonl",
        dataset_name="Qwen3-4B-I_MusiQue_128_without_2hop_reasonable_copypaste",
        preprocess_func=DeepSeekR1Preprocessor(
            system_prompt="copypaste",
            columns={
                "solution": "solution",
            }
        ),
    )
)

register_dataset(
    DatasetMeta(
        dataset_path="train/4-CopyPasteLRM-MusiQue/trainset/Qwen3_4B_I-musique_128_without_2hop_reasonable.jsonl",
        dataset_name="Qwen3-4B-I_MusiQue_128_without_2hop_reasonable_deepseek",
        preprocess_func=DeepSeekR1Preprocessor(
            system_prompt="deepseek",
            columns={
                "solution": "solution",
            }
        ),
    )
)

if __name__ == "__main__":
    dataset = load_dataset(
        ["Qwen3-4B-I_MusiQue_128_without_2hop_deepseek"], remove_unused_columns=False, download_mode="force_redownload"
    )

    print(f"dataset: {dataset}")
    print(f"dataset[0]: {dataset[0]}")
    print(f"dataset[0][0]: {json.dumps(dataset[0][0], indent=2)}")
