from swift.llm import DatasetMeta, ResponsePreprocessor, load_dataset, register_dataset
from copypastelrm.prompt.passAtKEqual0 import system_prompt, user_prompt_template 
from copypastelrm.datasets.MuSiQue import MuSiQue
from string import Template
from typing import Literal
from copypastelrm.utils.dataset import StringContainmentFilter
import json

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
                    "supporting_facts": filtered_sfs,
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
