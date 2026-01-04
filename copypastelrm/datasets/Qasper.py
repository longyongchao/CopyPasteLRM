from typing import Any, Dict, List
from copypastelrm.datasets.BaseDatasetLoader import BaseDatasetLoader
from datasets import load_dataset
from typing import Literal
from copypastelrm.utils.dataset import StringContainmentFilter


class Qasper(BaseDatasetLoader):
    """https://huggingface.co/datasets/allenai/qasper"""

    def __init__(
        self,
        split: Literal["train", "validation", "test"] = "test",
        reload: bool = False,
        max_samples: int = -1,
        distractor_docs: int = 8,
        unanswerable: bool = False,
    ):

        super().__init__(
            dataset_path="allenai/qasper",
            split=split,
            offline=True,
            reload=reload,
            max_samples=max_samples,
            distractor_docs=distractor_docs,
            unanswerable=unanswerable,  # 是否不包含gold context
        )

    def download_dataset(self) -> List[Dict[str, Any]]:
        """默认从huggingface下载数据"""
        print(f"正在加载 {self.dataset_path} 数据集...")
        if self.dataset_name:
            print(f"数据集子集: {self.dataset_name}")
        print(f"数据分割: {self.split}")

        origin_dataset = load_dataset(
            "parquet", data_files="data/qasper/qasper_test.parquet"
        )

        origin_dataset = list(origin_dataset["train"])

        print(len(origin_dataset))

        dataset = []

        def contruct_context(paper: dict):
            abstract = paper["abstract"]
            full_text = paper["full_text"]
            figures_and_tables = paper["figures_and_tables"]
            context = abstract + "\n"
            for sec_name, para in zip(
                full_text["section_name"], full_text["paragraphs"]
            ):
                part = "\n".join(para)
                context += f"{sec_name}\n{part}\n\n"
            captions = "\n".join(figures_and_tables["caption"])
            context += f"{captions}"
            return context

        for paper in origin_dataset:
            context = contruct_context(paper)
            title = paper["title"]
            paper_id = paper["id"]
            qas = paper["qas"]
            questions = qas["question"]
            question_ids = qas["question_id"]
            answers = qas["answers"]
            for question, question_id, answer_dict in zip(
                questions, question_ids, answers
            ):
                sample_id = f"{paper_id}_{question_id}"
                answers = []
                supporting_facts = []
                for ans_dict in answer_dict["answer"]:
                    if not ans_dict["unanswerable"]:
                        answers += ans_dict["extractive_spans"] + [
                            ans_dict["free_form_answer"]
                        ]
                        supporting_facts += ans_dict["highlighted_evidence"]

                answers = [s for s in answers if s.strip()]
                if len(answers) < 1 or len(supporting_facts) < 1:
                    continue

                dataset.append(
                    {
                        "id": sample_id,
                        "query": question,
                        "title": title,
                        "context": context,
                        "supporting_facts": supporting_facts,
                        "answers": [s for s in answers if s.strip()],
                    }
                )
        return dataset

    def format_item(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        facts = sample["supporting_facts"]

        facts_processor = StringContainmentFilter(facts)

        filtered_facts = facts_processor.filter_minimal_substrings()

        return {
            "id": sample["id"],
            "query": sample["query"],
            "answers": sample["answers"],
            "corpus": [
                {
                    "title": sample["title"],
                    "sentences": self.nlp.split_sentences_spacy(sample["context"]),
                    "facts": filtered_facts,
                }
            ],
            "extra": {}
        }


if __name__ == "__main__":
    import json
    loader = Qasper(reload=True, split="validation")
    dataset = loader.dataset
    dataset_list = loader.dataset_list
    print(json.dumps(dataset_list[0], indent=4))
