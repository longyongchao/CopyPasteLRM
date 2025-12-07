from typing import Any, Dict, List
from copypastelrm.datasets.BaseDatasetLoader import BaseDatasetLoader
from datasets import load_dataset


class Qasper(BaseDatasetLoader):
    """https://huggingface.co/datasets/allenai/qasper"""

    def __init__(self, reload: bool = False):
        super().__init__(
            dataset_path="allenai/qasper",
            split="test",
            cache_path="cache/qasper_test.jsonl",
            offline=True,
            reload=reload,
        )

    def download_dataset(self) -> List[Dict[str, Any]]:
        """默认从huggingface下载数据"""
        print(f"正在加载 {self.dataset_path} 数据集...")
        if self.dataset_name:
            print(f"数据集子集: {self.dataset_name}")
        print(f"数据分割: {self.split}")

        origin_dataset = load_dataset("parquet",data_files='data/qasper/qasper_test.parquet')

        origin_dataset = list(origin_dataset['train'])
    
        print(len(origin_dataset))

        dataset = []

        def contruct_context(paper: dict):
            title = paper['title']
            abstract = paper["abstract"]
            full_text = paper["full_text"]
            figures_and_tables = paper["figures_and_tables"]
            context = '# ' + title + '\n' + abstract + "\n"
            for sec_name, para in zip(
                full_text["section_name"], full_text["paragraphs"]
            ):
                part = "\n".join(para)
                context += f'{sec_name}\n{part}\n\n'
            captions = "\n".join(figures_and_tables['caption'])
            context += f"{captions}"
            return context

        for paper in origin_dataset:
            context = contruct_context(paper)
            paper_id = paper['id']
            qas = paper['qas']
            questions = qas['question']
            question_ids = qas['question_id']
            answers = qas['answers']
            for question, question_id, answer_dict in zip(questions, question_ids, answers):
                sample_id = f"{paper_id}_{question_id}"
                answer = []
                supporting_facts = []
                for ans_dict in answer_dict['answer']:
                    if not ans_dict['unanswerable']:
                        answer += ans_dict['extractive_spans'] + [ans_dict['free_form_answer']]
                        supporting_facts += ans_dict['highlighted_evidence']
                
                answer = [s for s in answer if s.strip()]
                if len(answer) < 1 or len(supporting_facts) < 1:
                    continue
                
                dataset.append({
                    "id": sample_id,
                    "query": question,
                    "context": context,
                    "supporting_facts": supporting_facts,
                    "answer": [s for s in answer if s.strip()]
                })
        return dataset
                    

if __name__ == "__main__":
    loader = Qasper(reload=True)
    dataset = loader.dataset
    print(loader.get_length())
    print(loader.random_sample())

