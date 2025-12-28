import os
import re
from copypastelrm.utils.json_tools import read_json, save_jsonl
from copypastelrm.datasets.MedAESQA import extract_citation_ids


data_path = 'data/medaesqa/medaesqa_v1.json'

if not os.path.exists(data_path):
    raise FileNotFoundError(f"{data_path} does not exist, 请先下载原始数据集")

data = read_json(data_path)
len(data)

expert_curated_answer_pubmed_ids = set()

machine_generated_answers_pubmed_ids = set()

for i, item in enumerate(data):
    expert_curated_answer = item['expert_curated_answer']
    ids = extract_citation_ids(expert_curated_answer)
    expert_curated_answer_pubmed_ids.update(ids)

    machine_generated_answers = item['machine_generated_answers']

    for key, machine in machine_generated_answers.items():
        mg_answer = machine['answer']
        mg_ids = extract_citation_ids(mg_answer)
        machine_generated_answers_pubmed_ids.update(mg_ids)

len(expert_curated_answer_pubmed_ids), len(machine_generated_answers_pubmed_ids)

save_jsonl(list(expert_curated_answer_pubmed_ids), 'data/medaesqa/expert_curated_answer_pubmed_ids.jsonl')
save_jsonl(list(machine_generated_answers_pubmed_ids), 'data/medaesqa/machine_generated_answers_pubmed_ids.jsonl')