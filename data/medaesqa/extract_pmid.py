import os
import re
from copypastelrm.utils.json_tools import read_json, save_jsonl
from copypastelrm.datasets.MedAESQA import extract_citation_ids

def extract_citation_ids(text):
    """
    从文本中提取所有方括号 [] 内的引用 ID。
    支持处理逗号分隔的多个 ID，例如 [123, 456]。
    """
    # 1. 正则表达式匹配：找到所有被 [] 包裹的内容
    # \[     : 匹配字面量的左方括号
    # (.*?)  : 非贪婪捕获组，匹配括号内的任意字符，直到遇到右括号
    # \]     : 匹配字面量的右方括号
    matches = re.findall(r'\[(.*?)\]', text)
    
    all_ids = []
    
    for match in matches:
        # 2. 处理括号内的内容
        # 如果是 "30466670, 38078610"，按逗号分割
        parts = match.split(',')
        
        for part in parts:
            # 去除可能存在的首尾空格 (例如逗号后的空格)
            clean_id = part.strip()
            
            # 3. 简单校验：确保提取的是数字（避免提取到非ID的内容）
            if clean_id.isdigit():
                all_ids.append(clean_id)
                
    return all_ids

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