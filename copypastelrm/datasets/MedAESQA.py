from typing import Any, Dict, List
from copypastelrm.datasets.BaseDatasetLoader import BaseDatasetLoader
from copypastelrm.utils.json_tools import read_json, read_jsonl_to_list
import re
import os
import json

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

class MedAESQA(BaseDatasetLoader):
    def __init__(self, reload: str = False, max_samples: int = -1):
        # 1. 优先加载并构建缓存，只读一次文件
        self.pubmed_cache = self._load_pubmed_db('data/medaesqa/medaesqa_context.jsonl')

        super().__init__(
            dataset_path="data/medaesqa/medaesqa_v1.json",
            split='test',
            offline=True,
            reload=reload,
            max_samples=max_samples
        )
    
    def _load_pubmed_db(self, filepath: str) -> Dict[str, Dict]:
        """
        加载 JSONL 文件并转换为字典缓存，具备容错能力。
        """
        print(f"正在加载 PubMed 数据库: {filepath} ...")
        cache = {}
        error_count = 0
        
        if not os.path.exists(filepath):
            print(f"警告: 文件不存在 {filepath}")
            return cache

        with open(filepath, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f, 1):
                line = line.strip()
                if not line: continue
                try:
                    entry = json.loads(line)
                    # 建立索引：PMID -> {title, abstract}
                    if 'pmid' in entry:
                        cache[str(entry['pmid'])] = {
                            'title': entry.get('title', ''),
                            'abstract': entry.get('abstract', '')
                        }
                except json.JSONDecodeError as e:
                    # 2. 捕获错误，打印具体哪一行坏了，然后继续执行，不要崩溃
                    if error_count < 5: # 只打印前5个错误，避免刷屏
                        print(f"[警告] 跳过损坏的 JSON 行 (第 {line_idx} 行): {e}")
                        # 打印出问题行的前100个字符方便调试
                        print(f"      内容片段: {line[:100]}...") 
                    error_count += 1
        
        print(f"数据库加载完成。有效记录: {len(cache)}, 跳过损坏行: {error_count}")
        return cache
    
    def download_dataset(self):
        data = read_json(self.dataset_path)
        return data
    
    def format_id(self, sample: Dict[str, Any]) -> str:
        return sample["question_id"]

    def format_context(self, sample: Dict[str, Any]) -> str:
        context, _ = self.extract_citation(sample)
        return "\n".join(context)

    def format_supporting_facts(self, sample: Dict[str, Any]) -> List[str]:
        _, supporting_facts = self.extract_citation(sample)
        return supporting_facts

    def format_answer(self, sample):
        return sample['nuggets']
        
    def get_pubmed_title_abstract(self, pmid: str) -> Dict[str, str]:
        """
        现在这里变成 O(1) 的内存查找，速度会飞快
        """
        # 确保 pmid 是字符串
        pmid = str(pmid).strip()
        return self.pubmed_cache.get(pmid, {'title': '', 'abstract': ''})
    
    def extract_citation(self, sample) -> List[str]:
        context_paper_ids = set()
        supporting_facts = []

        expert_curated_answer_ids = extract_citation_ids(sample['expert_curated_answer'])

        mg_answers_supporting_facts_ids = set() # 用于存储机器生成答案中支持事实的引用ID
        context_paper_ids.update(expert_curated_answer_ids) # 该问题的所有引用ID集合
        expert_curated_answer_ids = set(expert_curated_answer_ids)

        machine_generated_answers = sample.get('machine_generated_answers')

        for _, mg_answer in machine_generated_answers.items():
            answer_sentences = mg_answer['answer_sentences']
            is_answer_accurate = mg_answer['is_answer_accurate']
            answer = mg_answer['answer']

            context_paper_ids.update(extract_citation_ids(answer))

            if is_answer_accurate:
                for sent in answer_sentences:
                    citation_assessment = sent['citation_assessment']
                    answer_sentence_relevance = sent['answer_sentence_relevance']

                    if answer_sentence_relevance == 'required':
                        if citation_assessment:
                            for citation in citation_assessment:
                                if citation['evidence_relation'] == 'supporting':
                                    supporting_facts.append(citation['evidence_support'])
                                    mg_answers_supporting_facts_ids.add(citation['cited_pmid']) 
        

        for miss_id in list(expert_curated_answer_ids - mg_answers_supporting_facts_ids):
            supporting_facts.append(self.get_pubmed_title_abstract(miss_id)['abstract'])
        
        context = []
        
        for pid in context_paper_ids:
            paper_info = self.get_pubmed_title_abstract(pid)
            context.append(f"{paper_info['title'].capitalize()}: {paper_info['abstract']}")
        
        return context, supporting_facts
        
                                
if __name__ == "__main__":
    loader = MedAESQA(reload=False)
    dataset = loader.dataset_list
    print(f"数据集样本数: {len(loader.dataset)}")
    print(dataset[0])


