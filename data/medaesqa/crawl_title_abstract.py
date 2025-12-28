from copypastelrm.utils.json_tools import read_json, save_jsonl, read_jsonl_to_list
from copypastelrm.datasets.MedAESQA import get_pubmed_title_abstract
from tqdm import tqdm
import os
from Bio import Entrez

def get_pubmed_title_abstract(pmid):
    """
    根据 PMID 获取论文标题和摘要
    """
    # 必须提供 email，这是 NCBI 的规定，用于追踪滥用情况
    Entrez.email = "longyongchao@stud.tjut.edu.cn" 
    Entrez.api_key = "9d771126706090859b3e69abeda4c5cfe908"
    
    try:
        # 使用 efetch 获取数据，返回 XML 格式
        handle = Entrez.efetch(db="pubmed", id=pmid, retmode="xml")
        records = Entrez.read(handle)
        handle.close()

        # 解析返回的数据
        if not records['PubmedArticle']:
            print("未找到对应 PMID 的文章")
            return None

        article_data = records['PubmedArticle'][0]['MedlineCitation']['Article']
        
        # 1. 获取标题
        title = article_data.get('ArticleTitle', 'No Title')

        # 2. 获取摘要
        # 摘要可能是简单的字符串，也可能是结构化的列表（如 Background, Methods 等）
        abstract_text = ""
        if 'Abstract' in article_data and 'AbstractText' in article_data['Abstract']:
            abstract_source = article_data['Abstract']['AbstractText']
            
            parts = []
            for part in abstract_source:
                # 检查是否有标签（例如 "BACKGROUND", "METHODS"）
                if hasattr(part, 'attributes') and 'Label' in part.attributes:
                    parts.append(f"{part.attributes['Label']}: {part}")
                else:
                    parts.append(str(part))
            
            abstract_text = "\n".join(parts)
        else:
            abstract_text = "No Abstract Available"

        return {
            'pmid': pmid,
            'title': title,
            'abstract': abstract_text
        }

    except Exception as e:
        return {'error': str(e)}


expert_pubmed_ids_path = 'data/medaesqa/machine_generated_answers_pubmed_ids.jsonl'
expert_pubmed_ids = read_jsonl_to_list(expert_pubmed_ids_path)
machine_pubmed_ids_path = 'data/medaesqa/machine_generated_answers_pubmed_ids.jsonl'
machine_pubmed_ids = read_jsonl_to_list(machine_pubmed_ids_path)
pubmed_ids = list(set(expert_pubmed_ids + machine_pubmed_ids))
print(f'Total unique PubMed IDs: {len(pubmed_ids)}')

medaesqa_context_path = 'data/medaesqa/medaesqa_context.jsonl'

res = []
existing_ids = []

# 检查medaesqa_context_path文件是否存在，存在则读取已有数据，避免重复抓取
if os.path.exists(medaesqa_context_path):
    res = read_jsonl_to_list(medaesqa_context_path)
    existing_ids = set([item['pmid'] for item in res])
    print(f'Existing PubMed IDs in context file: {len(existing_ids)}')

for pmid in tqdm(pubmed_ids):
    if pmid in existing_ids:
        continue
    try:
        paper = get_pubmed_title_abstract(pmid)
        if paper:
            res.append(paper)
            save_jsonl(res, medaesqa_context_path)
    except Exception as e:
        print(f'Error fetching PMID {pmid}: {e}')

