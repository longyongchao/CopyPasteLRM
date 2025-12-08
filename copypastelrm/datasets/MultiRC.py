import json
from typing import Any, Dict, List
from copypastelrm.datasets.BaseDatasetLoader import BaseDatasetLoader
import re


class MultiRC(BaseDatasetLoader):
    def __init__(self, reload: bool = False, max_samples: int = -1):
        super().__init__(
            dataset_path="data/multirc/dev.json",
            split="dev",
            name="multirc_dev",
            offline=True,
            reload=reload,
            max_samples=max_samples
        )
    
    def extract_sentences_with_regex(self, input_text: str) -> dict:

    
        # 1. 定义用于匹配和分割句子的模式：
        #    这里使用 <br> 标签（或其变体 <br/>）作为分割点。
        sentences = re.split(r'<br\s*/?>', input_text, flags=re.IGNORECASE)
        
        # 2. 定义用于清理每个句子的模式：
        #    这个模式匹配并捕获 (Sent N: ) 及其周围的 HTML 标签：
        #    ^.*?<b>Sent\s+\d+:\s*<\/b>\s*
        #    - ^: 匹配字符串的开头
        #    - .*: 匹配开头的任意字符（用于处理潜在的空白或换行符）
        #    - <b>Sent\s+\d+:\s*<\/b>: 匹配例如 <b>Sent 1: </b> 这样的标签结构
        clean_pattern = re.compile(r'^.*?<b>Sent\s+\d+:\s*<\/b>\s*', flags=re.IGNORECASE | re.DOTALL)
        
        output_dict = {}
        sentence_number = 1
        
        for sentence in sentences:
            # 清除首尾空白
            stripped_sentence = sentence.strip()
            
            if not stripped_sentence:
                # 跳过空行
                continue
                
            # 使用正则表达式清理句子开头的编号和标签
            cleaned_content = re.sub(clean_pattern, '', stripped_sentence)
            
            # 再次清除首尾空白，以防清理后留下多余空格
            final_content = cleaned_content.strip()
            
            if final_content:
                output_dict[str(sentence_number)] = final_content
                sentence_number += 1
                
        return output_dict

    def download_dataset(self):

        dataset = []

        with open(self.dataset_path, 'r') as f:
            data = json.load(f)['data']
        
        for item in data:
            context = item['paragraph']['text']
            context_dict = self.extract_sentences_with_regex(context)
            questions = item['paragraph']['questions']
            id = item['id']
            for q_item in questions:
                q_idx = q_item['idx']
                sub_id = f"{id}_{q_idx}"
                question = q_item['question']
                supporting_facts = q_item['sentences_used']
                answer_item = q_item['answers']
                answers = []
                for a_item in answer_item:
                    if a_item['isAnswer']:
                        answers.append(a_item['text'])
                
                dataset.append({
                    "id": sub_id,
                    "question": question,
                    "answers": answers,
                    "context": context_dict,
                    "supporting_facts": [str(sf+1) for sf in supporting_facts]
                })
        
        return dataset
                
    def format_answer(self, sample):
        return sample['answers']
    
    def format_context(self, sample: Dict[str, Any]) -> str:
        context_text = ""
        context = sample['context']

        for _, sent in context.items():
            context_text += f'\n{sent}'
        return context_text.strip()
    
    def format_supporting_facts(self, sample: Dict[str, Any]) -> List[str]:

        context = sample['context']
        supporting_facts = sample['supporting_facts']

        sfs = []

        for sf_idx in supporting_facts:
            sfs.append(context[sf_idx])

        return sfs
        

if __name__ == "__main__":
    loader = MultiRC(reload=True)
    dataset = loader.dataset
    print(f"数据集样本数: {len(loader.dataset)}")
    print(loader.get_sample('News/CNN/cnn-3b5bbf3ba31e4775140f05a8b59db55b22ee3e63.txt_0'))
