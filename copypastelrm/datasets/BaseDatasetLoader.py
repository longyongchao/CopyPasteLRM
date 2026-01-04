import gzip  # <--- 新增引入
import os
import json
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from datasets import load_dataset
from tqdm import tqdm
from copypastelrm.utils.dataset import NLPTool
from copypastelrm.utils.bm25 import BM25Retriever 

class BaseDatasetLoader(ABC):
    def __init__(
        self,
        dataset_path: str,
        split: str,
        cache_dir: str = "/tmp/copypastelrm/cache/",
        dataset_name: Optional[str] = None,
        offline: bool = True,
        reload: bool = False,
        format: bool = True,
        max_samples: int = -1,
        filter_empty_answer: bool = True,
        distractor_docs: int = 8,
        unanswerable: bool = False, 
    ):
        self.nlp = NLPTool()
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.split = split
        self.offline = offline
        self.reload = reload
        self.format = format
        self.filter_empty_answer = filter_empty_answer
        self.unanswerable = unanswerable
        self.distractor_docs = distractor_docs
        
        # -----------------------------------------------------------
        # 修改点 1: 缓存文件名后缀改为 .jsonl.gz
        # -----------------------------------------------------------
        base_name = self.dataset_path.replace('.json', '').replace('/', '-')
        subset_name = f"-{self.dataset_name}" if self.dataset_name else ""
        file_name = f"{base_name}{subset_name}-{self.split}-noise_{self.distractor_docs}-{'unanswerable' if self.unanswerable else 'answerable'}"

        # 改用 .jsonl.gz
        self.cache_path = os.path.join(cache_dir, f"{file_name}.jsonl.gz")

        # -----------------------------------------------------------
        # Step 1: 加载数据
        # -----------------------------------------------------------
        self.dataset_list, self.is_from_cache = self.get_dataset()
        
        if not self.dataset_list:
            print("⚠️ Warning: Loaded dataset is empty.")

        # -----------------------------------------------------------
        # Step 2: 构建检索器
        # -----------------------------------------------------------
        if not self.is_from_cache:
            print('正在构建 BM25 语料库...')
            self.corpus = self.construct_corpus(self.dataset_list, is_formatted=self.is_from_cache)
            if not self.corpus:
                print("⚠️ Warning: Corpus is empty. BM25 index will fail.")
            else:
                print(f'语料库构建完成，共 {len(self.corpus)} 条文档，开始构建索引...')
                self.bm25 = BM25Retriever(self.corpus)

        # -----------------------------------------------------------
        # Step 3: 最终化数据集
        # -----------------------------------------------------------
        if self.is_from_cache:
            print('✅ 检测到数据来自缓存 (Compressed)，跳过格式化步骤，直接加载。')
            self.dataset = {sample["id"]: sample for sample in self.dataset_list}
        else:
            print('🔄 数据为原始格式，开始执行格式化与检索...')
            self.dataset = self.format_dataset(self.dataset_list)

        # -----------------------------------------------------------
        # Step 4: 采样
        # -----------------------------------------------------------
        if 0 < max_samples < len(self.dataset_list):
            print(f"Sampling {max_samples} samples from {len(self.dataset_list)} total.")
            self.dataset_list = random.sample(self.dataset_list, max_samples)
            self.dataset = {sample["id"]: sample for sample in self.dataset_list}

        assert len(self.dataset_list) == len(self.dataset), "数据集列表和字典长度不一致"
        print('🎉 数据集准备就绪')

    # ... download_dataset 保持不变 ...
    def download_dataset(self) -> List[Dict[str, Any]]:
        # (代码省略，保持原样)
        print(f"正在加载 {self.dataset_path} 数据集...")
        if self.dataset_name:
            print(f"数据集子集: {self.dataset_name}")
        print(f"数据分割: {self.split}")

        if self.dataset_name:
            dataset = load_dataset(
                path=self.dataset_path, name=self.dataset_name, split=self.split
            )
        else:
            dataset = load_dataset(path=self.dataset_path, split=self.split)
        
        print(f"数据集加载完成，共 {len(dataset)} 个样本")
        return list(dataset)

    def get_dataset(self) -> Tuple[List[Dict[str, Any]], bool]:
        """
        加载数据集 (支持 gzip 读取)。
        """
        if (
            not self.reload
            and self.offline
            and self.cache_path
            and os.path.exists(self.cache_path)
        ):
            try:
                # -----------------------------------------------------------
                # 修改点 2: 使用 gzip.open 读取，模式为 'rt' (read text)
                # -----------------------------------------------------------
                print(f"🎯 Loading dataset from compressed cache: {self.cache_path}")
                with gzip.open(self.cache_path, "rt", encoding='utf-8') as f:
                    formatted_dataset_list = json.load(f)
                    
                    if self.filter_empty_answer:
                        formatted_dataset_list = self.get_non_empty_answer(formatted_dataset_list)
                        random.shuffle(formatted_dataset_list)
                    
                    return formatted_dataset_list, True
            except Exception as e:
                print(f"⚠️ 读取缓存失败: {e}，将回退到下载模式。")

        # 无缓存或强制刷新
        origin_dataset = self.download_dataset()
        return origin_dataset, False
    
    def format_dataset(self, origin_dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        格式化数据集并保存 (支持 gzip 写入)。
        """
        formatted_dataset_dict = {}
        formatted_dataset_list = []

        iterator = tqdm(origin_dataset, desc="Formatting dataset", unit="sample")

        for sample in iterator:
            if self.format:
                formatted_sample = self.format_sample(sample)
            else:
                formatted_sample = sample
            
            formatted_dataset_list.append(formatted_sample)
            formatted_dataset_dict[formatted_sample["id"]] = formatted_sample

        if self.filter_empty_answer:
            formatted_dataset_list = self.get_non_empty_answer(formatted_dataset_list)
            formatted_dataset_dict = {sample["id"]: sample for sample in formatted_dataset_list}
            random.shuffle(formatted_dataset_list)
        
        self.dataset_list = formatted_dataset_list

        if self.offline and self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            # -----------------------------------------------------------
            # 修改点 3: 使用 gzip.open 写入，模式为 'wt' (write text)
            # -----------------------------------------------------------
            print(f"Saving formatted dataset to compressed cache: {self.cache_path}")
            with gzip.open(self.cache_path, "wt", encoding='utf-8') as f:
                json.dump(
                    formatted_dataset_list,
                    f,
                    ensure_ascii=False,
                    indent=4, # 如果为了极致空间，可以去掉 indent=4，变成紧凑格式
                )

        return formatted_dataset_dict

    # ... 其余函数 (get_non_empty_answer, format_sample, format_item, construct_corpus 等) 保持完全不变 ...
    def get_non_empty_answer(self, data: list) -> list:
        return [
            sample for sample in data 
            if "answers" in sample 
            and isinstance(sample["answers"], list) 
            and len(sample["answers"]) > 0 
            and str(sample["answers"][0]).strip() != ""
        ]

    def format_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        item = self.format_item(sample)
        context, facts = self.construct_context_and_facts(item)
        formatted_sample = {
            "id": item['id'],
            "query": item['query'],
            "answers": item['answers'],
            "context": "\n\n".join(context),
            "facts": facts,
            "corpus": item['corpus'],
            "extra": item.get('extra', None),
            "dataset": self.dataset_path if 'dataset' not in sample else sample['dataset'],
        }
        return formatted_sample

    @abstractmethod
    def format_item(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("子类必须实现 format_item 方法")

    @staticmethod
    def string_context(title: str, sentences: List[str]) -> str:
        return "###" + str(title).upper() + "\n" + " ".join(sentences)
    
    @staticmethod
    def string_sub_id(id: str, idx: int) -> str:
        return f"{id}___{idx}"

    def construct_corpus(self, data: List[Dict[str, Any]], is_formatted: bool = False) -> List[Dict[str, Any]]:
        global_corpus = []
        seen_texts = set()

        for sample in tqdm(data, desc="Constructing corpus"):
            if is_formatted:
                if 'corpus' in sample:
                    _id = sample['id']
                    corpus_items = sample['corpus']
                else:
                    continue
            else:
                formated_item = self.format_item(sample) 
                _id = formated_item["id"]
                corpus_items = formated_item['corpus']
            
            for idx, context in enumerate(corpus_items):
                text = self.string_context(context['title'], context['sentences'])
                if text not in seen_texts:
                    seen_texts.add(text)
                    global_corpus.append({
                        "id": self.string_sub_id(_id, idx),
                        "text": text,
                    })
        return global_corpus

    def construct_context_and_facts(self, format_item: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        _id = format_item["id"]
        query = format_item['query']
        single_corpus = format_item['corpus']
        gold_context = []
        gold_ctx_ids = set()
        facts = []
        for idx, item in enumerate(single_corpus):
            if item.get('facts'):
                ctx_str = self.string_context(item['title'], item['sentences'])
                gold_context.append(ctx_str)
                gold_ctx_ids.add(self.string_sub_id(_id, idx))
                facts.extend(item['facts'])
        
        distractor_context = []
        if self.distractor_docs > 0:
            k_val = self.distractor_docs + len(gold_context) + 10
            candidate_distractor_context = self.bm25.retrieve(query, k=k_val)
            distractor_count = 0
            for item in candidate_distractor_context:
                if item['id'] not in gold_ctx_ids and item['text'] not in gold_context:
                    distractor_context.append(item['text'])
                    distractor_count += 1
                    if distractor_count >= self.distractor_docs:
                        break
        
        if self.unanswerable:
            context = distractor_context
            facts = []
        else:
            context = gold_context + distractor_context

        rng = random.Random(42) 
        rng.shuffle(context)
        return context, facts

    def get_length(self) -> int:
        return len(self.dataset)

    def get_sample(self, sample_id=None) -> Dict[str, Any]:
        if not self.dataset: return None
        if sample_id: return self.dataset.get(sample_id)
        else:
            sample_id = random.choice(list(self.dataset.keys()))
            return self.dataset[sample_id]

    def random_sample(self) -> Dict[str, Any]:
        if not self.dataset:
            print("Dataset is empty.")
            return {}
        sample_id = random.choice(list(self.dataset.keys()))
        sample = self.dataset[sample_id]
        print(f"ID: {sample_id}")
        print("-" * 20)
        print(f"Query: {sample['query']}")
        print("-" * 20)
        print(f"Context (Preview): {sample['context'][:500]}..." if len(sample['context']) > 500 else f"Context: {sample['context']}")
        print("-" * 20)
        print(f"Answers: {sample['answers']}")
        print("-" * 20)
        if "facts" in sample:
            print(f"Supporting Facts: {sample['facts']}")
        return sample