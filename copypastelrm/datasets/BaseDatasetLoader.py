from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datasets import load_dataset
from tqdm import tqdm
import os
import json
import random
from copypastelrm.utils.tokenizer import ChatTokenCounter
from copypastelrm.utils.bm25 import BM25Retriever
from copypastelrm.utils.dataset import NLPTool


class BaseDatasetLoader(ABC):
    """
    数据集加载器的基类，提供统一的数据集加载和预处理接口
    """

    def __init__(
        self,
        dataset_path: str,
        split: str,
        cache_dir: str = "data/cache/",  # 缓存路径
        dataset_name: Optional[str] = None,
        offline: bool = True,
        reload: bool = False,
        format: bool = True,
        max_samples: int = -1,
        filter_empty_answer: bool = True,
        distractor_docs: int = 8,
        unanswerable: bool = False, # 是否不包含gold context
        # max_input_tokens: int = 1024 * 24,
        # tokenizer_path: str = 'Qwen/Qwen2.5-3B-Instruct',
    ):
        """
        初始化数据集加载器

        Args:
            dataset_path: HuggingFace 数据集路径
            dataset_name: 数据集子集名称（可选）
            split: 数据集分割（默认为 validation）
            offline: 是否离线模式
        """
        self.nlp = NLPTool()
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.split = split
        self.offline = offline
        self.reload = reload
        self.format = format  # 是否格式化数据集
        if self.dataset_name:
            file_name = f"{self.dataset_path.replace('.json', '').replace('/', '-')}-{self.dataset_name}-{self.split}"
        else:
            file_name = f"{self.dataset_path.replace('.json', '').replace('/', '-')}-{self.split}"
        self.cache_path = cache_dir + f"{file_name}.jsonl"
        self.filter_empty_answer = filter_empty_answer

        # 检查cache_path是否以.jsonl结尾
        if self.cache_path:
            if not self.cache_path.endswith(".jsonl"):
                raise ValueError("cache_path must end with .jsonl")

        self.unanswerable = unanswerable

        self.dataset_list = None
        self.distractor_docs = distractor_docs

        self.origin_dataset = self.get_dataset()
        self.corpus = self.construct_corpus(self.origin_dataset)
        self.bm25 = BM25Retriever(self.corpus)
        self.dataset = self.format_dataset(self.origin_dataset)

        if max_samples > 0 and max_samples < self.get_length():
            self.dataset_list = random.sample(self.dataset_list, max_samples)
            self.dataset_dict = {}
            for sample in self.dataset_list:
                self.dataset_dict[sample["id"]] = sample
            self.dataset = self.dataset_dict

        assert len(self.dataset_list) == len(self.dataset), "数据集列表和字典长度不一致"

    def download_dataset(self) -> List[Dict[str, Any]]:
        """默认从huggingface下载数据"""
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

        dataset = list(dataset)

        return dataset

    def get_dataset(self):
        """
        加载数据集

        Args:
            origin: 是否返回原始数据集（不进行格式化）

        Returns:
            List[Dict]: 包含数据样本的列表
        """

        if (
            not self.reload
            and self.offline
            and self.cache_path
            and os.path.exists(self.cache_path)
        ):
            with open(self.cache_path, "r") as f:
                print(f"🎯Loading dataset from cache: {self.cache_path}")
                formatted_dataset_list = json.load(f)
                if self.filter_empty_answer:
                    formatted_dataset_list = self.get_non_empty_answer(formatted_dataset_list)
                    random.shuffle(formatted_dataset_list)
                self.dataset_list = formatted_dataset_list
                dataset_dict = {}
                for sample in formatted_dataset_list:
                    dataset_dict[sample["id"]] = sample
                return dataset_dict

        origin_dataset = self.download_dataset()
        return origin_dataset
    
    def format_dataset(self, origin_dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        格式化数据集

        Args:
            dataset: 原始数据集

        Returns:
            Dict[str, Any]: 格式化后的数据集
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

        # 如果开启离线模式并且指定了缓存路径，则将格式化后的数据集保存到缓存文件
        if self.offline and self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "w") as f:
                print(f"Saving formatted dataset to cache: {self.cache_path}")
                json.dump(
                    list(formatted_dataset_dict.values()),
                    f,
                    ensure_ascii=False,
                    indent=4,
                )

        if self.filter_empty_answer:
            formatted_dataset_list = self.get_non_empty_answer(formatted_dataset_list)
            random.shuffle(formatted_dataset_list)
            formatted_dataset_dict = {}
            for sample in formatted_dataset_list:
                formatted_dataset_dict[sample["id"]] = sample
        self.dataset_list = formatted_dataset_list

        return formatted_dataset_dict
    
    def get_non_empty_answer(self, data: list) -> list:
        return [sample for sample in data if len(sample["answers"]) > 0 and sample["answers"][0].strip() != ""]

    def format_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        格式化单个数据样本

        Args:
            sample: 原始数据样本

        Returns:
            Dict[str, Any]: 格式化后的数据样本
        """
        item = self.format_item(sample)
        context, facts = self.construct_context_and_facts(item)

        formatted_sample = {
            "id": item['id'],
            "query": item['query'],
            "answers": item['answers'],
            "context": "\n\n".join(context),
            "facts": facts,
            "corpus": item['corpus'],
            "extra": item['extra'],
            "dataset": self.dataset_path if 'dataset' not in sample else sample['dataset'],
        }

        return formatted_sample


    # 子类必须实现的函数: format_corpus
    @abstractmethod
    def format_item(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        格式化语料库，必须返回数据结构

        {
            "id": str,
            "query": str,
            "answers": List[str],
            "corpus": [
                {
                    "title": Optinal[str, None],
                    "sentences": List[str], # 必须经过分句
                    "facts": Optional[List[str], None], # 如果没有，则返回 None
                }
            ],
            "extra": Optional[Dict[str, Any], None]
        }

        Args:
            sample: 原始数据样本

        Returns:
            str: 上下文文本
        """
        raise NotImplementedError("子类必须实现 format_corpus 方法")

    @staticmethod
    def string_context(title: str, sentences: List[str]) -> str:
        return "###" + title.upper() + "\n" + " ".join(sentences)
    
    @staticmethod
    def string_sub_id(id: str, idx: int) -> str:
        return id + "___" + str(idx)

    def construct_corpus(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        global_corpus = []
        for sample in data:
            formated_item = self.format_item(sample) 
            _id = formated_item["id"]
            corpus = formated_item['corpus']
            for idx, context in enumerate(corpus):
                text = self.string_context(context['title'], context['sentences'])
                global_corpus.append({
                    "id": self.string_sub_id(_id, idx),
                    "text": text,
                })
        
        # 针对text进行去重，保留corpus的结构，但是如果text的内容重复了，则保留第一个，后面的均剔除
        global_corpus = list({item['text']: item for item in global_corpus}.values())

        return global_corpus

    
    def construct_context_and_facts(self, format_item: Dict[str, Any]) -> tuple[str, str, str, str, List[str]]:
        """
        Args:
            context: 包含 title 和 sentences 的字典

        Returns:

        """

        _id = format_item["id"]
        query = format_item['query']
        single_corpus = format_item['corpus']

        gold_context = []
        gold_ctx_ids = []
        facts = []
        for idx, item in enumerate(single_corpus):
            if item['facts']:
                gold_context.append(self.string_context(item['title'], item['sentences']))
                gold_ctx_ids.append(self.string_sub_id(_id, idx))
                facts.extend(item['facts'])
        
        distractor_context = []
        
        if self.distractor_docs > 0:
            candidate_distractor_context = self.bm25.retrieve(query, k=self.distractor_docs + len(gold_context) + 10)

            distractor_count = 0

            for item in candidate_distractor_context:
                if item['id'] not in gold_ctx_ids and item['text'] not in gold_context:
                    distractor_context.append(item['text'])
                    distractor_count += 1
                    if distractor_count >= self.distractor_docs:
                        break
        
        # 合并 gold context 和 distractor context，并且打乱
        if self.unanswerable:
            context = distractor_context
            facts = []
        else:
            context = gold_context + distractor_context

        random.seed(42)
        random.shuffle(context)

        return context, facts


    def get_length(self) -> int:
        """
        获取数据集样本数量

        Returns:
            int: 数据集样本数量
        """
        return len(self.dataset)

    def get_sample(self, sample_id=None) -> Dict[str, Any]:
        """
        根据样本 ID 获取样本

        Args:
            sample_id: 样本 ID

        Returns:
            Dict[str, Any]: 样本数据
        """
        if sample_id:
            return self.dataset[sample_id]
        else:
            sample_id = random.choice(list(self.dataset.keys()))
        return self.dataset.get(sample_id, None)

    def random_sample(self) -> Dict[str, Any]:
        """
        随机获取一个样本

        Returns:
            Dict[str, Any]: 随机样本
        """
        sample_id = random.choice(list(self.dataset.keys()))
        sample = self.dataset[sample_id]
        print(f"ID: {sample_id}")
        print("-" * 20)
        print(f"Query: {sample['query']}")
        print("-" * 20)
        print(f"Context: {sample['context']}")
        print("-" * 20)
        print(f"Answers: {sample['answers']}")
        print("-" * 20)
        if "facts" in sample:
            print(f"Supporting Facts: {sample['facts']}")
