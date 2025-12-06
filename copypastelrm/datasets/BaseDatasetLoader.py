from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datasets import load_dataset
from tqdm import tqdm
import os
import json
import random


class BaseDatasetLoader(ABC):
    """
    数据集加载器的基类，提供统一的数据集加载和预处理接口
    """
    
    def __init__(
        self, 
        dataset_path: str, 
        split: str, 
        cache_path: str,  # 缓存路径
        dataset_name: Optional[str] = None, 
        offline: bool = True,
    ):
        """
        初始化数据集加载器
        
        Args:
            dataset_path: HuggingFace 数据集路径
            dataset_name: 数据集子集名称（可选）
            split: 数据集分割（默认为 validation）
            offline: 是否离线模式
        """
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.split = split
        self.offline = offline
        self.cache_path = cache_path

        # 检查cache_path是否以.jsonl结尾
        if self.cache_path:
            if not self.cache_path.endswith('.jsonl'):
                raise ValueError("cache_path must end with .jsonl")

        self.dataset = self.load_dataset()
    
    def load_dataset(self) -> List[Dict[str, Any]]:
        """
        加载数据集
        
        Args:
            origin: 是否返回原始数据集（不进行格式化）
            
        Returns:
            List[Dict]: 包含数据样本的列表
        """

        if self.offline and self.cache_path and os.path.exists(self.cache_path):
            with open(self.cache_path, 'r') as f:
                print(f"🎯Loading dataset from cache: {self.cache_path}")
                dataset_list = json.load(f)
                dataset_dict = {}
                for sample in dataset_list:
                    dataset_dict[sample["id"]] = sample
                return dataset_dict
        
        print(f"正在加载 {self.dataset_path} 数据集...")
        if self.dataset_name:
            print(f"数据集子集: {self.dataset_name}")
        print(f"数据分割: {self.split}")
        
        if self.dataset_name:
            dataset = load_dataset(path=self.dataset_path, name=self.dataset_name, split=self.split)
        else:
            dataset = load_dataset(path=self.dataset_path, split=self.split)
        
        dataset = list(dataset)
        
        formatted_dataset = {}
        
        
        iterator = tqdm(dataset, desc="Formatting dataset", unit="sample")
        
        for sample in iterator:
            formatted_sample = self.format_sample(sample)
            formatted_dataset[formatted_sample["id"]] = formatted_sample
        
        # 如果开启离线模式并且指定了缓存路径，则将格式化后的数据集保存到缓存文件
        if self.offline and self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, 'w') as f:
                print(f"Saving formatted dataset to cache: {self.cache_path}")
                json.dump(list(formatted_dataset.values()), f, ensure_ascii=False, indent=4)

        return formatted_dataset
    
    def format_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        格式化单个数据样本
        
        Args:
            sample: 原始数据样本
            
        Returns:
            Dict[str, Any]: 格式化后的数据样本
        """
        return {
            "id": self.format_id(sample),
            "query": self.format_query(sample),
            "context": self.format_context(sample),
            "answer": self.format_answer(sample),
            "sfs": self.format_supporting_facts(sample),
        }

    def format_id(self, sample: Dict[str, Any]) -> str:
        """
        格式化样本 ID
        
        Args:
            sample: 原始数据样本
            
        Returns:
            str: 样本 ID
        """
        if "id" in sample:
            return sample["id"]
        else:
            raise NotImplementedError("子类必须实现 format_id 方法")

    def format_query(self, sample: Dict[str, Any]) -> str:
        """
        格式化查询字段
        
        Args:
            sample: 原始数据样本
            
        Returns:
            str: 查询文本
        """
        if 'question' in sample:
            return sample["question"]
        elif "query" in sample:
            return sample["query"]
        else:
            raise NotImplementedError("子类必须实现 format_query 方法")
    
    @abstractmethod
    def format_context(self, sample: Dict[str, Any]) -> str:
        """
        标准的上下文格式化方法（适用于包含 title 和 sentences 的上下文）
        
        Args:
            context: 包含 title 和 sentences 的字典
            
        Returns:
            str: 格式化后的上下文文本
        """
        if 'context' in sample and isinstance(sample['context'], str):
            return sample['context']
        else:
            raise NotImplementedError("子类必须实现 format_context 方法")
    
    @abstractmethod
    def format_supporting_facts(self, sample: Dict[str, Any]) -> List[str]:
        """
        段落式的上下文格式化方法（适用于包含段落列表的上下文）
        
        Args:
            context: 包含段落的列表，每个段落包含 title 和 paragraph_text
            
        Returns:
            str: 格式化后的上下文文本
        """
        pass

    def format_answer(self, sample: Dict[str, Any]) -> List[str]:
        """
        格式化答案字段
        
        Args:
            sample: 原始数据样本
            
        Returns:
            str: 格式化后的答案文本
        """
        if 'answer' in sample:
            return [sample["answer"]]
        else:
            raise NotImplementedError("子类必须实现 format_answer 方法")
    
    def get_length(self) -> int:
        """
        获取数据集样本数量
        
        Returns:
            int: 数据集样本数量
        """
        return len(self.dataset)
    
    def get_sample(self, sample_id: str) -> Dict[str, Any]:
        """
        根据样本 ID 获取样本
        
        Args:
            sample_id: 样本 ID
            
        Returns:
            Dict[str, Any]: 样本数据
        """
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
        print('-'* 20)
        print(f"Query: {sample['query']}")
        print('-'* 20)
        print(f"Context: {sample['context']}")
        print('-'* 20)
        print(f"Answer: {sample['answer']}")
        print('-'* 20)
        if 'sfs' in sample:
            print(f"Supporting Facts: {sample['sfs']}")






