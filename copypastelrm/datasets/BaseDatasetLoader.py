from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from datasets import load_dataset
from tqdm import tqdm
import os
import json
import random
from copypastelrm.utils.dataset import NLPTool
# 假设 BM25Retriever 在这个路径，保持引用不变
from copypastelrm.utils.bm25 import BM25Retriever 

class BaseDatasetLoader(ABC):
    """
    数据集加载器的基类，提供统一的数据集加载、预处理、BM25索引构建及缓存管理接口。
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
        unanswerable: bool = False, 
    ):
        """
        初始化数据集加载器
        """
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
        
        # 构建缓存文件名
        base_name = self.dataset_path.replace('.json', '').replace('/', '-')
        subset_name = f"-{self.dataset_name}" if self.dataset_name else ""
        file_name = f"{base_name}{subset_name}-{self.split}-{self.distractor_docs}-{self.unanswerable}"
        self.cache_path = os.path.join(cache_dir, f"{file_name}.jsonl")

        if self.cache_path and not self.cache_path.endswith(".jsonl"):
            raise ValueError("cache_path must end with .jsonl")

        # -----------------------------------------------------------
        # Step 1: 加载数据 (Load Data)
        # 返回数据列表和来源标记 (是否来自缓存)
        # -----------------------------------------------------------
        self.dataset_list, self.is_from_cache = self.get_dataset()
        
        if not self.dataset_list:
            print("⚠️ Warning: Loaded dataset is empty.")

        # -----------------------------------------------------------
        # Step 2: 构建检索器 (Build Retriever)
        # 无论数据来自缓存还是原始源，为了支持外部调用 bm25，我们都需要构建索引。
        # construct_corpus 会根据 is_from_cache 自动决定如何解析数据。
        # -----------------------------------------------------------
        print('正在构建 BM25 语料库...')
        self.corpus = self.construct_corpus(self.dataset_list, is_formatted=self.is_from_cache)
        
        if not self.corpus:
            print("⚠️ Warning: Corpus is empty. BM25 index will fail.")
        else:
            print(f'语料库构建完成，共 {len(self.corpus)} 条文档，开始构建索引...')
            self.bm25 = BM25Retriever(self.corpus)

        # -----------------------------------------------------------
        # Step 3: 最终化数据集 (Finalize Dataset)
        # 如果来自缓存：直接映射为字典。
        # 如果是原始数据：执行 format_dataset 流程（含 BM25 检索干扰项）并缓存。
        # -----------------------------------------------------------
        if self.is_from_cache:
            print('✅ 检测到数据来自缓存，跳过格式化步骤，直接加载。')
            self.dataset = {sample["id"]: sample for sample in self.dataset_list}
        else:
            print('🔄 数据为原始格式，开始执行格式化与检索...')
            # 只有原始数据才调用 format_dataset，避免 KeyError
            self.dataset = self.format_dataset(self.dataset_list)

        # -----------------------------------------------------------
        # Step 4: 采样 (Optional Sampling)
        # -----------------------------------------------------------
        if 0 < max_samples < len(self.dataset_list):
            print(f"Sampling {max_samples} samples from {len(self.dataset_list)} total.")
            self.dataset_list = random.sample(self.dataset_list, max_samples)
            # 重建 dataset 字典映射
            self.dataset = {sample["id"]: sample for sample in self.dataset_list}

        # 最终一致性检查
        assert len(self.dataset_list) == len(self.dataset), "数据集列表和字典长度不一致"
        print('🎉 数据集准备就绪')

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
        
        print(f"数据集加载完成，共 {len(dataset)} 个样本")
        return list(dataset)

    def get_dataset(self) -> Tuple[List[Dict[str, Any]], bool]:
        """
        加载数据集。
        
        Returns:
            Tuple[List, bool]: (数据列表, 是否来自缓存)
        """
        # 1. 尝试读取缓存
        if (
            not self.reload
            and self.offline
            and self.cache_path
            and os.path.exists(self.cache_path)
        ):
            try:
                with open(self.cache_path, "r", encoding='utf-8') as f:
                    print(f"🎯 Loading dataset from cache: {self.cache_path}")
                    formatted_dataset_list = json.load(f)
                    
                    if self.filter_empty_answer:
                        formatted_dataset_list = self.get_non_empty_answer(formatted_dataset_list)
                        random.shuffle(formatted_dataset_list)
                    
                    return formatted_dataset_list, True
            except Exception as e:
                print(f"⚠️ 读取缓存失败: {e}，将回退到下载模式。")

        # 2. 如果无缓存或强制重载，下载原始数据
        origin_dataset = self.download_dataset()
        # 注意：这里我们不对原始数据做 filter_empty_answer，
        # 因为原始数据格式各异，'answers' 字段可能还未生成，过滤通常放在 format 之后或期间。
        return origin_dataset, False
    
    def format_dataset(self, origin_dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        格式化数据集（仅针对原始数据调用）。
        包含：格式转换 -> BM25检索干扰项 -> 保存缓存。
        """
        formatted_dataset_dict = {}
        formatted_dataset_list = []

        iterator = tqdm(origin_dataset, desc="Formatting dataset", unit="sample")

        for sample in iterator:
            if self.format:
                # 调用子类实现的 format_item 和基类的 construct_context_and_facts
                formatted_sample = self.format_sample(sample)
            else:
                formatted_sample = sample
            
            formatted_dataset_list.append(formatted_sample)
            formatted_dataset_dict[formatted_sample["id"]] = formatted_sample

        # 过滤与打乱
        if self.filter_empty_answer:
            formatted_dataset_list = self.get_non_empty_answer(formatted_dataset_list)
            # 更新 dict 以匹配 filter 后的 list
            formatted_dataset_dict = {sample["id"]: sample for sample in formatted_dataset_list}
            random.shuffle(formatted_dataset_list)
        
        # 更新类成员
        self.dataset_list = formatted_dataset_list

        # 保存缓存
        if self.offline and self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "w", encoding='utf-8') as f:
                print(f"Saving formatted dataset to cache: {self.cache_path}")
                # 保存的是 List
                json.dump(
                    formatted_dataset_list,
                    f,
                    ensure_ascii=False,
                    indent=4,
                )

        return formatted_dataset_dict
    
    def get_non_empty_answer(self, data: list) -> list:
        """过滤掉答案为空的样本"""
        return [
            sample for sample in data 
            if "answers" in sample 
            and isinstance(sample["answers"], list) 
            and len(sample["answers"]) > 0 
            and str(sample["answers"][0]).strip() != ""
        ]

    def format_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        格式化单个数据样本：原始数据 -> 中间格式 -> 添加BM25上下文 -> 最终格式
        """
        # 1. 调用子类转换逻辑 (Raw -> Standard Schema)
        item = self.format_item(sample)
        
        # 2. 构建上下文 (Retrieval & Distractors)
        context, facts = self.construct_context_and_facts(item)

        # 3. 组装最终样本
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

    # ------------------------------------------------------------------------
    # Abstract Methods
    # ------------------------------------------------------------------------
    @abstractmethod
    def format_item(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        [子类必须实现]
        将原始数据样本转换为包含 id, query, answers, corpus 的标准中间格式。
        """
        raise NotImplementedError("子类必须实现 format_item 方法")

    # ------------------------------------------------------------------------
    # Corpus & Context Construction
    # ------------------------------------------------------------------------
    @staticmethod
    def string_context(title: str, sentences: List[str]) -> str:
        return "###" + str(title).upper() + "\n" + " ".join(sentences)
    
    @staticmethod
    def string_sub_id(id: str, idx: int) -> str:
        return f"{id}___{idx}"

    def construct_corpus(self, data: List[Dict[str, Any]], is_formatted: bool = False) -> List[Dict[str, Any]]:
        """
        构建用于 BM25 检索的语料库。
        
        Args:
            data: 数据列表
            is_formatted: 如果为 True，表示 data 已经是格式化后的（包含 corpus 字段），
                          无需再次调用 format_item。
        """
        global_corpus = []
        
        # 使用 set 进行去重，key 为 text 内容
        seen_texts = set()

        for sample in tqdm(data, desc="Constructing corpus"):
            # 策略模式：根据数据来源决定如何解析
            if is_formatted:
                # 来自缓存，直接使用结构化数据
                if 'corpus' in sample:
                    _id = sample['id']
                    corpus_items = sample['corpus']
                else:
                    # 异常情况：缓存数据结构不完整，跳过
                    continue
            else:
                # 来自原始源，需要转换
                formated_item = self.format_item(sample) 
                _id = formated_item["id"]
                corpus_items = formated_item['corpus']
            
            # 展平 corpus
            for idx, context in enumerate(corpus_items):
                text = self.string_context(context['title'], context['sentences'])
                
                # 内存优化：直接在循环中去重，避免构建过大的列表后再去重
                if text not in seen_texts:
                    seen_texts.add(text)
                    global_corpus.append({
                        "id": self.string_sub_id(_id, idx),
                        "text": text,
                    })
        
        return global_corpus

    def construct_context_and_facts(self, format_item: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """
        基于 Gold Context 和 BM25 检索构建最终的 context 和 facts。
        """
        _id = format_item["id"]
        query = format_item['query']
        single_corpus = format_item['corpus']

        gold_context = []
        gold_ctx_ids = set() # 使用 set 加速查找
        facts = []
        
        # 1. 提取 Gold Context
        for idx, item in enumerate(single_corpus):
            if item.get('facts'): # 如果有 facts，视作 gold
                ctx_str = self.string_context(item['title'], item['sentences'])
                gold_context.append(ctx_str)
                gold_ctx_ids.add(self.string_sub_id(_id, idx))
                facts.extend(item['facts'])
        
        distractor_context = []
        
        # 2. BM25 检索干扰项
        if self.distractor_docs > 0:
            # 检索数量 = 需要的干扰项 + 已有的Gold项 + 缓冲(10)
            k_val = self.distractor_docs + len(gold_context) + 10
            candidate_distractor_context = self.bm25.retrieve(query, k=k_val)

            distractor_count = 0
            for item in candidate_distractor_context:
                # 排除已经是 gold 的文档
                if item['id'] not in gold_ctx_ids and item['text'] not in gold_context:
                    distractor_context.append(item['text'])
                    distractor_count += 1
                    if distractor_count >= self.distractor_docs:
                        break
        
        # 3. 合并与混洗
        if self.unanswerable:
            context = distractor_context
            facts = [] # 不可回答模式下清除 facts
        else:
            context = gold_context + distractor_context

        # 使用局部随机种子或全局种子，这里保持原有逻辑
        # 建议：如果希望多次运行不固定，可以去掉 seed(42)，或者在 init 里传入 seed
        rng = random.Random(42) 
        rng.shuffle(context)

        return context, facts

    # ------------------------------------------------------------------------
    # Utility Methods
    # ------------------------------------------------------------------------
    def get_length(self) -> int:
        return len(self.dataset)

    def get_sample(self, sample_id=None) -> Dict[str, Any]:
        if not self.dataset:
            return None
        
        if sample_id:
            return self.dataset.get(sample_id)
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
        # 截断过长的 context 显示，避免刷屏
        print(f"Context (Preview): {sample['context'][:500]}..." if len(sample['context']) > 500 else f"Context: {sample['context']}")
        print("-" * 20)
        print(f"Answers: {sample['answers']}")
        print("-" * 20)
        if "facts" in sample:
            print(f"Supporting Facts: {sample['facts']}")
        
        return sample