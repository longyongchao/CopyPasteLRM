import bm25s
import Stemmer  # 显式导入 PyStemmer 库
from typing import List, Dict

class BM25Retriever:
    def __init__(self, corpus: List[Dict[str, str]]):
        self.corpus = corpus
        print(f"正在构建 BM25S 索引 (English)，共 {len(corpus)} 条文档...")
        
        # 1. 显式创建 PyStemmer 的提取器 (使用 'english' 算法)
        self.stemmer = Stemmer.Stemmer("english")

        # 2. 将实例化好的 stemmer 对象传给 tokenize
        # 提取 text 列表
        corpus_texts = [doc['text'] for doc in corpus]
        
        # 使用自定义的 stemmer 进行分词
        self.corpus_tokens = bm25s.tokenize(corpus_texts, stemmer=self.stemmer)
        
        # 3. 创建并索引
        self.retriever = bm25s.BM25()
        self.retriever.index(self.corpus_tokens)
        print("索引构建完成。")

    def retrieve(self, query: str, k: int = 5):
        # 检索时，必须使用同一个 stemmer 对象（或者相同算法的新对象）
        query_tokens = bm25s.tokenize([query], stemmer=self.stemmer)
        
        # k 不能超过语料库大小
        k = min(k, len(self.corpus))
        
        # retrieve 返回结果
        results, _ = self.retriever.retrieve(query_tokens, k=k)
        
        # results[0] 是第一个查询命中的文档索引
        top_k_indices = results[0]
        
        return [self.corpus[i] for i in top_k_indices]

if __name__ == "__main__":
    # 测试代码
    my_corpus = [
        {"id": "doc1", "text": "DeepSeek R1 is a large language model based on reinforcement learning"},
        {"id": "doc2", "text": "Machine learning requires a lot of data for training"},
        {"id": "doc3", "text": "The weather in Beijing is very nice today, suitable for visiting the Great Wall"},
        {"id": "doc4", "text": "Reinforcement learning algorithms include PPO and GRPO"},
    ]

    query = "GRPO reinforcement learning"

    retriever = BM25Retriever(my_corpus)
    res = retriever.retrieve(query, k=3)
    print("检索结果：")
    for item in res:
        print(item)