from typing import List

import spacy

class NLPTool:
    def __init__(self):
        # 加载英文模型 (建议在全局加载一次，避免函数重复调用导致性能下降)
        # disable 参数禁用了不需要的组件（如命名实体识别），可以提高分句速度
        try:
            print('正在装载 spacy 模型...')
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
        except OSError:
            print("正在下载 spacy 模型...")
            from spacy.cli import download
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])

    def split_sentences_spacy(self, text):
        """
        使用 spaCy 对英文文本进行分句。
        
        Args:
            text (str): 输入的大段文本。
            
        Returns:
            list: 包含分句后字符串的列表。
        """
        # 处理文本
        # nlp() 会自动进行分词、词性标注和依存句法分析，从而确定句子边界
        doc = self.nlp(text)
        
        # doc.sents 是一个生成器，生成 Span 对象
        # 我们将其转换为文本并去除首尾空白
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        return sentences


class StringContainmentFilter:
    def __init__(self, data: List[str]):
        # 预处理：去重，防止 "a" 和 "a" 互相判断导致逻辑错误
        self.data = list(set(data))

    def filter_maximal_superstrings(self) -> List[str]:
        """
        对应你的 'Max过滤' (保留 'abcd')
        逻辑：保留那些【不是】其他任何元素子串的元素。
        术语：寻找偏序集中的极大元 (Maximal Elements)。
        """
        result = []
        # 按长度降序排序，这是一种简单的优化，但对于O(N^2)逻辑不是必须的
        # 主要是为了让结果看起来有序
        sorted_data = sorted(self.data, key=len, reverse=True)

        for current_str in sorted_data:
            is_substring = False
            for other_str in sorted_data:
                # 如果当前字符串是另一个字符串的一部分，且两者不相等
                if current_str != other_str and current_str in other_str:
                    is_substring = True
                    break
            
            # 只有当它不属于任何其他人时，才保留
            if not is_substring:
                result.append(current_str)
        
        return result

    def filter_minimal_substrings(self) -> List[str]:
        """
        对应你的 'Min过滤' (保留 'ab')
        逻辑：保留那些【不包含】其他任何元素的元素。
        术语：寻找偏序集中的极小元 (Minimal Elements)。
        """
        result = []
        # 按长度升序排序
        sorted_data = sorted(self.data, key=len)

        for current_str in sorted_data:
            contains_others = False
            for other_str in sorted_data:
                # 如果当前字符串包含另一个字符串，且两者不相等
                if current_str != other_str and other_str in current_str:
                    contains_others = True
                    break
            
            # 只有当它不包含任何其他人时，才保留
            if not contains_others:
                result.append(current_str)
        
        return result
