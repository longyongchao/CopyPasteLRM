from typing import List

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
