"""
测试 AnswerRewardCalculator 类的各个函数
"""

import os
from sys import path

import pytest

from reward import AnswerRewardCalculator

# 添加项目根目录到Python路径
path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def calculator():
    """AnswerRewardCalculator实例fixture"""
    return AnswerRewardCalculator()


class TestAnswerRewardCalculator:
    """测试AnswerRewardCalculator类的所有函数"""

    # ============================================================================
    # 测试 clean_text 函数
    # ============================================================================

    def test_clean_text_basic_case(self, calculator):
        """测试基本的文本清理"""
        text = "Hello, World!"
        result = calculator.clean_text(text)
        assert result == "hello world"

    def test_clean_text_with_punctuation(self, calculator):
        """测试包含标点符号的文本"""
        text = "Paris, France! Is it beautiful?"
        result = calculator.clean_text(text)
        assert result == "paris france is it beautiful"

    def test_clean_text_with_whitespace(self, calculator):
        """测试包含空白字符的文本"""
        text = "  Hello   World  \n\t  "
        result = calculator.clean_text(text)
        assert result == "hello world"

    def test_clean_text_empty_string(self, calculator):
        """测试空字符串"""
        text = ""
        result = calculator.clean_text(text)
        assert result == ""

    def test_clean_text_only_punctuation(self, calculator):
        """测试只有标点符号的文本"""
        text = "!@#$%^&*()"
        result = calculator.clean_text(text)
        assert result == ""

    def test_clean_text_only_whitespace(self, calculator):
        """测试只有空白字符的文本"""
        text = "   \n\t   "
        result = calculator.clean_text(text)
        assert result == ""

    def test_clean_text_mixed_case(self, calculator):
        """测试混合大小写的文本"""
        text = "HeLLo WoRLd"
        result = calculator.clean_text(text)
        assert result == "hello world"

    def test_clean_text_with_numbers(self, calculator):
        """测试包含数字的文本"""
        text = "Paris 2024 Olympics"
        result = calculator.clean_text(text)
        assert result == "paris 2024 olympics"

    def test_clean_text_with_special_chars(self, calculator):
        """测试包含特殊字符的文本"""
        text = "Hello@World.com"
        result = calculator.clean_text(text)
        assert result == "helloworldcom"

    def test_clean_text_newlines_and_tabs(self, calculator):
        """测试换行符和制表符"""
        text = "Line 1\nLine 2\tLine 3"
        result = calculator.clean_text(text)
        assert result == "line 1 line 2 line 3"

    def test_clean_text_multiple_punctuation(self, calculator):
        """测试多个连续标点符号"""
        text = "Hello!!! World???"
        result = calculator.clean_text(text)
        assert result == "hello world"

    # ============================================================================
    # 测试 calculate_answer_reward 函数
    # ============================================================================

    def test_calculate_answer_reward_exact_match(self, calculator):
        """测试精确匹配"""
        completion = "</think>some thinking<answer>Paris</answer>"
        target_answer = "Paris"
        reward = calculator.calculate_answer_reward(completion, target_answer)
        assert reward == 1.0

    def test_calculate_answer_reward_case_insensitive(self, calculator):
        """测试大小写不敏感"""
        completion = "</think>some thinking<answer>paris</answer>"
        target_answer = "PARIS"
        reward = calculator.calculate_answer_reward(completion, target_answer)
        assert reward == 1.0

    def test_calculate_answer_reward_with_punctuation(self, calculator):
        """测试包含标点符号"""
        completion = "</think>some thinking<answer>Paris!</answer>"
        target_answer = "Paris"
        reward = calculator.calculate_answer_reward(completion, target_answer)
        assert reward == 1.0

    def test_calculate_answer_reward_target_as_substring(self, calculator):
        """测试目标答案是生成答案的子串"""
        completion = "</think>some thinking<answer>The capital of France is Paris</answer>"
        target_answer = "Paris"
        reward = calculator.calculate_answer_reward(completion, target_answer)
        assert reward == 1.0

    def test_calculate_answer_reward_generated_as_substring(self, calculator):
        """测试生成答案是目标答案的子串"""
        completion = "</think>some thinking<answer>Paris</answer>"
        target_answer = "Paris is the capital"
        reward = calculator.calculate_answer_reward(completion, target_answer)
        assert reward == 0.0  # 目标答案不在生成答案中

    def test_calculate_answer_reward_no_match(self, calculator):
        """测试不匹配"""
        completion = "</think>some thinking<answer>London</answer>"
        target_answer = "Paris"
        reward = calculator.calculate_answer_reward(completion, target_answer)
        assert reward == 0.0

    def test_calculate_answer_reward_no_answer_tag(self, calculator):
        """测试没有answer标签"""
        completion = "</think>some thinking without answer tag"
        target_answer = "Paris"
        reward = calculator.calculate_answer_reward(completion, target_answer)
        assert reward == 0.0

    def test_calculate_answer_reward_empty_answer_tag(self, calculator):
        """测试空的answer标签"""
        completion = "</think>some thinking<answer></answer>"
        target_answer = "Paris"
        reward = calculator.calculate_answer_reward(completion, target_answer)
        assert reward == 0.0

    def test_calculate_answer_reward_empty_target(self, calculator):
        """测试空的目标答案"""
        completion = "</think>some thinking<answer>Paris</answer>"
        target_answer = ""
        reward = calculator.calculate_answer_reward(completion, target_answer)
        assert reward == 0.0  # 空目标答案不应该匹配

    def test_calculate_answer_reward_both_empty(self, calculator):
        """测试两者都为空"""
        completion = "</think>some thinking<answer></answer>"
        target_answer = ""
        reward = calculator.calculate_answer_reward(completion, target_answer)
        assert reward == 0.0

    def test_calculate_answer_reward_multiple_answer_tags(self, calculator):
        """测试多个answer标签（应该使用第一个）"""
        completion = "</think>some thinking<answer>Paris</answer><answer>London</answer>"
        target_answer = "Paris"
        reward = calculator.calculate_answer_reward(completion, target_answer)
        assert reward == 1.0

    def test_calculate_answer_reward_whitespace_handling(self, calculator):
        """测试空白字符处理"""
        completion = "</think>some thinking<answer>  Paris  </answer>"
        target_answer = "Paris"
        reward = calculator.calculate_answer_reward(completion, target_answer)
        assert reward == 1.0

    def test_calculate_answer_reward_complex_punctuation(self, calculator):
        """测试复杂标点符号"""
        completion = "</think>some thinking<answer>Paris, France! (capital)</answer>"
        target_answer = "Paris"
        reward = calculator.calculate_answer_reward(completion, target_answer)
        assert reward == 1.0

    def test_calculate_answer_reward_partial_match(self, calculator):
        """测试部分匹配"""
        completion = "</think>some thinking<answer>The capital city is Paris, France</answer>"
        target_answer = "Paris"
        reward = calculator.calculate_answer_reward(completion, target_answer)
        assert reward == 1.0

    def test_calculate_answer_reward_case_and_punctuation_mix(self, calculator):
        """测试大小写和标点符号混合"""
        completion = "</think>some thinking<answer>PARIS!</answer>"
        target_answer = "paris"
        reward = calculator.calculate_answer_reward(completion, target_answer)
        assert reward == 1.0

    def test_calculate_answer_reward_with_numbers(self, calculator):
        """测试包含数字"""
        completion = "</think>some thinking<answer>Paris 2024</answer>"
        target_answer = "Paris 2024"
        reward = calculator.calculate_answer_reward(completion, target_answer)
        assert reward == 1.0

    def test_calculate_answer_reward_malformed_answer_tag(self, calculator):
        """测试格式错误的answer标签"""
        completion = "</think>some thinking<answer>unclosed answer"
        target_answer = "unclosed answer"
        reward = calculator.calculate_answer_reward(completion, target_answer)
        assert reward == 0.0  # 正则表达式应该无法匹配

    def test_calculate_answer_reward_nested_tags(self, calculator):
        """测试嵌套标签"""
        completion = "</think>some thinking<answer>Paris <inner>France</inner></answer>"
        target_answer = "Paris"
        reward = calculator.calculate_answer_reward(completion, target_answer)
        assert reward == 1.0

    def test_calculate_answer_reward_unicode_content(self, calculator):
        """测试Unicode内容"""
        completion = "</think>some thinking<answer>巴黎</answer>"
        target_answer = "巴黎"
        reward = calculator.calculate_answer_reward(completion, target_answer)
        assert reward == 1.0

    def test_calculate_answer_reward_long_answer(self, calculator):
        """测试长答案"""
        completion = """<think>some thinking</think>
        <answer>This is a very long answer that contains the target word Paris somewhere in the middle of the sent</answer>"""
        target_answer = "Paris"
        reward = calculator.calculate_answer_reward(completion, target_answer)
        assert reward == 1.0

    def test_calculate_answer_reward_multiple_occurrences(self, calculator):
        """测试目标答案多次出现"""
        completion = "<think>some thinking</think><answer>Paris is beautiful, Paris is amazing</answer>"
        target_answer = "Paris"
        reward = calculator.calculate_answer_reward(completion, target_answer)
        assert reward == 1.0  # 只要出现一次就奖励

    def test_calculate_answer_reward_empty_completion(self, calculator):
        """测试空的completion"""
        completion = ""
        target_answer = "Paris"
        reward = calculator.calculate_answer_reward(completion, target_answer)
        assert reward == 0.0

    def test_calculate_answer_reward_only_answer_tag(self, calculator):
        """测试只有answer标签"""
        completion = "<answer>Paris</answer>"
        target_answer = "Paris"
        reward = calculator.calculate_answer_reward(completion, target_answer)
        assert reward == 1.0

    def test_calculate_answer_reward_with_newlines(self, calculator):
        """测试包含换行符"""
        completion = "<think>some thinking</think><answer>Paris\nFrance</answer>"
        target_answer = "Paris"
        reward = calculator.calculate_answer_reward(completion, target_answer)
        assert reward == 1.0

    def test_calculate_answer_reward_special_characters_in_target(self, calculator):
        """测试目标答案包含特殊字符"""
        completion = "<think>some thinking</think><answer>Hello@World</answer>"
        target_answer = "Hello@World"
        reward = calculator.calculate_answer_reward(completion, target_answer)
        assert reward == 1.0

    def test_calculate_answer_reward_mixed_language(self, calculator):
        """测试混合语言"""
        completion = "<think>some thinking</think><answer>Paris巴黎</answer>"
        target_answer = "Paris"
        reward = calculator.calculate_answer_reward(completion, target_answer)
        assert reward == 1.0
