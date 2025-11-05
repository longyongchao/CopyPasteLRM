"""
测试 CopyRewardCalculator 类的各个函数
"""

import os
from sys import path

import pytest

from reward import CopyRewardCalculator

# 添加项目根目录到Python路径
path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def calculator():
    """CopyRewardCalculator实例fixture"""
    return CopyRewardCalculator()


@pytest.fixture
def sample_facts_sentences():
    """示例facts句子"""
    return [
        "Paris is the capital city of France.",
        "The Eiffel Tower is a famous landmark in Paris.",
        "The Louvre Museum is the world's largest art museum.",
    ]


class TestCopyRewardCalculator:
    """测试CopyRewardCalculator类的所有函数"""

    # ============================================================================
    # 测试 calculate_copy_reward 函数
    # ============================================================================

    def test_calculate_copy_reward_perfect_match(self, calculator, sample_facts_sentences):
        """测试完美匹配的情况"""
        think_content = """
            <copy>Paris is the capital city of France.</copy>
            Paris is the capital city of France.
            Paris is the capital city of France.
            Paris is the capital city of France.
            Paris is the capital city of France.
            Paris is the capital city of France.
            Paris is the capital city of France.
        <copy>The Eiffel Tower is a famous landmark in Paris.</copy>"""
        reward = calculator.calculate_copy_reward(think_content, sample_facts_sentences)
        # 3个facts句子，每个有效copy奖励1/3分，2个有效copy共2/3分
        expected_reward = 2.0 / 3.0
        assert abs(reward - expected_reward) < 0.001

    def test_calculate_copy_reward_all_facts_matched(self, calculator, sample_facts_sentences):
        """测试所有facts都被匹配"""
        think_content = """
            <copy>Paris is the capital city of France.</copy>something
            <copy>The Eiffel Tower is a famous landmark in Paris.</copy>something
            <copy>The Louvre Museum is the world\'s largest art museum.</copy>"""
        reward = calculator.calculate_copy_reward(think_content, sample_facts_sentences)
        # 3个facts句子，3个有效copy，每个奖励1/3分，共1.0分
        assert abs(reward - 1.0) < 0.001

    def test_calculate_copy_reward_no_match(self, calculator, sample_facts_sentences):
        """测试没有匹配的情况"""
        think_content = """
            <copy>This content is not in any fact.</copy>
            <copy>Neither is this one.</copy>
        """
        reward = calculator.calculate_copy_reward(think_content, sample_facts_sentences)
        assert reward == 0.0

    def test_calculate_copy_reward_partial_match(self, calculator, sample_facts_sentences):
        """测试部分匹配"""
        think_content = """
            <copy>Paris is the capital city of France.</copy>
            something
            <copy>This is not in facts.</copy>
        """
        reward = calculator.calculate_copy_reward(think_content, sample_facts_sentences)
        # 只有第一个copy匹配，奖励1/3分
        expected_reward = 1.0 / 3.0
        assert abs(reward - expected_reward) < 0.001

    def test_calculate_copy_reward_empty_facts(self, calculator):
        """测试空的facts句子"""
        facts_sentences = []
        think_content = "<copy>some content</copy>"
        reward = calculator.calculate_copy_reward(think_content, facts_sentences)
        assert reward == 0.0

    def test_calculate_copy_reward_no_copy_tags(self, calculator, sample_facts_sentences):
        """测试没有copy标签"""
        think_content = "no copy tags here"
        reward = calculator.calculate_copy_reward(think_content, sample_facts_sentences)
        assert reward == 0.0

    def test_calculate_copy_reward_empty_copy_content(self, calculator, sample_facts_sentences):
        """测试空的copy内容"""
        think_content = "<copy></copy><copy></copy>"
        reward = calculator.calculate_copy_reward(think_content, sample_facts_sentences)
        assert reward == 0.0

    def test_calculate_copy_reward_length_sorting(self, calculator, sample_facts_sentences):
        """测试按长度排序（长copy优先）"""
        # 故意让短的copy在前面，长的在后面
        think_content = "<copy>Paris is the capital city</copy><copy>Paris is the capital city of France.</copy>"
        reward = calculator.calculate_copy_reward(think_content, sample_facts_sentences)
        # "Paris"是"Paris is the capital city of France."的子串，所以只有长的应该被奖励
        expected_reward = 1.0 / 3.0
        assert abs(reward - expected_reward) < 0.001

    def test_calculate_copy_reward_duplicate_content(self, calculator, sample_facts_sentences):
        """测试重复内容（一个copy是另一个的子串）"""
        think_content = "<copy>Paris is the capital city of France.</copy><copy>Paris is the capital</copy>"
        reward = calculator.calculate_copy_reward(think_content, sample_facts_sentences)
        # 只有第一个（较长的）copy应该被奖励
        expected_reward = 1.0 / 3.0
        assert abs(reward - expected_reward) < 0.001

    def test_calculate_copy_reward_multiple_same_fact(self, calculator, sample_facts_sentences):
        """测试多个copy匹配同一个fact"""
        think_content = "<copy>Paris is the capital city of France.</copy><copy>Paris is the capital</copy><copy>Paris</copy>"
        reward = calculator.calculate_copy_reward(think_content, sample_facts_sentences)
        # 只有最长的一个应该被奖励
        expected_reward = 1.0 / 3.0
        assert abs(reward - expected_reward) < 0.001

    def test_calculate_copy_reward_single_fact(self, calculator):
        """测试单个fact的情况"""
        facts_sentences = ["Paris is the capital city of France."]
        think_content = "<copy>Paris is the capital city of France.</copy>"
        reward = calculator.calculate_copy_reward(think_content, facts_sentences)
        # 1个fact，1个有效copy，奖励1.0分
        assert abs(reward - 1.0) < 0.001

    def test_calculate_copy_reward_overlapping_matches(self, calculator, sample_facts_sentences):
        """测试重叠的匹配"""
        think_content = "<copy>Eiffel Tower</copy>something text. <copy>Eiffel Tower is a famous landmark</copy>"
        reward = calculator.calculate_copy_reward(think_content, sample_facts_sentences)
        # 两个copy都匹配同一个fact，但第二个更长，应该只奖励第二个
        expected_reward = 1.0 / 3.0
        assert abs(reward - expected_reward) < 0.001

    def test_calculate_copy_reward_case_sensitive(self, calculator, sample_facts_sentences):
        """测试大小写敏感"""
        think_content = "<copy>paris is the capital city of france.</copy>"  # 小写
        reward = calculator.calculate_copy_reward(think_content, sample_facts_sentences)
        assert reward == 0.0  # 大小写不匹配

    def test_calculate_copy_reward_whitespace_handling(self, calculator, sample_facts_sentences):
        """测试空白字符处理"""
        think_content = "<copy>  Paris is the capital city of France.  </copy>"
        reward = calculator.calculate_copy_reward(think_content, sample_facts_sentences)
        # 应该strip后匹配
        expected_reward = 1.0 / 3.0
        assert abs(reward - expected_reward) < 0.001

    def test_calculate_copy_reward_partial_substring(self, calculator, sample_facts_sentences):
        """测试部分子串匹配"""
        think_content = "<copy>capital city of France</copy>"
        reward = calculator.calculate_copy_reward(think_content, sample_facts_sentences)
        # 应该匹配第一个fact
        expected_reward = 1.0 / 3.0
        assert abs(reward - expected_reward) < 0.001

    def test_calculate_copy_reward_single_character(self, calculator, sample_facts_sentences):
        """测试单字符匹配"""
        think_content = "<copy>P</copy>"
        reward = calculator.calculate_copy_reward(think_content, sample_facts_sentences)
        # 单字符应该匹配
        expected_reward = 1.0 / 3.0
        assert abs(reward - expected_reward) < 0.001

    def test_calculate_copy_reward_many_copies_few_facts(self, calculator):
        """测试很多copy但很少facts"""
        facts_sentences = ["Paris"]
        think_content = "<copy>Paris</copy><copy>Paris</copy><copy>Paris</copy>"
        reward = calculator.calculate_copy_reward(think_content, facts_sentences)
        # 虽然有3个copy，但只有1个fact，最多奖励1.0分
        # 而且由于重复内容，只有第一个被奖励
        assert abs(reward - 1.0) < 0.001

    def test_calculate_copy_reward_complex_scenario(self, calculator, sample_facts_sentences):
        """测试复杂场景"""
        think_content = """
        <copy>The Louvre Museum is the world\'s largest art museum.</copy>
        <copy>Paris</copy>
        <copy>The Eiffel Tower is a famous landmark in Paris.</copy>
        <copy>capital city</copy>
        <copy>not in facts</copy>
        """
        reward = calculator.calculate_copy_reward(think_content, sample_facts_sentences)
        # 应该奖励3个有效的copy（Louvre, Eiffel Tower, capital city）
        # Paris是capital city的子串，所以不会被重复奖励
        expected_reward = 3.0 / 3.0  # 3个有效copy，每个1/3分
        assert abs(reward - expected_reward) < 0.001

    def test_calculate_copy_reward_empty_think_content(self, calculator, sample_facts_sentences):
        """测试空的think内容"""
        think_content = ""
        reward = calculator.calculate_copy_reward(think_content, sample_facts_sentences)
        assert reward == 0.0

    def test_calculate_copy_reward_malformed_copy_tags(self, calculator, sample_facts_sentences):
        """测试格式错误的copy标签"""
        think_content = "<copy>unclosed copy<copy>another copy</copy>"
        reward = calculator.calculate_copy_reward(think_content, sample_facts_sentences)
        # 正则表达式应该能处理这种情况
        assert isinstance(reward, float)
        assert reward >= 0.0

    def test_calculate_copy_reward_nested_copy_tags(self, calculator, sample_facts_sentences):
        """测试嵌套的copy标签"""
        think_content = "<copy>outer<copy>inner</copy></copy>"
        reward = calculator.calculate_copy_reward(think_content, sample_facts_sentences)
        # 正则表达式是贪婪匹配，可能会产生意外结果
        # 我们只测试函数能正常执行
        assert isinstance(reward, float)
        assert reward >= 0.0

    def test_calculate_copy_reward_special_characters(self, calculator, sample_facts_sentences):
        """测试包含特殊字符的copy内容"""
        think_content = "<copy>Paris is the capital city of France! @#$%^&*()</copy>"
        reward = calculator.calculate_copy_reward(think_content, sample_facts_sentences)
        # 特殊字符应该被包含在匹配中
        expected_reward = 0.0 / 3.0
        assert abs(reward - expected_reward) < 0.001

    def test_calculate_copy_reward_unicode_characters(self, calculator):
        """测试Unicode字符"""
        facts_sentences = ["巴黎是法国的首都。"]
        think_content = "<copy>巴黎是法国的首都。</copy>"
        reward = calculator.calculate_copy_reward(think_content, facts_sentences)
        assert abs(reward - 1.0) < 0.001
