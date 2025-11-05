"""
测试 LengthValidator 类的各个函数
"""

import os
from sys import path

import pytest

from dataset import HotpotQAContext, HotpotQASupportingFacts
from reward import LengthValidator

# 添加项目根目录到Python路径
path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def validator():
    """LengthValidator实例fixture"""
    return LengthValidator()


@pytest.fixture
def sample_context():
    """示例上下文数据"""
    return HotpotQAContext(
        {
            "title": ["Article 1", "Article 2", "Article 3"],
            "sentences": [
                [
                    "Paris is the capital city of France.",
                    "It has a population of about 2.2 million.",
                ],
                [
                    "The Eiffel Tower is a famous landmark in Paris.",
                    "It was built in 1889.",
                ],
                [
                    "The Louvre Museum is the world's largest art museum.",
                    "It houses thousands of artworks.",
                ],
            ],
        }
    )


@pytest.fixture
def sample_facts():
    """示例支持事实数据"""
    return HotpotQASupportingFacts(
        {
            "title": ["Article 1", "Article 2", "Article 3"],
            "sent_id": [0, 0, 0],  # 每篇文章的第0句
        }
    )


class TestLengthValidator:
    """测试LengthValidator类的所有函数"""

    # ============================================================================
    # 测试 get_facts_sentences 函数
    # ============================================================================

    def test_get_facts_sentences_valid_case(self, validator, sample_context, sample_facts):
        """测试获取facts句子的正常情况"""
        sentences = validator.get_facts_sentences(sample_context, sample_facts)
        expected = [
            "Paris is the capital city of France.",
            "The Eiffel Tower is a famous landmark in Paris.",
            "The Louvre Museum is the world's largest art museum.",
        ]
        assert sentences == expected

    def test_get_facts_sentences_empty_context(self, validator):
        """测试空上下文"""
        ctx = HotpotQAContext({"title": [], "sentences": []})
        facts = HotpotQASupportingFacts({"title": [], "sent_id": []})
        sentences = validator.get_facts_sentences(ctx, facts)
        assert sentences == []

    def test_get_facts_sentences_empty_facts(self, validator, sample_context):
        """测试空facts"""
        facts = HotpotQASupportingFacts({"title": [], "sent_id": []})
        sentences = validator.get_facts_sentences(sample_context, facts)
        assert sentences == []

    def test_get_facts_sentences_title_not_found(self, validator, sample_context):
        """测试facts中的标题在context中不存在"""
        facts = HotpotQASupportingFacts({"title": ["Non-existent Article"], "sent_id": [0]})
        sentences = validator.get_facts_sentences(sample_context, facts)
        assert sentences == []

    def test_get_facts_sentences_invalid_sent_id(self, validator, sample_context):
        """测试无效的句子ID"""
        facts = HotpotQASupportingFacts({"title": ["Article 1"], "sent_id": [999]})  # 超出范围的句子ID
        sentences = validator.get_facts_sentences(sample_context, facts)
        assert sentences == []

    def test_get_facts_sentences_mixed_valid_invalid(self, validator, sample_context):
        """测试混合有效和无效的facts"""
        facts = HotpotQASupportingFacts(
            {
                "title": ["Article 1", "Non-existent", "Article 2"],
                "sent_id": [0, 0, 999],  # 第二个和三个无效
            }
        )
        sentences = validator.get_facts_sentences(sample_context, facts)
        expected = ["Paris is the capital city of France."]
        assert sentences == expected

    def test_get_facts_sentences_different_sent_ids(self, validator, sample_context):
        """测试不同的句子ID"""
        facts = HotpotQASupportingFacts({"title": ["Article 1", "Article 2"], "sent_id": [1, 1]})  # 每篇文章的第1句
        sentences = validator.get_facts_sentences(sample_context, facts)
        expected = [
            "It has a population of about 2.2 million.",
            "It was built in 1889.",
        ]
        assert sentences == expected

    # ============================================================================
    # 测试 validate_answer_length 函数
    # ============================================================================

    def test_validate_answer_length_valid_case(self, validator):
        """测试有效的答案长度"""
        answer_content = "Paris"
        target_answer = "Paris"
        assert validator.validate_answer_length(answer_content, target_answer)

    def test_validate_answer_length_exactly_three_times(self, validator):
        """测试答案长度正好是目标答案的3倍"""
        answer_content = "Par"  # 长度为3
        target_answer = "P"  # 长度为1
        assert validator.validate_answer_length(answer_content, target_answer)

    def test_validate_answer_length_over_three_times(self, validator):
        """测试答案长度超过目标答案的3倍"""
        answer_content = "Paris is a beautiful city xxxxx"  # 长度为31
        target_answer = "Paris"  # 长度为5, 3倍为15
        assert not validator.validate_answer_length(answer_content, target_answer)

    def test_validate_answer_length_empty_target(self, validator):
        """测试空的目标答案"""
        answer_content = "some content"
        target_answer = ""
        assert not validator.validate_answer_length(answer_content, target_answer)  # 如果目标答案为空，则判定无效

    def test_validate_answer_length_empty_answer(self, validator):
        """测试空的答案内容"""
        answer_content = ""
        target_answer = "Paris"
        assert not validator.validate_answer_length(answer_content, target_answer)

    def test_validate_answer_length_both_empty(self, validator):
        """测试两者都为空"""
        answer_content = ""
        target_answer = ""
        assert not validator.validate_answer_length(answer_content, target_answer)

    def test_validate_answer_length_with_whitespace(self, validator):
        """测试包含空白字符的答案"""
        answer_content = "  Paris  "  # 长度为9
        target_answer = "Paris"  # 长度为5
        assert validator.validate_answer_length(answer_content, target_answer)

    # ============================================================================
    # 测试 validate_think_vs_answer_length 函数
    # ============================================================================

    def test_validate_think_vs_answer_length_valid_case(self, validator):
        """测试有效的think vs answer长度"""
        think_content = "This is a longer thinking process"
        answer_content = "short"
        assert validator.validate_think_vs_answer_length(think_content, answer_content)

    def test_validate_think_vs_answer_length_equal_length(self, validator):
        """测试think和answer长度相等"""
        think_content = "equal"
        answer_content = "equal"
        assert not validator.validate_think_vs_answer_length(think_content, answer_content)

    def test_validate_think_vs_answer_length_think_shorter(self, validator):
        """测试think比answer短"""
        think_content = "short"
        answer_content = "much longer answer"
        assert not validator.validate_think_vs_answer_length(think_content, answer_content)

    def test_validate_think_vs_answer_length_empty_think(self, validator):
        """测试空的think内容"""
        think_content = ""
        answer_content = "some answer"
        assert not validator.validate_think_vs_answer_length(think_content, answer_content)

    def test_validate_think_vs_answer_length_empty_answer(self, validator):
        """测试空的answer内容"""
        think_content = "some thinking"
        answer_content = ""
        assert validator.validate_think_vs_answer_length(think_content, answer_content)

    def test_validate_think_vs_answer_length_both_empty(self, validator):
        """测试两者都为空"""
        think_content = ""
        answer_content = ""
        assert not validator.validate_think_vs_answer_length(think_content, answer_content)

    def test_validate_think_vs_answer_length_with_newlines(self, validator):
        """测试包含换行符的内容"""
        think_content = "thinking\nwith\nnewlines"
        answer_content = "answer"
        assert validator.validate_think_vs_answer_length(think_content, answer_content)

    # ============================================================================
    # 测试 validate_think_vs_facts_length 函数
    # ============================================================================

    def test_validate_think_vs_facts_length_valid_case(self, validator, sample_context, sample_facts):
        """测试有效的think vs facts长度"""
        facts_sentences = validator.get_facts_sentences(sample_context, sample_facts)
        think_content = "This is a very long thinking process that should be longer than the average facts sentence length"
        assert validator.validate_think_vs_facts_length(think_content, facts_sentences)

    def test_validate_think_vs_facts_length_too_short(self, validator, sample_context, sample_facts):
        """测试think内容太短"""
        facts_sentences = validator.get_facts_sentences(sample_context, sample_facts)
        think_content = "short"
        assert not validator.validate_think_vs_facts_length(think_content, facts_sentences)

    def test_validate_think_vs_facts_length_empty_facts(self, validator):
        """测试空的facts句子"""
        facts_sentences = []
        think_content = "some thinking"
        assert not validator.validate_think_vs_facts_length(think_content, facts_sentences)

    def test_validate_think_vs_facts_length_empty_think(self, validator, sample_context, sample_facts):
        """测试空的think内容"""
        facts_sentences = validator.get_facts_sentences(sample_context, sample_facts)
        think_content = ""
        assert not validator.validate_think_vs_facts_length(think_content, facts_sentences)

    def test_validate_think_vs_facts_length_single_fact(self, validator):
        """测试单个fact的情况"""
        facts_sentences = ["This is a single fact sentence."]
        think_content = "This thinking is longer than the single fact"
        assert validator.validate_think_vs_facts_length(think_content, facts_sentences)

    def test_validate_think_vs_facts_length_variable_lengths(self, validator):
        """测试不同长度的facts句子"""
        facts_sentences = [
            "Short",
            "This is a medium length sentence",
            "This is a very long sentence that contains many words and characters",
        ]
        think_content = "This thinking should be longer than the average length"
        avg_length = sum(len(sent) for sent in facts_sentences) / len(facts_sentences)
        assert len(think_content) > avg_length
        assert validator.validate_think_vs_facts_length(think_content, facts_sentences)

    # ============================================================================
    # 测试 validate_copy_in_facts 函数
    # ============================================================================

    def test_validate_copy_in_facts_valid_case(self, validator, sample_context, sample_facts):
        """测试有效的copy在facts中"""
        facts_sentences = validator.get_facts_sentences(sample_context, sample_facts)
        think_content = "<copy>Paris is the capital city of France.</copy><copy>some other content</copy>"
        assert validator.validate_copy_in_facts(think_content, facts_sentences)

    def test_validate_copy_in_facts_no_copy_in_facts(self, validator, sample_context, sample_facts):
        """测试copy内容不在facts中"""
        facts_sentences = validator.get_facts_sentences(sample_context, sample_facts)
        think_content = "<copy>This content is not in any fact sentence.</copy><copy>Neither is this one.</copy>"
        assert not validator.validate_copy_in_facts(think_content, facts_sentences)

    def test_validate_copy_in_facts_empty_facts(self, validator):
        """测试空的facts句子"""
        facts_sentences = []
        think_content = "<copy>some content</copy>"
        assert not validator.validate_copy_in_facts(think_content, facts_sentences)

    def test_validate_copy_in_facts_no_copy_tags(self, validator, sample_context, sample_facts):
        """测试没有copy标签"""
        facts_sentences = validator.get_facts_sentences(sample_context, sample_facts)
        think_content = "no copy tags here"
        assert not validator.validate_copy_in_facts(think_content, facts_sentences)

    def test_validate_copy_in_facts_empty_copy_content(self, validator, sample_context, sample_facts):
        """测试空的copy内容"""
        facts_sentences = validator.get_facts_sentences(sample_context, sample_facts)
        think_content = "<copy></copy><copy></copy>"
        assert not validator.validate_copy_in_facts(think_content, facts_sentences)

    def test_validate_copy_in_facts_partial_match(self, validator, sample_context, sample_facts):
        """测试部分匹配"""
        facts_sentences = validator.get_facts_sentences(sample_context, sample_facts)
        think_content = "<copy>Paris</copy><copy>not in facts</copy>"
        assert validator.validate_copy_in_facts(think_content, facts_sentences)  # 至少一个匹配

    def test_validate_copy_in_facts_case_sensitive(self, validator, sample_context, sample_facts):
        """测试大小写敏感"""
        facts_sentences = validator.get_facts_sentences(sample_context, sample_facts)
        think_content = "<copy>paris is the capital city of france.</copy>"  # 小写
        assert not validator.validate_copy_in_facts(think_content, facts_sentences)  # 大小写不匹配

    def test_validate_copy_in_facts_whitespace_handling(self, validator, sample_context, sample_facts):
        """测试空白字符处理"""
        facts_sentences = validator.get_facts_sentences(sample_context, sample_facts)
        think_content = "<copy>  Paris is the capital city of France.  </copy>"
        assert validator.validate_copy_in_facts(think_content, facts_sentences)  # 应该strip后比较

    # ============================================================================
    # 测试 validate_copy_min_length 函数
    # ============================================================================

    def test_validate_copy_min_length_valid_case(self, validator, sample_context, sample_facts):
        """测试有效的copy最小长度"""
        facts_sentences = validator.get_facts_sentences(sample_context, sample_facts)
        think_content = """
            <copy>Paris is the capital city of France.</copy>
            something
            <copy>The Eiffel Tower is a famous landmark in Paris.</copy>"""
        assert validator.validate_copy_min_length(think_content, facts_sentences)

    def test_validate_copy_min_length_too_short(self, validator, sample_context, sample_facts):
        """测试copy内容太短"""
        facts_sentences = validator.get_facts_sentences(sample_context, sample_facts)
        think_content = "<copy>Paris</copy><copy>Tower</copy>"
        assert not validator.validate_copy_min_length(think_content, facts_sentences)

    def test_validate_copy_min_length_empty_facts(self, validator):
        """测试空的facts句子"""
        facts_sentences = []
        think_content = "<copy>some content</copy>"
        assert not validator.validate_copy_min_length(think_content, facts_sentences)

    def test_validate_copy_min_length_no_copy_tags(self, validator, sample_context, sample_facts):
        """测试没有copy标签"""
        facts_sentences = validator.get_facts_sentences(sample_context, sample_facts)
        think_content = "no copy tags here"
        assert validator.validate_copy_min_length(think_content, facts_sentences)  # 没有copy标签，应该返回True

    def test_validate_copy_min_length_mixed_lengths(self, validator, sample_context, sample_facts):
        """测试混合长度的copy内容"""
        facts_sentences = validator.get_facts_sentences(sample_context, sample_facts)
        think_content = "<copy>Paris</copy><copy>The Eiffel Tower is a famous landmark in Paris.</copy>"
        assert not validator.validate_copy_min_length(think_content, facts_sentences)  # 第一个太短

    def test_validate_copy_min_length_single_fact(self, validator):
        """测试单个fact的情况"""
        facts_sentences = ["Short"]
        think_content = "<copy>Short</copy>"
        # avg_length = 5, min_length = 5, threshold = min(2.5, 5) = 2.5
        assert validator.validate_copy_min_length(think_content, facts_sentences)

    def test_validate_copy_min_length_exact_threshold(self, validator):
        """测试正好达到阈值"""
        facts_sentences = ["Exactly ten", "Much longer sentence here"]
        think_content = "<copy>Exactly ten</copy>"
        # avg_length = (10 + 23) / 2 = 16.5, min_length = 10, threshold = min(8.25, 10) = 8.25
        assert validator.validate_copy_min_length(think_content, facts_sentences)

    def test_validate_copy_min_length_with_whitespace(self, validator, sample_context, sample_facts):
        """测试包含空白字符的copy内容"""
        facts_sentences = validator.get_facts_sentences(sample_context, sample_facts)
        think_content = "<copy>  Paris is the capital city of France.  </copy>"
        assert validator.validate_copy_min_length(think_content, facts_sentences)  # 应该strip后计算长度

    def test_validate_copy_min_length_empty_copy_content(self, validator, sample_context, sample_facts):
        """测试空的copy内容"""
        facts_sentences = validator.get_facts_sentences(sample_context, sample_facts)
        think_content = "<copy></copy><copy></copy>"
        assert not validator.validate_copy_min_length(think_content, facts_sentences)
