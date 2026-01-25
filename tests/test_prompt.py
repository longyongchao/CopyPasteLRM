"""
测试 copypastelrm.prompt.prompt 模块的功能

该测试文件涵盖所有15种prompt类型的测试，包括：
- 系统提示词的正确性和区分度
- 用户提示词的格式验证
- 特殊逻辑（解压缩、分段、重复）
- 边界情况和异常处理
"""

import pytest
from copypastelrm.prompt.prompt import create_prompt, SYSTEM_PROMPT


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def basic_question():
    """基础问题"""
    return "What is the capital of France?"


@pytest.fixture
def basic_context():
    """基础上下文（单行）"""
    return "Paris is the capital city of France."


@pytest.fixture
def multi_line_context():
    """多行上下文（用于测试分段）"""
    return """Line 1: First piece of information.
Line 2: Second piece of information.
Line 3: Third piece of information.
Line 4: Fourth piece of information."""


@pytest.fixture
def four_line_context():
    """恰好4行上下文（用于测试25%-75%分段）"""
    return """Line 1
Line 2
Line 3
Line 4"""


@pytest.fixture
def eight_line_context():
    """8行上下文（用于测试25%-75%分段：2行-4行-2行）"""
    return """Line 1
Line 2
Line 3
Line 4
Line 5
Line 6
Line 7
Line 8"""


@pytest.fixture
def three_line_context():
    """3行上下文（边界情况：n//4 = 0）"""
    return """Line 1
Line 2
Line 3"""


@pytest.fixture
def single_line_context():
    """单行上下文（最小情况）"""
    return "Only one line."


@pytest.fixture
def two_document_context():
    """两个文档的上下文"""
    return "###PARIS\nParis is the capital city\n\n###LONDON\nLondon is in England"


@pytest.fixture
def three_document_context():
    """三个文档的上下文"""
    return "###PARIS\nParis\n\n###LONDON\nLondon\n\n###TOKYO\nTokyo"


@pytest.fixture
def single_document_context():
    """单个文档的上下文"""
    return "###PARIS\nParis is the capital city"


@pytest.fixture
def context_with_special_chars():
    """包含特殊字符的上下文"""
    return "Price: $100, Discount: 20%, Date: 2024-01-15"


@pytest.fixture
def question_with_unicode():
    """包含Unicode字符的问题"""
    return "What is 巴黎?"


@pytest.fixture
def question_with_special_chars():
    """包含特殊字符的问题"""
    return "What is the price? (USD: $100)"


@pytest.fixture
def empty_string():
    """空字符串fixture"""
    return ""


@pytest.fixture
def whitespace_only_string():
    """只有空白字符的字符串"""
    return "   \n\t   "


@pytest.fixture
def all_prompt_types():
    """所有16种prompt类型"""
    return [
        "direct_inference", "cot",
        "rag", "rag_rep_2", "rag_rep_q",
        "rag_qcq", "rag_qcq2",
        "rag_q_int_q", "rag_q_int2_q", "rag_q_int_docs_q",
        "rag_decompressed", "rag_decompressed_rep_q",
        "ircot", "deepseek", "copypaste", "find_facts"
    ]


@pytest.fixture
def rag_prompt_types():
    """所有RAG类型（共用system prompt）"""
    return [
        "rag", "rag_rep_2", "rag_rep_q",
        "rag_qcq", "rag_qcq2",
        "rag_q_int_q", "rag_q_int2_q", "rag_q_int_docs_q",
        "rag_decompressed", "rag_decompressed_rep_q"
    ]


@pytest.fixture
def no_context_types():
    """无上下文类型"""
    return ["direct_inference", "cot"]


@pytest.fixture
def special_format_types():
    """特殊格式类型"""
    return ["deepseek", "copypaste", "find_facts"]


# ============================================================================
# TestSystemPrompts - 测试系统提示词的正确性和区分度
# ============================================================================

class TestSystemPrompts:
    """测试系统提示词的正确性和区分度"""

    def test_direct_inference_system_prompt_contains_direct_answer(self):
        """验证direct_inference包含"Directly answer"关键词"""
        system_prompt = SYSTEM_PROMPT["direct_inference"]
        assert "Directly answer" in system_prompt

    def test_cot_system_prompt_contains_step_by_step(self):
        """验证cot包含"think step by step"关键词"""
        system_prompt = SYSTEM_PROMPT["cot"]
        assert "think step by step" in system_prompt

    def test_rag_system_prompts_all_identical(self, rag_prompt_types):
        """验证所有RAG类型使用相同的system prompt"""
        prompts = [SYSTEM_PROMPT[t] for t in rag_prompt_types]
        # 所有prompts应该相同
        assert len(set(prompts)) == 1

    def test_rag_system_prompt_contains_based_on_context(self):
        """验证RAG类型包含"Based on the context"关键词"""
        system_prompt = SYSTEM_PROMPT["rag"]
        assert "Based on the context" in system_prompt

    def test_ircot_system_prompt_contains_based_on_context(self):
        """验证ircot包含"Based on the context" """
        system_prompt = SYSTEM_PROMPT["ircot"]
        assert "Based on the context" in system_prompt

    def test_ircot_system_prompt_contains_think_step_by_step(self):
        """验证ircot包含"think step by step" """
        system_prompt = SYSTEM_PROMPT["ircot"]
        assert "think step by step" in system_prompt

    def test_deepseek_system_prompt_contains_think_answer_tags(self):
        """验证deepseek包含<|think|>和<|answer|>标签说明"""
        system_prompt = SYSTEM_PROMPT["deepseek"]
        assert "<|think|>" in system_prompt
        assert "<|answer|>" in system_prompt

    def test_copypaste_system_prompt_contains_evidence_tag(self):
        """验证copypaste包含<|EVIDENCE|>标签说明"""
        system_prompt = SYSTEM_PROMPT["copypaste"]
        assert "<|EVIDENCE|>" in system_prompt

    def test_find_facts_system_prompt_contains_extract_facts(self):
        """验证find_facts包含"extract all specific facts"说明"""
        system_prompt = SYSTEM_PROMPT["find_facts"]
        assert "extract all specific facts" in system_prompt

    def test_find_facts_system_prompt_no_answer(self):
        """验证find_facts明确说明不要提供答案"""
        system_prompt = SYSTEM_PROMPT["find_facts"]
        assert "Do NOT provide the answer" in system_prompt

    def test_all_system_prompts_are_non_empty(self, all_prompt_types):
        """验证所有system prompt非空"""
        for prompt_type in all_prompt_types:
            system_prompt = SYSTEM_PROMPT[prompt_type]
            assert len(system_prompt) > 0

    def test_all_system_prompts_are_strings(self, all_prompt_types):
        """验证所有system prompt是字符串类型"""
        for prompt_type in all_prompt_types:
            system_prompt = SYSTEM_PROMPT[prompt_type]
            assert isinstance(system_prompt, str)

    def test_rag_system_prompts_contain_answer_format_instructions(self, rag_prompt_types):
        """验证RAG类型包含格式说明"""
        for prompt_type in rag_prompt_types:
            system_prompt = SYSTEM_PROMPT[prompt_type]
            assert "## IMPORTANT:" in system_prompt
            assert "Answer:" in system_prompt

    def test_direct_inference_no_context_prefix(self):
        """验证direct_inference没有"Based on the context"前缀"""
        system_prompt = SYSTEM_PROMPT["direct_inference"]
        assert "Based on the context" not in system_prompt

    def test_cot_no_context_prefix(self):
        """验证cot没有"Based on the context"前缀"""
        system_prompt = SYSTEM_PROMPT["cot"]
        assert "Based on the context" not in system_prompt

    def test_direct_inference_vs_cot_different(self):
        """验证direct_inference和cot的system prompt不同"""
        prompt_direct = SYSTEM_PROMPT["direct_inference"]
        prompt_cot = SYSTEM_PROMPT["cot"]
        assert prompt_direct != prompt_cot

    def test_rag_vs_direct_inference_different(self):
        """验证RAG类型和direct_inference的system prompt不同"""
        prompt_rag = SYSTEM_PROMPT["rag"]
        prompt_direct = SYSTEM_PROMPT["direct_inference"]
        assert prompt_rag != prompt_direct

    def test_deepseek_vs_copypaste_different(self):
        """验证deepseek和copypaste的system prompt不同"""
        prompt_deepseek = SYSTEM_PROMPT["deepseek"]
        prompt_copypaste = SYSTEM_PROMPT["copypaste"]
        assert prompt_deepseek != prompt_copypaste

    def test_copypaste_contains_evidence_deepseek_does_not(self):
        """验证copypaste包含证据标签说明而deepseek不包含"""
        prompt_copypaste = SYSTEM_PROMPT["copypaste"]
        prompt_deepseek = SYSTEM_PROMPT["deepseek"]
        assert "<|EVIDENCE|>" in prompt_copypaste
        assert "<|EVIDENCE|>" not in prompt_deepseek

    def test_find_facts_is_distinct_from_others(self, all_prompt_types):
        """验证find_facts与其他类型明显不同"""
        find_facts_prompt = SYSTEM_PROMPT["find_facts"]
        for prompt_type in all_prompt_types:
            if prompt_type != "find_facts":
                assert SYSTEM_PROMPT[prompt_type] != find_facts_prompt


# ============================================================================
# TestDirectInferencePrompts - 测试无上下文直接回答类型
# ============================================================================

class TestDirectInferencePrompts:
    """测试direct_inference类型"""

    def test_user_prompt_only_question(self, basic_question):
        """验证user prompt只包含问题，不包含context"""
        system_prompt, user_prompt = create_prompt(basic_question, "ignored", "direct_inference")
        assert basic_question in user_prompt
        # 不应该有context部分
        assert "## Context" not in user_prompt

    def test_user_question_format_correct(self, basic_question):
        """验证问题格式正确"""
        system_prompt, user_prompt = create_prompt(basic_question, "ignored", "direct_inference")
        assert basic_question in user_prompt
        # 问题应该被包含

    def test_system_prompt_has_answer_instructions(self):
        """验证system prompt包含Answer:格式说明"""
        system_prompt, _ = create_prompt("Test?", "ignored", "direct_inference")
        assert "Answer:" in system_prompt

    def test_no_context_section_in_user_prompt(self):
        """验证user prompt中没有## Context部分"""
        system_prompt, user_prompt = create_prompt("Test?", "some context", "direct_inference")
        assert "## Context" not in user_prompt

    def test_question_preserved_exactly(self):
        """验证问题文本被完整保留"""
        question = "What is your name?"
        system_prompt, user_prompt = create_prompt(question, "ignored", "direct_inference")
        assert question in user_prompt

    def test_empty_question_handling(self, empty_string):
        """测试空问题的处理"""
        system_prompt, user_prompt = create_prompt(empty_string, "ignored", "direct_inference")
        # 应该仍然返回结果
        assert isinstance(system_prompt, str)
        assert isinstance(user_prompt, str)

    def test_multiline_question_handling(self):
        """测试多行问题的处理"""
        question = "Line 1\nLine 2"
        system_prompt, user_prompt = create_prompt(question, "ignored", "direct_inference")
        assert "Line 1" in user_prompt
        assert "Line 2" in user_prompt

    def test_question_with_special_chars(self, question_with_special_chars):
        """测试包含特殊字符的问题"""
        system_prompt, user_prompt = create_prompt(question_with_special_chars, "ignored", "direct_inference")
        assert "$100" in user_prompt

    def test_question_with_unicode(self, question_with_unicode):
        """测试包含Unicode的问题"""
        system_prompt, user_prompt = create_prompt(question_with_unicode, "ignored", "direct_inference")
        assert "巴黎" in user_prompt


# ============================================================================
# TestCoTPrompts - 测试CoT推理类型
# ============================================================================

class TestCoTPrompts:
    """测试cot类型"""

    def test_user_prompt_only_question(self, basic_question):
        """验证user prompt只包含问题"""
        system_prompt, user_prompt = create_prompt(basic_question, "ignored", "cot")
        assert basic_question in user_prompt
        assert "## Context" not in user_prompt

    def test_system_prompt_contains_thinking_instruction(self):
        """验证system prompt包含"think step by step" """
        system_prompt, _ = create_prompt("Test?", "ignored", "cot")
        assert "think step by step" in system_prompt

    def test_system_prompt_requires_answer_format(self):
        """验证system prompt仍然要求Answer:格式"""
        system_prompt, _ = create_prompt("Test?", "ignored", "cot")
        assert "Answer:" in system_prompt

    def test_cot_vs_direct_inference_system_diff(self):
        """验证cot和direct_inference的system prompt不同"""
        system_cot, _ = create_prompt("Test?", "ignored", "cot")
        system_direct, _ = create_prompt("Test?", "ignored", "direct_inference")
        assert system_cot != system_direct

    def test_cot_system_mentions_thinking_process(self):
        """验证cot的system prompt提及思考过程"""
        system_prompt, _ = create_prompt("Test?", "ignored", "cot")
        assert "thinking process" in system_prompt

    def test_empty_question_cot(self, empty_string):
        """测试空问题在CoT模式下的处理"""
        system_prompt, user_prompt = create_prompt(empty_string, "ignored", "cot")
        assert isinstance(system_prompt, str)
        assert isinstance(user_prompt, str)

    def test_multiline_question_cot(self):
        """测试多行问题在CoT模式下的处理"""
        question = "Line 1\nLine 2"
        system_prompt, user_prompt = create_prompt(question, "ignored", "cot")
        assert "Line 1" in user_prompt
        assert "Line 2" in user_prompt

    def test_question_with_special_chars_cot(self, question_with_special_chars):
        """测试特殊字符问题在CoT模式下的处理"""
        system_prompt, user_prompt = create_prompt(question_with_special_chars, "ignored", "cot")
        assert "$100" in user_prompt

    def test_cot_user_prompt_no_context(self):
        """验证CoT的user prompt不包含context"""
        system_prompt, user_prompt = create_prompt("Test?", "some context", "cot")
        assert "## Context" not in user_prompt


# ============================================================================
# TestRAGStandardPrompts - 测试标准RAG类型
# ============================================================================

class TestRAGStandardPrompts:
    """测试标准RAG类型（rag）"""

    def test_user_prompt_has_context_section(self, basic_context):
        """验证user prompt包含## Context部分"""
        system_prompt, user_prompt = create_prompt("Test?", basic_context, "rag")
        assert "## Context" in user_prompt

    def test_user_prompt_has_question_section(self, basic_question):
        """验证user prompt包含## Question部分"""
        system_prompt, user_prompt = create_prompt(basic_question, "context", "rag")
        assert "## Question" in user_prompt

    def test_context_before_question(self):
        """验证context在question之前"""
        system_prompt, user_prompt = create_prompt("Test?", "Some context", "rag")
        context_pos = user_prompt.find("## Context")
        question_pos = user_prompt.find("## Question")
        assert context_pos < question_pos

    def test_context_preserved_exactly(self, basic_context):
        """验证context文本被完整保留"""
        system_prompt, user_prompt = create_prompt("Test?", basic_context, "rag")
        assert basic_context in user_prompt

    def test_question_preserved_exactly(self, basic_question):
        """验证question文本被完整保留"""
        system_prompt, user_prompt = create_prompt(basic_question, "context", "rag")
        assert basic_question in user_prompt

    def test_system_prompt_based_on_context(self):
        """验证system prompt包含"Based on the context" """
        system_prompt, _ = create_prompt("Test?", "context", "rag")
        assert "Based on the context" in system_prompt

    def test_system_prompt_direct_answer(self):
        """验证system prompt要求直接回答"""
        system_prompt, _ = create_prompt("Test?", "context", "rag")
        assert "Directly answer" in system_prompt

    def test_empty_context_rag(self, empty_string):
        """测试空context的处理"""
        system_prompt, user_prompt = create_prompt("Test?", empty_string, "rag")
        # 即使为空，也应该有## Context部分
        assert "## Context" in user_prompt

    def test_empty_question_rag(self, empty_string):
        """测试空question的处理"""
        system_prompt, user_prompt = create_prompt(empty_string, "context", "rag")
        # 即使为空，也应该有## Question部分
        assert "## Question" in user_prompt

    def test_both_empty_rag(self, empty_string):
        """测试context和question都为空"""
        system_prompt, user_prompt = create_prompt(empty_string, empty_string, "rag")
        assert "## Context" in user_prompt
        assert "## Question" in user_prompt

    def test_multiline_context_preserved(self, multi_line_context):
        """验证多行context被完整保留"""
        system_prompt, user_prompt = create_prompt("Test?", multi_line_context, "rag")
        assert "Line 1" in user_prompt
        assert "Line 2" in user_prompt
        assert "Line 3" in user_prompt
        assert "Line 4" in user_prompt

    def test_context_with_special_chars(self, context_with_special_chars):
        """测试包含特殊字符的context"""
        system_prompt, user_prompt = create_prompt("Test?", context_with_special_chars, "rag")
        assert "$100" in user_prompt
        assert "20%" in user_prompt

    def test_question_with_special_chars(self, question_with_special_chars):
        """测试包含特殊字符的question"""
        system_prompt, user_prompt = create_prompt(question_with_special_chars, "context", "rag")
        assert "$100" in user_prompt

    def test_standard_separator_between_sections(self):
        """验证各部分之间使用标准分隔符（\n\n）"""
        system_prompt, user_prompt = create_prompt("Test?", "Some context", "rag")
        # context和question之间应该有\n\n分隔
        assert "\n\n" in user_prompt


# ============================================================================
# TestRAGRepetitionPrompts - 测试RAG重复类型
# ============================================================================

class TestRAGRepetitionPrompts:
    """测试RAG重复类型（rag_rep_2, rag_rep_q）"""

    # rag_rep_2 测试

    def test_rag_rep_2_has_two_repetitions(self):
        """验证包含两次完整的context+question"""
        system_prompt, user_prompt = create_prompt("Test?", "Some context", "rag_rep_2")
        # 应该有两个## Context和两个## Question
        assert user_prompt.count("## Context") == 2
        assert user_prompt.count("## Question") == 2

    def test_rag_rep_2_separator_is_equals(self):
        """验证两次重复之间使用50个=号分隔"""
        system_prompt, user_prompt = create_prompt("Test?", "Some context", "rag_rep_2")
        assert "=" * 50 in user_prompt

    def test_rag_rep_2_separator_length(self):
        """验证分隔符恰好是50个=号"""
        system_prompt, user_prompt = create_prompt("Test?", "Some context", "rag_rep_2")
        assert user_prompt.count("=" * 50) == 1

    def test_rag_rep_2_both_parts_have_context_and_question(self):
        """验证两个部分都包含完整的context和question"""
        system_prompt, user_prompt = create_prompt("Test?", "Some context", "rag_rep_2")
        parts = user_prompt.split("=" * 50)
        assert len(parts) == 2
        # 第一部分
        assert "## Context" in parts[0]
        assert "## Question" in parts[0]
        # 第二部分
        assert "## Context" in parts[1]
        assert "## Question" in parts[1]

    def test_rag_rep_2_with_empty_context(self, empty_string):
        """测试空context的rag_rep_2"""
        system_prompt, user_prompt = create_prompt("Test?", empty_string, "rag_rep_2")
        assert user_prompt.count("## Context") == 2

    def test_rag_rep_2_with_multiline_content(self, multi_line_context):
        """测试多行内容的rag_rep_2"""
        system_prompt, user_prompt = create_prompt("Test?", multi_line_context, "rag_rep_2")
        assert multi_line_context.count("Line") == user_prompt.count("Line") // 2

    def test_rag_rep_2_preserves_content_exactly(self):
        """验证内容在两次重复中都被完整保留"""
        question = "Test question?"
        context = "Test context."
        system_prompt, user_prompt = create_prompt(question, context, "rag_rep_2")
        assert user_prompt.count(question) == 2
        assert user_prompt.count(context) == 2

    # rag_rep_q 测试

    def test_rag_rep_q_context_once(self):
        """验证context只出现一次"""
        system_prompt, user_prompt = create_prompt("Test?", "Some context", "rag_rep_q")
        assert user_prompt.count("## Context") == 1

    def test_rag_rep_q_question_twice(self):
        """验证question出现两次"""
        system_prompt, user_prompt = create_prompt("Test?", "Some context", "rag_rep_q")
        assert user_prompt.count("## Question") == 2

    def test_rag_rep_q_first_part_has_context_and_question(self):
        """验证第一部分包含context+question"""
        system_prompt, user_prompt = create_prompt("Test?", "Some context", "rag_rep_q")
        parts = user_prompt.split("=" * 50)
        assert "## Context" in parts[0]
        assert "## Question" in parts[0]

    def test_rag_rep_q_second_part_only_question(self):
        """验证第二部分只包含question"""
        system_prompt, user_prompt = create_prompt("Test?", "Some context", "rag_rep_q")
        parts = user_prompt.split("=" * 50)
        assert "## Context" not in parts[1]
        assert "## Question" in parts[1]

    def test_rag_rep_q_separator_is_equals(self):
        """验证两部分之间使用50个=号分隔"""
        system_prompt, user_prompt = create_prompt("Test?", "Some context", "rag_rep_q")
        assert "=" * 50 in user_prompt

    def test_rag_rep_q_with_empty_inputs(self, empty_string):
        """测试空输入的rag_rep_q"""
        system_prompt, user_prompt = create_prompt(empty_string, empty_string, "rag_rep_q")
        assert user_prompt.count("## Context") == 1
        assert user_prompt.count("## Question") == 2

    def test_rag_rep_q_questions_identical(self):
        """验证两个question的内容完全相同"""
        question = "Test question?"
        system_prompt, user_prompt = create_prompt(question, "context", "rag_rep_q")
        parts = user_prompt.split("=" * 50)
        # 提取两个question部分
        assert parts[0].count(question) >= 1
        assert parts[1].count(question) >= 1

    def test_rag_rep_q_preserves_context_exactly(self):
        """验证context被完整保留"""
        context = "Test context."
        system_prompt, user_prompt = create_prompt("Test?", context, "rag_rep_q")
        assert context in user_prompt
        assert user_prompt.count(context) == 1


# ============================================================================
# TestRAGQuestionContextQuestionPrompts - 测试QCQ模式
# ============================================================================

class TestRAGQuestionContextQuestionPrompts:
    """测试QCQ模式（rag_qcq, rag_qcq2）"""

    # rag_qcq 测试

    def test_rag_qcq_three_parts(self):
        """验证包含三个部分：question - context - question"""
        system_prompt, user_prompt = create_prompt("Test?", "Some context", "rag_qcq")
        # 应该有2个question和1个context
        assert user_prompt.count("## Question") == 2
        assert user_prompt.count("## Context") == 1

    def test_rag_qcq_pattern_question_context_question(self):
        """验证Q-C-Q模式"""
        system_prompt, user_prompt = create_prompt("Test?", "Some context", "rag_qcq")
        parts = user_prompt.split("\n\n")
        # 第一部分应该是question
        assert "## Question" in parts[0]
        # 第二部分应该是context
        assert "## Context" in parts[1]
        # 第三部分应该是question
        assert "## Question" in parts[2]

    def test_rag_qcq_first_part_is_question(self):
        """验证第一部分是question"""
        system_prompt, user_prompt = create_prompt("Test?", "Some context", "rag_qcq")
        assert user_prompt.startswith("## Question")

    def test_rag_qcq_second_part_is_context(self):
        """验证第二部分是context（不包含## Question标题）"""
        system_prompt, user_prompt = create_prompt("Test?", "Some context", "rag_qcq")
        parts = user_prompt.split("\n\n")
        # 第二部分应该有## Context但没有## Question
        assert "## Context" in parts[1]
        assert "## Question" not in parts[1]

    def test_rag_qcq_third_part_is_question(self):
        """验证第三部分是question"""
        system_prompt, user_prompt = create_prompt("Test?", "Some context", "rag_qcq")
        parts = user_prompt.split("\n\n")
        assert "## Question" in parts[2]

    def test_rag_qcq_questions_identical(self):
        """验证两个question的内容相同"""
        question = "Test question?"
        system_prompt, user_prompt = create_prompt(question, "context", "rag_qcq")
        assert user_prompt.count(question) == 2

    def test_rag_qcq_with_empty_inputs(self, empty_string):
        """测试空输入的rag_qcq"""
        system_prompt, user_prompt = create_prompt(empty_string, empty_string, "rag_qcq")
        assert user_prompt.count("## Question") == 2
        assert user_prompt.count("## Context") == 1

    def test_rag_qcq_with_multiline_content(self, multi_line_context):
        """测试多行内容的rag_qcq"""
        system_prompt, user_prompt = create_prompt("Test?", multi_line_context, "rag_qcq")
        assert "Line 1" in user_prompt
        assert "Line 2" in user_prompt

    # rag_qcq2 测试

    def test_rag_qcq2_four_parts(self):
        """验证包含四个部分：question - context - question - question"""
        system_prompt, user_prompt = create_prompt("Test?", "Some context", "rag_qcq2")
        # 应该有3个question和1个context
        assert user_prompt.count("## Question") == 3
        assert user_prompt.count("## Context") == 1

    def test_rag_qcq2_pattern(self):
        """验证Q-C-Q-Q模式"""
        system_prompt, user_prompt = create_prompt("Test?", "Some context", "rag_qcq2")
        parts = user_prompt.split("\n\n")
        assert len(parts) == 4
        assert "## Question" in parts[0]
        assert "## Context" in parts[1]
        assert "## Question" in parts[2]
        assert "## Question" in parts[3]

    def test_rag_qcq2_all_questions_identical(self):
        """验证三个question的内容都相同"""
        question = "Test question?"
        system_prompt, user_prompt = create_prompt(question, "context", "rag_qcq2")
        assert user_prompt.count(question) == 3

    def test_rag_qcq2_with_empty_inputs(self, empty_string):
        """测试空输入的rag_qcq2"""
        system_prompt, user_prompt = create_prompt(empty_string, empty_string, "rag_qcq2")
        assert user_prompt.count("## Question") == 3
        assert user_prompt.count("## Context") == 1

    def test_rag_qcq2_structure_matches_spec(self):
        """验证结构完全符合规范"""
        system_prompt, user_prompt = create_prompt("Test?", "Some context", "rag_qcq2")
        parts = user_prompt.split("\n\n")
        # 验证顺序：Q-C-Q-Q
        assert "## Question" in parts[0] and "## Context" not in parts[0]
        assert "## Context" in parts[1] and "## Question" not in parts[1]
        assert "## Question" in parts[2] and "## Context" not in parts[2]
        assert "## Question" in parts[3] and "## Context" not in parts[3]


# ============================================================================
# TestRAGContextInterleavedPrompts - 测试上下文分段类型
# ============================================================================

class TestRAGContextInterleavedPrompts:
    """测试上下文分段类型（rag_q_int_q, rag_q_int2_q）"""

    # rag_q_int_q 测试

    def test_rag_q_int_q_six_parts(self):
        """验证包含6个部分：Q - C1 - Q - C2 - C3 - Q"""
        system_prompt, user_prompt = create_prompt("Test?", "Line 1\nLine 2\nLine 3\nLine 4", "rag_q_int_q")
        parts = user_prompt.split("\n\n")
        assert len(parts) == 6

    def test_rag_q_int_q_with_four_line_context(self, four_line_context):
        """测试4行context的分割：1行-2行-1行"""
        system_prompt, user_prompt = create_prompt("Test?", four_line_context, "rag_q_int_q")
        parts = user_prompt.split("\n\n")
        # 验证分段
        assert "Line 1" in parts[1]  # 第一部分：1行
        assert "Line 2" in parts[3] and "Line 3" in parts[3]  # 第二部分：2行
        assert "Line 4" in parts[4]  # 第三部分：1行 (注意：parts[4]不是parts[5])

    def test_rag_q_int_q_with_eight_line_context(self, eight_line_context):
        """测试8行context的分割：2行-4行-2行"""
        system_prompt, user_prompt = create_prompt("Test?", eight_line_context, "rag_q_int_q")
        parts = user_prompt.split("\n\n")
        # 验证分段
        assert "Line 1" in parts[1] and "Line 2" in parts[1]  # 2行
        assert "Line 3" in parts[3] and "Line 6" in parts[3]  # 4行 (Line 3-6)
        assert "Line 7" in parts[4] and "Line 8" in parts[4]  # 2行

    def test_rag_q_int_q_with_three_line_context(self, three_line_context):
        """测试3行context的分割：0行-2行-1行（整数除法）"""
        system_prompt, user_prompt = create_prompt("Test?", three_line_context, "rag_q_int_q")
        parts = user_prompt.split("\n\n")
        # idx1 = 3//4 = 0, idx2 = 9//4 = 2
        # context_parts[0] = []
        # context_parts[1] = [Line 1, Line 2]
        # context_parts[2] = [Line 3]
        assert "Line 1" in parts[3] and "Line 2" in parts[3]  # 中间部分
        assert "Line 3" in parts[4]  # 最后部分 (parts[4]不是parts[5])

    def test_rag_q_int_q_with_single_line_context(self, single_line_context):
        """测试单行context的分割：0行-0行-1行"""
        system_prompt, user_prompt = create_prompt("Test?", single_line_context, "rag_q_int_q")
        parts = user_prompt.split("\n\n")
        # idx1 = 1//4 = 0, idx2 = 3//4 = 0
        # context_parts[0] = []
        # context_parts[1] = []
        # context_parts[2] = [Only one line]
        # 第一和第二段context是空的
        assert "Only one line" in parts[4]  # 第三段context (parts[4]不是parts[5])

    def test_rag_q_int_q_all_questions_identical(self, four_line_context):
        """验证所有question内容相同"""
        question = "Test?"
        system_prompt, user_prompt = create_prompt(question, four_line_context, "rag_q_int_q")
        assert user_prompt.count(question) == 3

    def test_rag_q_int_q_with_empty_context(self, empty_string):
        """测试空context的处理"""
        system_prompt, user_prompt = create_prompt("Test?", empty_string, "rag_q_int_q")
        # 应该仍然有6个部分
        parts = user_prompt.split("\n\n")
        assert len(parts) == 6

    def test_rag_q_int_q_preserves_all_context_lines(self, four_line_context):
        """验证所有context行都被保留（无丢失）"""
        system_prompt, user_prompt = create_prompt("Test?", four_line_context, "rag_q_int_q")
        assert four_line_context.count("Line") == user_prompt.count("Line")

    # rag_q_int2_q 测试

    def test_rag_q_int2_q_seven_parts(self):
        """验证包含7个部分：Q - C1 - Q - C2 - C3 - Q - Q"""
        system_prompt, user_prompt = create_prompt("Test?", "Line 1\nLine 2\nLine 3\nLine 4", "rag_q_int2_q")
        parts = user_prompt.split("\n\n")
        assert len(parts) == 7

    def test_rag_q_int2_q_ends_with_two_questions(self):
        """验证最后有两个question"""
        system_prompt, user_prompt = create_prompt("Test?", "Line 1\nLine 2\nLine 3\nLine 4", "rag_q_int2_q")
        parts = user_prompt.split("\n\n")
        # 最后两个部分都应该是question
        assert "## Question" in parts[5]
        assert "## Question" in parts[6]
        assert "## Context" not in parts[5]
        assert "## Context" not in parts[6]

    def test_rag_q_int2_q_with_four_line_context(self, four_line_context):
        """测试4行context的分割"""
        system_prompt, user_prompt = create_prompt("Test?", four_line_context, "rag_q_int2_q")
        parts = user_prompt.split("\n\n")
        # 验证分段与rag_q_int_q相同：1行-2行-1行
        assert "Line 1" in parts[1]
        assert "Line 2" in parts[3] and "Line 3" in parts[3]
        assert "Line 4" in parts[4]

    def test_rag_q_int2_q_all_questions_identical(self, four_line_context):
        """验证所有question内容相同"""
        question = "Test?"
        system_prompt, user_prompt = create_prompt(question, four_line_context, "rag_q_int2_q")
        # 应该有4个question
        assert user_prompt.count(question) == 4

    def test_rag_q_int2_q_with_single_line_context(self, single_line_context):
        """测试单行context的分割"""
        system_prompt, user_prompt = create_prompt("Test?", single_line_context, "rag_q_int2_q")
        parts = user_prompt.split("\n\n")
        # 对于rag_q_int2_q，单行context应该在parts[4]（第三段context）
        assert "Only one line" in parts[4]

    def test_rag_q_int2_q_with_empty_context(self, empty_string):
        """测试空context的处理"""
        system_prompt, user_prompt = create_prompt("Test?", empty_string, "rag_q_int2_q")
        parts = user_prompt.split("\n\n")
        assert len(parts) == 7

    def test_rag_q_int2_q_vs_q_int_q_last_question(self):
        """验证rag_q_int2_q比rag_q_int_q多一个question"""
        system_prompt_q, user_prompt_q = create_prompt("Test?", "Line 1\nLine 2\nLine 3\nLine 4", "rag_q_int_q")
        system_prompt_q2, user_prompt_q2 = create_prompt("Test?", "Line 1\nLine 2\nLine 3\nLine 4", "rag_q_int2_q")
        # rag_q_int2_q应该多一个question部分
        assert user_prompt_q.count("## Question") == 3
        assert user_prompt_q2.count("## Question") == 4


# ============================================================================
# TestRAGDecompressedPrompts - 测试解压缩类型
# ============================================================================

class TestRAGDecompressedPrompts:
    """测试解压缩类型（rag_decompressed, rag_decompressed_rep_q）"""

    # rag_decompressed 测试

    def test_rag_decompressed_question_has_spaces(self):
        """验证question的每个字符之间都插入了空格"""
        question = "hello"
        context = "Test context"

        system_prompt, user_prompt = create_prompt(question, context, "rag_decompressed")

        expected = "h e l l o"
        assert expected in user_prompt

    def test_rag_decompressed_spaces_between_chars(self):
        """验证空格插入在每个字符之间"""
        question = "ABC"
        context = "Test"

        system_prompt, user_prompt = create_prompt(question, context, "rag_decompressed")

        expected = "A B C"
        assert expected in user_prompt

    def test_rag_decompressed_no_leading_trailing_spaces(self):
        """验证没有前导或尾随空格"""
        question = "test"
        context = "context"

        system_prompt, user_prompt = create_prompt(question, context, "rag_decompressed")

        # 解压后的问题应该在 ## Question 部分
        assert "## Question" in user_prompt
        # 验证 "t e s t" 存在，且没有前导空格（换行符是有的，但不是空格）
        assert "t e s t" in user_prompt
        # 确保没有 "t e s t " 这种带尾随空格的形式（后面直接是换行）

    def test_rag_decompressed_preserves_context(self):
        """验证context被完整保留，未修改"""
        context = "Original context here."
        question = "test"

        system_prompt, user_prompt = create_prompt(question, context, "rag_decompressed")

        assert context in user_prompt
        # context应该没有被修改（没有额外的空格）

    def test_rag_decompressed_context_question_format(self):
        """验证仍然是context-question格式"""
        system_prompt, user_prompt = create_prompt("Test?", "Some context", "rag_decompressed")

        # 应该有Context和Question部分
        assert "## Context" in user_prompt
        assert "## Question" in user_prompt

    def test_rag_decompressed_with_special_chars(self):
        """测试包含特殊字符的question的解压缩"""
        question = "Test?"
        context = "context"

        system_prompt, user_prompt = create_prompt(question, context, "rag_decompressed")

        # 特殊字符也应该有空格分隔
        assert "T e s t ?" in user_prompt

    def test_rag_decompressed_with_unicode(self, question_with_unicode):
        """测试包含Unicode的question的解压缩"""
        context = "context"
        system_prompt, user_prompt = create_prompt(question_with_unicode, context, "rag_decompressed")

        # Unicode字符也应该被分隔
        assert "巴 黎" in user_prompt

    # rag_decompressed_rep_q 测试

    def test_rag_decompressed_rep_q_three_parts(self):
        """验证包含3个部分：context - decompressed_q - decompressed_q"""
        system_prompt, user_prompt = create_prompt("Test?", "Some context", "rag_decompressed_rep_q")

        # 应该有1个context和2个question
        assert user_prompt.count("## Context") == 1
        assert user_prompt.count("## Question") == 2

    def test_rag_decompressed_rep_q_context_once(self):
        """验证context只出现一次"""
        system_prompt, user_prompt = create_prompt("Test?", "Some context", "rag_decompressed_rep_q")

        assert user_prompt.count("## Context") == 1

    def test_rag_decompressed_rep_q_question_twice(self):
        """验证decompressed question出现两次"""
        question = "test"
        context = "context"

        system_prompt, user_prompt = create_prompt(question, context, "rag_decompressed_rep_q")

        # 解压后的question应该出现两次
        assert "t e s t" in user_prompt

    def test_rag_decompressed_rep_q_both_questions_decompressed(self):
        """验证两个question都被解压缩"""
        question = "hello"
        context = "context"

        system_prompt, user_prompt = create_prompt(question, context, "rag_decompressed_rep_q")

        expected = "h e l l o"
        # 应该出现两次
        assert user_prompt.count(expected) == 2

    def test_rag_decompressed_rep_q_questions_identical(self):
        """验证两个decompressed question内容相同"""
        question = "test"
        context = "context"

        system_prompt, user_prompt = create_prompt(question, context, "rag_decompressed_rep_q")

        parts = user_prompt.split("\n\n")
        # 第二个和第三个部分都应该是question
        question_parts = [p for p in parts if "## Question" in p]
        assert len(question_parts) == 2

    def test_rag_decompressed_rep_q_format(self):
        """验证格式：## Context \n\n ## Question \n\n ## Question"""
        system_prompt, user_prompt = create_prompt("Test?", "context", "rag_decompressed_rep_q")

        parts = user_prompt.split("\n\n")
        # 第一部分是context
        assert "## Context" in parts[0]
        # 第二和第三部分是question
        assert "## Question" in parts[1]
        assert "## Question" in parts[2]


# ============================================================================
# TestIRCoTPrompts - 测试RAG CoT类型
# ============================================================================

class TestIRCoTPrompts:
    """测试ircot类型（RAG + CoT）"""

    def test_ircot_system_prompt_based_on_context(self):
        """验证system prompt包含"Based on the context" """
        system_prompt = SYSTEM_PROMPT["ircot"]
        assert "Based on the context" in system_prompt

    def test_ircot_system_prompt_think_step_by_step(self):
        """验证system prompt包含"think step by step" """
        system_prompt = SYSTEM_PROMPT["ircot"]
        assert "think step by step" in system_prompt

    def test_ircot_system_prompt_requires_answer_format(self):
        """验证system prompt仍然要求Answer:格式"""
        system_prompt = SYSTEM_PROMPT["ircot"]
        assert "Answer:" in system_prompt

    def test_ircot_user_prompt_standard_rag_format(self):
        """验证user prompt使用标准RAG格式（context+question）"""
        system_prompt, user_prompt = create_prompt("Test?", "Some context", "ircot")

        # 应该是标准的context-question格式
        assert "## Context" in user_prompt
        assert "## Question" in user_prompt
        # 只有一个context和一个question
        assert user_prompt.count("## Context") == 1
        assert user_prompt.count("## Question") == 1

    def test_ircot_vs_rag_system_diff(self):
        """验证ircot和rag的system prompt不同"""
        ircot_system = SYSTEM_PROMPT["ircot"]
        rag_system = SYSTEM_PROMPT["rag"]
        assert ircot_system != rag_system

    def test_ircot_vs_cot_system_diff(self):
        """验证ircot和cot的system prompt不同"""
        ircot_system = SYSTEM_PROMPT["ircot"]
        cot_system = SYSTEM_PROMPT["cot"]
        assert ircot_system != cot_system

    def test_ircot_combines_rag_and_cot(self):
        """验证ircot结合了RAG和CoT的特点"""
        system_prompt = SYSTEM_PROMPT["ircot"]
        # 包含RAG特点
        assert "Based on the context" in system_prompt
        # 包含CoT特点
        assert "think step by step" in system_prompt

    def test_ircot_with_empty_inputs(self, empty_string):
        """测试空输入的ircot"""
        system_prompt, user_prompt = create_prompt(empty_string, empty_string, "ircot")
        assert "## Context" in user_prompt
        assert "## Question" in user_prompt

    def test_ircot_with_multiline_content(self, multi_line_context):
        """测试多行内容的ircot"""
        system_prompt, user_prompt = create_prompt("Test?", multi_line_context, "ircot")
        assert "Line 1" in user_prompt
        assert "Line 2" in user_prompt

    def test_ircot_preserves_context_and_question(self):
        """验证context和question被完整保留"""
        context = "Test context."
        question = "Test question?"
        system_prompt, user_prompt = create_prompt(question, context, "ircot")
        assert context in user_prompt
        assert question in user_prompt


# ============================================================================
# TestSpecialFormatPrompts - 测试特殊格式类型
# ============================================================================

class TestSpecialFormatPrompts:
    """测试特殊格式类型（deepseek, copypaste, find_facts）"""

    # deepseek 测试

    def test_deepseek_system_prompt_has_think_tags(self):
        """验证system prompt包含<|think|> </|think|>说明"""
        system_prompt = SYSTEM_PROMPT["deepseek"]
        assert "<|think|>" in system_prompt
        assert "</|think|>" in system_prompt

    def test_deepseek_system_prompt_has_answer_tags(self):
        """验证system prompt包含<|answer|> </|answer|>说明"""
        system_prompt = SYSTEM_PROMPT["deepseek"]
        assert "<|answer|>" in system_prompt
        assert "</|answer|>" in system_prompt

    def test_deepseek_system_prompt_example_included(self):
        """验证system prompt包含示例"""
        system_prompt = SYSTEM_PROMPT["deepseek"]
        assert "Example" in system_prompt or "example" in system_prompt.lower()

    def test_deepseek_system_prompt_no_evidence_tag(self):
        """验证不包含<|EVIDENCE|>标签说明"""
        system_prompt = SYSTEM_PROMPT["deepseek"]
        assert "<|EVIDENCE|>" not in system_prompt

    def test_deepseek_user_prompt_standard_rag(self):
        """验证user prompt使用标准RAG格式"""
        system_prompt, user_prompt = create_prompt("Test?", "Some context", "deepseek")

        assert "## Context" in user_prompt
        assert "## Question" in user_prompt
        assert user_prompt.count("## Context") == 1
        assert user_prompt.count("## Question") == 1

    def test_deepseek_with_empty_inputs(self, empty_string):
        """测试空输入的deepseek"""
        system_prompt, user_prompt = create_prompt(empty_string, empty_string, "deepseek")
        assert isinstance(system_prompt, str)
        assert isinstance(user_prompt, str)

    def test_deepseek_preserves_content(self):
        """验证context和question被完整保留"""
        context = "Test context."
        question = "Test?"
        system_prompt, user_prompt = create_prompt(question, context, "deepseek")
        assert context in user_prompt
        assert question in user_prompt

    # copypaste 测试

    def test_copypaste_system_prompt_has_think_tags(self):
        """验证system prompt包含<|think|> </|think|>说明"""
        system_prompt = SYSTEM_PROMPT["copypaste"]
        assert "<|think|>" in system_prompt
        assert "</|think|>" in system_prompt

    def test_copypaste_system_prompt_has_answer_tags(self):
        """验证system prompt包含<|answer|> </|answer|>说明"""
        system_prompt = SYSTEM_PROMPT["copypaste"]
        assert "<|answer|>" in system_prompt
        assert "</|answer|>" in system_prompt

    def test_copypaste_system_prompt_has_evidence_tags(self):
        """验证system prompt包含<|EVIDENCE|> </|EVIDENCE|>说明"""
        system_prompt = SYSTEM_PROMPT["copypaste"]
        assert "<|EVIDENCE|>" in system_prompt

    def test_copypaste_system_prompt_evidence_guidelines(self):
        """验证包含证据提取指南"""
        system_prompt = SYSTEM_PROMPT["copypaste"]
        assert "Evidence Extraction" in system_prompt or "evidence" in system_prompt.lower()

    def test_copypaste_system_prompt_example_with_evidence(self):
        """验证示例包含<|EVIDENCE|>标签使用"""
        system_prompt = SYSTEM_PROMPT["copypaste"]
        # 应该在示例中展示EVIDENCE标签的使用
        assert system_prompt.count("<|EVIDENCE|>") >= 2  # 至少出现在说明和示例中

    def test_copypaste_vs_deepseek_difference(self):
        """验证copypaste和deepseek的关键区别（证据标签）"""
        copypaste_system = SYSTEM_PROMPT["copypaste"]
        deepseek_system = SYSTEM_PROMPT["deepseek"]
        # copypaste有证据标签，deepseek没有
        assert "<|EVIDENCE|>" in copypaste_system
        assert "<|EVIDENCE|>" not in deepseek_system

    def test_copypaste_user_prompt_standard_rag(self):
        """验证user prompt使用标准RAG格式"""
        system_prompt, user_prompt = create_prompt("Test?", "Some context", "copypaste")
        assert "## Context" in user_prompt
        assert "## Question" in user_prompt

    # find_facts 测试

    def test_find_facts_system_prompt_extract_facts(self):
        """验证system prompt要求提取事实"""
        system_prompt = SYSTEM_PROMPT["find_facts"]
        assert "extract" in system_prompt.lower()
        assert "facts" in system_prompt.lower()

    def test_find_facts_system_prompt_no_answer(self):
        """验证system prompt明确说明不要提供答案"""
        system_prompt = SYSTEM_PROMPT["find_facts"]
        assert "Do NOT provide the answer" in system_prompt or "not provide" in system_prompt.lower()

    def test_find_facts_system_prompt_evidence_tag_format(self):
        """验证要求使用<|EVIDENCE|>标签"""
        system_prompt = SYSTEM_PROMPT["find_facts"]
        assert "<|EVIDENCE|>" in system_prompt

    def test_find_facts_system_prompt_no_extra_text(self):
        """验证说明不要包含额外文本"""
        system_prompt = SYSTEM_PROMPT["find_facts"]
        assert "No extra text" in system_prompt or "No Extra Text" in system_prompt

    def test_find_facts_user_prompt_standard_rag(self):
        """验证user prompt使用标准RAG格式"""
        system_prompt, user_prompt = create_prompt("Test?", "Some context", "find_facts")
        assert "## Context" in user_prompt
        assert "## Question" in user_prompt


# ============================================================================
# TestEdgeCases - 测试边界情况和异常情况
# ============================================================================

class TestEdgeCases:
    """测试边界情况和异常情况"""

    # 空输入测试

    def test_empty_question_all_types(self, all_prompt_types):
        """测试所有类型的空question处理"""
        for prompt_type in all_prompt_types:
            system_prompt, user_prompt = create_prompt("", "context", prompt_type)
            # 应该仍然返回有效的字符串
            assert isinstance(system_prompt, str)
            assert isinstance(user_prompt, str)
            assert len(system_prompt) > 0
            # 对于 direct_inference 和 cot，空question会导致空user_prompt（这是预期的）
            if prompt_type in ["direct_inference", "cot"]:
                assert len(user_prompt) == 0  # 这两种类型只有question，空question=空user_prompt
            else:
                assert len(user_prompt) > 0  # 其他类型有context，所以不会为空

    def test_empty_context_rag_types(self, rag_prompt_types):
        """测试RAG类型的空context处理"""
        for prompt_type in rag_prompt_types:
            system_prompt, user_prompt = create_prompt("Test?", "", prompt_type)
            # rag_q_int_docs_q 使用不同的格式（Document 1, Document 2 等），不包含 ## Context
            if prompt_type == "rag_q_int_docs_q":
                # 对于 rag_q_int_docs_q，应该有问题在空上下文时
                assert "## Question" in user_prompt
            else:
                assert "## Context" in user_prompt

    def test_both_empty_rag_types(self, rag_prompt_types):
        """测试RAG类型的context和question都为空"""
        for prompt_type in rag_prompt_types:
            system_prompt, user_prompt = create_prompt("", "", prompt_type)
            assert isinstance(system_prompt, str)
            assert isinstance(user_prompt, str)

    # 特殊字符测试

    def test_question_with_newlines(self):
        """测试包含换行符的question"""
        question = "Line 1\nLine 2"
        context = "context"
        system_prompt, user_prompt = create_prompt(question, context, "rag")
        assert "Line 1" in user_prompt
        assert "Line 2" in user_prompt

    def test_context_with_newlines(self, multi_line_context):
        """测试包含换行符的context（验证分段逻辑）"""
        system_prompt, user_prompt = create_prompt("Test?", multi_line_context, "rag")
        assert "Line 1" in user_prompt
        assert "Line 2" in user_prompt

    def test_question_with_tabs(self):
        """测试包含制表符的question"""
        question = "Test\tquestion"
        context = "context"
        system_prompt, user_prompt = create_prompt(question, context, "rag")
        assert question in user_prompt

    def test_context_with_tabs(self):
        """测试包含制表符的context"""
        context = "Line\t1\nLine\t2"
        system_prompt, user_prompt = create_prompt("Test?", context, "rag")
        assert context in user_prompt

    def test_question_with_quotes(self):
        """测试包含引号的question"""
        question = 'What is "test"?'
        context = "context"
        system_prompt, user_prompt = create_prompt(question, context, "rag")
        assert question in user_prompt

    def test_context_with_quotes(self):
        """测试包含引号的context"""
        context = 'He said "test"'
        system_prompt, user_prompt = create_prompt("Test?", context, "rag")
        assert context in user_prompt

    # Unicode测试

    def test_chinese_characters_question(self, question_with_unicode):
        """测试包含中文字符的question"""
        context = "context"
        system_prompt, user_prompt = create_prompt(question_with_unicode, context, "rag")
        assert "巴黎" in user_prompt

    def test_chinese_characters_context(self):
        """测试包含中文字符的context"""
        question = "Test?"
        context = "这是中文内容"
        system_prompt, user_prompt = create_prompt(question, context, "rag")
        assert context in user_prompt

    def test_mixed_language_question(self, question_with_unicode):
        """测试混合语言的question"""
        context = "context"
        system_prompt, user_prompt = create_prompt(question_with_unicode, context, "rag")
        # 应该包含所有字符
        assert "What" in user_prompt
        assert "巴黎" in user_prompt

    # 极端长度测试

    def test_single_character_question(self):
        """测试单个字符的question"""
        question = "A"
        context = "context"
        system_prompt, user_prompt = create_prompt(question, context, "rag")
        assert question in user_prompt

    def test_single_character_context(self, single_line_context):
        """测试单个字符的context"""
        question = "Test?"
        system_prompt, user_prompt = create_prompt(question, single_line_context, "rag")
        assert single_line_context in user_prompt

    def test_single_line_vs_multiline_context(self):
        """对比单行和多行context在分段逻辑中的差异"""
        question = "Test?"
        single = "Only one line."
        multi = "Line 1\nLine 2\nLine 3\nLine 4"

        # 单行context在rag中
        _, single_prompt = create_prompt(question, single, "rag")
        # 多行context在rag中
        _, multi_prompt = create_prompt(question, multi, "rag")

        assert single in single_prompt
        assert "Line 1" in multi_prompt
        assert "Line 4" in multi_prompt

    # 无效类型测试

    def test_invalid_prompt_type_raises_error(self):
        """测试无效的prompt_type会引发错误"""
        with pytest.raises(KeyError):
            create_prompt("Test?", "context", "invalid_type")


# ============================================================================
# TestPromptStructure - 测试提示词结构和格式
# ============================================================================

class TestPromptStructure:
    """测试提示词的结构和格式"""

    # 返回值类型测试

    def test_create_prompt_returns_tuple(self):
        """验证返回值是tuple类型"""
        result = create_prompt("Test?", "context", "rag")
        assert isinstance(result, tuple)

    def test_create_prompt_returns_two_elements(self):
        """验证返回值包含2个元素"""
        system_prompt, user_prompt = create_prompt("Test?", "context", "rag")
        # 解包后应该有两个变量
        pass  # 如果解包成功，说明有2个元素

    def test_create_prompt_elements_are_strings(self):
        """验证两个元素都是字符串类型"""
        system_prompt, user_prompt = create_prompt("Test?", "context", "rag")
        assert isinstance(system_prompt, str)
        assert isinstance(user_prompt, str)

    # 分隔符和格式测试

    def test_standard_context_question_separator(self):
        """验证标准RAG格式中context和question的分隔符"""
        system_prompt, user_prompt = create_prompt("Test?", "Some context", "rag")

        # 应该有标准的分隔符：context内容后面跟\n\n## Question
        assert "\n\n## Question" in user_prompt or "\r\n\r\n## Question" in user_prompt
        # 验证完整的格式：## Context\n<content>\n\n## Question
        assert "## Context\n" in user_prompt or "## Context\r\n" in user_prompt

    def test_question_section_format(self):
        """验证## Question标题的格式"""
        system_prompt, user_prompt = create_prompt("Test?", "context", "rag")
        assert "## Question" in user_prompt

    def test_context_section_format(self):
        """验证## Context标题的格式"""
        system_prompt, user_prompt = create_prompt("Test?", "context", "rag")
        assert "## Context" in user_prompt

    def test_newline_separators_between_parts(self):
        """验证各部分之间使用\n\n分隔"""
        system_prompt, user_prompt = create_prompt("Test?", "Some context", "rag")
        # 至少应该有一个\n\n分隔符（在context和question之间）
        assert "\n\n" in user_prompt

    # 内容完整性测试

    def test_context_not_truncated(self):
        """验证context没有被截断"""
        context = "A" * 1000  # 很长的context
        question = "Test?"
        system_prompt, user_prompt = create_prompt(question, context, "rag")
        assert context in user_prompt

    def test_question_not_truncated(self):
        """验证question没有被截断"""
        question = "B" * 1000  # 很长的question
        context = "context"
        system_prompt, user_prompt = create_prompt(question, context, "rag")
        assert question in user_prompt

    def test_multiline_context_preserves_line_count(self, multi_line_context):
        """验证多行context保留所有行"""
        question = "Test?"
        system_prompt, user_prompt = create_prompt(question, multi_line_context, "rag")
        # 应该包含所有4行
        assert "Line 1" in user_prompt
        assert "Line 2" in user_prompt
        assert "Line 3" in user_prompt
        assert "Line 4" in user_prompt

    def test_interleaved_context_preserves_all_lines(self, four_line_context):
        """验证分段context保留所有行（无丢失）"""
        question = "Test?"
        system_prompt, user_prompt = create_prompt(question, four_line_context, "rag_q_int_q")
        # 验证所有4行都还在
        assert four_line_context.count("Line") == user_prompt.count("Line")

    def test_decompressed_question_char_count(self):
        """验证解压缩后question的字符数正确"""
        question = "test"
        context = "context"
        system_prompt, user_prompt = create_prompt(question, context, "rag_decompressed")

        # 解压后应该是 4个字符 + 3个空格 = 7个字符
        assert "t e s t" in user_prompt

    def test_repeated_content_identical(self):
        """验证重复的内容完全相同"""
        question = "Same question?"
        context = "Same context."
        system_prompt, user_prompt = create_prompt(question, context, "rag_rep_2")

        parts = user_prompt.split("=" * 50)
        # 两部分应该都包含完整的question和context
        assert question in parts[0] and question in parts[1]
        assert context in parts[0] and context in parts[1]

    def test_no_unexpected_modifications(self):
        """验证输入文本没有被意外修改"""
        question = "Test question?"
        context = "Test context."
        system_prompt, user_prompt = create_prompt(question, context, "rag")

        # 应该包含原始文本
        assert question in user_prompt
        assert context in user_prompt

    # 批量类型验证

    @pytest.mark.parametrize("prompt_type", [
        "rag", "ircot", "deepseek", "copypaste", "find_facts"
    ])
    def test_rag_types_have_context_section(self, prompt_type):
        """验证所有RAG类型都包含context部分"""
        system_prompt, user_prompt = create_prompt("Test?", "context", prompt_type)
        assert "## Context" in user_prompt

    @pytest.mark.parametrize("prompt_type", ["direct_inference", "cot"])
    def test_no_context_types_have_no_context_section(self, prompt_type):
        """验证无上下文类型不包含context部分"""
        system_prompt, user_prompt = create_prompt("Test?", "context", prompt_type)
        assert "## Context" not in user_prompt


# ============================================================================
# TestRAGDocumentInterleavedPrompts - 测试文档交错类型
# ============================================================================

class TestRAGDocumentInterleavedPrompts:
    """测试文档交错类型（rag_q_int_docs_q）"""

    def test_rag_q_int_docs_q_two_documents_structure(self):
        """验证两个文档的交错结构：Q → Doc1 → Q → Doc2 → Q → Q"""
        question = "What is the capital?"
        context = "###PARIS\nParis is the capital city\n\n###LONDON\nLondon is in England"

        system_prompt, user_prompt = create_prompt(question, context, "rag_q_int_docs_q")

        # 验证问题数量：起始1个 + 每个文档后1个(2个) + 结尾1个 = 4个
        assert user_prompt.count("## Question") == 4

        # 验证文档标签
        assert "Document 1" in user_prompt
        assert "Document 2" in user_prompt

        # 验证文档内容保留
        assert "###PARIS" in user_prompt
        assert "###LONDON" in user_prompt

        # 验证问题内容
        assert question in user_prompt

    def test_rag_q_int_docs_q_three_documents_structure(self):
        """验证三个文档的交错结构：Q → Doc1 → Q → Doc2 → Q → Doc3 → Q → Q"""
        question = "Test question?"
        context = "###PARIS\nParis\n\n###LONDON\nLondon\n\n###TOKYO\nTokyo"

        system_prompt, user_prompt = create_prompt(question, context, "rag_q_int_docs_q")

        # 验证问题数量：起始1个 + 每个文档后1个(3个) + 结尾1个 = 5个
        assert user_prompt.count("## Question") == 5

        # 验证所有文档标签
        assert "Document 1" in user_prompt
        assert "Document 2" in user_prompt
        assert "Document 3" in user_prompt

    def test_rag_q_int_docs_q_single_document(self):
        """验证单个文档的处理：Q → Doc1 → Q → Q"""
        question = "Test question?"
        context = "###PARIS\nParis is the capital city"

        system_prompt, user_prompt = create_prompt(question, context, "rag_q_int_docs_q")

        # 验证问题数量：起始1个 + 文档后1个 + 结尾1个 = 3个
        assert user_prompt.count("## Question") == 3

        # 验证文档标签
        assert "Document 1" in user_prompt
        assert "Document 2" not in user_prompt

        # 验证文档内容保留
        assert "###PARIS" in user_prompt

    def test_rag_q_int_docs_q_empty_context(self):
        """验证空上下文的处理"""
        question = "Test question?"
        context = ""

        system_prompt, user_prompt = create_prompt(question, context, "rag_q_int_docs_q")

        # 空上下文应该仍然有起始问题和结尾问题
        assert "## Question" in user_prompt
        assert question in user_prompt

    def test_rag_q_int_docs_q_document_content_preserved(self):
        """验证文档内容完整保留"""
        question = "Test question?"
        context = "###PARIS\nParis is the capital city\n\n###LONDON\nLondon is in England"

        system_prompt, user_prompt = create_prompt(question, context, "rag_q_int_docs_q")

        # 验证原始文档内容未被修改
        assert "###PARIS\nParis is the capital city" in user_prompt
        assert "###LONDON\nLondon is in England" in user_prompt

    def test_rag_q_int_docs_q_questions_identical(self):
        """验证所有问题内容相同"""
        question = "What is the capital?"
        context = "###PARIS\nParis\n\n###LONDON\nLondon"

        system_prompt, user_prompt = create_prompt(question, context, "rag_q_int_docs_q")

        # 验证问题出现次数正确（起始1个 + 每个文档后1个(2个) + 结尾1个 = 4个）
        assert user_prompt.count(question) == 4

        # 验证每个 "## Question" 后面紧跟的都是同一个问题文本
        # 使用正则表达式提取所有 ## Question 后面的内容直到换行
        import re
        pattern = r"## Question\n([^\n]+)"
        matches = re.findall(pattern, user_prompt)
        # 所有匹配的问题应该相同
        assert len(set(matches)) == 1
        assert matches[0] == question

    def test_rag_q_int_docs_q_document_labeling(self):
        """验证文档编号正确"""
        question = "Test?"
        context = "###A\nA\n\n###B\nB\n\n###C\nC"

        system_prompt, user_prompt = create_prompt(question, context, "rag_q_int_docs_q")

        # 验证文档编号从1开始
        assert "Document 1" in user_prompt
        assert "Document 2" in user_prompt
        assert "Document 3" in user_prompt

        # 验证没有 Document 0
        assert "Document 0" not in user_prompt

    def test_rag_q_int_docs_q_system_prompt(self):
        """验证使用标准RAG system prompt"""
        system_prompt = SYSTEM_PROMPT["rag_q_int_docs_q"]
        rag_system_prompt = SYSTEM_PROMPT["rag"]

        # 应该与其他RAG类型使用相同的system prompt
        assert system_prompt == rag_system_prompt
        assert "Based on the context" in system_prompt

    def test_rag_q_int_docs_q_handles_extra_newlines(self):
        """验证处理多余空行"""
        question = "Test?"
        context = "###A\nA\n\n\n\n###B\nB"

        system_prompt, user_prompt = create_prompt(question, context, "rag_q_int_docs_q")

        # 应该只有2个文档（空文档被过滤）
        assert "Document 1" in user_prompt
        assert "Document 2" in user_prompt
        assert "Document 3" not in user_prompt

    def test_rag_q_int_docs_q_preserves_document_internal_newlines(self):
        """验证保留文档内部的单个换行符"""
        question = "Test?"
        context = "###DOC1\nLine 1\nLine 2\n\n###DOC2\nLine 3\nLine 4"

        system_prompt, user_prompt = create_prompt(question, context, "rag_q_int_docs_q")

        # 验证文档内部的换行被保留
        assert "Line 1\nLine 2" in user_prompt
        assert "Line 3\nLine 4" in user_prompt

    def test_rag_q_int_docs_q_order(self):
        """验证输出顺序：起始问题 → 文档 → 问题 → 文档 → 问题 → ... → 结尾问题"""
        question = "Test?"
        context = "###A\nA\n\n###B\nB"

        system_prompt, user_prompt = create_prompt(question, context, "rag_q_int_docs_q")

        # 验证顺序：第一个元素应该是问题
        assert user_prompt.startswith("## Question")

        # 验证顺序：问题 → Document 1 → 问题
        parts = user_prompt.split("\n\n")
        assert "## Question" in parts[0]
        assert "Document 1" in parts[1]
        assert "## Question" in parts[2]

    def test_rag_q_int_docs_q_all_questions_same(self, two_document_context):
        """验证所有问题内容完全相同"""
        question = "Same question?"

        system_prompt, user_prompt = create_prompt(question, two_document_context, "rag_q_int_docs_q")

        # 验证问题出现次数
        assert user_prompt.count(question) == 4

        # 每个问题后都应该紧跟文档或结束
        question_indices = [i for i, part in enumerate(user_prompt.split("\n\n")) if "## Question" in part]

        # 验证问题数量
        assert len(question_indices) == 4
