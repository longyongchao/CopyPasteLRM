"""
测试 CopyPasteUniORM 的功能

## 奖励函数设计说明

> 注意，如果完美复制命中，且答案命中，则总奖励为2.0分，最大奖励为2.0分

### 1 格式校验
    1.1 校验completion中的所有HTML标签是否合法（用先进先出栈），如果不合法则直接奖励0分
    1.2 校验completion中的<think></think>和<answer></answer>各出现一次且不嵌套，否则直接奖励0分
    1.3 校验completion中的<think></think>是否出现在<answer></answer>之前，且<think></think>之前没有任何内容，
    <answer></answer>之后没有任何内容，<think></think>和<answer></answer>之间，除了空白字符之外，没有其它内容，否则直接奖励0分
    1.4 校验在<think></think>之中是否存在两对及以上的<copy></copy>标签，并且确保每对<copy></copy>不直接相邻，否则直接奖励0分

### 2 长度校验
    2.1 校验completion中<answer></answer>中的内容的长度小于等于 len(answer) * 3，否则直接奖励0分
    2.2 校验completion中<think></think>中内容的长度大于<answer></answer>中内容的长度，否则直接奖励0分
    2.3 校验completion中<think></think>中内容的长度大于 facts 中所有内容长度的两倍，否则直接奖励0分
    2.4 校验completion中每一个<copy></copy>中内容均是facts中某一句话的子串，否则直接奖励0分
    2.5 校验completion中每一个<copy></copy>中内容的长度大于等于 min(facts中平均长度的一半, facts中最短的一句话的长度)，否则直接奖励0分

### 3 复制命中奖励计算。
    3.0 copy命中奖励的总分值是1.0，按照facts中句子的数量进行分配，比如facts中有5句话，则每一个满足条件的<copy></copy>奖励1/5分
    3.1 首先按照<copy></copy>标签内的内容长度进行排序，从长到短，然后依次进行判断：
    3.2 如果当前<copy></copy>标签内的内容是facts中某一句话的子串，并且不是先前任一<copy></copy>内容的子串（防止重复复制），
    则奖励1/facts中句子的数量分，否则不奖励
    3.3 判断累积的复制命中奖励分数是否大于0.0，如果大于0.0则继续下面答案命中奖励的计算，否则直接奖励0分（防止没有任何有效复制命中却答案正确的情况）

### 4 答案命中奖励计算
    4.1 提取completion中<answer></answer>标签内的内容，去除前后空白、去除标点符号，并转为小写，记为 generated_answer
    4.2 将answer转为小写，去除标点符号，去除前后空白，记为 target_answer
    4.3 如果 target_answer 非空且出现在 generated_answer 中，则奖励+1.0分，否则直接奖励0分
"""

import os
from sys import path

import pytest

from dataset import HotpotQAContext, HotpotQASupportingFacts
from reward import CopyPasteUniORM

# 添加项目根目录到Python路径
path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def test_data():
    """创建测试数据fixture"""
    ctx: HotpotQAContext = {
        "title": ["Article 1", "Article 2"],
        "sentences": [
            [
                "Paris is the capital city of France.",
                "It has a population of about 2.2 million.",
            ],
            [
                "The Eiffel Tower is a famous landmark in Paris.",
                "It was built in 1889.",
            ],
        ],
    }

    facts: HotpotQASupportingFacts = {
        "title": ["Article 1", "Article 2"],
        "sent_id": [0, 0],  # 第一篇文章的第0句，第二篇文章的第0句
    }

    answer = "Paris"

    return ctx, facts, answer


@pytest.fixture
def orm():
    """CopyPasteUniORM实例fixture"""
    return CopyPasteUniORM()


# ============================================================================
# 1. 格式校验测试
# ============================================================================


class TestFormatValidation:
    """测试格式校验规则"""

    def test_1_1_invalid_html_tags_unclosed(self, orm, test_data):
        """测试1.1: HTML标签不合法 - 未闭合标签"""
        ctx, facts, answer = test_data
        completion = """<think>
            <copy>Paris is the capital city of France.</copy>
            <copy>The Eiffel Tower is a famous landmark in Paris.
        </arg_value>
        <answer>Paris</answer>"""

        solution = [{"context": ctx, "supporting_facts": facts, "response": answer}]
        rewards = orm([completion], solution)
        assert rewards[0] == 0.0

    def test_1_1_invalid_html_tags_mismatched(self, orm, test_data):
        """测试1.1: HTML标签不合法 - 标签不匹配"""
        ctx, facts, answer = test_data
        completion = """
        <copy>Paris is the capital city of France.</answer>
        <copy>The Eiffel Tower is a famous landmark in Paris.</copy>
        </think><answer>Paris</answer>"""

        solution = [{"context": ctx, "supporting_facts": facts, "response": answer}]
        rewards = orm([completion], solution)
        assert rewards[0] == 0.0

    def test_1_2_missing_think_tag(self, orm, test_data):
        """测试1.2: 缺少think标签"""
        ctx, facts, answer = test_data
        completion = """
            <copy>Paris is the capital city of France.</copy>
            something
            <copy>The Eiffel Tower is a famous landmark in Paris.</copy>
            <answer>Paris</answer>"""

        solution = [{"context": ctx, "supporting_facts": facts, "response": answer}]
        rewards = orm([completion], solution)
        assert rewards[0] == 0.0

    def test_1_2_missing_answer_tag(self, orm, test_data):
        """测试1.2: 缺少answer标签"""
        ctx, facts, answer = test_data
        completion = """<think>
<copy>Paris is the capital city of France.</copy>The Eiffel Tower is a famous landmark in Paris.</think>"""

        solution = [{"context": ctx, "supporting_facts": facts, "response": answer}]
        rewards = orm([completion], solution)
        assert rewards[0] == 0.0

    def test_1_2_multiple_think_tags(self, orm, test_data):
        """测试1.2: 多个think标签"""
        ctx, facts, answer = test_data
        completion = """<think>
            <copy>Paris is the capital city of France.</copy>
        </think>
        <copy>The Eiffel Tower is a famous landmark in Paris.</copy>
        <think></think><answer>Paris</answer>"""

        solution = [{"context": ctx, "supporting_facts": facts, "response": answer}]
        rewards = orm([completion], solution)
        assert rewards[0] == 0.0

    def test_1_2_multiple_answer_tags(self, orm, test_data):
        """测试1.2: 多个answer标签"""
        ctx, facts, answer = test_data
        completion = """<think>
            <copy>Paris is the capital city of France.</copy>
            <copy>The Eiffel Tower is a famous landmark in Paris.</copy>
        </think>
        <answer>Paris</answer><answer>Paris</answer>"""

        solution = [{"context": ctx, "supporting_facts": facts, "response": answer}]
        rewards = orm([completion], solution)
        assert rewards[0] == 0.0

    def test_1_3_answer_before_think(self, orm, test_data):
        """测试1.3: answer标签在think标签之前"""
        ctx, facts, answer = test_data
        completion = """<answer>Paris</answer>
        <think>
            <copy>Paris is the capital city of France.</copy>
            something
            <copy>The Eiffel Tower is a famous landmark in Paris.</copy>
        </think>"""

        solution = [{"context": ctx, "supporting_facts": facts, "response": answer}]
        rewards = orm([completion], solution)
        assert rewards[0] == 0.0

    def test_1_3_content_before_think(self, orm, test_data):
        """测试1.3: think标签之前有内容"""
        ctx, facts, answer = test_data
        completion = """Some content before think</arg_value>
            <copy>Paris is the capital city of France.</copy>
            something
            <copy>The Eiffel Tower is a famous landmark in Paris.</copy>
        </think>
        <answer>Paris</answer>"""

        solution = [{"context": ctx, "supporting_facts": facts, "response": answer}]
        rewards = orm([completion], solution)
        assert rewards[0] == 0.0

    def test_1_3_content_after_answer(self, orm, test_data):
        """测试1.3: answer标签之后有内容"""
        ctx, facts, answer = test_data
        completion = """<think>
            <copy>Paris is the capital city of France.</copy>
            something
            <copy>The Eiffel Tower is a famous landmark in Paris.</copy>
        </think>
        <answer>Paris</answer>Some content after answer"""

        solution = [{"context": ctx, "supporting_facts": facts, "response": answer}]
        rewards = orm([completion], solution)
        assert rewards[0] == 0.0

    def test_1_3_content_between_think_and_answer(self, orm, test_data):
        """测试1.3: think和answer标签之间有内容"""
        ctx, facts, answer = test_data
        completion = """<think>
            <copy>Paris is the capital city of France.</copy>
            something
            <copy>The Eiffel Tower is a famous landmark in Paris.</copy>
            Some content between think and answer
            <answer>Paris</answer>"""

        solution = [{"context": ctx, "supporting_facts": facts, "response": answer}]
        rewards = orm([completion], solution)
        assert rewards[0] == 0.0

    def test_1_4_insufficient_copy_tags(self, orm, test_data):
        """测试1.4: copy标签数量不足（少于2个）"""
        ctx, facts, answer = test_data
        completion = """<think>
            <copy>Paris is the capital city of France.</copy>
        </think>
        <answer>Paris</answer>"""

        solution = [{"context": ctx, "supporting_facts": facts, "response": answer}]
        rewards = orm([completion], solution)
        assert rewards[0] == 0.0

    def test_1_4_content_between_copy_tags(self, orm, test_data):
        """测试1.4: copy标签之间不存在其他内容"""
        ctx, facts, answer = test_data
        completion = """<think>
            <copy>Paris is the capital city of France.</copy>
            <copy>The Eiffel Tower is a famous landmark in Paris.</copy>
        </think>
        <answer>Paris</answer>"""

        solution = [{"context": ctx, "supporting_facts": facts, "response": answer}]
        rewards = orm([completion], solution)
        assert rewards[0] == 0.0


# ============================================================================
# 2. 长度校验测试
# ============================================================================


class TestLengthValidation:
    """测试长度校验规则"""

    def test_2_1_answer_too_long(self, orm, test_data):
        """测试2.1: answer内容长度超过目标答案的6倍"""
        ctx, facts, answer = test_data
        completion = """<think>
            <copy>Paris is the capital city of France.</copy>
            something
            <copy>The Eiffel Tower is a famous landmark in Paris.</copy>
        </think>
        <answer>This is a very long answer that definitely exceeds three times the length of Paris</answer>"""

        solution = [{"context": ctx, "supporting_facts": facts, "response": answer}]
        rewards = orm([completion], solution)
        assert rewards[0] == 0.0

    def test_2_2_think_shorter_than_answer(self, orm, test_data):
        """测试2.2: think内容长度不大于answer内容长度"""
        ctx, facts, answer = test_data
        completion = """<think><copy>Paris</copy>something<copy>Tower</copy></think>
        <answer>This is a much longer answer than the think content</answer>"""

        solution = [{"context": ctx, "supporting_facts": facts, "response": answer}]
        rewards = orm([completion], solution)
        assert rewards[0] == 0.0

    def test_2_3_think_not_longer_than_facts_average(self, orm, test_data):
        """测试2.3: think内容长度不大于facts中所有句子的平均长度"""
        ctx, facts, answer = test_data
        completion = """<think><copy>Paris</copy>something<copy>Tower</copy></think><answer>Paris</answer>"""

        solution = [{"context": ctx, "supporting_facts": facts, "response": answer}]
        rewards = orm([completion], solution)
        assert rewards[0] == 0.0

    def test_2_4_copy_not_in_facts(self, orm, test_data):
        """测试2.4: copy内容不是facts中某句话的子串"""
        ctx, facts, answer = test_data
        completion = """<think>
            <copy>This content is not in the facts at all.</copy>
            <copy>Neither is this content.</copy>
        </think>
        <answer>Paris</answer>"""

        solution = [{"context": ctx, "supporting_facts": facts, "response": answer}]
        rewards = orm([completion], solution)
        assert rewards[0] == 0.0

    def test_2_5_copy_too_short(self, orm, test_data):
        """测试2.5: copy内容长度小于阈值"""
        ctx, facts, answer = test_data
        completion = """<think>
            <copy>Paris</copy>something<copy>Tower</copy>
        </think>
        <answer>Paris</answer>"""

        solution = [{"context": ctx, "supporting_facts": facts, "response": answer}]
        rewards = orm([completion], solution)
        assert rewards[0] == 0.0


# ============================================================================
# 3. 复制命中奖励计算测试
# ============================================================================


class TestCopyRewardCalculation:
    """测试复制命中奖励计算"""

    def test_3_0_perfect_copy_reward(self, orm, test_data):
        """测试3.0: 完美复制命中奖励"""
        ctx, facts, answer = test_data
        completion = """<think>
            <copy>Paris is the capital city of France.</copy>
            Paris is the capital city of France.
            Paris is the capital city of France.
            Paris is the capital city of France.
            Paris is the capital city of France.
            Paris is the capital city of France.
            Paris is the capital city of France.
            <copy>The Eiffel Tower is a famous landmark in Paris.</copy>
        </think>
        <answer>
            Paris
        </answer>"""

        solution = [{"context": ctx, "supporting_facts": facts, "response": answer}]
        rewards = orm([completion], solution)
        # 2个facts句子，每个有效copy奖励0.5分，共1.0分复制奖励 + 1.0分答案奖励 = 2.0分
        assert abs(rewards[0] - 2.0) < 0.001

    def test_3_1_copy_length_sorting(self, orm, test_data):
        """测试3.1: copy标签按长度排序"""
        ctx, facts, answer = test_data
        # 故意让短的copy在前面，长的在后面
        completion = """<think>
            <copy>Paris is the capital city</copy>
            Paris is the capital city of France.
            Paris is the capital city of France.
            Paris is the capital city of France.
            Paris is the capital city of France.
            Paris is the capital city of France.
            Paris is the capital city of France.
            <copy>Paris is the capital city of France.</copy>
        </think>
        <answer>
            Paris
        </answer>"""

        solution = [{"context": ctx, "supporting_facts": facts, "response": answer}]
        rewards = orm([completion], solution)
        # 只有长的copy应该被奖励，因为"Paris"是"Paris is the capital city of France."的子串
        expected_reward = 0.5 + 1.0  # 0.5复制奖励 + 1.0答案奖励
        assert abs(rewards[0] - expected_reward) < 0.001

    def test_3_2_duplicate_copy_content(self, orm, test_data):
        """测试3.2: 重复复制内容（一个copy是另一个的子串）"""
        ctx, facts, answer = test_data
        completion = """<think>
            <copy>Paris is the capital city of France.</copy>
            something
            <copy>Paris is the capital city of</copy>
        </think>
        <answer>
            Paris
        </answer>"""

        solution = [{"context": ctx, "supporting_facts": facts, "response": answer}]
        rewards = orm([completion], solution)
        # 只有第一个（较长的）copy应该被奖励
        expected_reward = 0.5 + 1.0  # 0.5复制奖励 + 1.0答案奖励
        assert abs(rewards[0] - expected_reward) < 0.001

    def test_3_3_no_valid_copy_but_correct_answer(self, orm, test_data):
        """测试3.3: 没有有效复制命中但答案正确"""
        ctx, facts, answer = test_data
        completion = """<think>
            <copy>Invalid content not in facts</copy>
            something
            <copy>Another invalid content</copy>
        </think>
        <answer>Paris</answer>"""

        solution = [{"context": ctx, "supporting_facts": facts, "response": answer}]
        rewards = orm([completion], solution)
        # 应该返回0分，因为没有有效复制命中
        assert rewards[0] == 0.0


# ============================================================================
# 4. 答案命中奖励计算测试
# ============================================================================


class TestAnswerRewardCalculation:
    """测试答案命中奖励计算"""

    def test_4_1_exact_answer_match(self, orm, test_data):
        """测试4.1-4.3: 精确答案匹配"""
        ctx, facts, answer = test_data
        completion = """<think>
            <copy>Paris is the capital city of France.</copy>
            something
            <copy>The Eiffel Tower is a famous landmark in Paris.</copy>
        </think>
        <answer>Paris</answer>"""

        solution = [{"context": ctx, "supporting_facts": facts, "response": answer}]
        rewards = orm([completion], solution)
        # 应该获得完整的答案奖励
        assert rewards[0] > 1.0  # 至少包含答案奖励

    def test_4_2_answer_with_punctuation(self, orm, test_data):
        """测试4.2: 答案包含标点符号"""
        ctx, facts, answer = test_data
        completion = """<think>
            <copy>Paris is the capital city of France.</copy>
            something
            <copy>The Eiffel Tower is a famous landmark in Paris.</copy>
        </think>
        <answer>Paris!</answer>"""

        solution = [{"context": ctx, "supporting_facts": facts, "response": answer}]
        rewards = orm([completion], solution)
        # 标点符号应该被忽略，仍然匹配
        assert rewards[0] > 1.0

    def test_4_3_answer_case_insensitive(self, orm, test_data):
        """测试4.3: 答案大小写不敏感"""
        ctx, facts, answer = test_data
        completion = """<think>
            <copy>Paris is the capital city of France.</copy>
            something
            <copy>The Eiffel Tower is a famous landmark in Paris.</copy>
        </think>
        <answer>PARIS</answer>"""

        solution = [{"context": ctx, "supporting_facts": facts, "response": answer}]
        rewards = orm([completion], solution)
        # 大小写应该被忽略，仍然匹配
        assert rewards[0] > 1.0

    def test_4_3_answer_as_substring(self, orm, test_data):
        """测试4.3: 目标答案是生成答案的子串"""
        ctx, facts, answer = test_data
        completion = """<think>
            <copy>Paris is the capital city of France.</copy>
            something
            <copy>The Eiffel Tower is a famous landmark in Paris.</copy>
        </think>
        <answer>The capital of France is Paris</answer>"""

        solution = [{"context": ctx, "supporting_facts": facts, "response": answer}]
        rewards = orm([completion], solution)
        # "Paris"应该被包含在生成的答案中
        assert rewards[0] > 1.0

    def test_4_3_wrong_answer(self, orm, test_data):
        """测试4.3: 错误答案"""
        ctx, facts, answer = test_data
        completion = """<think>
            <copy>Paris is the capital city of France.</copy>
            something
            <copy>The Eiffel Tower is a famous landmark in Paris.</copy>
        </think>
        <answer>London</answer>"""

        solution = [{"context": ctx, "supporting_facts": facts, "response": answer}]
        rewards = orm([completion], solution)
        # 错误答案应该返回0分
        assert rewards[0] == 0.0


# ============================================================================
# 综合测试和边界情况
# ============================================================================


class TestComprehensiveAndEdgeCases:
    """综合测试和边界情况"""

    def test_perfect_case_maximum_reward(self, orm, test_data):
        """测试完美情况：获得最大奖励2.0分"""
        ctx, facts, answer = test_data
        completion = """<think>
            <copy>Paris is the capital city of France.</copy>
            something
            <copy>The Eiffel Tower is a famous landmark in Paris.</copy>
        </think>
        <answer>Paris</answer>"""

        solution = [{"context": ctx, "supporting_facts": facts, "response": answer}]
        rewards = orm([completion], solution)
        assert abs(rewards[0] - 2.0) < 0.001

    def test_empty_completion(self, orm, test_data):
        """测试空completion"""
        ctx, facts, answer = test_data
        completion = ""

        solution = [{"context": ctx, "supporting_facts": facts, "response": answer}]
        rewards = orm([completion], solution)
        assert rewards[0] == 0.0

    def test_multiple_completions_batch_processing(self, orm, test_data):
        """测试多个completion的批量处理"""
        ctx, facts, answer = test_data

        completions = [
            # 完美情况
            """<think>
                <copy>Paris is the capital city of France.</copy>
                something
                <copy>The Eiffel Tower is a famous landmark in Paris.</copy>
            </think>
            <answer>Paris</answer>""",
            # 格式错误
            """<answer>Paris</answer>""",
            # 复制内容错误
            """<think>
                <copy>Invalid content</copy>
                something
                <copy>Another invalid</copy>
            </think>
            <answer>Paris</answer>""",
            # 答案错误
            """<think>
                <copy>Paris is the capital city of France.</copy>
                something
                <copy>The Eiffel Tower is a famous landmark in Paris.</copy>
            </think>
            <answer>London</answer>""",
        ]

        solutions = [
            {"context": ctx, "supporting_facts": facts, "response": answer},
            {"context": ctx, "supporting_facts": facts, "response": answer},
            {"context": ctx, "supporting_facts": facts, "response": answer},
            {"context": ctx, "supporting_facts": facts, "response": answer},
        ]

        rewards = orm(completions, solutions)

        assert abs(rewards[0] - 2.0) < 0.001  # 完美情况
        assert rewards[1] == 0.0  # 格式错误
        assert rewards[2] == 0.0  # 复制内容错误
        assert rewards[3] == 0.0  # 答案错误

    def test_partial_copy_reward_with_correct_answer(self, orm, test_data):
        """测试部分复制命中但答案正确"""
        ctx, facts, answer = test_data
        completion = """<think>
            <copy>Paris is the capital city of France.</copy>
            something
            <copy>Invalid content not in facts</copy>
        </think>
        <answer>Paris</answer>"""

        solution = [{"context": ctx, "supporting_facts": facts, "response": answer}]
        rewards = orm([completion], solution)
        expected_reward = 0.5 + 1.0  # 0.5复制奖励 + 1.0答案奖励
        assert abs(rewards[0] - expected_reward) < 0.001

    def test_copy_tags_with_whitespace(self, orm, test_data):
        """测试copy标签之间只有空白字符（不允许允许）"""
        ctx, facts, answer = test_data
        completion = """<think>
            <copy>Paris is the capital city of France.</copy>   \n\t
            <copy>The Eiffel Tower is a famous landmark in Paris.</copy>
        </think>
        <answer>Paris</answer>"""

        solution = [{"context": ctx, "supporting_facts": facts, "response": answer}]
        rewards = orm([completion], solution)
        # 应该获得满分，因为空白字符是允许的
        assert rewards[0] == 0

    def test_empty_facts_handling(self, orm):
        """测试空facts的处理"""
        ctx: HotpotQAContext = {"title": [], "sentences": []}
        facts: HotpotQASupportingFacts = {"title": [], "sent_id": []}
        answer = "Paris"

        completion = """<think>
            <copy>Paris is the capital city of France.</copy>
            something
            <copy>The Eiffel Tower is a famous landmark in Paris.</copy>
        </think>
        <answer>Paris</answer>"""

        solution = [{"context": ctx, "supporting_facts": facts, "response": answer}]
        rewards = orm([completion], solution)
        # 空facts应该导致0分
        assert rewards[0] == 0.0

    def test_single_fact_scenario(self, orm):
        """测试只有一个fact的场景"""
        ctx: HotpotQAContext = {
            "title": ["Article 1"],
            "sentences": [["Paris is the capital city of France."]],
        }

        facts: HotpotQASupportingFacts = {"title": ["Article 1"], "sent_id": [0]}

        answer = "Paris"

        completion = """<think>
            <copy>Paris is the capital city of France.</copy>
            something
            <copy>Paris is the capital city of</copy>
        </think>
        <answer>Paris</answer>"""

        solution = [{"context": ctx, "supporting_facts": facts, "response": answer}]
        rewards = orm([completion], solution)
        # 1个fact，每个有效copy奖励1.0分，但最大复制奖励为1.0分
        expected_reward = 1.0 + 1.0  # 1.0复制奖励 + 1.0答案奖励
        assert abs(rewards[0] - expected_reward) < 0.001


# ============================================================================
# 性能和压力测试
# ============================================================================


class TestPerformanceAndStress:
    """性能和压力测试"""

    def test_large_batch_processing(self, orm, test_data):
        """测试大批量处理"""
        ctx, facts, answer = test_data
        completion = """<think>
            <copy>Paris is the capital city of France.</copy>
            something
            <copy>The Eiffel Tower is a famous landmark in Paris.</copy>
        </think>
        <answer>Paris</answer>"""

        # 创建1000个相同的completion
        completions = [completion] * 1000
        solutions = [{"context": ctx, "supporting_facts": facts, "response": answer}] * 1000

        rewards = orm(completions, solutions)

        # 所有奖励都应该是2.0
        assert all(abs(r - 2.0) < 0.001 for r in rewards)
        assert len(rewards) == 1000

    def test_very_long_completion(self, orm, test_data):
        """测试非常长的completion"""
        ctx, facts, answer = test_data

        # 创建一个很长的completion
        long_copy = "Paris is the capital city of France. " * 100
        completion = f"""<think>
            <copy>{long_copy}</copy>
            something
            <copy>The Eiffel Tower is a famous landmark in Paris.</copy>
        </think>
        <answer>Paris</answer>"""

        solution = [{"context": ctx, "supporting_facts": facts, "response": answer}]
        rewards = orm([completion], solution)

        # 应该正常处理并返回分数
        assert rewards[0] == 1.5
