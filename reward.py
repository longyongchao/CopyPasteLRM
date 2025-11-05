import re
import string
from typing import List, Optional, Tuple

from swift.plugin import ORM, orms

from dataset import HotpotQAContext, HotpotQASupportingFacts


class FormatValidator:
    """格式校验模块"""

    def __init__(self):
        self.html_tag_pattern = re.compile(r"<(/?)(\w+)>")
        self.think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
        self.answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
        self.copy_pattern = re.compile(r"<copy>(.*?)</copy>", re.DOTALL | re.IGNORECASE)

    def validate_html_tags(self, text: str) -> bool:
        """校验HTML标签是否合法（使用栈进行匹配）"""
        stack = []
        matches = self.html_tag_pattern.finditer(text)

        for match in matches:
            is_closing = match.group(1) == "/"
            tag_name = match.group(2).lower()

            # 只检查我们关心的HTML标签（copy, answer）
            if tag_name not in ["copy", "answer", "think"]:
                continue

            if is_closing:
                if not stack or stack[-1] != tag_name:
                    return False
                stack.pop()
            else:
                stack.append(tag_name)

        return len(stack) == 0

    def validate_structure(self, text: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """校验文本结构，返回(是否有效, think内容, answer内容)"""
        think_matches = list(self.think_pattern.finditer(text))
        answer_matches = list(self.answer_pattern.finditer(text))

        # 检查各出现一次
        if len(think_matches) != 1 or len(answer_matches) != 1:
            return False, None, None

        think_match = think_matches[0]
        answer_match = answer_matches[0]

        think_content = think_match.group(1)
        answer_content = answer_match.group(1)

        # 检查不嵌套（think在answer之前）
        if think_match.start() >= answer_match.start():
            return False, None, None

        # 检查 think 之前无内容, answer 之后无内容
        if think_match.start() > 0 or answer_match.end() < len(text):
            return False, None, None

        # 检查 think 和 answer 之间只允许空白字符
        between_text = text[think_match.end() : answer_match.start()]
        if between_text.strip() != "":
            return False, None, None

        return True, think_content, answer_content

    def validate_copy_tags(self, think_content: str) -> bool:
        """校验think中的copy标签"""
        # 使用大小写敏感的正则表达式，只匹配小写copy标签
        copy_pattern_case_sensitive = re.compile(r"<copy>(.*?)</copy>", re.DOTALL)
        copy_matches = list(copy_pattern_case_sensitive.finditer(think_content))

        # 至少两对copy标签
        if len(copy_matches) < 2:
            return False

        # 检查每个copy标签的内容不能为空
        for match in copy_matches:
            copy_content = match.group(1).strip()
            if not copy_content:  # 如果内容为空
                return False

        # 检查copy标签之间必须有非空白字符
        for i in range(len(copy_matches) - 1):
            between_content = think_content[copy_matches[i].end() : copy_matches[i + 1].start()]
            if not between_content.strip():  # 如果之间只有空白字符
                return False

        return True


class LengthValidator:
    """长度校验模块"""

    def __init__(self):
        self.copy_pattern = re.compile(r"<copy>(.*?)</copy>", re.DOTALL | re.IGNORECASE)

    def get_facts_sentences(self, ctx: HotpotQAContext, facts: HotpotQASupportingFacts) -> List[str]:
        """获取facts中的所有句子"""
        sentences = []
        titles = facts.get("title", [])
        sent_ids = facts.get("sent_id", [])

        for title, sent_id in zip(titles, sent_ids):
            if title in ctx.get("title", []):
                title_idx = ctx["title"].index(title)
                if title_idx < len(ctx.get("sentences", [])) and sent_id < len(ctx["sentences"][title_idx]):
                    sentences.append(ctx["sentences"][title_idx][sent_id])

        return sentences

    def validate_answer_length(self, answer_content: str, target_answer: str) -> bool:
        """校验答案长度不超过目标答案的3倍"""
        if len(target_answer.strip()) == 0 or len(answer_content.strip()) == 0:
            # 如果Ground Truth Answer为空，则答案无效
            return False
        return len(answer_content.strip()) <= len(target_answer.strip()) * 6

    def validate_think_vs_answer_length(self, think_content: str, answer_content: str) -> bool:
        """校验think内容长度大于answer内容长度"""
        return len(think_content) > len(answer_content)

    def validate_think_vs_facts_length(self, think_content: str, facts_sentences: List[str]) -> bool:
        """校验think内容长度大于facts中所有句子的平均长度"""
        if not facts_sentences:
            return False

        think_length = len(think_content)
        avg_facts_length = sum(len(sent) for sent in facts_sentences) / len(facts_sentences)

        return think_length > avg_facts_length

    def validate_copy_in_facts(self, think_content: str, facts_sentences: List[str]) -> bool:
        """校验至少一个copy内容是facts中某句话的子串"""
        copies = self.copy_pattern.findall(think_content)

        for copy_content in copies:
            copy_content = copy_content.strip()
            if not copy_content:
                continue

            is_substring = any(copy_content in sent for sent in facts_sentences)
            if is_substring:
                return True

        return False

    def validate_copy_min_length(self, think_content: str, facts_sentences: List[str]) -> bool:
        """校验每个copy内容的长度>=min(facts平均长度的一半, facts最短句子长度)"""
        copies = self.copy_pattern.findall(think_content)

        if not facts_sentences:
            return False

        avg_length = sum(len(sent) for sent in facts_sentences) / len(facts_sentences)
        min_length = min(len(sent) for sent in facts_sentences)
        threshold = min(avg_length / 2, min_length)

        for copy_content in copies:
            copy_content = copy_content.strip()
            if len(copy_content) < threshold:
                return False

        return True


class CopyRewardCalculator:
    """复制命中奖励计算模块"""

    def __init__(self):
        self.copy_pattern = re.compile(r"<copy>(.*?)</copy>", re.DOTALL | re.IGNORECASE)

    def calculate_copy_reward(self, think_content: str, facts_sentences: List[str]) -> float:
        """计算复制命中奖励"""
        copies = self.copy_pattern.findall(think_content)

        if not facts_sentences:
            return 0.0

        # 按长度从长到短排序
        sorted_copies = sorted(copies, key=len, reverse=True)

        reward_per_sentence = 1.0 / len(facts_sentences)
        total_reward = 0.0
        used_substrings = set()

        for copy_content in sorted_copies:
            copy_content = copy_content.strip()
            if not copy_content:
                continue

            # 检查是否是facts中某句话的子串
            is_valid = False
            for sent in facts_sentences:
                if copy_content in sent:
                    # 检查是否不是先前copy内容的子串
                    is_not_subset = all(copy_content not in used for used in used_substrings)
                    if is_not_subset:
                        is_valid = True
                        used_substrings.add(copy_content)
                        break

            if is_valid:
                total_reward += reward_per_sentence

        return total_reward


class AnswerRewardCalculator:
    """答案命中奖励计算模块"""

    def __init__(self):
        self.answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)

    def clean_text(self, text: str) -> str:
        """清理文本：去除空白、标点符号，转为小写"""
        # 去除标点符号
        text_no_punct = "".join(char for char in text if char not in string.punctuation)
        # 将连续的空白字符（包括空格、制表符、换行符）替换为单个空格
        text_normalized = " ".join(text_no_punct.split())
        # 去除首尾空白并转为小写
        return text_normalized.strip().lower()

    def calculate_answer_reward(self, completion: str, target_answer: str) -> float:
        """计算答案命中奖励"""
        match = self.answer_pattern.search(completion)
        if not match:
            return 0.0

        generated_answer = self.clean_text(match.group(1))
        target_answer_clean = self.clean_text(target_answer)

        if target_answer_clean and target_answer_clean in generated_answer:
            return 1.0

        return 0.0


class CopyPasteUniORM(ORM):
    """统一的复制粘贴奖励函数"""

    def __init__(self):
        self.format_validator = FormatValidator()
        self.length_validator = LengthValidator()
        self.copy_reward_calculator = CopyRewardCalculator()
        self.answer_reward_calculator = AnswerRewardCalculator()

    def __call__(self, completions: List[str], solution: List[dict], **kwargs) -> List[float]:
        """计算奖励分数"""
        rewards = []

        for completion, sol in zip(completions, solution):
            ctx: HotpotQAContext = sol.get("context", {})
            facts: HotpotQASupportingFacts = sol.get("supporting_facts", {})
            answer: str = sol.get("response", "")

            reward = self._calculate_single_reward(completion, ctx, facts, answer)
            rewards.append(reward)

        return rewards

    def _calculate_single_reward(
        self,
        completion: str,
        ctx: HotpotQAContext,
        facts: HotpotQASupportingFacts,
        answer: str,
    ) -> float:
        """计算单个completion的奖励分数"""

        # 1. 格式校验
        if not self.format_validator.validate_html_tags(completion):
            return 0.0

        (
            is_valid,
            think_content,
            answer_content,
        ) = self.format_validator.validate_structure(completion)
        if not is_valid or not think_content or not answer_content:
            return 0.0

        if not self.format_validator.validate_copy_tags(think_content):
            return 0.0

        # 获取facts句子
        facts_sentences = self.length_validator.get_facts_sentences(ctx, facts)

        # 2. 长度校验
        if not self.length_validator.validate_answer_length(answer_content, answer):
            return 0.0

        if not self.length_validator.validate_think_vs_answer_length(think_content, answer_content):
            return 0.0

        if not self.length_validator.validate_think_vs_facts_length(think_content, facts_sentences):
            return 0.0

        if not self.length_validator.validate_copy_in_facts(think_content, facts_sentences):
            return 0.0

        if not self.length_validator.validate_copy_min_length(think_content, facts_sentences):
            return 0.0

        # 3. 复制命中奖励计算
        copy_reward = self.copy_reward_calculator.calculate_copy_reward(think_content, facts_sentences)

        # 如果没有有效复制命中，直接返回0分
        if copy_reward <= 0.0:
            return 0.0

        # 4. 答案命中奖励计算
        answer_reward = self.answer_reward_calculator.calculate_answer_reward(completion, answer)

        # 如果答案错误，直接返回0分（防止没有任何有效复制命中却答案正确的情况）
        if answer_reward <= 0.0:
            return 0.0

        # 总奖励：复制命中奖励 + 答案命中奖励，最大2.0分
        total_reward = copy_reward + answer_reward
        return min(total_reward, 2.0)


orms["copypaste_uni"] = CopyPasteUniORM
