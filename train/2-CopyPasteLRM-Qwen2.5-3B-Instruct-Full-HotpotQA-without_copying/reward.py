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

    def validate_html_tags(self, text: str) -> bool:
        """校验HTML标签是否合法（使用栈进行匹配）"""
        stack = []
        matches = self.html_tag_pattern.finditer(text)

        for match in matches:
            is_closing = match.group(1) == "/"
            tag_name = match.group(2).lower()

            # 只检查我们关心的HTML标签（copy, answer）
            if tag_name not in ["answer", "think"]:
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


class LengthValidator:
    """长度校验模块"""

    def __init__(self):
        pass

    def validate_answer_length(self, answer_content: str, target_answer: str) -> bool:
        """校验答案长度不超过目标答案的3倍"""
        if len(target_answer.strip()) == 0 or len(answer_content.strip()) == 0:
            # 如果Ground Truth Answer为空，则答案无效
            return False
        return len(answer_content.strip()) <= len(target_answer.strip()) * 6

    def validate_think_vs_answer_length(self, think_content: str, answer_content: str) -> bool:
        """校验think内容长度大于answer内容长度"""
        return len(think_content) > len(answer_content)


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


class CopyPasteWithOutCopyingORM(ORM):
    """统一的复制粘贴奖励函数"""

    def __init__(self):
        self.format_validator = FormatValidator()
        self.length_validator = LengthValidator()
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

        # 2. 校验答案不超过目标答案长度的3倍
        if not self.length_validator.validate_answer_length(answer_content, answer):
            return 0.0

        # 3. 校验think内容长度大于answer内容长度
        if not self.length_validator.validate_think_vs_answer_length(think_content, answer_content):
            return 0.0

        # 4. 答案命中奖励计算
        answer_reward = self.answer_reward_calculator.calculate_answer_reward(completion, answer)

        return answer_reward


orms["copypaste_without_copying"] = CopyPasteWithOutCopyingORM
