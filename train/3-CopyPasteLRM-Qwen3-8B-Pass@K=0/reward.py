import re
from typing import List, Optional, Tuple

from swift.plugin import ORM, orms
from copypastelrm.metrics.HotpotQA import update_sp, f1_score, exact_match_score, hit_answer
from copypastelrm.metrics.utils import remove_evidence_tags


class FormatValidator:
    """格式校验模块"""

    def __init__(self):
        self.html_tag_pattern = re.compile(r"<(/?)(\w+)>")
        self.think_pattern = re.compile(
            r"<|think|>(.*?)</|think|>", re.DOTALL | re.IGNORECASE
        )
        self.answer_pattern = re.compile(
            r"<|answer|>(.*?)</|answer|>", re.DOTALL | re.IGNORECASE
        )
        self.copy_pattern = re.compile(
            r"<|EVIDENCE|>(.*?)</|EVIDENCE|>", re.DOTALL | re.IGNORECASE
        )

    def validate_html_tags(self, text: str) -> bool:
        """校验HTML标签是否合法（使用栈进行匹配）"""
        stack = []
        matches = self.html_tag_pattern.finditer(text)

        for match in matches:
            is_closing = match.group(1) == "/"
            tag_name = match.group(2).lower()

            # 只检查我们关心的HTML标签（copy, answer）
            if tag_name not in ["evidence", "answer", "think"]:
                continue

            if is_closing:
                if not stack or stack[-1] != tag_name:
                    return False
                stack.pop()
            else:
                stack.append(tag_name)

        return len(stack) == 0

    def validate_structure(
        self, text: str
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """_summary_

        Args:
            text (str): _description_

        Returns:
            Tuple[Optional[str], Optional[str], bool]:
                - is_all_valid: 是否全部合法
                - think_content: think标签中的内容
                - answer_content: answer标签中的内容
        """
        think_matches = list(self.think_pattern.finditer(text))
        answer_matches = list(self.answer_pattern.finditer(text))

        is_all_valid = True

        if len(think_matches) == 1:
            think_match = think_matches[0]
            think_content = think_match.group(1)
        else:
            think_content = None
            is_all_valid = False

        if len(answer_matches) == 1:
            answer_match = answer_matches[0]
            answer_content = answer_match.group(1)
        else:
            answer_content = None
            is_all_valid = False

        if think_content and answer_content:
            # 检查不嵌套（think在answer之前）
            if think_match.start() >= answer_match.start():
                is_all_valid = False

            # 检查 think 之前无内容, answer 之后无内容
            if think_match.start() > 0 or answer_match.end() < len(text):
                is_all_valid = False

            # 检查 think 和 answer 之间只允许空白字符
            between_text = text[think_match.end() : answer_match.start()]
            if between_text.strip() != "":
                is_all_valid = False

        return is_all_valid, think_content, answer_content

    def validate_evidence_tags(self, think_content: str) -> bool:
        """校验think中的copy标签"""

        if think_content is None:
            return False
        
        # 使用大小写敏感的正则表达式
        copy_matches = list(self.copy_pattern.finditer(think_content))

        # 至少一对copy标签
        if len(copy_matches) < 1:
            return False

        # 检查每个copy标签的内容不能为空
        for match in copy_matches:
            copy_content = match.group(1).strip()
            if not copy_content:  # 如果内容为空
                return False

        # 检查copy标签之间必须有非空白字符
        for i in range(len(copy_matches) - 1):
            between_content = think_content[
                copy_matches[i].end() : copy_matches[i + 1].start()
            ]
            if not between_content.strip():  # 如果之间只有空白字符
                return False

        return True

    def get_predict_facts(self, think_content: str) -> List[str]:
        """获取predict_facts"""
        copy_matches = list(self.copy_pattern.finditer(think_content))

        predict_facts = []
        # 检查每个copy标签的内容不能为空
        for match in copy_matches:
            copy_content = match.group(1).strip()
            if copy_content:
                predict_facts.append(copy_content.strip())

        return predict_facts


class LengthValidator:
    """长度校验模块"""

    def __init__(self):
        self.format_validator = FormatValidator()

    def validate_think_vs_answer_length(
        self, think_content: str, answer_content: str
    ) -> bool:
        """think内容长度大于answer内容长度"""
        return len(think_content) > len(answer_content)

    def validate_think_vs_facts_length(self, think_content: str, times: float = 2.0) -> bool:
        predict_facts = self.format_validator.get_predict_facts(think_content)

        predict_facts_length = sum(len(fact) for fact in predict_facts)

        return len(think_content) > predict_facts_length * times


class FormatReward(ORM):
    """Think和Answer格式奖励"""

    def __init__(self):
        self.format_validator = FormatValidator()  # 初始化格式验证器

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        rewards = []

        for completion in completions:
            valid_html_tags = self.format_validator.validate_html_tags(completion)

            reward_score = 0.0

            # 如果标签不闭合，没有任何奖励
            if not valid_html_tags:
                rewards.append(0.0)
                continue

            is_all_valid, think_content, answer_content = (
                self.format_validator.validate_structure(completion)
            )

            # 如果完美匹配，则奖励0.5，下面的两个0.25自然也会被加上，即1.0满分
            if is_all_valid:
                reward_score += 0.5

            # 如果think格式正确，奖励0.25
            if think_content:
                reward_score += 0.25

            # 如果answer格式正确，奖励0.25
            if answer_content:
                reward_score += 0.25

            rewards.append(reward_score)

        return rewards


class LengthtReward(ORM):
    """计算模型生成内容的长度结构奖励。
    
    该类通过对比推理过程（think）与最终答案（answer）及预测事实（facts）的长度关系，
    鼓励模型产生详尽的推理链条。

    奖励逻辑说明：
    1. 格式校验：首先检查 completion 是否符合基础的结构化格式要求。
    2. 深度推理奖励 (+0.6)：
       如果模型推理内容（think_content）的长度显著大于最终答案（answer_content），
       说明模型进行了充分的思考，而非直接跳到结论。
    3. 推理独立性奖励 (+0.4)：
       如果推理内容（think_content）的长度大于其提取的预测事实（predict_facts）的两倍，
       说明模型在引用事实之外，还进行了额外的逻辑推导或分析，而非简单的资料堆砌。

    Attributes:
        length_validator: 长度校验器，内部封装了具体的长度比较逻辑（如字符数或 Token 数对比）。
        format_validator: 格式验证器，用于拆分内容中的 think、answer 以及 facts 部分。
    """

    def __init__(self):
        self.length_validator = LengthValidator()
        self.format_validator = FormatValidator()

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        rewards = []

        for completion in completions:

            is_all_valid, think_content, answer_content = (
                self.format_validator.validate_structure(completion)
            )

            reward_score = 0.0

            # 思考长度是否大于答案长度
            if is_all_valid and self.length_validator.validate_think_vs_answer_length(
                think_content, answer_content
            ):
                reward_score += 0.6

            # 思考长度是否大于预测事实长度
            if think_content and self.length_validator.validate_think_vs_facts_length(
                think_content,
                times=2.0
            ):
                reward_score += 0.4

            rewards.append(reward_score)

        return rewards


class CopyReward(ORM):
    """计算模型在推理过程中对关键事实提取（Copying）能力的奖励分值。
    
    该类通过对比模型 `think` 模块提取的片段与参考答案中的 `supporting_facts`（支撑事实）以及 
    `context`（全文背景）的重合度来计算奖励。

    奖励逻辑说明：
    1. 格式校验：首选通过 format_validator 检查 completion 是否符合预定义结构。若不合规，返回 0.0。
    2. 核心事实提取 (Facts Copied)：计算模型提取的内容与金标准支撑事实（gold_sfs）之间的 F1 分数。
    3. 干扰项控制 (Non-facts Copied)：计算模型提取的内容中，属于背景信息但非核心事实部分的精确率（Precision）。
    4. 加权求和：最终奖励 = (facts_copied_weight * F1) + (non_facts_copied_weight * Precision)。

    Attributes:
        format_validator: 格式验证器，负责解析推理内容并提取模型预测的支撑事实。
        non_facts_copied_weight (float): 对复制非核心事实行为的奖励权重（默认为 0.1）。
        facts_copied_weight (float): 对正确复制支撑事实行为的奖励权重（默认为 0.9）。
    """

    def __init__(self):
        self.format_validator = FormatValidator()

        self.non_facts_copied_weight = 0.05
        self.facts_copied_weight = 0.95

    def __call__(
        self, completions: List[str], solution: List[dict], **kwargs
    ) -> List[float]:
        rewards = []

        for completion, sol in zip(completions, solution):
            ctx = sol["context"]
            facts = sol["supporting_facts"]
            is_think_answer_valid, think_content, _ = self.format_validator.validate_structure(
                completion
            )

            # 如果EVIDENCE格式不合规，则不给予任何奖励
            is_evidence_valid = self.format_validator.validate_evidence_tags(think_content)

            if not is_think_answer_valid or not is_evidence_valid:
                rewards.append(0.0)
                continue

            predict_facts = self.format_validator.get_predict_facts(think_content)

            metrics = {
                "sp_em": 0.0,
                "sp_f1": 0.0,
                "sp_prec": 0.0,
                "sp_recall": 0.0,
            }

            _, _, _, facts_copied_f1 = update_sp(
                metrics=metrics, predict_sfs=predict_facts, gold_sfs=facts
            )

            ctx_without_facts = remove_evidence_tags(ctx)
            _, non_facts_copied_prec, _, _ = update_sp(
                metrics=metrics, predict_sfs=predict_facts, gold_sfs=ctx_without_facts
            )

            reward = (
                self.non_facts_copied_weight * non_facts_copied_prec
                + self.facts_copied_weight * facts_copied_f1
            )

            rewards.append(reward)

        return rewards


class AnswerLooseReward(ORM):
    """计算模型生成回答的奖励分值。
    
    该类通过验证回答的结构格式、F1 分数来评估生成内容的质量。
    奖励逻辑遵循以下优先级降级机制：
    1. 格式校验：如果格式非法，奖励为 0.0。
    3. 模糊匹配：计算与参考答案的最佳 F1 分数，并按 f1_weight 比例进行缩放。

    Attributes:
        format_validator: 用于校验生成内容是否符合预定义结构的验证器。
    """

    def __init__(self):
        self.format_validator = FormatValidator()

    def __call__(
        self, completions: List[str], solution: List[dict], **kwargs
    ) -> List[float]:
        rewards = []

        for completion, sol in zip(completions, solution):
            is_all_valid, _, answer_content = self.format_validator.validate_structure(completion)
            if not is_all_valid:
                rewards.append(0.0)
                continue

            gold_answers = sol["answers"]

            best_f1 = 0.0

            for gold_answer in gold_answers:
                ans_f1, _, _ = f1_score(answer_content, gold_answer)
                best_f1 = max(best_f1, ans_f1)

            rewards.append(best_f1)

        return rewards


class AnswerStrictReward(ORM):
    """计算模型生成回答的奖励分值。
    
    严格模式，计算EM，如果EM不匹配，则降级计算HIT，并且按照答案长度进行惩罚。

    1. 如果EM==True，则奖励为1.0
    2. HIT的奖励等于，HIT的长度/答案的长度

    Attributes:
        format_validator: 用于校验生成内容是否符合预定义结构的验证器。
    """

    def __init__(self):
        self.format_validator = FormatValidator()

    def __call__(
        self, completions: List[str], solution: List[dict], **kwargs
    ) -> List[float]:
        rewards = []

        for completion, sol in zip(completions, solution):
            is_all_valid, _, answer_content = self.format_validator.validate_structure(completion)
            if not is_all_valid:
                rewards.append(0.0)
                continue

            gold_answers = sol["answers"]

            em = exact_match_score(answer_content, gold_answers)
            if em:
                reward = 1.0

            hit_answer = hit_answer(answer_content, gold_answers)
            if hit_answer:
                reward = len(hit_answer) / len(answer_content)

            rewards.append(reward)

        return rewards


orms["cplrm_format"] = FormatReward
orms["cplrm_length"] = LengthtReward
orms["cplrm_copy"] = CopyReward
orms["cplrm_loose_answer"] = AnswerLooseReward
orms["cplrm_strict_answer"] = AnswerStrictReward