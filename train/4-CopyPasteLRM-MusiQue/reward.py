import re
from typing import List, Optional, Tuple, Literal
import os
import numpy as np

from swift.plugin import ORM, orms
from copypastelrm.metrics.HotpotQA import (
    update_sp,
    f1_score,
    exact_match_score,
    hit_answer,
)
from copypastelrm.metrics.utils import remove_evidence_tags
from copypastelrm.metrics.HotpotQA import normalize_answer, hit_answer


class FormatValidator:
    """格式校验模块"""

    def __init__(self):
        self.html_tag_pattern = re.compile(r"<(/?)(\w+)>")
        self.think_pattern = re.compile(
            r"<\|think\|>(.*?)</\|think\|>", re.DOTALL | re.IGNORECASE
        )
        self.answer_pattern = re.compile(
            r"<\|answer\|>(.*?)</\|answer\|>", re.DOTALL | re.IGNORECASE
        )
        self.copy_pattern = re.compile(
            r"<\|EVIDENCE\|>(.*?)</\|EVIDENCE\|>", re.DOTALL | re.IGNORECASE
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

    def validate_evidence_tags(self, think_content: str, at_least: int = 1) -> bool:
        """校验think中的copy标签"""

        if think_content is None:
            return False

        # 使用大小写敏感的正则表达式
        copy_matches = list(self.copy_pattern.finditer(think_content))

        # 至少一对copy标签
        if len(copy_matches) < at_least:
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

    def validate_think_vs_facts_length(
        self, think_content: str, times: float = 2.0
    ) -> bool:
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
                think_content, times=2.0
            ):
                reward_score += 0.4

            rewards.append(reward_score)

        return rewards


class CopyReward(ORM):
    """
    CopyReward 类：用于计算“复制”行为的奖励。
    该奖励函数旨在鼓励模型从上下文中提取（复制）正确的支撑事实（Supporting Facts）。
    """

    def __init__(self):
        # 从环境变量获取奖励模式，默认为 'dense' (稠密模式)
        # dense: 必须找全所有支撑事实才给满分
        # sparse: 找到部分支撑事实也给分
        # 检查COPY_REWARD_MODE变量是否存在
        if "COPY_REWARD_MODE" in os.environ:
            self.copy_reward_mode: Literal['dense', 'sparse'] = os.getenv("COPY_REWARD_MODE", "dense") 
        else:
            raise ValueError("Environment variable 'COPY_REWARD_MODE' is not set. Please set it to 'dense' or 'sparse'.")
        
        # 初始化格式验证器，用于检查模型输出是否包含规范的 XML 标签（如 <think>, <evidence> 等）
        self.format_validator = FormatValidator()

        self.perfert_match_bonus = 1.5  # 完美匹配奖励系数
    
    @staticmethod
    def _compute_supporting_facts_reward(predict_sfs: List[str], gold_sfs: List[List[str]]) -> float:
        """
        计算预测的支撑事实与真实标签（Gold Supporting Facts）之间的匹配情况。
        
        Args:
            predict_sfs: 模型预测出的支撑事实列表（字符串列表）。
            gold_sfs: 真实的支撑事实列表。结构通常为 [[句子1片段...], [句子2片段...]]。

        Returns:
            reward_across_facts: 一个由 0 和 1 组成的列表，表示每个真实事实是否被成功预测。
        """
        # 初始化每个真实事实的奖励为 0
        reward_across_facts = [0] * len(gold_sfs)
        # 标记每个真实事实是否已被匹配过，防止重复计算
        hit_tag = [False] * len(gold_sfs)

        # 将每个真实事实（可能被分割为列表）拼接成完整的字符串
        gold_sfs_flat = [" ".join(fact) for fact in gold_sfs] 

        for predict_sf in predict_sfs:
            # 使用 hit_answer 函数（外部定义）判断预测片段是否命中某个真实事实
            hit_gold_sf_flat = hit_answer(predict_sf, gold_sfs_flat)
            
            # 如果命中了某个真实事实
            if hit_gold_sf_flat:
                # 找到该事实在列表中的索引
                idx = gold_sfs_flat.index(hit_gold_sf_flat)
                
                # 计算该真实事实中最短部分的长度（可能是为了防止模型只复制极短的片段来骗取奖励）
                # 注意：这里假设 gold_sfs[idx] 是一个列表，包含该事实的各个组成部分
                miniemum_sf_length = min([len(gold_sf) for gold_sf in gold_sfs[idx]])
                
                # 核心判断逻辑：
                # 1. 确实命中了 (hit_gold_sf_flat 非空)
                # 2. 该事实之前没有被命中过 (hit_tag[idx] == False) -> 避免模型重复输出同一事实刷分
                if hit_gold_sf_flat and hit_tag[idx] == False:
                    # 3. 长度校验：模型预测（复制）的内容长度必须大于等于真实事实中最短部分的长度
                    if len(predict_sf) >= miniemum_sf_length: 
                        reward_across_facts[idx] = 1 # 标记该事实已解决，奖励置为 1
                        hit_tag[idx] = True          # 更新命中标记
        
        return reward_across_facts

    def __call__(
        self, completions: List[str], solution: List[dict], **kwargs
    ) -> List[float]:
        """
        计算一批样本的奖励分数。
        
        Args:
            completions: 模型生成的文本列表。
            solution: 包含标准答案和支撑事实的数据列表。
        """
        rewards = []

        for completion, sol in zip(completions, solution):
            # 获取该样本的标准支撑事实 (Gold Supporting Facts)
            facts = sol["supporting_facts"]
            facts_count = len(facts)

            # 1. 结构完整性校验
            # 检查输出是否包含合法的思维链结构（如 <think>...</think>）
            is_think_answer_valid, think_content, _ = (
                self.format_validator.validate_structure(completion)
            )

            # 2. Evidence 标签校验
            # 检查是否包含 <evidence> 标签，且数量是否达标
            # strict模式：必须包含与真实事实数量一致的 evidence 标签
            # loose模式：至少包含 1 个 evidence 标签
            is_evidence_valid = self.format_validator.validate_evidence_tags(
                think_content=think_content,
                at_least=facts_count if self.copy_reward_mode == 'sparse' else 1,
            )

            # 如果格式校验不通过（结构错误 或 evidence标签数量不足），直接给 0 分
            if not is_think_answer_valid or not is_evidence_valid:
                rewards.append(0.0)
                continue

            # 3. 提取内容
            # 从 <think> 部分提取出模型预测的所有 evidence 内容
            predict_facts = self.format_validator.get_predict_facts(think_content)

            # 4. 计算匹配度
            # 调用静态方法，对比预测事实与真实事实
            reward_across_facts = self._compute_supporting_facts_reward(
                predict_sfs=predict_facts,
                gold_sfs=facts,
            ) 

            predict_count = len(predict_facts)
            gold_count = len(facts)
            hit_count = sum(reward_across_facts)

            # 计算 Precision 和 Recall
            precision = hit_count / (predict_count + 1e-9) # 防止除零
            recall = hit_count / (gold_count + 1e-9)

            # 使用 F1 Score 作为奖励
            f1 = 2 * (precision * recall) / (precision + recall + 1e-9)

            # 5. 按照稠密和稀疏模式，计算最终奖励
            if self.copy_reward_mode == 'sparse':
                # 【稀疏模式】：全对才给分
                # 只有当命中的事实总数等于真实事实总数时，奖励为 1.0，否则为 0.0
                if f1 == 1.0:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            else:
                # 【稠密模式】：按比例给分
                # 奖励分为：命中的事实数量 / 总事实数量
                rewards.append( f1 * self.perfert_match_bonus if f1 == 1.0 else f1 )

        return rewards


class CopyFormatReward(ORM):
    def __init__(self):

        self.format_validator = FormatValidator()

    def __call__(
        self, completions: List[str], solution: List[dict], **kwargs
    ) -> List[float]:
        """
        计算一批样本的奖励分数。
        
        Args:
            completions: 模型生成的文本列表。
            solution: 包含标准答案和支撑事实的数据列表。
        """
        rewards = []

        for completion, sol in zip(completions, solution):
            # 获取该样本的标准支撑事实 (Gold Supporting Facts)
            facts = sol["supporting_facts"]

            # 1. 结构完整性校验
            # 检查输出是否包含合法的思维链结构（如 <think>...</think>）
            is_think_answer_valid, think_content, _ = (
                self.format_validator.validate_structure(completion)
            )
            
            # 2. Evidence 标签校验
            # 检查是否包含 <evidence> 标签，且数量是否达标
            # strict模式：必须包含与真实事实数量一致的 evidence 标签
            # loose模式：至少包含 1 个 evidence 标签
            is_evidence_valid = self.format_validator.validate_evidence_tags(
                think_content=think_content,
                at_least=1,
            )

            # 如果格式校验不通过（结构错误 或 evidence标签数量不足），直接给 0 分
            if not is_think_answer_valid or not is_evidence_valid:
                rewards.append(0.0)
            else:
                rewards.append(1.0)


        return rewards


class AnswerF1Reward(ORM):
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
            is_all_valid, _, answer_content = self.format_validator.validate_structure(
                completion
            )
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


class AnswerEMReward(ORM):
    """计算模型生成回答的奖励分值。

    严格模式，计算EM，如果EM不匹配，则降级计算HIT，并且按照答案长度进行惩罚。

    如果EM==True，则奖励为1.0

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
            is_all_valid, _, answer_content = self.format_validator.validate_structure(
                completion
            )
            if not is_all_valid:
                rewards.append(0.0)
                continue

            gold_answers = sol["answers"]

            em = exact_match_score(answer_content, gold_answers)
            if em:
                reward = 1.0
            else:
                reward = 0

            rewards.append(reward)

        return rewards


class AnswerHitReward(ORM):
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
            is_all_valid, _, answer_content = self.format_validator.validate_structure(
                completion
            )
            if not is_all_valid:
                rewards.append(0.0)
                continue

            gold_answers = sol["answers"]

            hit_gold = hit_answer(answer_content, gold_answers)
            if hit_gold:
                reward = len(hit_gold) / len(answer_content)
            else:
                reward = 0

            rewards.append(reward)

        return rewards

class CopyAnswerCombinedReward(ORM):
    """
    结合 Copy 和 Answer 的复合奖励函数（带错误惩罚机制）。
    
    设计理念 (基于 #CopyPasteLRM 新想法 + 惩罚机制):
    1. 耦合性：Answer 的奖励建立在 Copy 正确的基础上。
    2. 鼓励机制：
       - Copy F1 决定基础分。
       - Answer 正确 (F1>0.8) 提供翻倍奖励 (Bonus)。
    3. 惩罚机制 (New):
       - 对于每一个“抄错”或“冗余”的事实 (Wrong/Redundant Facts)，扣除固定分数。
       - 目的：抑制模型为了凑数而通过胡乱复制来“撞” F1 的行为。
    
    计算公式:
       Base_Score = Copy_F1 * (1.0 + Answer_Correct_Bonus)
       Penalty = (Predict_Count - Unique_Hit_Count) * Penalty_Weight
       
       Final_Reward = Base_Score - Penalty
    """

    def __init__(self):
        """
        Args:
            wrong_penalty_weight (float): 每一个错误或冗余引用扣除的分数。默认 0.1。
        """
        self.format_validator = FormatValidator()
        # 复用 CopyReward 中的静态方法计算 facts 匹配度
        self.compute_facts_logic = CopyReward._compute_supporting_facts_reward
        self.wrong_penalty_weight = 0.1
        self.answer_f1_perfect_threshold = 0.7

    def __call__(
        self, completions: List[str], solution: List[dict], **kwargs
    ) -> List[float]:
        rewards = []

        for completion, sol in zip(completions, solution):
            # --- 1. 基础结构校验 ---
            is_valid_structure, think_content, answer_content = (
                self.format_validator.validate_structure(completion)
            )
            
            # 格式错误直接给0分（或者也可以给一个负分，这里保持0分）
            if not is_valid_structure:
                rewards.append(0.0)
                continue

            # --- 2. 计算 Copy 相关的指标 ---
            gold_facts = sol["supporting_facts"]
            predict_facts = self.format_validator.get_predict_facts(think_content)
            
            # 如果没有预测任何事实，给 0 分
            if not predict_facts:
                rewards.append(0.0)
                continue

            # 计算事实匹配情况 (返回的是 boolean list, 对应 gold_facts 是否被命中)
            # 注意：这个方法只统计 Unique 的命中。重复引用同一个事实不会增加 hit_count。
            reward_across_facts = self.compute_facts_logic(predict_facts, gold_facts)
            
            predict_count = len(predict_facts)
            gold_count = len(gold_facts)
            hit_count = sum(reward_across_facts) # 命中了多少个唯一的正确事实

            # 2.1 计算 F1
            precision = hit_count / (predict_count + 1e-9)
            recall = hit_count / (gold_count + 1e-9)
            copy_f1 = 2 * (precision * recall) / (precision + recall + 1e-9)

            # 2.2 计算错误/冗余引用的数量 (用于惩罚)
            # 预测总数 - 命中正确事实数 = 没用的引用数 (可能是错的，也可能是重复的)
            wrong_count = predict_count - hit_count

            # --- 3. 计算 Answer Score ---
            gold_answers = sol["answers"]
            best_ans_f1 = 0.0
            
            if answer_content:
                for gold_answer in gold_answers:
                    f1, _, _ = f1_score(answer_content, gold_answer)
                    best_ans_f1 = max(best_ans_f1, f1)

            # --- 4. 融合奖励与惩罚 ---
            
            # 判定答案是否正确 (阈值 > 0.8)
            is_answer_correct = 1.0 if best_ans_f1 > self.answer_f1_perfect_threshold else 0.0
            
            # 基础奖励：CopyF1 * (1 + 答对奖励)
            # 范围: [0, 2.0] (假设 F1=1.0 且答对)
            base_reward = copy_f1 * (1.0 + is_answer_correct)
            
            # 惩罚项：错误数量 * 权重
            # 例如：抄错了3条，扣 0.3 分
            penalty = wrong_count * self.wrong_penalty_weight
            
            final_reward = base_reward - penalty

            rewards.append(final_reward)

        return rewards

# 注册新的奖励函数
orms["cplrm_combined"] = CopyAnswerCombinedReward

orms["cplrm_format"] = FormatReward
orms["cplrm_length"] = LengthtReward

orms["cplrm_copy"] = CopyReward
orms["cplrm_copy_format"] = CopyFormatReward

orms["cplrm_answer_f1"] = AnswerF1Reward
orms["cplrm_answer_em"] = AnswerEMReward

orms["cplrm_answer_hit"] = AnswerHitReward
