import re
from typing import List

from swift.plugin import ORM, orms


class CopyPasteAnswerORM(ORM):

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """奖励最终输出是否与答案一致

        Args:
            completions (List[str]): 模型生成的输出文本
            answer (str): 期望答案（字符串）

        Returns:
            List[float]: 每个输出的奖励分数（1或0）
        """

        # print('golden_response', solution[0])


        rewards = []
        answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
        # 兼容多种来源：自定义gt字段、标准response、objects内的gt字段

        for text, s in zip(completions, solution):
            # 提取模型输出中的 <answer>...</answer> 内容
            match = answer_pattern.search(text)
            if not match:
                rewards.append(0.0)
                continue

            answer = s['response']

            generated_answer = match.group(1).strip().lower()
            target_answer = (answer or "").strip().lower()

            # 如果 answer 被包含，则奖励 1
            if target_answer and target_answer in generated_answer:
                rewards.append(1.0)
            else:
                rewards.append(0.0)


        return rewards

orms['copypaste_answer'] = CopyPasteAnswerORM


class CopyPasteFormatORM(ORM):

    def __call__(self, completions, context, **kwargs) -> List[float]:
        """奖励推理过程是否使用并正确使用了上下文片段

        Args:
            completions (List[str]): 模型生成的输出文本
            context (str): 提供的上下文字符串

        Returns:
            List[float]: 每个输出的奖励分数（1或0）
        """

        # print('context', context[0][:50])

        rewards = []
        think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
        copy_pattern = re.compile(r"<copy>(.*?)</copy>", re.DOTALL | re.IGNORECASE)

        for text, ctx in zip(completions, context):
            think_match = think_pattern.search(text)
            if not think_match:
                rewards.append(0.0)
                continue

            think_block = think_match.group(1)
            copies = copy_pattern.findall(think_block)

            if not copies:
                rewards.append(0.0)
                continue

            # 检查每个 <copy> 内容是否为 context 子串
            valid_copies = [cp for cp in copies if cp.strip() and cp.strip() in ctx]

            if len(valid_copies) == len(copies):
                rewards.append(1.0)
            else:
                rewards.append(0.0)

        return rewards

orms['copypaste_format'] = CopyPasteFormatORM
