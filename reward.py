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

        # print('test solution input: ', solution[0])

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

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """奖励推理过程是否使用并正确使用了上下文片段

        Args:
            completions (List[str]): 模型生成的输出文本
            solution(str): 提供数据集的其它字段

        Returns:
            List[float]: 每个输出的奖励分数（1或0）
        """

        rewards = []
        think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
        copy_pattern = re.compile(r"<copy>(.*?)</copy>", re.DOTALL | re.IGNORECASE)

        for text, s in zip(completions, solution):
            ctx_dict = s.get('context', {})
            supporting_facts = s.get('supporting_facts', [])

            ctx = ''
            for title, sent_id in zip(supporting_facts.get('title', []), supporting_facts.get('sent_id', [])):
                idx_title = ctx_dict.get('title', []).index(title)
                if idx_title == -1:
                    continue
                ctx_sentences = ctx_dict.get('sentences', [])
                if not ctx_sentences:
                    continue
                if idx_title >= len(ctx_sentences):
                    continue
                if sent_id >= len(ctx_sentences[idx_title]):
                    continue
                ctx += ctx_dict.get('sentences', [])[idx_title][sent_id]
            
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


class CopyPasteUniORM(ORM):

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        rewards = []

        return rewards

orms['copypaste_uni'] = CopyPasteUniORM
