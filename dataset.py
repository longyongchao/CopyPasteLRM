# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Optional

from swift.llm import DatasetMeta, ResponsePreprocessor, load_dataset, register_dataset


# class FaithEvalPreprocessor(ResponsePreprocessor):
#     prompt = """
# Answer questions using the provided Context.

# Formatting rules:
# 1) Explain your reasoning in a natural, fluent way inside a single <think>...</think> block.
# 2) Give the concise final conclusion inside a single <answer>...</answer> block.
# 3) Whenever you use an exact phrase or sentence taken verbatim from the Context as part of your reasoning,
# embed that exact substring with <copy>...</copy> tags. The content inside <copy> must be an exact substring of Context—do not paraphrase or modify it.
# 4) If no direct supporting sentence exists in the Context for a claim, explicitly acknowledge uncertainty in <think> instead of inventing facts.
# 5) Prefer natural, paragraph-style reasoning (not numbered steps). It is encouraged to integrate short <copy>...</copy> evidence snippets seamlessly into your reasoning text to show traceability.

# i.e., <think> reasoning process (may include <copy>evidence from Context</copy>) </think><answer> final answer here </answer>\n

# ---

# Context: {context}
# Question: {question}
# """.strip()

#     def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
#         # 将需要在奖励函数中使用的字段同时放入标准字段和objects，便于trainer透传
#         row.update({
#             'query': self.prompt.format(question=row['query'], context=row['context']),
#             'response': f"{row['response']}",
#             # 'solution': {
#             #     'context': row['context'],
#             #     'response': row['response'],
#             # }
#         })
#         return super().preprocess(row)

# register_dataset(
#     DatasetMeta(
#         hf_dataset_id='Salesforce/FaithEval-counterfactual-v1.0',
#         ms_dataset_id='Salesforce/FaithEval-counterfactual-v1.0',
#         dataset_name='faitheval',
#         preprocess_func=FaithEvalPreprocessor(columns={
#             'context': 'context',
#             'solution': 'response',
#         }),
#         split=['test'],
#     ))


class HotpotQAPreprocessor(ResponsePreprocessor):
    """
    FaithEvalPreprocessor类继承自ResponsePreprocessor，用于处理问答任务的输入数据。
    该类主要功能是将原始数据格式化为符合特定提示模板的格式，并添加必要的字段。
    """
    prompt = """
Answer questions using the provided Context.

Formatting rules:
1) Explain your reasoning in a natural, fluent way inside a single <think>...</think> block.
2) Give the concise final conclusion inside a single <answer>...</answer> block.
3) Whenever you use an exact phrase or sentence taken verbatim from the Context as part of your reasoning,
embed that exact substring with <copy>...</copy> tags. The content inside <copy> must be an exact substring of Context—do not paraphrase or modify it.
4) If no direct supporting sentence exists in the Context for a claim, explicitly acknowledge uncertainty in <think> instead of inventing facts.
5) Prefer natural, paragraph-style reasoning (not numbered steps). It is encouraged to integrate short <copy>...</copy> evidence snippets seamlessly into your reasoning text to show traceability.

i.e., <think> reasoning process (may include <copy>evidence from Context</copy>) </think><answer> final answer here </answer>\n

---

Context: {context}
Question: {question}
""".strip()

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        context_dict = row['context']
        context_text = ''
        for title, sentences in zip(context_dict['title'], context_dict['sentences']):
            context_text += f"{title}. " + ''.join(sentences) + "\n"

        row.update({
            'query': self.prompt.format(question=row['query'], context=context_text),
            'response': f"{row['response']}",
            'solution': {
                'context': row['context'],
                'response': row['response'],
                'supporting_facts': row['supporting_facts']
            }

        })
        return super().preprocess(row)

register_dataset(
    DatasetMeta(
        hf_dataset_id='hotpotqa/hotpot_qa',
        dataset_name='hotpot_qa',
        preprocess_func=HotpotQAPreprocessor(columns={
            'solution': 'solution',
            'context': 'context',
            'suporting_facts': 'supporting_facts',
        }),
        subsets=['distractor'],
        split=['train'],
    ))

if __name__ == '__main__':
    dataset = load_dataset(['hotpot_qa'], use_hf=True, split_dataset_ratio=0.01)
    
    
    print(f'dataset: {dataset}')
    # print(f'dataset[0]: {dataset[0]}')
    # print(f'length: {len(dataset)}')