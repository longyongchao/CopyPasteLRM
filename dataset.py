# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Optional

from swift.llm import DatasetMeta, ResponsePreprocessor, load_dataset, register_dataset


class CustomPreprocessor(ResponsePreprocessor):
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
        # 将需要在奖励函数中使用的字段同时放入标准字段和objects，便于trainer透传
        row.update({
            'query': self.prompt.format(question=row['query'], context=row['context']),
            'response': f"{row['response']}",
            # 'solution': {
            #     'context': row['context'],
            #     'response': row['response'],
            # }
        })
        return super().preprocess(row)

register_dataset(
    DatasetMeta(
        hf_dataset_id='Salesforce/FaithEval-counterfactual-v1.0',
        ms_dataset_id='Salesforce/FaithEval-counterfactual-v1.0',
        preprocess_func=CustomPreprocessor(columns={
            'context': 'context',
            'solution': 'response',
        }),
        split=['test'],
    ))

if __name__ == '__main__':
    dataset = load_dataset(['Salesforce/FaithEval-counterfactual-v1.0'])[0]
    print(f'dataset: {dataset}')
    print(f'dataset[0]: {dataset[0]}')