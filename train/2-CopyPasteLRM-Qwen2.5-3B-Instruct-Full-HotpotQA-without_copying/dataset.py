# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, List, Optional, TypedDict

from swift.llm import DatasetMeta, ResponsePreprocessor, load_dataset, register_dataset


class HotpotQAContext(TypedDict):
    title: List[str]
    sentences: List[List[str]]


class HotpotQASupportingFacts(TypedDict):
    title: List[str]
    sent_id: List[int]


class HotpotQARow(TypedDict):
    """TypedDict for HotpotQA dataset row structure."""

    _id: str
    question: str
    answer: str
    context: HotpotQAContext
    supporting_facts: HotpotQASupportingFacts
    level: str
    type: str


class ProcessedHotpotQARow(TypedDict):
    """TypedDict for processed HotpotQA dataset row structure."""

    query: str
    response: str
    solution: Dict[str, Any]


class HotpotQAPreprocessor(ResponsePreprocessor):
    """
    HotpotQAPreprocessor类继承自ResponsePreprocessor，用于处理问答任务的输入数据。
    该类主要功能是将原始数据格式化为符合特定提示模板的格式，并添加必要的字段。
    """

    prompt = """
Context: {context}

Question: {question}

---

Answer questions using the provided Context.

Formatting rules:
1) Explain your reasoning in a natural, fluent way inside a single <think>...</think> block.
2) Give the concise final answer inside a single <answer>...</answer> block.

i.e., <think> reasoning process </think><answer> final answer here </answer>\n
""".strip()

    def preprocess(self, row: HotpotQARow) -> Optional[ProcessedHotpotQARow]:
        """Process HotpotQA row with proper type annotations.

        Args:
            row: Input row containing HotpotQA data with strongly typed context

        Returns:
            Processed row with query, response, and solution fields
        """
        context: HotpotQAContext = row["context"]
        context_text = ""
        for title, sentences in zip(context["title"], context["sentences"]):
            context_text += f"{title}. " + "".join(sentences) + "\n"

        # Use proper typing for the row update
        processed_row: ProcessedHotpotQARow = {
            "query": self.prompt.format(question=row["query"], context=context_text),
            "response": f"{row['response']}",
            "solution": {
                "context": row["context"],
                "response": row["response"],
                "supporting_facts": row["supporting_facts"],
            },
        }

        # Update the original row for compatibility
        row.update(processed_row)  # type: ignore[arg-type]
        return super().preprocess(row)  # type: ignore[return-value]


register_dataset(
    DatasetMeta(
        hf_dataset_id="hotpotqa/hotpot_qa",
        dataset_name="hotpot_qa",
        preprocess_func=HotpotQAPreprocessor(
            columns={
                "solution": "solution",
                "context": "context",
                "supporting_facts": "supporting_facts",
            }
        ),
        subsets=["distractor"],
        split=["train"],
    )
)

if __name__ == "__main__":
    dataset = load_dataset(["hotpot_qa"], use_hf=True, split_dataset_ratio=0.001)

    print(f"dataset: {dataset}")
    print(f"dataset[0]: {dataset[0]}")
