from typing import Dict, Any, Literal
from string import Template

from copypastelrm.datasets import load, AvailebleDatasets


TEMPLATE = {
    "pass@K": """User provides the following CONTEXT:

$context

QUESTION: $question

Answer the above question directly without additional explanation in the exact format 'Answer: xyz'.""".strip(),
    "pass@K_with_wrong_tips": """User provides the following CONTEXT:

$context

QUESTION: $question

Note that these answers is incorrect: $prior_answer

Answer the above question directly without additional explanation in the exact format 'Answer: xyz'.""".strip(),
    # 适用于推理模型的CoT
    "direct": """User provides the following CONTEXT:

$context

QUESTION: $question

Answer the above question in the exact format 'Answer: <Your final answer>'.""".strip(),
    # 适用于通用模型
    "reasoning": """User provides the following CONTEXT:

$context

QUESTION: $question

First, let's think step by step, provide your thinking process, and then answer the above question in the exact format 'Answer: <|answer|>'.""".strip(),
    # 完整的CopyPaste模式，只能用来评估
    "reasoning_with_copypaste": """User provides the following CONTEXT:

$context

QUESTION: $question

First, let's think step by step, provide your thinking process. During the thinking process, you must identify supporting facts (spans) from the context to form your reasoning. Copy the relevant span from the context and enclose it with the <|EVIDENCE|>copied span from context</|EVIDENCE|> tag. Crucially, ensure the <|EVIDENCE|>...</|EVIDENCE|> tags are naturally and fluently integrated within the sentences of your reasoning, serving as direct components of your argument. Then, answer the above question in the exact format 'Answer: <|answer|>'.""".strip(),
    "reasoning_with_copypaste_old": """

Context:
$context

Question: $question

---

Answer questions using the provided Context.

Formatting rules:
1) Explain your reasoning in a natural, fluent way inside a single <|think|>...</|think|> block.
2) Give the concise final answer inside a single <|answer|>...</|answer|> block.
3) Whenever you use an exact phrase or sentence taken verbatim from the Context as part of your reasoning, embed that exact substring with <copy>...</copy> tags. The content inside <copy> must be an exact substring of Context—do not paraphrase or modify it.
4) If no direct supporting sentence exists in the Context for a claim, explicitly acknowledge uncertainty in <|think|></|think|> instead of inventing facts.
5) Prefer natural, paragraph-style reasoning (not numbered steps). It is encouraged to integrate <copy>...</copy> evidence sentences seamlessly into your reasoning text to show traceability.

i.e., <|think|> reasoning process (must include <copy>evidence from Context</copy> naturally) </|think|><|answer|> final answer here </|answer|>
""".strip(),
}


def create_prompt(
    question: str,
    context: str,
    prompt_type: Literal[
        "pass@K", # 用于筛选模型无法回答的samples
        "direct",
        "reasoning",
        "reasoning_with_copypaste",
        "reasoning_with_copypaste_old",
    ],
    evidence: str = "",
    prior_answer: str = "",
) -> str:


    if "pass@K" in prompt_type:
        if prior_answer:
            template = TEMPLATE['pass@K_with_wrong_tips']
            template = Template(template)
            full_prompt = template.substitute(
                context=context,
                question=question,
                prior_answer=prior_answer,
            )
        else:
            template = TEMPLATE['pass@K']
            template = Template(template)
            full_prompt = template.substitute(
                context=context,
                question=question,
            )
    else: 
        # 使用 substitute 替换模板中的占位符
        template = TEMPLATE[prompt_type]
        template = Template(template)
        full_prompt = template.substitute(
            context=context,
            question=question,
        )

    return full_prompt.strip()


def main():
    # 示例用法
    dataset_loader = load(AvailebleDatasets.TWOWIKI)
    sample = dataset_loader.get_sample()

    question = sample["query"]
    context = sample["context"]
    prompt_type = "reasoning_with_copypaste_old"
    prompt = create_prompt(question, context, prompt_type)

    print(prompt)


if __name__ == "__main__":
    main()
