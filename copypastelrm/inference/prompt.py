from typing import Dict, Any
from string import Template

from copypastelrm.datasets import load, AvailebleDatasets


TEMPLATE = {
    # 适用于推理模型的CoT
    "direct": """User provides the following CONTEXT:

$context

QUESTION: $question

Answer the above question in the exact format 'Answer: <answer>'.""".strip(),
    # 适用于通用模型
    "reasoning": """User provides the following CONTEXT:

$context

QUESTION: $question

First, let's think step by step, provide your thinking process, and then answer the above question in the exact format 'Answer: <answer>'.""".strip(),
    # 完整的CopyPaste模式，只能用来评估
    "reasoning_with_copypaste": """User provides the following CONTEXT:

$context

QUESTION: $question

First, let's think step by step, provide your thinking process. During the thinking process, you must identify supporting facts (spans) from the context to form your reasoning. Copy the relevant span from the context and enclose it with the <EVIDENCE>copied span from context</EVIDENCE> tag. Crucially, ensure the <EVIDENCE>...</EVIDENCE> tags are naturally and fluently integrated within the sentences of your reasoning, serving as direct components of your argument. Then, answer the above question in the exact format 'Answer: <answer>'.""".strip()
}

def create_prompt(question: str, context: str, prompt_type: str) -> str:

    template = TEMPLATE[prompt_type]
    template = Template(template)

    # 使用 substitute 替换模板中的占位符
    full_prompt = template.substitute(
        context=context,
        question=question,
    )

    return full_prompt.strip()


def main():
    # 示例用法
    dataset_loader = load(AvailebleDatasets.TWOWIKI)
    sample = dataset_loader.get_sample()

    question = sample['query']
    context = sample['context']
    prompt_type = 'reasoning_with_copypaste'
    prompt = create_prompt(question, context, prompt_type)

    print(prompt)

if __name__ == "__main__":
    main()
