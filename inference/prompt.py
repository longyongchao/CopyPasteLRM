from typing import Dict, Any
from string import Template


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
    question = "Which historical decade saw the launch of the Apollo 11 mission?"
    context = (
        "The Apollo 11 mission was the spaceflight that landed the first two humans on the Moon. "
        "The launch took place in 1969. The 1960s were a decade marked by significant political "
        "and social change globally, including the US space race."
    )
    prompt_type = "direct"
    prompt = create_prompt(question, context, prompt_type)
    print(prompt)

if __name__ == "__main__":
    main()
