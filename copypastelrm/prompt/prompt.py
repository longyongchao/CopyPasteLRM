from typing import Dict, Any, Literal
from string import Template

from copypastelrm.datasets import load, AvailebleDatasets


SYSTEM_PROMPT = {
    # 适用于推理模型的CoT
    "direct_inference": """Directly answer the above question without any additional explanation in the exact format 'Answer: xyz'.""".strip(),
    "cot": """First, let's think step by step about the above question, provide your thinking process, and then give your final answer without any additional explanation in the exact format 'Answer: xyz'.""".strip(),
    # 适用于通用模型
    "rag": """
Based on the context, directly answer the above question without any additional explanation in the exact format 'Answer: xyz'.

## IMPORTANT:
- The content after `Answer:` must be a direct answer to the question.
- Do NOT include any explanation, reasoning, justification, or extra words after `Answer:`.
- After the `Answer:` should contain ONLY the answer itself.
    
    """.strip(),
    "ircot": """Based on the context, let's think step by step about the above question, provide your thinking process, and then give your final answer without any additional explanation in the exact format 'Answer: xyz'.""".strip(),
    "deepseek": """
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <|think|> </|think|> and <|answer|> </|answer|> tags, respectively, i.e., <|think|> reasoning process here </|think|> <|answer|> answer here </|answer|>.

## IMPORTANT:
- The content inside <|answer|> must be a direct answer to the question.
- Do NOT include any explanation, reasoning, justification, or extra words in <|answer|>.
- The <|answer|> tag should contain ONLY the answer itself.

## Example of Desired Style
<|think|>After analyzing the test results, I observe that the patient’s white blood cell count has increased by 30% following antibiotic therapy—this points to progress in controlling the infection. That said, their C-reactive protein level has only dropped by 10%, so the inflammatory response may not have eased notably.</|think|>
<|answer|>inflammation lingers</|answer|>
""".strip(),
    "copypaste": """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <|think|> </|think|> and <|answer|> </|answer|> tags, respectively, i.e., <|think|> reasoning process here </|think|><|answer|> answer here </|answer|>.

## IMPORTANT:
- The content inside <|answer|> must be a direct answer to the question.
- Do NOT include any explanation, reasoning, justification, or extra words in <|answer|>.
- The <|answer|> tag should contain ONLY the answer itself.

## Reasoning Guidelines (The <|think|> block)
1. **Evidence Extraction:** You must support your reasoning by extracting **exact text spans** from the context.
2. **Evidence Formatting:** Wrap these exact spans in `<|EVIDENCE|>...</|EVIDENCE|>` tags.
3. **Natural Integration:** Do not list evidence separately. The `<|EVIDENCE|>...</|EVIDENCE|>` tags must be naturally and fluently integrated into the sentences of your reasoning as grammatical components.

## Example of Desired Style
<|think|>Upon reviewing the test results, I notice that <|EVIDENCE|>the patient's white blood cell count rose by 30% after antibiotic treatment</|EVIDENCE|>, which suggests infection control progress. However, since <|EVIDENCE|>their C-reactive protein level only decreased by 10%</|EVIDENCE|>, the inflammatory response might not have subsided significantly.</|think|>
<|answer|>inflammation lingers</|answer|>""".strip(),
}

only_query_prompt_template = """
$question
""".strip()

context_query_prompt_template = """
## Context
$context

## Question
$question
""".strip()


def create_prompt(
    question: str,
    context: str,
    prompt_type: Literal[
        "direct_inference",
        "cot",
        "rag",
        "ircot",
        "deepseek",
        "copypaste",
    ],
) -> str:

    system_prompt = SYSTEM_PROMPT[prompt_type]

    if prompt_type in ["direct_inference", "cot"]:
        # 使用 substitute 替换模板中的占位符
        template = Template(only_query_prompt_template)
        user_prompt = template.substitute(
            question=question,
        )
    else:
        # 使用 substitute 替换模板中的占位符
        template = Template(context_query_prompt_template)
        user_prompt = template.substitute(
            context=context,
            question=question,
        )

    return system_prompt, user_prompt


def main():
    # 示例用法
    dataset_loader = load(AvailebleDatasets.TWOWIKI)
    sample = dataset_loader.get_sample()

    question = sample["query"]
    context = sample["context"]
    prompt_type = "cot"
    system_prompt, user_prompt = create_prompt(question, context, prompt_type)

    print(system_prompt)
    print(user_prompt)


if __name__ == "__main__":
    main()
