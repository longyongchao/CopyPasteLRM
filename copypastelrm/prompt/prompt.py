from typing import Dict, Any, Literal
from string import Template

from copypastelrm.datasets import load, AvailableDataset


# 基础系统提示词模板
_ANSWER_FORMAT_INSTRUCTIONS = """
## IMPORTANT:
- The content after `Answer:` must be a direct answer to the question.
- Do NOT include any explanation, reasoning, justification, or extra words after `Answer:`.
- After the `Answer:` should contain ONLY the answer itself.
""".strip()

# 系统提示词工厂函数
def _create_direct_answer_prompt(prefix: str = "") -> str:
    """创建直接回答类型的系统提示词"""
    base = "Directly answer the above question without any additional explanation in the exact format 'Answer: xyz'."
    if prefix:
        return f"{prefix}\n\n{base}\n\n{_ANSWER_FORMAT_INSTRUCTIONS}"
    return f"{base}\n\n{_ANSWER_FORMAT_INSTRUCTIONS}"

def _create_cot_prompt(prefix: str = "") -> str:
    """创建CoT类型的系统提示词"""
    base = "First, let's think step by step about the above question, provide your thinking process, and then give your final answer without any additional explanation in the exact format 'Answer: xyz'."
    if prefix:
        return f"{prefix}\n\n{base}\n\n{_ANSWER_FORMAT_INSTRUCTIONS}"
    return f"{base}\n\n{_ANSWER_FORMAT_INSTRUCTIONS}"

# 所有使用 RAG 格式的提示类型（基于上下文直接回答）
_RAG_PROMPT = _create_direct_answer_prompt("Based on the context,")

# 所有使用 RAG CoT 格式的提示类型（基于上下文的CoT）
_RAG_COT_PROMPT = _create_cot_prompt("Based on the context,")

SYSTEM_PROMPT = {
    # 适用于推理模型（无上下文）
    "direct_inference": _create_direct_answer_prompt(),
    "cot": _create_cot_prompt(),

    # 适用于通用模型（基于上下文直接回答）- 所有 RAG 变体共用
    "rag": _RAG_PROMPT,
    "rag_rep_2": _RAG_PROMPT,
    "rag_rep_q": _RAG_PROMPT,
    "rag_qcq": _RAG_PROMPT,
    "rag_qcq2": _RAG_PROMPT,
    "rag_q_int_q": _RAG_PROMPT,
    "rag_q_int2_q": _RAG_PROMPT,
    "rag_decompressed": _RAG_PROMPT,
    "rag_decompressed_rep_q": _RAG_PROMPT,

    # 基于上下文的 CoT
    "ircot": _RAG_COT_PROMPT,

    # DeepSeek 风格（特殊格式）
    "deepseek": """
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <|think|> </|think|> and <|answer|> </|answer|> tags, respectively, i.e., <|think|> reasoning process here </|think|> <|answer|> answer here </|answer|>.

## IMPORTANT:
- The content inside <|answer|> must be a direct answer to the question.
- Do NOT include any explanation, reasoning, justification, or extra words in <|answer|>.
- The <|answer|> tag should contain ONLY the answer itself.

## Example of Desired Style
<|think|>After analyzing the test results, I observe that the patient's white blood cell count has increased by 30% following antibiotic therapy—this points to progress in controlling the infection. That said, their C-reactive protein level has only dropped by 10%, so the inflammatory response may not have eased notably.</|think|>
<|answer|>inflammation lingers</|answer|
""".strip(),

    # CopyPaste 风格（带证据提取）
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

    # 提取事实任务
    "find_facts": """
Based on the provided context, identify and extract all specific facts or sentences that are helpful or necessary to answer the question.

## Instructions:
1. **Identify Support:** Scan the context for information that directly supports the reasoning path to the answer.
2. **Extract Only:** Extract these facts exactly as they appear in the context or slightly paraphrased for clarity.
3. **Format:** Wrap each distinct fact in <|EVIDENCE|>...</|EVIDENCE|> tags.
4. **No Answer:** Do NOT provide the answer to the question.
5. **No Extra Text:** Do NOT include any introduction, explanation, or conversational filler. Your output should strictly be a list of tagged facts.

## Desired Output Format
<|EVIDENCE|>Supporting fact 1 extracted from context</|EVIDENCE|>
<|EVIDENCE|>Supporting fact 2 extracted from context</|EVIDENCE|>
...
""".strip(),
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
        "rag_rep_2",
        "rag_rep_q",
        "rag_qcq",
        "rag_qcq2",
        "rag_q_int_q",
        "rag_q_int2_q",
        "rag_decompressed",
        "rag_decompressed_rep_q",
        "ircot",
        "deepseek",
        "copypaste",
        "find_facts",
    ],
) -> tuple[str, str]:
    """
    Create system and user prompts for the given question and context.
    """
    system_prompt = SYSTEM_PROMPT[prompt_type]

    # Helper template for question
    question_template = Template("## Question\n$question")

    # Build prompt based on prompt_type
    if prompt_type in ["direct_inference", "cot"]:
        template = Template(only_query_prompt_template)
        user_prompt = template.substitute(question=question)
    elif prompt_type == "rag_rep_2":
        # Hardcoded: repeat context+question twice
        template = Template(context_query_prompt_template)
        part1 = template.substitute(context=context, question=question)
        part2 = template.substitute(context=context, question=question)
        user_prompt = part1 + "\n\n" + "=" * 50 + "\n\n" + part2
    elif prompt_type == "rag_rep_q":
        # Hardcoded: context once, question twice
        template = Template(context_query_prompt_template)
        part1 = template.substitute(context=context, question=question)
        part2 = question_template.substitute(question=question)
        user_prompt = part1 + "\n\n" + "=" * 50 + "\n\n" + part2
    elif prompt_type == "rag_qcq":
        # <question><context><question>
        part1 = question_template.substitute(question=question)
        template = Template("## Context\n$context")
        part2 = template.substitute(context=context)
        part3 = question_template.substitute(question=question)
        user_prompt = part1 + "\n\n" + part2 + "\n\n" + part3
    elif prompt_type == "rag_qcq2":
        # <question><context><question><question>
        part1 = question_template.substitute(question=question)
        template = Template("## Context\n$context")
        part2 = template.substitute(context=context)
        part3 = question_template.substitute(question=question)
        part4 = question_template.substitute(question=question)
        user_prompt = part1 + "\n\n" + part2 + "\n\n" + part3 + "\n\n" + part4
    elif prompt_type == "rag_q_int_q":
        # <question><context_parts><question> (internal 25%, 75%)
        part1 = question_template.substitute(question=question)
        # Split context into 4 parts by newlines
        context_lines = context.split('\n')
        n = len(context_lines)
        idx1 = n // 4
        idx2 = (3 * n) // 4
        context_parts = [
            '\n'.join(context_lines[:idx1]),
            '\n'.join(context_lines[idx1:idx2]),
            '\n'.join(context_lines[idx2:]),
        ]
        template = Template("## Context\n$context_part")
        part2 = template.substitute(context_part=context_parts[0])
        part3 = question_template.substitute(question=question)
        part4 = template.substitute(context_part=context_parts[1])
        part5 = template.substitute(context_part=context_parts[2])
        part6 = question_template.substitute(question=question)
        user_prompt = part1 + "\n\n" + part2 + "\n\n" + part3 + "\n\n" + part4 + "\n\n" + part5 + "\n\n" + part6
    elif prompt_type == "rag_q_int2_q":
        # <question><context_parts><question><question> (internal 25%, 75%, plus 2 at end)
        part1 = question_template.substitute(question=question)
        # Split context into 4 parts by newlines
        context_lines = context.split('\n')
        n = len(context_lines)
        idx1 = n // 4
        idx2 = (3 * n) // 4
        context_parts = [
            '\n'.join(context_lines[:idx1]),
            '\n'.join(context_lines[idx1:idx2]),
            '\n'.join(context_lines[idx2:]),
        ]
        template = Template("## Context\n$context_part")
        part2 = template.substitute(context_part=context_parts[0])
        part3 = question_template.substitute(question=question)
        part4 = template.substitute(context_part=context_parts[1])
        part5 = template.substitute(context_part=context_parts[2])
        part6 = question_template.substitute(question=question)
        part7 = question_template.substitute(question=question)
        user_prompt = part1 + "\n\n" + part2 + "\n\n" + part3 + "\n\n" + part4 + "\n\n" + part5 + "\n\n" + part6 + "\n\n" + part7
    elif prompt_type == "rag_decompressed":
        # Decompress question by adding spaces between characters
        decompressed_question = ' '.join(question)
        template = Template(context_query_prompt_template)
        user_prompt = template.substitute(context=context, question=decompressed_question)
    elif prompt_type == "rag_decompressed_rep_q":
        # <context><decompressed_question><decompressed_question>
        decompressed_question = ' '.join(question)
        template = Template("## Context\n$context")
        part1 = template.substitute(context=context)
        question_template = Template("## Question\n$question")
        part2 = question_template.substitute(question=decompressed_question)
        part3 = question_template.substitute(question=decompressed_question)
        user_prompt = part1 + "\n\n" + part2 + "\n\n" + part3
    else:
        # rag, ircot, deepseek, copypaste, find_facts: use standard context+question
        template = Template(context_query_prompt_template)
        user_prompt = template.substitute(context=context, question=question)

    return system_prompt, user_prompt


def main():
    # 示例用法
    dataset_loader = load(AvailableDataset.TWO_WIKI_MULTI_HOP_QA, reload=True)
    sample = dataset_loader.get_sample()

    question = sample["query"]
    context = sample["context"]
    prompt_type = "find_facts"
    system_prompt, user_prompt = create_prompt(question, context, prompt_type)

    print(system_prompt)
    print(user_prompt)


if __name__ == "__main__":
    main()
