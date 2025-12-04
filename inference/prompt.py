def create_prompt(question: str, context: str, prompt_type: str) -> str:
    """
    创建推理提示

    Args:
        question: 问题文本
        context: 上下文文本

    Returns:
        str: 完整的提示文本
    """
    reasoning_with_copy_paste = f"""
Context: {context}

Question: {question}

---

Answer questions using the provided Context.

Formatting rules:
1) Explain your reasoning in a natural, fluent way inside a single <think>...</think> block.
2) Give the concise final answer inside a single <answer>...</answer> block.
3) Whenever you use an exact phrase or sentence taken verbatim from the Context as part of your reasoning, embed that exact substring with <copy>...</copy> tags. The content inside <copy> must be an exact substring of Context—do not paraphrase or modify it.
4) If no direct supporting sentence exists in the Context for a claim, explicitly acknowledge uncertainty in <think></think> instead of inventing facts.
5) Prefer natural, paragraph-style reasoning (not numbered steps). It is encouraged to integrate <copy>...</copy> evidence sentences seamlessly into your reasoning text to show traceability.

i.e., <think> reasoning process (must include <copy>evidence from Context</copy> naturally) </think><answer> final answer here </answer>
""".strip()

    reasoning = f"""
Context: {context}

Question: {question}

---

Answer questions using the provided Context.

Formatting rules:
1) Explain your reasoning before giving the final answer.
2) Give the concise final answer inside a single <answer>...</answer> block.

i.e., reasoning process... <answer> final answer here </answer>
""".strip()

    direct = f"""
Context: {context}

Question: {question}

---

Answer questions using the provided Context.

Formatting rules:
1) Give the concise final answer inside a single <answer>...</answer> block directly.

i.e., <answer> final answer here </answer>
""".strip()

    if prompt_type == "reasoning with copy-paste":
        prompt = reasoning_with_copy_paste
    elif prompt_type == "reasoning":
        prompt = reasoning
    elif prompt_type == "direct":
        prompt = direct
    else:
        raise ValueError(f"未知的 prompt_type: {prompt_type}")

    return prompt
