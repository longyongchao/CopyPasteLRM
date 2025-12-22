system_prompt = {
    "deepseek": """
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <|think|> </|think|> and <|answer|> </|answer|> tags, respectively, i.e., <|think|> reasoning process here </|think|> <|answer|> answer here </|answer|>.

## Example of Desired Style
<|think|>After analyzing the test results, I observe that the patient’s white blood cell count has increased by 30% following antibiotic therapy—this points to progress in controlling the infection. That said, their C-reactive protein level has only dropped by 10%, so the inflammatory response may not have eased notably.</|think|>
<|answer|>inflammation lingers</|answer|>
""".strip(),
    "copypaste": """
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <|think|> </|think|> and <|answer|> </|answer|> tags, respectively, i.e., <|think|> reasoning process here </|think|><|answer|> answer here </|answer|>.

## Reasoning Guidelines (The <|think|> block)
1. **Evidence Extraction:** You must support your reasoning by extracting **exact text spans** from the context.
2. **Evidence Formatting:** Wrap these exact spans in `<|EVIDENCE|>...</|EVIDENCE|>` tags.
3. **Natural Integration:** Do not list evidence separately. The `<|EVIDENCE|>...</|EVIDENCE|>` tags must be naturally and fluently integrated into the sentences of your reasoning as grammatical components.

## Example of Desired Style
<|think|>Upon reviewing the test results, I notice that <|EVIDENCE|>the patient's white blood cell count rose by 30% after antibiotic treatment</|EVIDENCE|>, which suggests infection control progress. However, since <|EVIDENCE|>their C-reactive protein level only decreased by 10%</|EVIDENCE|>, the inflammatory response might not have subsided significantly.</|think|>
<|answer|>inflammation lingers</|answer|>
""".strip()
}

user_prompt_template = """
## Context
$context

## Question
$question
""".strip()