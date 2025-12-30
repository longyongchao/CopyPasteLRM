import dspy
import re
from dspy.teleprompt import MIPROv2

# ==========================================
# 1. 配置模型 (Teacher 最好用强模型，Student 可以用小模型)
# ==========================================

# Teacher: 负责提出更好的 Prompt 指令 (建议 GPT-4o)
teacher_lm = dspy.LM(
    "openai/glm-4.7",
    api_key="3054b366b2868b315f5a4cb6f5e5fae9.GVbQ1PQVYmgDzm4K",
    api_base="https://open.bigmodel.cn/api/paas/v4",
)
# Student: 实际运行任务的模型 (可以是 gpt-4o-mini 或 llama3)
student_lm = dspy.LM(
    "openai/Qwen2.5-3B-Instruct",
    api_base="http://localhost:8124/v1",  # ensure this points to your port
    api_key="local",
    model_type="chat",
)

dspy.configure(lm=student_lm)

# ==========================================
# 2. 准备数据
# ==========================================
raw_data = [
    {
        "context": "Patient A has a history of hypertension. Their BP today is 150/95. The doctor prescribed Amlodipine 5mg.",
        "question": "What medication was prescribed?",
        "answer": "Amlodipine 5mg",
    },
    {
        "context": "The experiment results show a 5% increase in accuracy using the new algorithm, but latency increased by 20ms.",
        "question": "What was the downside of the new algorithm?",
        "answer": "latency increased by 20ms",
    },
    {
        "context": "Q3 revenue was $40M, up from $30M in Q2. However, net profit dropped due to a one-time tax charge.",
        "question": "Why did net profit drop?",
        "answer": "one-time tax charge",
    },
    {
        "context": "The suspect was seen wearing a red jacket and blue jeans leaving the bank at 3:00 PM.",
        "question": "What color was the suspect's jacket?",
        "answer": "red",
    },
    {
        "context": "Solar panels efficiency drops as temperature rises. The coefficient is usually around -0.5% per degree Celsius.",
        "question": "What happens to efficiency when it gets hotter?",
        "answer": "efficiency drops",
    },
]

dataset = [
    dspy.Example(
        context=x["context"], question=x["question"], answer=x["answer"]
    ).with_inputs("context", "question")
    for x in raw_data
]
trainset = dataset[:3]
valset = dataset[3:]


# ==========================================
# 3. 定义 Signature
# ==========================================
class CopyPasteSignature(dspy.Signature):
    """
    You are an expert analytical engine.
    Given a Context and a Question, perform a strict reasoning process to find the answer.
    """

    context = dspy.InputField(desc="The source text containing the evidence.")
    question = dspy.InputField(desc="The specific query to answer.")
    answer = dspy.OutputField(
        desc="The final concise answer derived from the evidence."
    )


# ==========================================
# 4. 定义 Module
# ==========================================
class CopyPasteLRM(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(CopyPasteSignature)

    def forward(self, context, question):
        return self.prog(context=context, question=question)


# ==========================================
# 5. 定义 Metric
# ==========================================
def strict_copypaste_metric(example, pred, trace=None):
    if pred.answer.strip().lower() not in example.answer.strip().lower():
        return 0
    rationale = pred.rationale
    evidence_spans = re.findall(r"<\|EVIDENCE\|>(.*?)<\/\|EVIDENCE\|>", rationale)
    if not evidence_spans:
        return 0
    context_text = example.context
    for span in evidence_spans:
        if span.strip() not in context_text:
            return 0
    return 1


# ==========================================
# 6. 运行 MIPROv2 优化 (关键修复版)
# ==========================================
print("开始 MIPROv2 优化...")

# 【关键修复点 1】
# 在初始化时显式设置 auto=None。
# 这会解除"自动模式"锁，允许我们在后面手动设置 num_trials。
teleprompter = MIPROv2(
    metric=strict_copypaste_metric,
    prompt_model=teacher_lm,
    task_model=student_lm,
    num_candidates=7,  # 这里可以设置，因为 auto=None
    init_temperature=1.0,
    verbose=True,
    auto=None,  # <--- 必须加这行！强制关闭自动模式
)

# 【关键修复点 2】
# compile 中不要传 auto，但要传 minibatch=False
compiled_program = teleprompter.compile(
    CopyPasteLRM(),
    trainset=trainset,
    valset=valset,
    num_trials=15,  # 只有 auto=None 时，这里才能生效
    max_bootstrapped_demos=3,
    max_labeled_demos=2,
    minibatch=False,  # 小数据集必须为 False
    requires_permission_to_run=False,
)

# ==========================================
# 7. 保存与测试
# ==========================================
save_path = "optimized_copypaste_lrm.json"
compiled_program.save(save_path)
print(f"\n优化完成！最佳 Prompt 已保存至 {save_path}")

test_context = "The software update v2.1 causing the crash was released on Monday. The patch v2.2 was deployed on Tuesday."
test_question = "When was the crash-causing update released?"
pred = compiled_program(context=test_context, question=test_question)
print(f"Question: {test_question}")
print(f"Rationale:\n{pred.rationale}")
print(f"Answer: {pred.answer}")
