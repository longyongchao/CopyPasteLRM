from copypastelrm.utils.json_tools import read_jsonl_to_list, read_json, save_jsonl
from string import Template
from tqdm import tqdm
from copypastelrm.utils.dataset import StringContainmentFilter
import random

system_prompt = """
You are CopyPasteLRM. The user asks a question, and the CopyPasteLRM solves it. The CopyPasteLRM first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <|think|> </|think|> and <|answer|> </|answer|> tags, respectively, i.e., <|think|> reasoning process here </|think|><|answer|> answer here </|answer|>

## Reasoning Guidelines (The <|think|> block)
1. **Evidence Extraction:** You must support your reasoning by extracting **exact text spans** from the context.
2. **Evidence Formatting:** Wrap these exact spans in `<|EVIDENCE|>...</|EVIDENCE|>` tags.
3. **Natural Integration:** Do not list evidence separately. The `<|EVIDENCE|>...</|EVIDENCE|>` tags must be naturally and fluently integrated into the sentences of your reasoning as grammatical components.

## Example of Desired Style
<|think|>
Upon reviewing the report, I notice that <|EVIDENCE|>revenue increased by 20% in Q3</|EVIDENCE|>, which suggests a positive trend. However, since <|EVIDENCE|>operating costs also rose by 15%</|EVIDENCE|>, the net profit margin might not have improved significantly.
</|think|>
<|answer|>
While revenue grew, the increase in costs offset some of the gains.
</|answer|>
""".strip()

user_prompt_template = """
## Context
$context

## Question
$question
""".strip()

paths = {
    "2wiki": "/data/lyc/CopyPasteLRM/pass_at_42/2wikimultihopqa-resamples_2000-tpr_0.6-tpp_0.95-Qwen3-8B-enable_thinking_True-tips_threshold_21-1765529926.jsonl",
    "hotpotqa": "/data/lyc/CopyPasteLRM/pass_at_42/hotpotqa-resamples_2000-tpr_0.6-tpp_0.95-Qwen3-8B-enable_thinking_True-tips_threshold_21-1765337233.jsonl",
    "multirc": "/data/lyc/CopyPasteLRM/pass_at_42/multirc-resamples_2000-tpr_0.6-tpp_0.95-Qwen3-8B-enable_thinking_True-tips_threshold_21-1765351563.jsonl",
    "musiqua": "/data/lyc/CopyPasteLRM/pass_at_42/musique-resamples_2000-tpr_0.6-tpp_0.95-Qwen3-8B-enable_thinking_True-tips_threshold_21-1765561271.jsonl",
    "qasper": "/data/lyc/CopyPasteLRM/pass_at_42/qasper-resamples_2000-tpr_0.6-tpp_0.95-Qwen3-8B-enable_thinking_True-tips_threshold_21-1765488919.jsonl",
    "popqa": "/data/lyc/CopyPasteLRM/pass_at_42/popqa-resamples_2000-tpr_0.6-tpp_0.95-Qwen3-8B-enable_thinking_True-tips_threshold_21-1765422382.jsonl",
}

paths_235B = {
    "2wiki": "results/infer/qwen3-235b-thinking/2wikimultihopqa/enable_thinking_False-prompt_direct-1765691551.json",
    "hotpotqa": "results/infer/qwen3-235b-thinking/hotpotqa/enable_thinking_False-prompt_direct-1765635169.json",
    "multirc": "results/infer/qwen3-235b-thinking/multirc/enable_thinking_False-prompt_direct-1765674954.json",
    "musiqua": "results/infer/qwen3-235b-thinking/musique/enable_thinking_False-prompt_direct-1765675179.json",
    "qasper": "results/infer/qwen3-235b-thinking/qasper/enable_thinking_False-prompt_direct-1765702159.json",
    "popqa": "results/infer/qwen3-235b-thinking/popqa/enable_thinking_False-prompt_direct-1765696326.json",
}


def get_passAtK0_samples_ids(data: list):
    """获取模型无法回答的samples的id列表"""
    is_correct_dict = {}
    for item in data:
        sample_id = item["id"]
        if "is_correct" not in item:
            # 不存在is_correct字段，说明该样本的推理出现了错误（jsonl的空行？），直接跳过
            continue
        is_correct = item["is_correct"]
        if sample_id not in is_correct_dict:
            is_correct_dict[sample_id] = is_correct
        else:
            if is_correct_dict[sample_id]:
                continue
            else:
                is_correct_dict[sample_id] = is_correct

    return [
        sample_id for sample_id, is_correct in is_correct_dict.items() if not is_correct
    ]


datas = {}
datas_235B = {}

for key, path in paths.items():
    datas[key] = read_jsonl_to_list(path)
    datas_235B[key] = read_json(paths_235B[key])["data"]


passAtK0_data = []

context_query_template = Template(user_prompt_template)


for key, data in datas.items():
    print("=" * 10, key, "=" * 10)
    passAtK0_samples_ids = get_passAtK0_samples_ids(data)
    data_235B = datas_235B[key]

    empty_sfs_count = 0
    empty_ans_count = 0

    sfs_before_count = 0
    sfs_after_count = 0

    for id in tqdm(passAtK0_samples_ids):
        id = str(id)
        query = data_235B[id]["query"]
        context = data_235B[id]["context"]
        answer = data_235B[id]["answer"]
        sfs = data_235B[id]["sfs"]

        if len(sfs) == 0:
            empty_sfs_count += 1
        if len(answer) == 0:
            empty_ans_count += 1
        if len(sfs) == 0 or len(answer) == 0:
            continue

        sfs_before_count += len(sfs)
        sfs_processor = StringContainmentFilter(sfs)
        sfs = sfs_processor.filter_maximal_superstrings()
        sfs_after_count += len(sfs)

        passAtK0_data.append(
            {
                "system": system_prompt,
                "query": context_query_template.substitute(
                    context=context,
                    question=query,
                ),
                "supporting_facts": sfs_processor.filter_maximal_superstrings(),
                "answer_candidates": answer,
                "context": context,
                "dataset": key,
                "id": id,
            }
        )

    print("empty sfs count:", empty_sfs_count)
    print("empty ans count:", empty_ans_count)
    print("sfs before count:", sfs_before_count)
    print("sfs after count:", sfs_after_count)

# 打乱passAtK0_data顺序
random.seed(42)
random.shuffle(passAtK0_data)

save_jsonl(passAtK0_data, "passAtK0_data.jsonl")
