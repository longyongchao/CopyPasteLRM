from string import Template
import random
from tqdm import tqdm
import os 

from copypastelrm.utils.json_tools import read_jsonl, save_jsonl, read_jsonl_to_list

from copypastelrm.datasets.HotpotQA import HotpotQA
from copypastelrm.datasets.MultiRC import MultiRC
from copypastelrm.datasets.MuSiQue import MuSiQue
from copypastelrm.datasets.TwoWikiMultiHopQA import TwoWikiMultihopQA
from copypastelrm.datasets.PopQA import PopQA
from copypastelrm.datasets.Qasper import Qasper

from copypastelrm.utils.llm_server import LLMServer

BASE_URL = "https://api.siliconflow.cn/v1"
MODEL = "deepseek-ai/DeepSeek-V3.2"
API_KEY = "sk-lqztxtcbxxoonlmsxvdhllhdnoegywnvuhfnoqnxvpphrhkh"

llm_server = LLMServer(BASE_URL, API_KEY)

user_prompt_template = """你的任务是对一个数据集样本进行质量判定。

请基于给定的 Context 及其标注的 Supporting Facts，判断 是否可以通过多跳推理，得到 Answer。判断标准如下：

1. **问题匹配性**：Answer 必须明确、直接地回答 Question，而不是无关或只部分相关。
2. **证据依赖性**：Answer 中的关键信息必须能够从多个 Supporting Facts 中得到支持，不能引入未在 Supporting Facts 中出现的关键事实、推断或常识性补充。
3. **证据充分性**：各个Supporting Facts 必须是回答 Question 所必需的关键证据，而 Answer 的结论应当可以主要或完全由这些 Supporting Facts 推导得到。

如果 Answer 同时满足以上所有条件，请返回 **"yes"**；  
否则（例如 Answer 未回答 Question、依赖 Context 中但未被标注为 Supporting Facts 的内容、存在关键事实缺失或推断），请返回 **"no"**。

除 "yes" 或 "no" 之外，不要输出任何其他内容。

## Question
$question

## Context
$context

## Supporting Facts
$sup_facts

## Answer
$answer
"""

paths = {
    "musique": "key_data/hard/musique_hard.jsonl",
}

dataloader = {
    # "hotpotqa": HotpotQA(split='validation').dataset,
    # "2wiki":    TwoWikiMultihopQA(split="dev").dataset,
    # "popqa":    PopQA('train').dataset,
    # "multirc":  MultiRC(split='train').dataset,
    "musique":  MuSiQue(split='validation').dataset,
    # "qasper":   Qasper('train').dataset,
}


template = Template(user_prompt_template)

for key, path in paths.items():
    print(f"Processing {key}...")
    data = read_jsonl(path)
    ids = read_jsonl_to_list(path)
    print(f"{key} has {len(ids)} samples that pass at k=0")

    random.shuffle(ids)
    
    origin_data = dataloader[key]

    avaible_data = []

    for i in tqdm(ids):
        if i not in origin_data:
            print('id not in origin_data, id: ', i)
            continue
        origin_item = origin_data[i]
        query = origin_item["query"]
        context = origin_item["context"]
        answers = origin_item['answer']
        sfs = origin_item['sfs']

        if len(answers) == 0 or len(sfs) == 0 or len(context) == 0 or len(query) == 0:
            continue

        prompt = template.substitute(
            question=query,
            context=context,
            sup_facts=sfs,
            answer="\n- ".join(answers),
        )
        response = llm_server.call(MODEL, prompt, enable_thinking=True)
        print(response)

        avaible_data.append({
            "id": i,
            "llm_response": response,
        })
        
    # 获取path所在的文件夹
    folder = os.path.dirname(path)
    save_jsonl(avaible_data, folder + f"/avaible_pass@K=0_subset_{key}.jsonl")