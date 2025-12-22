from copypastelrm.utils.json_tools import read_jsonl_to_list, read_json, save_jsonl
from string import Template
from tqdm import tqdm
from copypastelrm.utils.dataset import StringContainmentFilter
import random
from typing import Literal

import os
import glob

from copypastelrm.prompt.passAtKEqual0 import system_prompt, user_prompt_template 

#############################

MODEL_NAME: Literal['Qwen3-4B-Instruct-2507', 'Qwen3-8B', 'Qwen2.5-3B-Instruct', 'Qwen2.5-7B-Instruct'] = 'Qwen3-4B-Instruct-2507'
RESAMPLES: Literal[2000, 5000, 10000] = 2000
SYSTEM_PROMPT: Literal['deepseek', 'copypaste'] = 'deepseek'

#############################


def get_subset_paths(base_dir):
    """
    遍历指定目录，寻找符合 'avaible_pass@K=0_subset_*.jsonl' 格式的文件，
    并返回以子集名称为 key，完整路径为 value 的字典。
    """
    paths = {}
    
    # 构建匹配模式：base_dir + 文件通配符
    # 使用 os.path.join 确保路径分隔符在不同系统下都能正常工作
    pattern = os.path.join(base_dir, "avaible_pass@K=0_subset_*.jsonl")
    
    # 查找所有匹配的文件路径
    file_list = glob.glob(pattern)
    
    for file_path in file_list:
        # 获取文件名（例如：avaible_pass@K=0_subset_2wiki.jsonl）
        file_name = os.path.basename(file_path)
        
        # 提取子集名称：
        # 1. 去掉前缀 "avaible_pass@K=0_subset_"
        # 2. 去掉后缀 ".jsonl"
        subset_name = file_name.replace("avaible_pass@K=0_subset_", "").replace(".jsonl", "")
        
        paths[subset_name] = file_path
        
    return paths


if __name__ == "__main__":

    folder_path = f"/data/lyc/CopyPasteLRM/pass_at_42/{MODEL_NAME}/resamples_{RESAMPLES}/"
    paths = get_subset_paths(folder_path)
    dataset_count = ""

    datas = {}

    for key, path in paths.items():
        datas[key] = read_jsonl_to_list(path)

    possible_passAtKEqual0_subset = []

    user_template = Template(user_prompt_template)

    for dataset_name, data in datas.items():
        sfs_before_count = 0
        sfs_after_count = 0
        count = 0
        for item in tqdm(data):
            id = str(id)
            query = item["query"]
            context = item["context"]
            answer = item["answer"]
            sfs = item["sfs"]

            sfs_before_count += len(sfs)
            sfs_processor = StringContainmentFilter(sfs)
            sfs = sfs_processor.filter_maximal_superstrings()
            sfs_after_count += len(sfs)

            possible_passAtKEqual0_subset.append(
                {
                    "system": system_prompt[SYSTEM_PROMPT],
                    "query": user_template.substitute(
                        context=context,
                        question=query,
                    ),
                    "supporting_facts": sfs_processor.filter_maximal_superstrings(),
                    "answer_candidates": answer,
                    "context": context,
                    "dataset": dataset_name,
                    "id": id,
                }
            )
            count += 1

        dataset_count += f"_{dataset_name}-{count}"

        print("sfs before count:", sfs_before_count)
        print("sfs after count:", sfs_after_count)

    # 打乱passAtK0_data顺序
    random.seed(42)
    random.shuffle(possible_passAtKEqual0_subset)

    save_jsonl(possible_passAtKEqual0_subset, f"data/possiblePassAtKEqual0Subset/{MODEL_NAME}_{RESAMPLES}_{SYSTEM_PROMPT}{dataset_count}.jsonl")
