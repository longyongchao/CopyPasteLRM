"""
评估脚本：从文件夹读取推理结果并计算指标

支持新的结果格式：每个样本一个JSON文件，保存在文件夹中
计算指标：Answer EM、Answer F1、Supporting F1
"""
import argparse
import json
import os
import sys
from typing import List

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from copypastelrm.datasets import load, AvailableDataset
from copypastelrm.metrics.HotpotQA import compute_answer_em_hit_f1, update_sp
from copypastelrm.metrics.utils import extract_answer_and_facts, extract_answer_and_facts_old


def load_samples_from_folder(folder_path: str):
    """
    从文件夹中加载所有样本JSON文件

    Args:
        folder_path: 包含样本文件的文件夹路径

    Returns:
        dict: 样本ID到样本数据的映射
    """
    samples = {}
    metadata_file = os.path.join(folder_path, "metadata.json")

    # 从metadata获取prompt_type
    prompt_type = "rag"  # 默认值
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            prompt_type = metadata.get("info", {}).get("prompt_type", "rag")

    # 遍历文件夹中的所有JSON文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".json") and filename != "metadata.json":
            filepath = os.path.join(folder_path, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    sample_data = json.load(f)
                    sample_id = sample_data.get("id")
                    if sample_id:
                        samples[sample_id] = {
                            "query": sample_data.get("query", ""),
                            "predict": sample_data.get("predict"),
                            "context": sample_data.get("context"),
                            "answers": sample_data.get("answers", []),
                            "prompt_type": prompt_type
                        }
            except Exception as e:
                print(f"[Warning] Failed to load {filename}: {e}")

    return samples


def evaluate_folder(folder_path: str):
    """
    评估文件夹中的推理结果

    Args:
        folder_path: 包含样本文件的文件夹路径
    """
    # 1. 加载推理结果
    print(f"Loading results from: {folder_path}")
    samples = load_samples_from_folder(folder_path)

    if not samples:
        print("No valid samples found!")
        return

    print(f"Loaded {len(samples)} samples")

    # 2. 从metadata获取数据集信息
    metadata_file = os.path.join(folder_path, "metadata.json")
    if not os.path.exists(metadata_file):
        print("[Error] metadata.json not found!")
        return

    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
        info = metadata.get("info", {})
        dataset_name = info.get("dataset")
        prompt_type = info.get("prompt_type", "rag")

    if not dataset_name:
        print("[Error] Dataset name not found in metadata!")
        return

    print(f"Dataset: {dataset_name}")
    print(f"Prompt type: {prompt_type}")

    # 3. 加载gold标准答案
    try:
        dataset_enum = AvailableDataset(dataset_name)
        dataset_loader = load(name=dataset_enum, split="test", distractor_docs=0, reload=False)
        dataset_dict = dataset_loader.dataset
    except Exception as e:
        print(f"[Error] Failed to load dataset: {e}")
        return

    # 4. 选择提取函数
    extract_func = extract_answer_and_facts_old if prompt_type in ['copypaste', 'deepseek', 'find_facts'] else extract_answer_and_facts

    # 5. 计算指标
    answer_em_list = []
    answer_f1_list = []
    supporting_f1_list = []

    for sample_id, sample in samples.items():
        predict = sample.get("predict")
        if not predict or predict.startswith("ERROR:"):
            continue

        # 获取gold答案和facts
        gold_sample = dataset_dict.get(sample_id)
        if not gold_sample:
            continue

        gold_answers = gold_sample.get("answers", [])
        gold_facts = gold_sample.get("facts", [])

        # 提取预测答案和facts
        predict_answer, predict_facts = extract_func(predict)

        # 计算Answer EM和F1
        em, _, f1 = compute_answer_em_hit_f1(predict_answer, gold_answers)
        answer_em_list.append(em)
        answer_f1_list.append(f1)

        # 计算Supporting F1
        _, sp_f1, _, _ = update_sp(predict_facts, gold_facts)
        supporting_f1_list.append(sp_f1)

    # 6. 计算平均值
    total_samples = len(answer_em_list)
    if total_samples == 0:
        print("[Warning] No valid samples for evaluation!")
        return

    avg_answer_em = sum(answer_em_list) / total_samples
    avg_answer_f1 = sum(answer_f1_list) / total_samples
    avg_supporting_f1 = sum(supporting_f1_list) / total_samples

    # 7. 输出结果
    print(f"\n{'='*60}")
    print(f"Evaluation Results ({dataset_name})")
    print(f"{'='*60}")
    print(f"Total samples evaluated: {total_samples}")
    print(f"Answer EM:  {avg_answer_em:.4f}")
    print(f"Answer F1:  {avg_answer_f1:.4f}")
    print(f"Supporting F1: {avg_supporting_f1:.4f}")
    print(f"{'='*60}")

    # 8. 保存结果
    output_file = folder_path + "_eval.json"
    result = {
        "dataset": dataset_name,
        "prompt_type": prompt_type,
        "total_samples": total_samples,
        "answer_em": avg_answer_em,
        "answer_f1": avg_answer_f1,
        "supporting_f1": avg_supporting_f1
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate inference results from folder")
    parser.add_argument("folder_path", type=str, help="Path to folder containing sample JSON files")
    args = parser.parse_args()

    if not os.path.isdir(args.folder_path):
        print(f"[Error] {args.folder_path} is not a valid directory!")
        return

    evaluate_folder(args.folder_path)


if __name__ == "__main__":
    main()
