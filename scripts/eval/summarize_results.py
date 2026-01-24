#!/usr/bin/env python3
"""
结果汇总脚本：生成 RAG vs RAG×2 对比表格
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple


def load_eval_results(base_path: str) -> Dict[str, Dict]:
    """
    加载所有评估结果

    Args:
        base_path: 基础路径

    Returns:
        dict: {(prompt_type, dataset): eval_results}
    """
    base_path = Path(base_path)
    results = {}

    prompt_types = ["prompt_rag", "prompt_rag_rep_2"]

    for prompt_type in prompt_types:
        prompt_dir = base_path / "seed_42" / "tpr_0.0" / prompt_type

        if not prompt_dir.exists():
            continue

        for dataset_dir in sorted(prompt_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue

            # 查找评估结果文件
            for timestamp_dir in dataset_dir.iterdir():
                if not timestamp_dir.is_dir():
                    continue

                eval_file = timestamp_dir.parent / (timestamp_dir.name + "_eval.json")

                if eval_file.exists():
                    try:
                        with open(eval_file, 'r') as f:
                            eval_data = json.load(f)
                            # 提取数据集名称（去掉 -noise_8-answerable 后缀）
                            dataset_name = dataset_dir.name.replace("-noise_8-answerable", "")
                            results[(prompt_type, dataset_name)] = eval_data
                        break  # 找到就停止
                    except Exception as e:
                        print(f"[Warning] Failed to load {eval_file}: {e}")

    return results


def calculate_delta(val1: float, val2: float) -> str:
    """
    计算差异百分比

    Args:
        val1: 基准值
        val2: 对比值

    Returns:
        str: 格式化的差异字符串
    """
    if val1 == 0:
        delta_pct = 0
    else:
        delta_pct = (val2 - val1) / val1 * 100

    sign = "+" if delta_pct > 0 else ""
    return f"{sign}{delta_pct:.1f}%"


def generate_comparison_table(results: Dict[str, Dict]) -> str:
    """
    生成对比表格

    Args:
        results: 评估结果字典

    Returns:
        str: Markdown 格式的表格
    """
    # 提取所有数据集名称（排序）
    datasets = sorted(set(k[1] for k in results.keys()))

    # 表头
    header = "| Dataset | RAG Answer EM | RAG×2 Answer EM | Δ EM | RAG Answer F1 | RAG×2 Answer F1 | Δ F1 | RAG Supp. F1 | RAG×2 Supp. F1 | Δ Supp. F1 |\n"
    header += "|" + "|".join(["-" * 10] * 10) + "|\n"

    rows = []
    em_rag_list, em_rep2_list = [], []
    f1_rag_list, f1_rep2_list = [], []
    supp_f1_rag_list, supp_f1_rep2_list = [], []

    for dataset in datasets:
        rag_key = ("prompt_rag", dataset)
        rep2_key = ("prompt_rag_rep_2", dataset)

        rag_result = results.get(rag_key)
        rep2_result = results.get(rep2_key)

        if not rag_result or not rep2_result:
            continue  # 跳过不完整的数据

        # 提取指标
        rag_em = rag_result.get("answer_em", 0)
        rep2_em = rep2_result.get("answer_em", 0)

        rag_f1 = rag_result.get("answer_f1", 0)
        rep2_f1 = rep2_result.get("answer_f1", 0)

        rag_supp_f1 = rag_result.get("supporting_f1", 0)
        rep2_supp_f1 = rep2_result.get("supporting_f1", 0)

        # 计算差异
        delta_em = calculate_delta(rag_em, rep2_em)
        delta_f1 = calculate_delta(rag_f1, rep2_f1)
        delta_supp_f1 = calculate_delta(rag_supp_f1, rep2_supp_f1)

        # 添加行
        row = f"| {dataset} | {rag_em:.4f} | {rep2_em:.4f} | {delta_em} | {rag_f1:.4f} | {rep2_f1:.4f} | {delta_f1} | {rag_supp_f1:.4f} | {rep2_supp_f1:.4f} | {delta_supp_f1} |"
        rows.append(row)

        # 收集数据用于计算平均值
        em_rag_list.append(rag_em)
        em_rep2_list.append(rep2_em)
        f1_rag_list.append(rag_f1)
        f1_rep2_list.append(rep2_f1)
        supp_f1_rag_list.append(rag_supp_f1)
        supp_f1_rep2_list.append(rep2_supp_f1)

    # 计算平均行
    if em_rag_list:
        avg_em_rag = sum(em_rag_list) / len(em_rag_list)
        avg_em_rep2 = sum(em_rep2_list) / len(em_rep2_list)
        avg_f1_rag = sum(f1_rag_list) / len(f1_rag_list)
        avg_f1_rep2 = sum(f1_rep2_list) / len(f1_rep2_list)
        avg_supp_f1_rag = sum(supp_f1_rag_list) / len(supp_f1_rag_list)
        avg_supp_f1_rep2 = sum(supp_f1_rep2_list) / len(supp_f1_rep2_list)

        delta_em_avg = calculate_delta(avg_em_rag, avg_em_rep2)
        delta_f1_avg = calculate_delta(avg_f1_rag, avg_f1_rep2)
        delta_supp_f1_avg = calculate_delta(avg_supp_f1_rag, avg_supp_f1_rep2)

        avg_row = f"| **平均** | **{avg_em_rag:.4f}** | **{avg_em_rep2:.4f}** | **{delta_em_avg}** | **{avg_f1_rag:.4f}** | **{avg_f1_rep2:.4f}** | **{delta_f1_avg}** | **{avg_supp_f1_rag:.4f}** | **{avg_supp_f1_rep2:.4f}** | **{delta_supp_f1_avg}** |"
        rows.append(avg_row)

    return header + "\n".join(rows)


def calculate_stats(results: Dict[str, Dict]) -> Dict:
    """
    计算统计信息

    Args:
        results: 评估结果字典

    Returns:
        dict: 统计信息
    """
    datasets = sorted(set(k[1] for k in results.keys()))

    stats = {
        "em_rag_better": 0,
        "em_rep2_better": 0,
        "em_tie": 0,
        "f1_rag_better": 0,
        "f1_rep2_better": 0,
        "f1_tie": 0,
        "supp_f1_rag_better": 0,
        "supp_f1_rep2_better": 0,
        "supp_f1_tie": 0,
    }

    for dataset in datasets:
        rag_result = results.get(("prompt_rag", dataset))
        rep2_result = results.get(("prompt_rag_rep_2", dataset))

        if not rag_result or not rep2_result:
            continue

        # EM 统计
        if rag_result["answer_em"] > rep2_result["answer_em"]:
            stats["em_rag_better"] += 1
        elif rag_result["answer_em"] < rep2_result["answer_em"]:
            stats["em_rep2_better"] += 1
        else:
            stats["em_tie"] += 1

        # F1 统计
        if rag_result["answer_f1"] > rep2_result["answer_f1"]:
            stats["f1_rag_better"] += 1
        elif rag_result["answer_f1"] < rep2_result["answer_f1"]:
            stats["f1_rep2_better"] += 1
        else:
            stats["f1_tie"] += 1

        # Supporting F1 统计
        if rag_result["supporting_f1"] > rep2_result["supporting_f1"]:
            stats["supp_f1_rag_better"] += 1
        elif rag_result["supporting_f1"] < rep2_result["supporting_f1"]:
            stats["supp_f1_rep2_better"] += 1
        else:
            stats["supp_f1_tie"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="汇总 RAG vs RAG×2 对比结果")
    parser.add_argument(
        "--base-path",
        type=str,
        default="results/infer/test/Qwen2.5-7B-Instruct/resamples_1000",
        help="基础路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出文件路径（默认打印到终端）"
    )
    args = parser.parse_args()

    # 加载评估结果
    print(f"Loading results from: {args.base_path}")
    results = load_eval_results(args.base_path)

    if not results:
        print("[Error] No evaluation results found!")
        print("Please run the batch evaluation first:")
        print(f"  python scripts/eval/batch_eval_compare.py --base-path {args.base_path}")
        return

    print(f"Loaded {len(results)} evaluation results\n")

    # 生成对比表格
    table = generate_comparison_table(results)

    # 计算统计信息
    stats = calculate_stats(results)

    # 输出结果
    output = f"\n{table}\n\n"

    output += "## 统计信息\n\n"
    output += "### Answer EM\n"
    output += f"- RAG 胜出: {stats['em_rag_better']} 个数据集\n"
    output += f"- RAG×2 胜出: {stats['em_rep2_better']} 个数据集\n"
    output += f"- 持平: {stats['em_tie']} 个数据集\n\n"

    output += "### Answer F1\n"
    output += f"- RAG 胜出: {stats['f1_rag_better']} 个数据集\n"
    output += f"- RAG×2 胜出: {stats['f1_rep2_better']} 个数据集\n"
    output += f"- 持平: {stats['f1_tie']} 个数据集\n\n"

    output += "### Supporting F1\n"
    output += f"- RAG 胜出: {stats['supp_f1_rag_better']} 个数据集\n"
    output += f"- RAG×2 胜出: {stats['supp_f1_rep2_better']} 个数据集\n"
    output += f"- 持平: {stats['supp_f1_tie']} 个数据集\n"

    # 输出到终端或文件
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"\nResults saved to: {args.output}\n")
    else:
        print(output)


if __name__ == "__main__":
    main()
