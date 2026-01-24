#!/usr/bin/env python3
"""
对比两个文件夹的评估结果

传入两个文件夹路径（基线 vs 对比方法），输出对比表格
按 Δ EM 降序排序
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple


def load_eval_results_from_folder(folder_path: str) -> Dict[str, Dict]:
    """
    从文件夹中加载所有评估结果

    Args:
        folder_path: 文件夹路径

    Returns:
        dict: {dataset_name: eval_results}
    """
    folder_path = Path(folder_path)
    results = {}

    if not folder_path.exists() or not folder_path.is_dir():
        return results

    # 遍历所有数据集子目录
    for dataset_dir in sorted(folder_path.iterdir()):
        if not dataset_dir.is_dir():
            continue

        # 查找评估结果文件
        eval_file = None
        for item in dataset_dir.iterdir():
            if item.is_dir():
                # 检查是否有对应的 _eval.json 文件
                potential_eval = dataset_dir / (item.name + "_eval.json")
                if potential_eval.exists():
                    eval_file = potential_eval
                    break

        if not eval_file:
            # 可能直接是 *_eval.json 文件
            for item in dataset_dir.glob("*_eval.json"):
                eval_file = item
                break

        if eval_file and eval_file.exists():
            try:
                with open(eval_file, 'r') as f:
                    eval_data = json.load(f)
                    # 提取数据集名称
                    dataset_name = dataset_dir.name
                    # 去掉常见的后缀
                    for suffix in ["-noise_8-answerable", "-noise_8-unanswerable", "-noise_0-answerable"]:
                        if dataset_name.endswith(suffix):
                            dataset_name = dataset_name[:-len(suffix)]
                            break
                    results[dataset_name] = eval_data
            except Exception as e:
                print(f"[Warning] Failed to load {eval_file}: {e}")

    return results


def calculate_delta(val1: float, val2: float) -> Tuple[float, str]:
    """
    计算差异百分比

    Args:
        val1: 基准值
        val2: 对比值

    Returns:
        tuple: (delta_value, formatted_string)
    """
    if val1 == 0:
        delta_pct = 0.0
    else:
        delta_pct = (val2 - val1) / val1 * 100

    sign = "+" if delta_pct > 0 else ""
    return delta_pct, f"{sign}{delta_pct:.1f}%"


def generate_comparison_table(
    baseline_results: Dict[str, Dict],
    compare_results: Dict[str, Dict],
    baseline_name: str = "Baseline",
    compare_name: str = "Method B"
) -> str:
    """
    生成对比表格，按 Δ EM 降序排序

    Args:
        baseline_results: 基线结果字典
        compare_results: 对比结果字典
        baseline_name: 基线方法名称
        compare_name: 对比方法名称

    Returns:
        str: Markdown 格式的表格
    """
    # 找出两个文件夹中都有的数据集
    common_datasets = set(baseline_results.keys()) & set(compare_results.keys())

    if not common_datasets:
        return "No common datasets found!"

    # 表头
    header = f"| Dataset | {baseline_name} EM | {compare_name} EM | Δ EM | {baseline_name} F1 | {compare_name} F1 | Δ F1 |\n"
    header += "|" + "|".join(["-" * 15] * 7) + "|\n"

    # 收集数据并计算 Δ EM
    rows_data = []
    em_baseline_list, em_compare_list = [], []
    f1_baseline_list, f1_compare_list = [], []

    for dataset in common_datasets:
        baseline = baseline_results[dataset]
        compare = compare_results[dataset]

        baseline_em = baseline.get("answer_em", 0)
        compare_em = compare.get("answer_em", 0)
        baseline_f1 = baseline.get("answer_f1", 0)
        compare_f1 = compare.get("answer_f1", 0)

        delta_em_pct, delta_em_str = calculate_delta(baseline_em, compare_em)
        delta_f1_pct, delta_f1_str = calculate_delta(baseline_f1, compare_f1)

        rows_data.append({
            "dataset": dataset,
            "baseline_em": baseline_em,
            "compare_em": compare_em,
            "delta_em_pct": delta_em_pct,
            "delta_em_str": delta_em_str,
            "baseline_f1": baseline_f1,
            "compare_f1": compare_f1,
            "delta_f1_str": delta_f1_str,
        })

        # 收集数据用于计算平均值
        em_baseline_list.append(baseline_em)
        em_compare_list.append(compare_em)
        f1_baseline_list.append(baseline_f1)
        f1_compare_list.append(compare_f1)

    # 按 Δ EM 降序排序
    rows_data.sort(key=lambda x: x["delta_em_pct"], reverse=True)

    # 生成行
    rows = []
    for row in rows_data:
        rows.append(
            f"| {row['dataset']} | {row['baseline_em']:.4f} | {row['compare_em']:.4f} | "
            f"**{row['delta_em_str']}** | {row['baseline_f1']:.4f} | {row['compare_f1']:.4f} | {row['delta_f1_str']} |"
        )

    # 计算平均行
    if em_baseline_list:
        avg_em_baseline = sum(em_baseline_list) / len(em_baseline_list)
        avg_em_compare = sum(em_compare_list) / len(em_compare_list)
        avg_f1_baseline = sum(f1_baseline_list) / len(f1_baseline_list)
        avg_f1_compare = sum(f1_compare_list) / len(f1_compare_list)

        _, avg_delta_em_str = calculate_delta(avg_em_baseline, avg_em_compare)
        _, avg_delta_f1_str = calculate_delta(avg_f1_baseline, avg_f1_compare)

        avg_row = (
            f"| **平均** | **{avg_em_baseline:.4f}** | **{avg_em_compare:.4f}** | "
            f"**{avg_delta_em_str}** | **{avg_f1_baseline:.4f}** | **{avg_f1_compare:.4f}** | **{avg_delta_f1_str}** |"
        )
        rows.append(avg_row)

    return header + "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(
        description="对比两个文件夹的评估结果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/eval/compare_two_folders.py \\
      results/infer/test/.../prompt_rag \\
      results/infer/test/.../prompt_rag_rep_2

  python scripts/eval/compare_two_folders.py \\
      results/infer/test/.../prompt_rag \\
      results/infer/test/.../prompt_rag_rep_2 \\
      --baseline-name "RAG" --compare-name "RAG×2" \\
      --output comparison_table.md
        """
    )
    parser.add_argument("baseline_folder", type=str, help="基线方法文件夹路径")
    parser.add_argument("compare_folder", type=str, help="对比方法文件夹路径")
    parser.add_argument(
        "--baseline-name",
        type=str,
        default="Baseline",
        help="基线方法显示名称（默认：Baseline）"
    )
    parser.add_argument(
        "--compare-name",
        type=str,
        default="Method B",
        help="对比方法显示名称（默认：Method B）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出文件路径（默认打印到终端）"
    )
    args = parser.parse_args()

    # 加载评估结果
    print(f"Loading baseline results from: {args.baseline_folder}")
    baseline_results = load_eval_results_from_folder(args.baseline_folder)
    print(f"  Found {len(baseline_results)} datasets\n")

    print(f"Loading compare results from: {args.compare_folder}")
    compare_results = load_eval_results_from_folder(args.compare_folder)
    print(f"  Found {len(compare_results)} datasets\n")

    if not baseline_results:
        print(f"[Error] No evaluation results found in baseline folder!")
        print("Please ensure *_eval.json files exist in the dataset subfolders.")
        return

    if not compare_results:
        print(f"[Error] No evaluation results found in compare folder!")
        print("Please ensure *_eval.json files exist in the dataset subfolders.")
        return

    # 生成对比表格
    table = generate_comparison_table(
        baseline_results,
        compare_results,
        args.baseline_name,
        args.compare_name
    )

    # 输出
    if args.output:
        with open(args.output, 'w') as f:
            f.write(table)
        print(f"\nComparison table saved to: {args.output}\n")
    else:
        print("\n" + table + "\n")


if __name__ == "__main__":
    main()
