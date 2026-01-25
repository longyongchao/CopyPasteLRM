#!/usr/bin/env python3
"""
对比多个方法在各个数据集上的表现

输入包含多个方法子文件夹的父目录，输出对比表格
- 行：不同的方法
- 列：各个数据集的 EM 和 F1 指标
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple


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


def discover_methods(parent_path: str) -> List[str]:
    """
    发现父目录中所有的方法文件夹（以 prompt_ 开头）

    Args:
        parent_path: 父目录路径

    Returns:
        list: 方法文件夹名称列表（已排序）
    """
    parent_path = Path(parent_path)
    if not parent_path.exists() or not parent_path.is_dir():
        return []

    methods = []
    for item in sorted(parent_path.iterdir()):
        if item.is_dir() and item.name.startswith('prompt_'):
            methods.append(item.name)

    return methods


def load_all_methods_results(parent_path: str) -> Dict[str, Dict[str, Dict]]:
    """
    加载所有方法的评估结果

    Args:
        parent_path: 父目录路径

    Returns:
        dict: {method_name: {dataset_name: eval_results}}
    """
    methods = discover_methods(parent_path)
    all_results = {}

    parent_path = Path(parent_path)
    for method_name in methods:
        method_path = parent_path / method_name
        results = load_eval_results_from_folder(str(method_path))
        all_results[method_name] = results

    return all_results


def get_all_datasets(all_results: Dict[str, Dict[str, Dict]]) -> List[str]:
    """
    获取所有方法中出现过的唯一数据集名称

    Args:
        all_results: 所有方法的结果字典

    Returns:
        list: 排序后的数据集名称列表
    """
    datasets = set()
    for method_results in all_results.values():
        datasets.update(method_results.keys())
    return sorted(datasets)


def format_method_name(method_name: str, strip_prefix: bool = True) -> str:
    """
    格式化方法名称用于显示

    Args:
        method_name: 原始方法名（如 'prompt_rag_rep_2'）
        strip_prefix: 是否移除 'prompt_' 前缀

    Returns:
        str: 格式化后的名称（如 'RAG×2'）
    """
    if strip_prefix:
        name = method_name.replace('prompt_', '')
        # 转换为可读格式
        name = name.replace('_', ' ').title()
        # 特殊符号替换
        name = name.replace('Rep 2', '×2')
        name = name.replace('Rep Q', '×Q')
        return name
    return method_name


def calculate_average(
    method_results: Dict[str, Dict],
    datasets: List[str],
    metric: str
) -> Optional[float]:
    """
    计算某个方法在所有数据集上的平均指标

    Args:
        method_results: 单个方法的结果字典
        datasets: 数据集列表
        metric: 指标名称（如 'answer_em', 'answer_f1'）

    Returns:
        float or None: 平均值，如果没有有效数据则返回 None
    """
    values = []
    for dataset in datasets:
        if dataset in method_results and metric in method_results[dataset]:
            values.append(method_results[dataset][metric])

    if not values:
        return None
    return sum(values) / len(values)


def build_comparison_table(
    all_results: Dict[str, Dict[str, Dict]],
    datasets: List[str],
    sort_by: str = "avg_em",
    sort_order: str = "desc",
    strip_prefix: bool = True,
    missing_indicator: str = ""
) -> str:
    """
    构建多方法对比表格

    Args:
        all_results: 所有方法的结果字典
        datasets: 数据集列表
        sort_by: 排序依据 ('avg_em', 'avg_f1', 'name')
        sort_order: 排序顺序 ('desc', 'asc')
        strip_prefix: 是否移除方法名前缀
        missing_indicator: 缺失数据的显示符号

    Returns:
        str: CSV 格式的表格
    """
    if not all_results or not datasets:
        return "No data to display!"

    # 构建表头
    header_parts = ["Method"]
    for dataset in datasets:
        header_parts.append(f"{dataset} EM")
        header_parts.append(f"{dataset} F1")
    header_parts.extend(["Avg EM", "Avg F1"])

    header = ",".join(header_parts)

    # 为每个方法收集数据
    rows_data = []
    for method_name, method_results in all_results.items():
        row = {
            "method": method_name,
            "display_name": format_method_name(method_name, strip_prefix),
            "metrics": {}
        }

        # 收集每个数据集的指标
        for dataset in datasets:
            if dataset in method_results:
                row["metrics"][dataset] = {
                    "em": method_results[dataset].get("answer_em"),
                    "f1": method_results[dataset].get("answer_f1")
                }
            else:
                row["metrics"][dataset] = {"em": None, "f1": None}

        # 计算平均值
        row["avg_em"] = calculate_average(method_results, datasets, "answer_em")
        row["avg_f1"] = calculate_average(method_results, datasets, "answer_f1")

        rows_data.append(row)

    # 排序
    reverse = (sort_order == "desc")
    if sort_by == "name":
        rows_data.sort(key=lambda x: x["method"], reverse=reverse)
    elif sort_by == "avg_em":
        rows_data.sort(key=lambda x: x["avg_em"] or 0, reverse=reverse)
    elif sort_by == "avg_f1":
        rows_data.sort(key=lambda x: x["avg_f1"] or 0, reverse=reverse)

    # 生成表格行
    rows = []
    for row in rows_data:
        row_parts = [row["display_name"]]

        for dataset in datasets:
            em = row["metrics"][dataset]["em"]
            f1 = row["metrics"][dataset]["f1"]
            em_str = f"{em:.4f}" if em is not None else missing_indicator
            f1_str = f"{f1:.4f}" if f1 is not None else missing_indicator
            row_parts.append(em_str)
            row_parts.append(f1_str)

        # 平均值列
        avg_em_str = f"{row['avg_em']:.4f}" if row['avg_em'] is not None else missing_indicator
        avg_f1_str = f"{row['avg_f1']:.4f}" if row['avg_f1'] is not None else missing_indicator
        row_parts.append(avg_em_str)
        row_parts.append(avg_f1_str)

        rows.append(",".join(row_parts))

    return header + "\n" + "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(
        description="对比多个方法在各个数据集上的表现",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法：对比所有方法
  python scripts/eval/compare_multiple_methods.py \\
      results/infer/test/Qwen2.5-7B-Instruct/resamples_1500/seed_42/tpr_0.0

  # 按平均 F1 排序
  python scripts/eval/compare_multiple_methods.py \\
      results/infer/test/.../tpr_0.0 \\
      --sort-by avg_f1

  # 按方法名称排序
  python scripts/eval/compare_multiple_methods.py \\
      results/infer/test/.../tpr_0.0 \\
      --sort-by name

  # 保存到文件
  python scripts/eval/compare_multiple_methods.py \\
      results/infer/test/.../tpr_0.0 \\
      --output comparison.md
        """
    )
    parser.add_argument(
        "parent_folder",
        type=str,
        help="包含多个方法子文件夹的父目录路径"
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="avg_em",
        choices=["avg_em", "avg_f1", "name"],
        help="排序依据：avg_em（默认）、avg_f1、name"
    )
    parser.add_argument(
        "--sort-order",
        type=str,
        default="desc",
        choices=["desc", "asc"],
        help="排序顺序：desc（降序，默认）、asc（升序）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出文件路径（默认打印到终端）"
    )

    args = parser.parse_args()

    # 检查父目录
    parent_path = Path(args.parent_folder)
    if not parent_path.exists() or not parent_path.is_dir():
        print(f"[Error] Invalid path: {args.parent_folder}")
        return

    # 发现方法
    methods = discover_methods(args.parent_folder)
    if not methods:
        print(f"[Error] No method folders (prompt_*) found in {args.parent_folder}")
        return

    print(f"Found {len(methods)} method folders: {', '.join(methods)}\n")

    # 加载所有方法的结果
    print("Loading evaluation results...")
    all_results = load_all_methods_results(args.parent_folder)

    # 获取所有数据集
    datasets = get_all_datasets(all_results)
    print(f"Found {len(datasets)} datasets: {', '.join(datasets)}\n")

    if not all_results:
        print("[Error] No evaluation results loaded!")
        return

    # 生成对比表格
    table = build_comparison_table(
        all_results,
        datasets,
        sort_by=args.sort_by,
        sort_order=args.sort_order
    )

    # 输出
    if args.output:
        with open(args.output, 'w') as f:
            f.write(table)
        print(f"Comparison table saved to: {args.output}\n")
    else:
        print("\n" + table + "\n")


if __name__ == "__main__":
    main()
