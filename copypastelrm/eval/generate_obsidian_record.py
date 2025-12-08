#!/usr/bin/env python3
"""
生成 Obsidian 格式的 Markdown 报告

该脚本从评估结果中提取信息，生成包含以下内容的 Obsidian Markdown 文件：
1. YAML properties（基于 obsidian_card）
2. 从不同类别中随机抽取的样本展示
"""

import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from copypastelrm.metrics.utils import extract_answer_and_facts


def load_all_done_results(results_dir: str = "results/eval") -> List[Dict[str, Any]]:
    """
    加载所有已完成的评估结果文件

    Args:
        results_dir: 结果目录路径

    Returns:
        所有已完成的评估结果列表
    """
    done_results = []

    for root, dirs, files in os.walk(results_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    result = json.load(f)
                    result["file_path"] = file_path
                    done_results.append(result)
            except Exception as e:
                print(f"警告：无法加载文件 {file_path}: {e}")

    return done_results


def generate_obsidian_markdown_for_single(result: Dict[str, Any]):
    """
    为单个评估结果生成 Obsidian 格式的 Markdown 文件

    Args:
        result: 单个评估结果
    """
    if not result:
        print("评估结果为空")
        return

    # 获取原始 JSON 文件路径
    json_file_path = result.get("file_path", "")
    if not json_file_path:
        print("无法找到原始 JSON 文件路径")
        return

    # 生成对应的 MD 文件路径
    md_file_path = json_file_path.replace(".json", ".md").replace(
        "results/eval", "results/markdown"
    )
    os.makedirs(os.path.dirname(md_file_path), exist_ok=True)

    print(f"为 {json_file_path} 生成对应的 Markdown 文件: {md_file_path}")

    # 生成 YAML properties
    yaml_props = "---\n"

    # 将result直接保持json的样式附加到yaml_props
    yaml_props += f"{json.dumps(result, ensure_ascii=False, indent=2)}\n"

    yaml_props += "---\n\n"

    # 开始构建 Markdown 内容
    markdown_content = yaml_props

    try:
        with open(md_file_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        print(f"📝 Obsidian Markdown 报告已生成: {md_file_path}")
    except Exception as e:
        print(f"错误：无法写入文件 {md_file_path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="生成 Obsidian 格式的 Markdown 评估报告"
    )
    parser.add_argument("--results-dir", default="results", help="结果目录路径")
    parser.add_argument(
        "--output",
        default="evaluation_report.md",
        help="输出文件名（已弃用，每个JSON文件生成对应的MD文件）",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()

    # 设置随机种子以确保可重现性
    random.seed(args.seed)

    # 加载所有已完成的结果
    done_results = load_all_done_results(args.results_dir)

    if not done_results:
        print("没有找到已完成的评估结果文件")
        return

    print(f"找到 {len(done_results)} 个已完成的评估结果")

    # 为每个结果生成对应的 Markdown 文件
    for result in done_results:
        generate_obsidian_markdown_for_single(result)

    print(f"已为所有 {len(done_results)} 个评估结果生成对应的 Markdown 文件")


if __name__ == "__main__":
    main()
