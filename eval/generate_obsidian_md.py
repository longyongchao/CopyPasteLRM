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

from eval.utils import extract_answer_and_facts


def load_all_done_results(results_dir: str = "results") -> List[Dict[str, Any]]:
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
            if file.endswith("_done.json"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        result = json.load(f)
                        result["file_path"] = file_path
                        done_results.append(result)
                except Exception as e:
                    print(f"警告：无法加载文件 {file_path}: {e}")

    return done_results


def categorize_samples(data: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    将样本按照不同条件分类

    Args:
        data: 原始数据字典

    Returns:
        分类后的样本ID字典
    """
    without_answer_ids = set()
    without_facts_ids = set()

    for id, item in data.items():
        predicted_answer, predicted_facts = extract_answer_and_facts(item["predict"])

        if predicted_answer is None:
            without_answer_ids.add(id)

        if len(predicted_facts) == 0:
            without_facts_ids.add(id)

    categories = {"without_answer_only": list(without_answer_ids - without_facts_ids), "without_facts_only": list(without_facts_ids - without_answer_ids), "without_answer_and_facts": list(without_answer_ids & without_facts_ids), "with_answer_and_facts": []}

    # 计算有答案和事实的样本
    all_ids = set(data.keys())
    problematic_ids = without_answer_ids | without_facts_ids
    categories["with_answer_and_facts"] = list(all_ids - problematic_ids)

    return categories


def format_sample(sample_id: str, sample_data: Dict[str, Any], dataset: Dict[str, Any] = None) -> str:
    """
    格式化单个样本为 Markdown

    Args:
        sample_id: 样本ID
        sample_data: 样本数据
        dataset: 完整数据集（用于获取sfs）

    Returns:
        格式化后的 Markdown 字符串
    """
    predicted_answer, predicted_facts = extract_answer_and_facts(sample_data["predict"])

    markdown = f"#### 样本 {sample_id}\n\n"

    # Query
    markdown += f"##### 问题\n\n{sample_data['query']}\n\n"

    # Context (截取前500字符以保持简洁)
    context = sample_data["context"]
    markdown += f"##### 上下文\n\n{context}\n\n"

    # Answer
    markdown += f"##### 标准答案\n\n{sample_data['answer']}\n\n"

    # Supporting Facts (如果有)
    if dataset and sample_id in dataset and "sfs" in dataset[sample_id]:
        sfs = dataset[sample_id]["sfs"]
        markdown += "##### 支持事实\n\n"
        for i, sf in enumerate(sfs, 1):
            markdown += f"{i}. {sf}\n"
        markdown += "\n"

    # Predict
    markdown += "##### 模型预测\n\n"
    predict_text = sample_data.get("predict", "")
    markdown += f"```\n{predict_text}\n```\n\n"

    # 提取的答案和事实
    markdown += "##### 提取的信息\n\n"
    markdown += f"- **提取的答案**: {predicted_answer if predicted_answer else '未提取到答案'}\n"
    markdown += f"- **提取的事实数量**: {len(predicted_facts)}\n"
    if predicted_facts:
        markdown += "- **提取的事实**:\n"
        for i, fact in enumerate(predicted_facts, 1):
            markdown += f"  {i}. {fact}\n"
    markdown += "\n"

    markdown += "---\n\n"

    return markdown


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
    md_file_path = json_file_path.replace("_done.json", ".md")

    print(f"为 {json_file_path} 生成对应的 Markdown 文件: {md_file_path}")

    # 生成 YAML properties
    yaml_props = "---\n"

    # 将result直接保持json的样式附加到yaml_props
    yaml_props += f"{json.dumps(result, ensure_ascii=False, indent=2)}\n"

    yaml_props += "---\n\n"

    # 开始构建 Markdown 内容
    markdown_content = yaml_props

    # 添加标题和概述
    markdown_content += "# 样本抽样\n\n"

    # 加载原始数据以获取样本
    original_file = result.get("output file", "")
    if not original_file:
        print("警告：无法找到原始数据文件路径")
        return

    try:
        with open(original_file, "r", encoding="utf-8") as f:
            original_data = json.load(f)
    except Exception as e:
        print(f"错误：无法加载原始数据文件 {original_file}: {e}")
        return

    # 分类样本
    categories = categorize_samples(original_data.get("data", {}))

    # 加载数据集以获取支持事实
    dataset_name = result.get("dataset", "")
    dataset = None
    if dataset_name:
        try:
            from load_datasets.load import data_loader

            dataset = data_loader(dataset_name, mode="dict")
        except Exception as e:
            print(f"警告：无法加载数据集 {dataset_name}: {e}")

    # 添加样本展示
    markdown_content += "## 样本展示\n\n"

    category_names = {"without_answer_only": "无答案但有事实", "without_facts_only": "有答案但无事实", "without_answer_and_facts": "无答案且无事实", "with_answer_and_facts": "有答案且有事实"}

    for category_key, category_name in category_names.items():
        sample_ids = categories[category_key]

        if sample_ids:
            markdown_content += f"### {category_name} (随机抽取3个)\n\n"

            # 随机选择3个样本
            selected_samples = random.sample(sample_ids, min(3, len(sample_ids)))

            for sample_id in selected_samples:
                if sample_id in original_data.get("data", {}):
                    sample_data = original_data["data"][sample_id]
                    markdown_content += format_sample(sample_id, sample_data, dataset)

    # 写入文件
    try:
        with open(md_file_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        print(f"Obsidian Markdown 报告已生成: {md_file_path}")
    except Exception as e:
        print(f"错误：无法写入文件 {md_file_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="生成 Obsidian 格式的 Markdown 评估报告")
    parser.add_argument("--results-dir", default="results", help="结果目录路径")
    parser.add_argument("--output", default="evaluation_report.md", help="输出文件名（已弃用，每个JSON文件生成对应的MD文件）")
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
