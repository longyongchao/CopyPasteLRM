#!/usr/bin/env python3
"""
批量评估脚本：自动检测并评估所有时间戳文件夹

支持任意层级的路径输入，自动检测时间戳文件夹并调用 eval_folder.py 进行评估
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def is_timestamp_folder(dirname: str) -> bool:
    """
    检查是否是时间戳文件夹（14位数字格式，如 20260124152132）

    Args:
        dirname: 目录名

    Returns:
        bool: 是否是时间戳文件夹
    """
    return dirname.isdigit() and len(dirname) == 14


def detect_path_level(base_path: Path) -> dict:
    """
    智能检测输入路径的层级

    Args:
        base_path: 用户输入的路径（Path对象）

    Returns:
        dict: {
            'level': 'root'|'seed_tpr'|'prompt_type'|'dataset',
            'path_parts': 路径组成部分列表
        }
    """
    path_str = str(base_path)
    parts = []

    # 分解路径，识别各层级
    for part in base_path.parts:
        parts.append(part)

    # 检测层级
    # level 1: 如果路径包含 "seed_XX" 和 "tpr_X.X" → seed_tpr 级别
    # level 2: 如果当前目录名是 "prompt_*" → prompt_type 级别
    # level 3: 如果当前目录名包含 "-noise_" → dataset 级别
    # level 0: 其他情况视为 root 级别

    level = "root"

    # 检查是否包含 seed/tpr
    for i, part in enumerate(parts):
        if part.startswith("seed_") and i + 1 < len(parts) and parts[i + 1].startswith("tpr_"):
            level = "seed_tpr"
            break

    # 检查最后一级是否是 prompt_* 目录
    if base_path.name.startswith("prompt_"):
        level = "prompt_type"

    # 检查最后一级是否是数据集目录（包含 "-noise_" 或 "-answerable"）
    elif "-noise_" in base_path.name or "-answerable" in base_path.name:
        level = "dataset"

    return {
        'level': level,
        'path_parts': parts
    }


def find_result_folders(base_path: str, recursive: bool = True) -> List[Tuple[str, str, Path]]:
    """
    查找所有需要评估的时间戳文件夹

    Args:
        base_path: 基础路径（任意层级）
        recursive: 是否递归查找子目录（暂未使用，保留扩展性）

    Returns:
        list: [(prompt_type, dataset_name, timestamp_folder_path), ...]
    """
    base_path = Path(base_path)

    if not base_path.exists():
        print(f"[Error] Path not found: {base_path}")
        return []

    # 检测路径层级
    path_info = detect_path_level(base_path)
    level = path_info['level']

    folders = []

    # 根据不同层级处理
    if level == "prompt_type":
        # 用户指定了 prompt_* 目录，直接查找该目录下的所有数据集
        prompt_type = base_path.name
        print(f"检测到 prompt_type 级别: {prompt_type}")

        # 遍历所有数据集目录
        for dataset_dir in sorted(base_path.iterdir()):
            if not dataset_dir.is_dir():
                continue

            dataset_name = dataset_dir.name

            # 查找所有时间戳文件夹
            for timestamp_dir in dataset_dir.iterdir():
                if timestamp_dir.is_dir() and is_timestamp_folder(timestamp_dir.name):
                    folders.append((prompt_type, dataset_name, timestamp_dir))

    elif level == "seed_tpr":
        # 用户指定了 seed_XX/tpr_X.X 目录，查找所有 prompt_* 目录
        print(f"检测到 seed_tpr 级别")

        # 查找所有 prompt_* 目录
        for prompt_dir in sorted(base_path.iterdir()):
            if not prompt_dir.is_dir() or not prompt_dir.name.startswith("prompt_"):
                continue

            prompt_type = prompt_dir.name

            # 遍历该 prompt_type 下的所有数据集
            for dataset_dir in sorted(prompt_dir.iterdir()):
                if not dataset_dir.is_dir():
                    continue

                dataset_name = dataset_dir.name

                # 查找所有时间戳文件夹
                for timestamp_dir in dataset_dir.iterdir():
                    if timestamp_dir.is_dir() and is_timestamp_folder(timestamp_dir.name):
                        folders.append((prompt_type, dataset_name, timestamp_dir))

    elif level == "dataset":
        # 用户指定了数据集目录，只查找该数据集下的时间戳文件夹
        dataset_name = base_path.name
        # 尝试从路径中提取 prompt_type
        prompt_type = "unknown"
        for part in base_path.parts:
            if part.startswith("prompt_"):
                prompt_type = part
                break

        print(f"检测到 dataset 级别: {dataset_name} (prompt_type: {prompt_type})")

        # 查找所有时间戳文件夹
        for timestamp_dir in base_path.iterdir():
            if timestamp_dir.is_dir() and is_timestamp_folder(timestamp_dir.name):
                folders.append((prompt_type, dataset_name, timestamp_dir))

    else:
        # root 级别：尝试遍历所有子目录
        print(f"检测到 root 级别，尝试查找所有子目录...")

        # 查找所有 prompt_* 目录（可能在任何子层级）
        for prompt_dir in base_path.rglob("prompt_*"):
            if not prompt_dir.is_dir():
                continue

            prompt_type = prompt_dir.name

            # 遍历该 prompt_type 下的所有数据集
            for dataset_dir in sorted(prompt_dir.iterdir()):
                if not dataset_dir.is_dir():
                    continue

                dataset_name = dataset_dir.name

                # 查找所有时间戳文件夹
                for timestamp_dir in dataset_dir.iterdir():
                    if timestamp_dir.is_dir() and is_timestamp_folder(timestamp_dir.name):
                        folders.append((prompt_type, dataset_name, timestamp_dir))

    return folders


def evaluate_folder(result_folder: Path) -> bool:
    """
    调用 eval_folder.py 评估单个文件夹

    Args:
        result_folder: 结果文件夹路径

    Returns:
        bool: 是否成功
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "copypastelrm.eval.eval_folder", str(result_folder)],
            capture_output=True,
            text=True,
            timeout=300  # 5分钟超时
        )
        return result.returncode == 0
    except Exception as e:
        print(f"[Error] Failed to evaluate {result_folder}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="批量评估并对比所有 prompt 类型的结果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 从 prompt_type 目录开始
  python scripts/eval/batch_eval_compare.py \\
      --base-path results/infer/test/Qwen2.5-7B-Instruct/resamples_1000/seed_42/tpr_0.0/prompt_rag_rep_q

  # 从 seed_tpr 目录开始（评估所有 prompt 类型）
  python scripts/eval/batch_eval_compare.py \\
      --base-path results/infer/test/Qwen2.5-7B-Instruct/resamples_1000/seed_42/tpr_0.0

  # 从更上层目录开始
  python scripts/eval/batch_eval_compare.py \\
      --base-path results/infer/test/Qwen2.5-7B-Instruct/resamples_1000
        """
    )
    parser.add_argument(
        "--base-path",
        type=str,
        default="results/infer/test/Qwen2.5-7B-Instruct/resamples_1000",
        help="基础路径（支持任意层级）"
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="跳过评估，直接汇总已有结果"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="递归查找所有时间戳文件夹（默认启用）"
    )
    parser.add_argument(
        "--no-recursive",
        action="store_false",
        dest="recursive",
        help="只查找直接子目录"
    )
    args = parser.parse_args()

    # 查找所有结果文件夹
    print("=" * 70)
    print(f"扫描路径: {args.base_path}")
    print("=" * 70)

    folders = find_result_folders(args.base_path, args.recursive)

    if not folders:
        print("[Error] No result folders found!")
        print("\n提示：请检查路径是否正确，或使用以下格式：")
        print("  - .../prompt_* (推荐)")
        print("  - .../seed_XX/tpr_X.X/")
        print("  - .../Dataset-noise_X-answerable/")
        return

    # 按数据集名称排序
    folders.sort(key=lambda x: (x[0], x[1]))

    print(f"\n找到 {len(folders)} 个时间戳文件夹:\n")

    # 显示找到的所有文件夹（按 prompt_type 分组）
    current_prompt = None
    for prompt_type, dataset, folder in folders:
        if prompt_type != current_prompt:
            current_prompt = prompt_type
            print(f"\n[{prompt_type}]")
        print(f"  - {dataset}/{folder.name}")

    print("\n" + "=" * 70)

    # 评估所有文件夹
    if not args.skip_eval:
        print(f"\n开始评估 {len(folders)} 个文件夹...\n")

        success_count = 0
        fail_count = 0

        for i, (prompt_type, dataset, folder) in enumerate(folders, 1):
            print(f"[{i}/{len(folders)}] 评估 {prompt_type}/{dataset}/{folder.name}...")

            success = evaluate_folder(folder)
            if success:
                print(f"  ✓ Success")
                success_count += 1
            else:
                print(f"  ✗ Failed")
                fail_count += 1
            print()

        print("=" * 70)
        print(f"评估完成！成功: {success_count}, 失败: {fail_count}")
        print("=" * 70)

    print("\n提示：运行以下命令查看汇总结果：")
    print(f"python scripts/eval/summarize_results.py --base-path {args.base_path}")


if __name__ == "__main__":
    main()
