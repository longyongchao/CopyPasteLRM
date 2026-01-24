#!/usr/bin/env python3
"""
批量评估脚本：对比 prompt_rag 和 prompt_rag_rep_2 的结果

遍历指定目录下的所有结果文件夹，调用 eval_folder.py 进行评估
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path


def find_result_folders(base_path: str):
    """
    查找所有需要评估的结果文件夹

    Args:
        base_path: 基础路径，如 results/infer/test/Qwen2.5-7B-Instruct/resamples_1000

    Returns:
        list: 所有结果文件夹路径
    """
    base_path = Path(base_path)

    if not base_path.exists():
        print(f"[Error] Path not found: {base_path}")
        return []

    # 遍历 prompt_rag 和 prompt_rag_rep_2 目录
    folders = []
    prompt_types = ["prompt_rag", "prompt_rag_rep_2"]

    for prompt_type in prompt_types:
        prompt_dir = base_path / "seed_42" / "tpr_0.0" / prompt_type
        if not prompt_dir.exists():
            print(f"[Warning] Directory not found: {prompt_dir}")
            continue

        # 遍历每个数据集文件夹
        for dataset_dir in sorted(prompt_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue

            # 找到时间戳文件夹（应该只有一个）
            timestamp_dirs = list(dataset_dir.iterdir())
            timestamp_dirs = [d for d in timestamp_dirs if d.is_dir() and d.name != ""]

            if not timestamp_dirs:
                print(f"[Warning] No timestamp folder found in: {dataset_dir}")
                continue

            # 取第一个时间戳文件夹
            result_folder = timestamp_dirs[0]
            folders.append((prompt_type, dataset_dir.name, result_folder))

    return folders


def evaluate_folder(result_folder: Path):
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
    parser = argparse.ArgumentParser(description="批量评估并对比 RAG vs RAG×2")
    parser.add_argument(
        "--base-path",
        type=str,
        default="results/infer/test/Qwen2.5-7B-Instruct/resamples_1000",
        help="基础路径"
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="跳过评估，直接汇总已有结果"
    )
    args = parser.parse_args()

    # 查找所有结果文件夹
    print(f"Searching for result folders in: {args.base_path}")
    folders = find_result_folders(args.base_path)

    if not folders:
        print("[Error] No result folders found!")
        return

    print(f"Found {len(folders)} result folders to evaluate\n")

    # 评估所有文件夹
    if not args.skip_eval:
        for i, (prompt_type, dataset, folder) in enumerate(folders, 1):
            print(f"[{i}/{len(folders)}] Evaluating {prompt_type}/{dataset}...")
            success = evaluate_folder(folder)
            if success:
                print(f"  ✓ Success")
            else:
                print(f"  ✗ Failed")
            print()

    print("\n" + "="*60)
    print("All evaluations completed!")
    print("="*60)
    print("\nNow run: python scripts/eval/summarize_results.py --base-path", args.base_path)


if __name__ == "__main__":
    main()
