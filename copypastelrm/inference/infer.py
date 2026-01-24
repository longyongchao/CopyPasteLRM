"""
推理脚本（支持断点续传 & 原子写入）

修改日志：
1. 移除文件名时间戳，固定文件路径以便续传。
2. 新增断点续传功能：自动检测是否存在结果文件，加载已有结果并跳过已推理样本。
3. 新增 --overwrite 参数：强制从头开始。
4. 保持原子写入和动态保存策略。
"""

import argparse
import json
import os
import re
import threading
import time
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Set, Tuple
import random
from datetime import datetime

from tqdm import tqdm

from copypastelrm.prompt.prompt import create_prompt
from copypastelrm.datasets import load, AvailableDataset
from copypastelrm.utils.git import get_git_commit_id
from copypastelrm.utils.llm_server import LLMServer


def sanitize_question_for_filename(question: str, max_length: int = 100) -> str:
    """
    Sanitize a question string for safe use in filenames.

    Args:
        question: The question text to sanitize
        max_length: Maximum length of the sanitized question (default 100)

    Returns:
        Sanitized question string safe for filenames
    """
    # Replace spaces with underscores
    sanitized = question.replace(" ", "_")

    # Remove or replace special characters that are unsafe for filenames
    # Unsafe chars: / \\ : * ? < > | " ' and control characters
    sanitized = re.sub(r'[\\/:*?"<>|\'\x00-\x1f]', '', sanitized)

    # Replace multiple consecutive underscores with single underscore
    sanitized = re.sub(r'_+', '_', sanitized)

    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')

    # Limit length (leaving room for sample_id prefix and .json extension)
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].strip('_')

    # Fallback if empty after sanitization
    if not sanitized:
        sanitized = "question"

    return sanitized


def sanitize_id_for_filename(sample_id: str) -> str:
    """
    Sanitize a sample ID string for safe use in filenames.

    This is needed because some datasets (like MultiRC) have sample IDs
    containing slashes or other special characters that are unsafe for filenames.

    Args:
        sample_id: The sample ID to sanitize

    Returns:
        Sanitized sample ID string safe for filenames
    """
    # Replace slashes and other unsafe characters with underscores
    # Unsafe chars: / \ : * ? < > | " ' and control characters
    sanitized = re.sub(r'[\\/:*?"<>|\'\x00-\x1f]', '_', sample_id)

    # Replace multiple consecutive underscores with single underscore
    sanitized = re.sub(r'_+', '_', sanitized)

    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')

    # Fallback if empty after sanitization
    if not sanitized:
        sanitized = "sample_id"

    return sanitized


# --- 辅助类：单样本保存器 (新版) ---
class SampleSaver:
    """
    Saves each sample as an individual file with format: {sample_id}-{sanitized_question}.json
    Also maintains a metadata.json file with experiment info and progress tracking.
    """

    def __init__(
        self,
        output_dir: str,
        info: Dict[str, Any],
        total_samples: int,
    ):
        """
        Initialize the SampleSaver.

        Args:
            output_dir: Directory to save sample files
            info: Experiment metadata dictionary
            total_samples: Total number of samples in the dataset
        """
        self.output_dir = output_dir
        self.info = info
        self.total_samples = total_samples
        self.completed_samples = 0
        self.failed_samples = 0
        self.lock = threading.Lock()

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize or load metadata
        self.metadata_file = os.path.join(self.output_dir, "metadata.json")
        self._load_metadata()

    def _load_metadata(self):
        """Load existing metadata or create new one."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    self.completed_samples = metadata.get("completed_samples", 0)
                    self.failed_samples = metadata.get("failed_samples", 0)
            except Exception as e:
                print(f"[Warning] Failed to load metadata: {e}")

    def get_completed_ids(self) -> Set[str]:
        """
        Scan the output directory for existing sample files and return set of completed sample IDs.

        Returns:
            Set of sample IDs that have been completed
        """
        completed_ids = set()
        try:
            for filename in os.listdir(self.output_dir):
                if filename.endswith(".json") and filename != "metadata.json":
                    # Parse sample_id from filename: {sample_id}-{question}.json
                    parts = filename[:-5].split("-", 1)  # Remove .json and split on first -
                    if parts:
                        sample_id = parts[0]
                        completed_ids.add(sample_id)
        except Exception as e:
            print(f"[Warning] Failed to scan completed IDs: {e}")

        return completed_ids

    def save_sample(
        self,
        sample_id: str,
        query: str,
        predict: str,
        context: str = None,
        answers: List[str] = None,
    ):
        """
        Save a single sample as an individual file.

        Args:
            sample_id: The sample ID
            query: The original question/query
            predict: The prediction result
            context: Optional context
            answers: Optional ground truth answers
        """
        with self.lock:
            try:
                # Generate filename
                sanitized_id = sanitize_id_for_filename(sample_id)
                sanitized_question = sanitize_question_for_filename(query)
                filename = f"{sanitized_id}-{sanitized_question}.json"
                filepath = os.path.join(self.output_dir, filename)

                # Prepare sample data
                sample_data = {
                    "id": sample_id,
                    "query": query,
                    "predict": predict,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }

                if context is not None:
                    sample_data["context"] = context
                if answers is not None:
                    sample_data["answers"] = answers

                # Atomic write using temp file
                temp_file = f"{filepath}.tmp"
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(sample_data, f, ensure_ascii=False, indent=2)
                shutil.move(temp_file, filepath)

                # Update counters
                if predict is None or str(predict).startswith("ERROR:"):
                    self.failed_samples += 1
                else:
                    self.completed_samples += 1

            except Exception as e:
                print(f"[Warning] Failed to save sample {sample_id}: {e}")
                self.failed_samples += 1

    def save_metadata(self, end_time: str = None):
        """
        Save the metadata.json file with experiment info and progress.

        Args:
            end_time: Optional end time string
        """
        with self.lock:
            try:
                metadata = {
                    "info": self.info,
                    "total_samples": self.total_samples,
                    "completed_samples": self.completed_samples,
                    "failed_samples": self.failed_samples,
                }

                if end_time:
                    metadata["info"]["end_time"] = end_time

                # Atomic write
                temp_file = f"{self.metadata_file}.tmp"
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                shutil.move(temp_file, self.metadata_file)

            except Exception as e:
                print(f"[Warning] Failed to save metadata: {e}")


# --- 辅助类：安全保存器 (旧版本，保留兼容性) ---
class SafeSaver:
    def __init__(
        self,
        output_file: str,
        save_interval_seconds: int = 60,
        save_interval_samples: int = 10,
    ):
        """
        安全保存器，支持基于时间间隔和样本数量的双重保存策略

        Args:
            output_file: 输出文件路径
            save_interval_seconds: 时间间隔（秒），默认60秒
            save_interval_samples: 样本间隔，默认每10个样本保存一次
        """
        self.output_file = output_file
        self.save_interval_seconds = save_interval_seconds
        self.save_interval_samples = save_interval_samples
        self.last_save_time = time.time()
        self.samples_since_last_save = 0
        self.lock = threading.Lock()

    def save(self, data: Dict[str, Any], force: bool = False, increment_sample_count: bool = True):
        """
        保存数据，支持原子写入和时间/样本数量双重间隔控制

        Args:
            data: 要保存的数据
            force: 是否强制保存
            increment_sample_count: 是否增加样本计数（默认True）
        """
        current_time = time.time()
        should_save_by_time = (current_time - self.last_save_time >= self.save_interval_seconds)
        should_save_by_samples = (self.samples_since_last_save >= self.save_interval_samples)

        if not force and not should_save_by_time and not should_save_by_samples:
            if increment_sample_count:
                self.samples_since_last_save += 1
            return

        with self.lock:
            temp_file = f"{self.output_file}.tmp"
            try:
                os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                shutil.move(temp_file, self.output_file)
                self.last_save_time = current_time
                if should_save_by_samples or force:
                    self.samples_since_last_save = 0
            except Exception as e:
                print(f"\n[Warning] 自动保存失败: {e}")

    def reset_sample_count(self):
        """重置样本计数（用于批量处理时）"""
        with self.lock:
            self.samples_since_last_save = 0


def process_single_sample(
    sample: Dict[str, Any],
    llm_server: LLMServer,
    model_name: str,
    prompt_type: str,
    temperature: float,
    enable_thinking: bool = False,
) -> Tuple[str, str]:
    """处理单个样本的推理任务"""
    sample_id = sample["id"]

    try:
        system_prompt, user_prompt = create_prompt(
            sample["query"], sample["context"], prompt_type
        )

        predict = llm_server.call(
            model_name=model_name,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            enable_thinking=enable_thinking,
            max_retries=3
        )

        if not predict:
            return sample_id, None

        return sample_id, predict

    except Exception as e:
        return sample_id, f"ERROR: {str(e)}"


def process_batch(
    samples: list,
    llm_server: LLMServer,
    model_name: str,
    prompt_type: str,
    temperature: float,
    enable_thinking: bool = False,
) -> Dict[str, str]:
    """
    批量处理多个样本的推理任务

    Args:
        samples: 样本列表
        llm_server: LLM服务器实例
        model_name: 模型名称
        prompt_type: 提示类型
        temperature: 温度参数
        enable_thinking: 是否启用思考模式

    Returns:
        Dict[str, str]: 样本ID到预测结果的映射
    """
    batch_results = {}
    user_prompts = []
    system_prompts = []
    sample_ids = []

    # 构建批量请求
    for sample in samples:
        sample_id = sample["id"]
        sample_ids.append(sample_id)

        try:
            system_prompt, user_prompt = create_prompt(
                sample["query"], sample["context"], prompt_type
            )
            user_prompts.append(user_prompt)
            system_prompts.append(system_prompt)
        except Exception as e:
            batch_results[sample_id] = f"ERROR: {str(e)}"

    # 调用批量推理
    try:
        predictions = llm_server.batch_call(
            model_name=model_name,
            user_prompts=user_prompts,
            system_prompts=system_prompts,
            temperature=temperature,
            enable_thinking=enable_thinking,
            max_retries=3
        )

        # 将结果映射回样本ID
        for i, sample_id in enumerate(sample_ids):
            if sample_id not in batch_results:  # 如果没有错误
                if predictions and i < len(predictions):
                    batch_results[sample_id] = predictions[i]
                else:
                    batch_results[sample_id] = None

    except Exception as e:
        # 批量调用失败，全部标记为错误
        for sample_id in sample_ids:
            if sample_id not in batch_results:
                batch_results[sample_id] = f"ERROR: {str(e)}"

    return batch_results


def run_inference(
    server_url: str,
    model_name: str,
    output_dir: str,
    max_samples: int = -1,
    num_threads: int = 4,
    batch_size: int = 1,
    prompt_type: str = "cot",
    dataset_name: str = "hotpotqa",
    api_key: str = "sk-wingchiu",
    temperature: float = 0.7,
    enable_thinking: bool = False,
    split: str = "test",
    distractor_docs: int = 0,
    unanswerable: bool = False,
    reload: bool = False,
    overwrite: bool = False,
    save_interval_samples: int = 10,
):
    # 1. 初始化 LLMServer
    llm_server = LLMServer(base_url=server_url, api_key=api_key)

    # 2. 加载数据集
    print(f"正在加载数据集: {dataset_name} (split={split})...")
    try:
        dataset_enum = AvailableDataset(dataset_name)
        dataset_loader = load(
            name=dataset_enum,
            split=split,
            max_samples=max_samples,
            distractor_docs=distractor_docs,
            unanswerable=unanswerable,
            reload=reload,
        )
    except Exception as e:
        raise ValueError(f"数据集加载失败: {e}")

    dataset_dict = dataset_loader.dataset
    # 转换为列表，方便后续处理
    full_dataset = list(dataset_dict.values())
    print(f"原始数据集共 {len(full_dataset)} 个样本")

    # 3. 【核心修改】断点续传逻辑 - 使用 SampleSaver
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # 快照用于记录
    system_prompt_snapshot, user_prompt_snapshot = create_prompt(
        "示例问题", "示例上下文", prompt_type
    )

    # Prepare experiment info
    experiment_info = {
        "server_url": server_url,
        "model_name": model_name,
        "prompt_type": prompt_type,
        "system_prompt_snapshot": system_prompt_snapshot,
        "user_prompt_snapshot": user_prompt_snapshot,
        "start_time": start_time,
        "temperature": temperature,
        "dataset": dataset_name,
        "max_samples": max_samples,
        "num_threads": num_threads,
        "batch_size": batch_size,
        "save_interval_samples": save_interval_samples,
        "output_dir": output_dir,
        "infer_git_commit_id": get_git_commit_id(),
        "噪音文档数量": distractor_docs,
        "是否剔除金标上下文": unanswerable,
        "是否重启了数据集构建": reload,
        "是否覆盖重跑": overwrite,
    }

    # Initialize SampleSaver
    saver = SampleSaver(
        output_dir=output_dir,
        info=experiment_info,
        total_samples=len(full_dataset),
    )

    # Get completed sample IDs for checkpointing
    existing_ids = set()
    if overwrite and os.path.exists(output_dir):
        print(f">>> 用户指定 --overwrite，将清空输出目录: {output_dir}")
        shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        # Reinitialize saver after clearing
        saver = SampleSaver(
            output_dir=output_dir,
            info=experiment_info,
            total_samples=len(full_dataset),
        )
    else:
        existing_ids = saver.get_completed_ids()
        if existing_ids:
            print(f">>> 检测到已完成 {len(existing_ids)} 个样本，将跳过这些样本继续推理。")

    # 4. 过滤掉已经做过的样本
    remaining_dataset = [s for s in full_dataset if sanitize_id_for_filename(s['id']) not in existing_ids]

    if not remaining_dataset:
        print("所有样本均已完成推理！")
        return

    print(f"本次需推理 {len(remaining_dataset)} 个样本，使用 {num_threads} 个线程，批次大小 {batch_size}")

    try:
        # 分批处理数据集
        batches = []
        for i in range(0, len(remaining_dataset), batch_size):
            batch = remaining_dataset[i:i + batch_size]
            batches.append(batch)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # 提交批处理任务
            future_to_batch = {}
            for batch in batches:
                if batch_size == 1:
                    # 单个样本处理模式
                    future_to_batch[executor.submit(
                        process_single_sample,
                        batch[0],
                        llm_server,
                        model_name,
                        prompt_type,
                        temperature,
                        enable_thinking,
                    )] = batch
                else:
                    # 批量处理模式
                    future_to_batch[executor.submit(
                        process_batch,
                        batch,
                        llm_server,
                        model_name,
                        prompt_type,
                        temperature,
                        enable_thinking,
                    )] = batch

            # 进度条总量是本次需要跑的任务量
            with tqdm(total=len(remaining_dataset), desc="推理进度", unit="样本") as pbar:
                for future in as_completed(future_to_batch):
                    batch = future_to_batch[future]

                    try:
                        result = future.result()

                        if batch_size == 1:
                            # 单个样本模式的结果处理
                            result_sample_id, predict = result
                            # Get the original sample from batch
                            sample = batch[0]
                            saver.save_sample(
                                sample_id=result_sample_id,
                                query=sample.get("query", ""),
                                predict=predict,
                                context=sample.get("context"),
                                answers=sample.get("answers"),
                            )
                        else:
                            # 批处理模式的结果处理
                            for sample in batch:
                                sample_id = sample["id"]
                                predict = result.get(sample_id)
                                saver.save_sample(
                                    sample_id=sample_id,
                                    query=sample.get("query", ""),
                                    predict=predict,
                                    context=sample.get("context"),
                                    answers=sample.get("answers"),
                                )

                        # Update metadata after each batch
                        saver.save_metadata()

                    except Exception as e:
                        for sample in batch:
                            sample_id = sample["id"]
                            tqdm.write(f"样本 {sample_id} 异常: {e}")
                            saver.save_sample(
                                sample_id=sample_id,
                                query=sample.get("query", ""),
                                predict=f"ERROR: {e}",
                                context=sample.get("context"),
                                answers=sample.get("answers"),
                            )

                    pbar.update(len(batch))

    except KeyboardInterrupt:
        print("\n\n检测到用户中断 (Ctrl+C)，正在保存当前进度...")
    except Exception as e:
        print(f"\n\n发生严重错误: {e}，正在尝试保存进度...")
    finally:
        # 强制保存最后状态
        end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        saver.save_metadata(end_time=end_time)
        print(f"推理结束，结果已保存到: {output_dir}")
        print(f"已完成 {saver.completed_samples}/{len(full_dataset)} 个样本，失败 {saver.failed_samples} 个")


def main():
    parser = argparse.ArgumentParser(description="HotpotQA 推理脚本（断点续传版）")
    parser.add_argument("--server-url", type=str, default="http://localhost:8124/v1", help="vLLM 服务地址")
    parser.add_argument("--model-name", type=str, required=True, help="模型名称")
    parser.add_argument("--max-samples", type=int, default=-1, help="最大处理样本数")
    parser.add_argument("--num-threads", type=int, default=4, help="并行线程数")
    parser.add_argument("--prompt-type", type=str, default="direct",
                        choices=["direct_inference", "cot", "rag", "rag_rep_2", "rag_rep_q", "rag_qcq", "rag_qcq2", "rag_q_int_q", "rag_q_int2_q", "ircot", "deepseek", "copypaste", "find_facts"],
                        help="提示模板选择")
    parser.add_argument("--dataset", type=str, default=AvailableDataset.HOTPOTQA.value,
                        choices=[e.value for e in AvailableDataset], help="数据集名称")
    parser.add_argument("--api-key", type=str, default="sk-lqztxtcbxxoonlmsxvdhllhdnoegywnvuhfnoqnxvpphrhkh", help="API Key")
    parser.add_argument("--temperature", type=float, default=0.7, help="温度")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--distractor-docs", type=int, default=4, help="噪音文档数")
    parser.add_argument("--unanswerable", action="store_true", help="是否剔除gold context")
    parser.add_argument("--reload", action="store_true", help="是否重构数据集")
    parser.add_argument("--enable-thinking", action="store_true", help="启用思考")
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"], help="数据集划分")

    # 【新增参数】
    parser.add_argument("--overwrite", action="store_true", help="如果输出文件存在，是否强制覆盖（默认：否，即进行断点续传）")
    parser.add_argument("--batch-size", type=int, default=1, help="批量推理时的批次大小（默认：1，即单个样本处理）")
    parser.add_argument("--save-interval-samples", type=int, default=10, help="每隔N个样本保存一次检查点（默认：10）")

    args = parser.parse_args()

    random.seed(args.seed)

    # Generate timestamp folder (YYYYMMDDHHmmss) - will be the last folder
    timestamp_folder = datetime.now().strftime("%Y%m%d%H%M%S")

    # 输出目录（不带 .json 扩展名）
    model_name_clean = args.model_name.replace("/", "_").replace(" ", "_")
    output_dir = f"results/infer/{args.split}/{model_name_clean}/resamples_{args.max_samples}/seed_{args.seed}/tpr_{args.temperature}/prompt_{args.prompt_type.replace(' ', '_')}/{args.dataset}-noise_{args.distractor_docs}-{'unanswerable' if args.unanswerable else 'answerable'}/{timestamp_folder}"

    run_inference(
        server_url=args.server_url,
        model_name=args.model_name,
        output_dir=output_dir,
        max_samples=args.max_samples,
        num_threads=args.num_threads,
        batch_size=args.batch_size,
        prompt_type=args.prompt_type,
        dataset_name=args.dataset,
        api_key=args.api_key,
        temperature=args.temperature,
        enable_thinking=args.enable_thinking,
        split=args.split,
        distractor_docs=args.distractor_docs,
        unanswerable=args.unanswerable,
        reload=args.reload,
        overwrite=args.overwrite,
        save_interval_samples=args.save_interval_samples,
    )

if __name__ == "__main__":
    main()