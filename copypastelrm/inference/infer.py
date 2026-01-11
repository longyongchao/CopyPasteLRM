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
import threading
import time
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Tuple
import random

from tqdm import tqdm

from copypastelrm.prompt.prompt import create_prompt
from copypastelrm.datasets import load, AvailableDataset
from copypastelrm.utils.git import get_git_commit_id
from copypastelrm.utils.llm_server import LLMServer


# --- 辅助类：安全保存器 (保持不变) ---
class SafeSaver:
    def __init__(self, output_file: str, save_interval_seconds: int = 60):
        self.output_file = output_file
        self.save_interval_seconds = save_interval_seconds
        self.last_save_time = time.time()
        self.lock = threading.Lock()
        
    def save(self, data: Dict[str, Any], force: bool = False):
        """保存数据，支持原子写入和时间间隔控制"""
        current_time = time.time()
        if not force and (current_time - self.last_save_time < self.save_interval_seconds):
            return

        with self.lock:
            temp_file = f"{self.output_file}.tmp"
            try:
                os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                shutil.move(temp_file, self.output_file)
                self.last_save_time = current_time
            except Exception as e:
                print(f"\n[Warning] 自动保存失败: {e}")


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


def run_inference(
    server_url: str,
    model_name: str,
    output_file: str,
    max_samples: int = -1,
    num_threads: int = 4,
    prompt_type: str = "cot",
    dataset_name: str = "hotpotqa",
    api_key: str = "sk-wingchiu",
    temperature: float = 0.7,
    enable_thinking: bool = False,
    split: str = "test",
    distractor_docs: int = 0,
    unanswerable: bool = False,
    reload: bool = False,
    overwrite: bool = False, # 【新增参数】
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

    # 3. 【核心修改】断点续传逻辑
    results = {}
    existing_ids = set()
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    if os.path.exists(output_file) and not overwrite:
        print(f"检测到已存在的结果文件: {output_file}")
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 兼容不同格式，有些可能直接存的dict，有些存的 {"data": ...}
                if "data" in data:
                    results = data["data"]
                    # 如果有 info，可以考虑保留原始 start_time，这里简单处理，使用当前时间更新
                else:
                    # 兼容旧格式（如果不小心直接存了results）
                    results = data 
            
            # 【新增】只将 predict 不为 null 的样本加入 existing_ids，predict 为 null 的需要重新推理
            existing_ids = set()
            null_predict_ids = []
            for sample_id, sample_result in results.items():
                predict = sample_result.get("predict")
                if predict is not None and not str(predict).startswith("ERROR:"):
                    existing_ids.add(sample_id)
                else:
                    null_predict_ids.append(sample_id)
            
            if null_predict_ids:
                print(f">>> 检测到 {len(null_predict_ids)} 个样本的 predict 为 null 或包含错误，将重新推理这些样本。")
                if len(null_predict_ids) <= 5:
                    print(f">>> 样本ID列表: {', '.join(null_predict_ids)}")
            
            print(f">>> 成功加载断点，已完成 {len(existing_ids)} 个有效样本，将继续推理剩余部分。")
        except json.JSONDecodeError:
            print(">>> [警告] 现有文件损坏或为空，将从头开始。")
        except Exception as e:
            print(f">>> [警告] 读取断点文件失败: {e}，将从头开始。")
    elif overwrite and os.path.exists(output_file):
        print(f">>> 用户指定 --overwrite，将覆盖现有文件: {output_file}")
    
    # 4. 过滤掉已经做过的样本
    remaining_dataset = [s for s in full_dataset if s['id'] not in existing_ids]

    if not remaining_dataset:
        print("所有样本均已完成推理！")
        return

    print(f"本次需推理 {len(remaining_dataset)} 个样本，使用 {num_threads} 个线程")

    # 快照用于记录
    system_prompt_snapshot, user_prompt_snapshot = create_prompt(
        "示例问题", "示例上下文", prompt_type
    )

    # 初始化安全保存器 (每 30 秒保存一次)
    saver = SafeSaver(output_file, save_interval_seconds=30)

    # 闭包函数：获取包含（旧结果 + 新结果）的完整数据
    def get_full_results():
        end_time_dynamic = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        experiment_info = {
            "server_url": server_url,
            "model_name": model_name,
            "prompt_type": prompt_type,
            "system_prompt_snapshot": system_prompt_snapshot,
            "user_prompt_snapshot": user_prompt_snapshot,
            "start_time": start_time,
            "end_time": end_time_dynamic,
            "temperature": temperature,
            "dataset": dataset_name,
            "max_samples": max_samples,
            "num_threads": num_threads,
            "output_file": output_file,
            "infer_git_commit_id": get_git_commit_id(),
            "噪音文档数量": distractor_docs,
            "是否剔除金标上下文": unanswerable,
            "是否重启了数据集构建": reload,
            "是否覆盖重跑": overwrite
        }
        return {"info": experiment_info, "data": results}

    try:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # 只提交剩余的任务
            future_to_sample = {
                executor.submit(
                    process_single_sample,
                    sample,
                    llm_server,
                    model_name,
                    prompt_type,
                    temperature,
                    enable_thinking,
                ): sample
                for sample in remaining_dataset
            }

            # 进度条总量是本次需要跑的任务量
            with tqdm(total=len(remaining_dataset), desc="推理进度", unit="样本") as pbar:
                for future in as_completed(future_to_sample):
                    sample = future_to_sample[future]
                    sample_id = sample["id"]
                    
                    try:
                        result_sample_id, predict = future.result()
                        
                        # 更新内存结果 (results 包含了旧数据 + 新数据)
                        results[result_sample_id] = {"predict": predict}
                        
                        # 尝试保存
                        saver.save(get_full_results())
                        
                    except Exception as e:
                        tqdm.write(f"样本 {sample_id} 异常: {e}")
                        results[sample_id] = {"predict": f"Error: {e}"}
                    
                    pbar.update(1)

    except KeyboardInterrupt:
        print("\n\n检测到用户中断 (Ctrl+C)，正在保存当前进度...")
    except Exception as e:
        print(f"\n\n发生严重错误: {e}，正在尝试保存进度...")
    finally:
        # 强制保存最后状态
        saver.save(get_full_results(), force=True)
        print(f"推理结束，结果已保存到: {output_file}")
        print(f"当前文件共包含 {len(results)}/{len(full_dataset)} 个样本的结果")


def main():
    parser = argparse.ArgumentParser(description="HotpotQA 推理脚本（断点续传版）")
    parser.add_argument("--server-url", type=str, default="http://localhost:8124/v1", help="vLLM 服务地址")
    parser.add_argument("--model-name", type=str, required=True, help="模型名称")
    parser.add_argument("--max-samples", type=int, default=-1, help="最大处理样本数")
    parser.add_argument("--num-threads", type=int, default=4, help="并行线程数")
    parser.add_argument("--prompt-type", type=str, default="direct", 
                        choices=["direct_inference", "cot", "rag", "ircot", "deepseek", "copypaste", "find_facts"], 
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

    args = parser.parse_args()

    random.seed(args.seed)
    
    # 移除了时间戳，确保文件名确定性
    model_name_clean = args.model_name.replace("/", "_").replace(" ", "_")
    output_file = f"results/infer/{args.split}/{model_name_clean}/resamples_{args.max_samples}/seed_{args.seed}/tpr_{args.temperature}/prompt_{args.prompt_type.replace(' ', '_')}/{args.dataset}-noise_{args.distractor_docs}-{'unanswerable' if args.unanswerable else 'answerable'}.json"
    
    run_inference(
        server_url=args.server_url,
        model_name=args.model_name,
        output_file=output_file,
        max_samples=args.max_samples,
        num_threads=args.num_threads,
        prompt_type=args.prompt_type,
        dataset_name=args.dataset,
        api_key=args.api_key,
        temperature=args.temperature,
        enable_thinking=args.enable_thinking,
        split=args.split,
        distractor_docs=args.distractor_docs,
        unanswerable=args.unanswerable,
        reload=args.reload,
        overwrite=args.overwrite, # 传入新参数
    )

if __name__ == "__main__":
    main()