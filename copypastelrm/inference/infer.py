"""
HotpotQA 推理脚本（多线程版本）

该脚本用于在 HotpotQA 数据集上进行模型推理，并生成符合评估格式的结果文件。

功能：
1. 从 HuggingFace 加载 HotpotQA 数据集的 distractor 子集 validation split
2. 使用多线程并行调用 vLLM 部署的模型进行推理，显著提升推理速度
3. 解析模型输出，提取答案和支持事实
4. 保存为 eval/hotpotqa.py 所需的格式
5. 支持中间结果保存，防止意外中断导致数据丢失

使用方法：
python inference/infer.py --server-url https://api.siliconflow.cn/v1 --model-name Qwen/Qwen2.5-14B-Instruct  --num-threads 2 --prompt-type direct --max-samples 10

新增参数：
--num-threads: 并行推理的线程数量，默认为 4
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Tuple
import random

from openai import OpenAI
from tqdm import tqdm

from copypastelrm.inference.prompt import create_prompt
from copypastelrm.datasets.HotpotQA import HotpotQA
from copypastelrm.datasets.PubMedQA import PubMedQA
from copypastelrm.datasets.MultiRC import MultiRC
from copypastelrm.datasets.MuSiQue import MuSiQue
from copypastelrm.datasets.TwoWikiMultiHopQA import TwoWikiMultihopQA
from copypastelrm.datasets.PopQA import PopQA
from copypastelrm.datasets.FaithEval import FaithEval
from copypastelrm.datasets.Qasper import Qasper
from copypastelrm.datasets.CopyPaste import CopyPaste


from copypastelrm.utils.git import get_git_commit_id


def call_model(
    client: OpenAI,
    model_name: str,
    prompt: str,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    top_p=0.95,
    enable_thinking: bool = False,
) -> str:
    """
    调用模型进行推理

    Args:
        client: OpenAI 客户端
        model_name: 模型名称
        prompt: 输入提示
        max_tokens: 最大生成 token 数

    Returns:
        str: 模型生成的回答
    """
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": enable_thinking},
            },
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"模型调用失败: {e}")
        return ""


def process_single_sample(
    sample: Dict[str, Any],
    client: OpenAI,
    model_name: str,
    prompt_type: str,
    temperature: float,
    top_p: float,
    enable_thinking: bool = False,
) -> Tuple[str, str]:
    """
    处理单个样本的推理任务

    Args:
        sample: 数据样本
        client: OpenAI 客户端
        model_name: 模型名称

    Returns:
        Tuple[str, str]: (样本ID, 预测响应)
    """
    sample_id = sample["id"]

    try:
        # 格式化上下文和创建提示
        prompt = create_prompt(sample["query"], sample["context"], prompt_type)

        # 调用模型
        predict = call_model(
            client,
            model_name,
            prompt,
            temperature=temperature,
            top_p=top_p,
            enable_thinking=enable_thinking,
        )

        if not predict:
            return sample_id, None

        return sample_id, predict

    except Exception as e:
        print(f"处理样本 {sample_id} 时发生错误: {e}")
        return sample_id, e


def save_intermediate_results(
    results: Dict[str, Any], output_file: str, processed_count: int, total_count: int
):
    """
    保存中间结果

    Args:
        results: 当前结果字典
        output_file: 输出文件路径
        processed_count: 已处理样本数
        total_count: 总样本数
    """
    # 创建中间结果文件名
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def run_inference(
    server_url: str,
    model_name: str,
    output_file: str,
    max_samples: int = None,
    num_threads: int = 4,
    prompt_type: str = "reasoning with copy-paste",
    dataset_name: str = "hotpotqa",
    api_key: str = "sk-wingchiu",
    temperature: float = 0.7,
    top_p: float = 0.95,
    enable_thinking: bool = False,
):
    """
    运行完整的推理流程（多线程版本）

    Args:
        model_url: vLLM 服务地址
        model_name: 模型名称
        output_file: 输出文件路径
        max_samples: 最大处理样本数，None 表示处理全部
        num_threads: 线程数量，默认为 4
    """
    # 初始化 OpenAI 客户端
    client = OpenAI(base_url=server_url, api_key=api_key)

    # 加载数据集
    if dataset_name == "hotpotqa":
        dataset_loader = HotpotQA(max_samples=max_samples)
    elif dataset_name == "multirc":
        dataset_loader = MultiRC(max_samples=max_samples)
    elif dataset_name == "pubmedqa":
        dataset_loader = PubMedQA(max_samples=max_samples)
    elif dataset_name == "musique":
        dataset_loader = MuSiQue(max_samples=max_samples)
    elif dataset_name == "2wikimultihopqa":
        dataset_loader = TwoWikiMultihopQA(max_samples=max_samples)
    elif dataset_name == "popqa":
        dataset_loader = PopQA(max_samples=max_samples)
    elif dataset_name == "faitheval":
        dataset_loader = FaithEval(max_samples=max_samples)
    elif dataset_name == "qasper":
        dataset_loader = Qasper(max_samples=max_samples)
    elif dataset_name == "copypaste":
        dataset_loader = CopyPaste(max_samples=max_samples)
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

    dataset_dict = dataset_loader.dataset
    dataset = []
    for id, sample in dataset_dict.items():
        dataset.append(sample)

    print(f"使用 {num_threads} 个线程进行并行推理")

    prompt_snapshot = create_prompt("示例问题", "示例上下文", prompt_type)
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # 初始化结果存储（线程安全）
    results = {}

    # 线程锁，用于保护共享资源
    results_lock = threading.Lock()
    processed_count = 0

    # 设置保存间隔
    save_interval = 1000

    def update_results(sample_id: str, sample: dict, predict: str):
        """线程安全地更新结果"""
        nonlocal processed_count

        with results_lock:
            results[sample_id] = sample
            results[sample_id]["predict"] = predict
            processed_count += 1

            # 每间隔指定数量样本保存一次中间结果
            if processed_count % save_interval == 0 or processed_count == len(dataset):
                save_intermediate_results(
                    results, output_file, processed_count, len(dataset)
                )

    # 使用线程池进行并行处理
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 提交所有任务
        future_to_sample = {
            executor.submit(
                process_single_sample,
                sample,
                client,
                model_name,
                prompt_type,
                temperature,
                top_p,
                enable_thinking,
            ): sample
            for sample in dataset
        }

        # 使用 tqdm 显示进度条
        with tqdm(total=len(dataset), desc="推理进度", unit="样本") as pbar:
            for future in as_completed(future_to_sample):
                sample = future_to_sample[future]
                sample_id = sample["id"]

                try:
                    # 获取结果
                    result_sample_id, predict = future.result()

                    # 更新进度条描述
                    pbar.set_description(f"处理样本 {sample_id}")

                    # 更新结果
                    update_results(result_sample_id, sample, predict)

                except Exception as e:
                    tqdm.write(f"样本 {sample_id} 处理失败: {e}")
                    # 即使失败也要更新结果（空结果）
                    update_results(sample_id, sample, [])

                # 更新进度条
                pbar.update(1)

    end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    commit_id = get_git_commit_id()

    # 记录实验信息
    experiment_info = {
        "server_url": server_url,
        "model_name": model_name,
        "prompt_type": prompt_type,
        "prompt_snapshot": prompt_snapshot,
        "start_time": start_time,
        "end_time": end_time,
        "temperature": temperature,
        "top_p": top_p,
        "dataset": dataset_name,
        "max_samples": max_samples,
        "num_threads": num_threads,
        "output_file": output_file,
        "infer_git_commit_id": commit_id,
    }

    # 保存最终结果
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {"info": experiment_info, "data": results}, f, ensure_ascii=False, indent=2
        )

    print(f"推理完成，结果已保存到: {output_file}")
    print(f"总共处理了 {len(results)} 个样本")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="HotpotQA 推理脚本（多线程版本）")
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:8124/v1",
        help="vLLM 服务地址，例如: https://api.siliconflow.cn/v1",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="模型名称",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None, help="最大处理样本数，用于测试"
    )
    parser.add_argument(
        "--num-threads", type=int, default=4, help="并行推理的线程数量，默认为 4"
    )
    parser.add_argument(
        "--prompt-type",
        type=str,
        default="direct",
        choices=["direct", "reasoning", "reasoning_with_copypaste", "reasoning_with_copypaste_old"],
        help="提示模板选择",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="hotpotqa",
        choices=[
            "hotpotqa",
            "multirc",
            "pubmedqa",
            "musique",
            "2wikimultihopqa",
            "popqa",
            "faitheval",
            "qasper",
            "copypaste",
        ],
        help="数据集名称，当前仅支持 hotpotqa",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="sk-lqztxtcbxxoonlmsxvdhllhdnoegywnvuhfnoqnxvpphrhkh",
        help="API Key，用于访问 第三方 服务",
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="模型生成温度")
    parser.add_argument("--top-p", type=float, default=0.95, help="模型生成 top-p 采样")
    parser.add_argument("--seed", type=int, default=42, help="模型生成 top-p 采样")
    parser.add_argument(
        "--enable-thinking", type=bool, default=False, help="是否启用思考过程"
    )

    args = parser.parse_args()

    random.seed(args.seed)

    timestamp = int(time.time())
    model_name_clean = args.model_name.replace("/", "_").replace(" ", "_")
    output_file = f"results/infer/resamples_{args.max_samples}/seed_{args.seed}/tpr_{args.temperature}-tpp_{args.top_p}/{model_name_clean}/{args.dataset}/enable_thinking_{args.enable_thinking}-prompt_{args.prompt_type.replace(' ', '_')}-{timestamp}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

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
        top_p=args.top_p,
        enable_thinking=args.enable_thinking,
    )


if __name__ == "__main__":
    main()
