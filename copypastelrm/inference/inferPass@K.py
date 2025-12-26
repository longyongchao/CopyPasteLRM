import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Set

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

from copypastelrm.metrics.HotpotQA import exact_match_score, compute_answer_em_hit_f1
from copypastelrm.metrics.utils import extract_answer_and_facts


def check_correctness(sample: Dict[str, Any], prediction: str) -> bool:
    """
    判断模型响应是否正确。

    Args:
        sample: 原始样本数据，包含 ground truth
        prediction: 模型生成的文本

    Returns:
        bool: 是否正确
    """
    em, hit, f1 = compute_answer_em_hit_f1(prediction, sample["answer"])

    return f1 > 0


def call_model(
    client: OpenAI,
    model_name: str,
    prompt: str,
    temperature: float = 0.6,
    top_p=0.95,
    enable_thinking: bool = True,
) -> str:
    """调用模型进行推理"""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": enable_thinking},
            },
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"模型调用失败: {e}")
        return ""


def process_single_sample_pass_at_k(
    sample: Dict[str, Any],
    client: OpenAI,
    model_name: str,
    temperature: float,
    top_p: float,
    k: int,
    enable_thinking: bool = True,
    prior_threshold: int = 64,
) -> List[Dict[str, Any]]:
    """
    处理单个样本的 Pass@K 推理任务
    """
    sample_id = sample["id"]
    results_list = []

    try:
        # 格式化上下文和创建提示 (提示对于同一样本是固定的)
        tips_wrong_answer = set()

        # Pass@K 循环
        for attempt_idx in range(1, k + 1):
            if prior_threshold is None or attempt_idx < prior_threshold:
                prompt = create_prompt(
                    sample["query"],
                    sample["context"],
                    "pass@K",
                    "\n".join(sample["sfs"]),
                )
            else:
                prompt = create_prompt(
                    sample["query"],
                    sample["context"],
                    "pass@K",
                    "\n".join(sample["sfs"]),
                    "; ".join(tips_wrong_answer),
                )

            # 调用模型
            predict = call_model(
                client,
                model_name,
                prompt,
                temperature=temperature,
                top_p=top_p,
                enable_thinking=enable_thinking,
            )

            # 判断正确性
            # 如果调用失败（空字符串），通常视作错误，但也可以根据需求处理
            is_correct = False
            if predict:
                predict_answer, _ = extract_answer_and_facts(predict)
                is_correct = exact_match_score(predict_answer, sample["answer"])

                if not is_correct:
                    tips_wrong_answer.add(predict_answer)

                if attempt_idx > 32 and is_correct:
                    # 这是一个简单的日志，用于观察长尾采样的效果
                    # print("✅", sample_id, attempt_idx, is_correct)
                    pass

            # 记录当前采样结果
            record = {
                "id": sample_id,
                "attempt_idx": attempt_idx,  # 第几次采样
                "predict": predict,
                "is_correct": is_correct,
                "tips_wrong_answer": list(tips_wrong_answer),
                "recieved_tips": (prior_threshold is not None) and (attempt_idx > prior_threshold),
            }

            if not is_correct and attempt_idx == k:
                results_list.append(record)
            elif is_correct:
                results_list.append(record)
                break

            # 论文逻辑：如果回答正确，终止对该样本的后续采样

        return results_list

    except Exception as e:
        print(f"处理样本 {sample_id} 时发生错误: {e}")
        # 发生错误时返回包含错误信息的记录
        return [{"id": sample_id, "error": str(e), "attempt_idx": -1}]


def run_inference(
    server_url: str,
    model_name: str,
    output_file: str,
    max_samples: int = -1,
    prior_threshold: int = 64,
    num_threads: int = 16,
    dataset_name: str = "hotpotqa",
    split: str = "train",
    api_key: str = "sk-wingchiu",
    temperature: float = 0.6,
    top_p: float = 0.95,
    k: int = 1024,  # 新增 K 参数
    enable_thinking: bool = True,
):
    """
    运行完整的 Pass@K 推理流程
    """
    # 初始化 OpenAI 客户端
    client = OpenAI(base_url=server_url, api_key=api_key)

    # 加载数据集
    if dataset_name == "hotpotqa":
        dataset_loader = HotpotQA(max_samples=max_samples, split="train" if split == "train" else "validation")
    elif dataset_name == "multirc":
        dataset_loader = MultiRC(max_samples=max_samples, split='train' if split == "train" else "dev")
    elif dataset_name == "musique":
        dataset_loader = MuSiQue(max_samples=max_samples, split="train" if split == "train" else 'validation')
    elif dataset_name == "popqa":
        dataset_loader = PopQA(max_samples=max_samples, split="train" if split == 'train' else 'test')
    elif dataset_name == "qasper":
        dataset_loader = Qasper(max_samples=max_samples, split="train" if split == 'train' else 'test')
    elif dataset_name == "2wikimultihopqa":
        dataset_loader = TwoWikiMultihopQA(max_samples=max_samples, split="dev" if split == 'train' else 'test')
    elif dataset_name == "pubmedqa":
        dataset_loader = PubMedQA(max_samples=max_samples, dataset_name='pqa_artificial' if split == 'train' else 'pqa_labeled')
    elif dataset_name == "faitheval":
        dataset_loader = FaithEval(max_samples=max_samples)
    elif dataset_name == "copypaste":
        dataset_loader = CopyPaste(max_samples=max_samples)
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

    dataset = dataset_loader.dataset_list

    # --------------------------------------------------------------------------
    # 断点重启逻辑
    # --------------------------------------------------------------------------
    processed_ids: Set[str] = set()
    if os.path.exists(output_file):
        print(f"检测到输出文件已存在: {output_file}")
        print("正在检查断点...")
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        if "id" in record:
                            processed_ids.add(record["id"])
                    except json.JSONDecodeError:
                        continue
            print(f"已找到 {len(processed_ids)} 个已处理的样本，将跳过。")
        except Exception as e:
            print(f"读取断点文件时发生错误，将重新开始: {e}")
    
    # 过滤掉已经处理过的样本
    original_count = len(dataset)
    if processed_ids:
        dataset = [d for d in dataset if d["id"] not in processed_ids]
        print(f"剩余需处理样本数: {len(dataset)} / {original_count}")
    
    if len(dataset) == 0:
        print("所有样本均已处理完毕。")
        return
    # --------------------------------------------------------------------------

    print(f"使用 {num_threads} 个线程进行并行推理，Pass@{k}")

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 线程锁，用于写入文件
    file_lock = threading.Lock()

    # 使用线程池进行并行处理
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_sample = {
            executor.submit(
                process_single_sample_pass_at_k,
                sample,
                client,
                model_name,
                temperature,
                top_p,
                k,
                enable_thinking,
                prior_threshold,
            ): sample
            for sample in dataset
        }

        # 打开文件准备写入 (JSONL 模式 - 追加模式 'a')
        with open(output_file, "a", encoding="utf-8") as f_out:
            with tqdm(
                total=len(dataset), desc=f"Pass@{k} 推理进度", unit="样本"
            ) as pbar:
                for future in as_completed(future_to_sample):
                    sample = future_to_sample[future]
                    sample_id = sample["id"]

                    try:
                        # 获取该样本的所有尝试记录 (List[Dict])
                        attempts_records = future.result()

                        # 线程安全地写入文件
                        with file_lock:
                            for record in attempts_records:
                                # 写入一行 JSON
                                f_out.write(
                                    json.dumps(record, ensure_ascii=False) + "\n"
                                )
                                f_out.flush()  # 确保实时写入

                        # 更新进度条
                        # 如果是 Pass@K，这里进度条代表处理完一个样本（无论内部尝试了多少次）
                        pbar.set_description(f"完成样本 {sample_id}")

                    except Exception as e:
                        tqdm.write(f"样本 {sample_id} 主循环异常: {e}")

                    pbar.update(1)

    print(f"推理完成，结果已保存到: {output_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Pass@K 推理脚本")
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:8124/v1",
        help="vLLM 服务地址",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="模型名称",
    )
    parser.add_argument("--max-samples", type=int, default=-1, help="最大处理样本数")
    parser.add_argument(
        "--prior-threshold",
        type=int,
        default=64,
        help="尝试多少次之后提供错误答案提示",
    )
    parser.add_argument("--num-threads", type=int, default=4, help="线程数量")
    parser.add_argument(
        "--k", type=int, default=1, help="Pass@K 的 K 值，即最大采样次数"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="hotpotqa",
        help="数据集名称",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="数据集划分",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="sk-wingchiu",
        help="API Key",
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="温度")
    parser.add_argument("--top-p", type=float, default=0.95, help="top-p")
    parser.add_argument("--enable-thinking", action="store_true", help="是否启用思考")
    
    # 新增 output-file 参数
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="指定输出文件路径，如果指定且文件存在，则启用断点续传",
    )

    args = parser.parse_args()

    # 确定输出文件路径
    if args.output_file:
        output_file = args.output_file
    else:
        # 如果未指定 output_file，则使用默认的自动生成逻辑
        timestamp = int(time.time())
        model_name_clean = args.model_name.replace("/", "_").replace(" ", "_")
        target_root = "/mnt/lustre/DATA/longyongchao"
        
        # 路径判断逻辑
        if os.path.exists(target_root) and os.path.isdir(target_root):
            save_dir_root = "/mnt/lustre/DATA/longyongchao/CopyPasteLRM"
        else:
            save_dir_root = "/data/lyc/CopyPasteLRM"
        
        output_file = f"{save_dir_root}/pass_at_{args.k}/{model_name_clean}/resamples_{args.max_samples}/{args.split}/{args.dataset}-tpr_{args.temperature}-tpp_{args.top_p}-enable_thinking_{args.enable_thinking}-tips_threshold_{args.prior_threshold}-{timestamp}.jsonl"

    if args.prior_threshold is not None:
        assert args.prior_threshold < args.k, "错误提示阈值不能大于最大采样次数"

    print(f"Output File: {output_file}")

    run_inference(
        server_url=args.server_url,
        model_name=args.model_name,
        output_file=output_file,
        max_samples=args.max_samples,
        prior_threshold=args.prior_threshold,
        num_threads=args.num_threads,
        dataset_name=args.dataset,
        split=args.split,
        api_key=args.api_key,
        temperature=args.temperature,
        top_p=args.top_p,
        k=args.k,
        enable_thinking=args.enable_thinking,
    )


if __name__ == "__main__":
    main()
