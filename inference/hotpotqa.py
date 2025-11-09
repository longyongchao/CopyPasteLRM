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
python inference/hotpotqa.py --model-url http://localhost:8124/v1 --model-name qwen2.5-3b-instruct --output-file results/hotpotqa/3b.json --num-threads 96

新增参数：
--num-threads: 并行推理的线程数量，默认为 4
"""

import argparse
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm


def load_hotpotqa_dataset() -> List[Dict[str, Any]]:
    """
    从 HuggingFace 加载 HotpotQA 数据集的 distractor 子集 validation split

    Returns:
        List[Dict]: 包含数据样本的列表
    """
    print("正在加载 HotpotQA 数据集...")
    try:
        dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
        print(f"成功加载数据集，共 {len(dataset)} 个样本")
        return list(dataset)
    except Exception as e:
        print(f"加载数据集失败: {e}")
        sys.exit(1)


def format_context(context: Dict[str, Any]) -> str:
    """
    格式化上下文信息

    Args:
        context: 包含 title 和 sentences 的字典

    Returns:
        str: 格式化后的上下文文本
    """
    context_text = ""
    for title, sentences in zip(context["title"], context["sentences"]):
        context_text += f"{title}. " + "".join(sentences) + "\n"
    return context_text


def create_prompt(question: str, context: str) -> str:
    """
    创建推理提示

    Args:
        question: 问题文本
        context: 上下文文本

    Returns:
        str: 完整的提示文本
    """
    prompt = f"""
Context: {context}

Question: {question}

---

Answer questions using the provided Context.

Formatting rules:
1) Explain your reasoning in a natural, fluent way inside a single </think>...</think> block.
2) Give the concise final answer inside a single <answer>...</answer> block.
3) Whenever you use an exact phrase or sentence taken verbatim from the Context as part of your reasoning, embed that exact substring with <copy>...</copy> tags. The content inside <copy> must be an exact substring of Context—do not paraphrase or modify it.
4) If no direct supporting sentence exists in the Context for a claim, explicitly acknowledge uncertainty in </think> instead of inventing facts.
5) Prefer natural, paragraph-style reasoning (not numbered steps). It is encouraged to integrate <copy>...</copy> evidence sentences seamlessly into your reasoning text to show traceability.

i.e., </think> reasoning process (must include <copy>evidence from Context</copy> naturally) </think><answer> final answer here </answer>
""".strip()
    return prompt


def call_model(client: OpenAI, model_name: str, prompt: str, max_tokens: int = 4096) -> str:
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
            temperature=1.0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"模型调用失败: {e}")
        return ""


def extract_answer_and_facts(response: str, context: Dict[str, Any]) -> Tuple[str, List[List[str]]]:
    """
    从模型响应中提取答案和支持事实

    Args:
        response: 模型生成的响应
        context: 原始上下文信息

    Returns:
        Tuple[str, List[List[str]]]: (答案, 支持事实列表)
    """
    # 提取答案
    answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()
    else:
        answer = ""

    # 提取所有 <copy> 标签中的内容
    copy_matches = re.findall(r"<copy>(.*?)</copy>", response, re.DOTALL)
    copied_texts = [match.strip() for match in copy_matches]

    # 将复制的文本映射回支持事实
    supporting_facts = []
    context_text = format_context(context)

    for copied_text in copied_texts:
        # 在上下文中查找复制的文本
        title, sent_id = find_text_in_context(copied_text, context)
        if title and sent_id is not None:
            if [title, sent_id] not in supporting_facts:
                supporting_facts.append([title, sent_id])

    return answer, supporting_facts


def find_text_in_context(text: str, context: Dict[str, Any]) -> Tuple[str, int]:
    """
    在上下文中查找文本，返回标题和句子ID

    Args:
        text: 要查找的文本
        context: 上下文信息

    Returns:
        Tuple[str, int]: (标题, 句子ID)，如果找不到返回 ("", None)
    """
    for title, sentences in zip(context["title"], context["sentences"]):
        for sent_id, sentence in enumerate(sentences):
            if text.strip() in sentence.strip():
                return title, sent_id

    return "", None


def process_single_sample(sample: Dict[str, Any], client: OpenAI, model_name: str) -> Tuple[str, str, List[List[str]]]:
    """
    处理单个样本的推理任务

    Args:
        sample: 数据样本
        client: OpenAI 客户端
        model_name: 模型名称

    Returns:
        Tuple[str, str, List[List[str]]]: (样本ID, 答案, 支持事实列表)
    """
    sample_id = sample["id"]

    try:
        # 格式化上下文和创建提示
        context_text = format_context(sample["context"])
        prompt = create_prompt(sample["question"], context_text)

        # 调用模型
        response = call_model(client, model_name, prompt)

        if not response:
            return sample_id, "", []

        # 提取答案和支持事实
        answer, supporting_facts = extract_answer_and_facts(response, sample["context"])

        return sample_id, answer, supporting_facts

    except Exception as e:
        print(f"处理样本 {sample_id} 时发生错误: {e}")
        return sample_id, "", []


def save_intermediate_results(results: Dict[str, Any], output_file: str, processed_count: int, total_count: int):
    """
    保存中间结果

    Args:
        results: 当前结果字典
        output_file: 输出文件路径
        processed_count: 已处理样本数
        total_count: 总样本数
    """
    # 创建中间结果文件名
    base_name = output_file.rsplit(".", 1)[0]
    extension = output_file.rsplit(".", 1)[1] if "." in output_file else "json"
    intermediate_file = f"{base_name}_intermediate_{processed_count}.{extension}"

    with open(intermediate_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"已保存中间结果 ({processed_count}/{total_count}): {intermediate_file}")


def run_inference(model_url: str, model_name: str, output_file: str, max_samples: int = None, num_threads: int = 4):
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
    client = OpenAI(base_url=model_url, api_key="sk-lqztxtcbxxoonlmsxvdhllhdnoegywnvuhfnoqnxvpphrhkh")  # vLLM 通常不需要真实的 API key

    # 加载数据集
    dataset = load_hotpotqa_dataset()

    if max_samples:
        dataset = dataset[:max_samples]
        print(f"限制处理样本数量为: {max_samples}")

    print(f"使用 {num_threads} 个线程进行并行推理")

    # 初始化结果存储（线程安全）
    results = {"answer": {}, "sp": {}}

    # 线程锁，用于保护共享资源
    results_lock = threading.Lock()
    processed_count = 0

    # 设置保存间隔
    save_interval = 1000

    def update_results(sample_id: str, answer: str, supporting_facts: List[List[str]]):
        """线程安全地更新结果"""
        nonlocal processed_count

        with results_lock:
            results["answer"][sample_id] = answer
            results["sp"][sample_id] = supporting_facts
            processed_count += 1

            # 显示当前样本信息
            tqdm.write(f"样本 {sample_id} 完成")
            tqdm.write(f"  答案: {answer}")
            tqdm.write(f"  支持事实数量: {len(supporting_facts)}")

            # 每间隔指定数量样本保存一次中间结果
            if processed_count % save_interval == 0 or processed_count == len(dataset):
                save_intermediate_results(results, output_file, processed_count, len(dataset))

    # 使用线程池进行并行处理
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 提交所有任务
        future_to_sample = {executor.submit(process_single_sample, sample, client, model_name): sample for sample in dataset}

        # 使用 tqdm 显示进度条
        with tqdm(total=len(dataset), desc="推理进度", unit="样本") as pbar:
            for future in as_completed(future_to_sample):
                sample = future_to_sample[future]
                sample_id = sample["id"]

                try:
                    # 获取结果
                    result_sample_id, answer, supporting_facts = future.result()

                    # 更新进度条描述
                    pbar.set_description(f"处理样本 {sample_id}")

                    # 更新结果
                    update_results(result_sample_id, answer, supporting_facts)

                except Exception as e:
                    tqdm.write(f"样本 {sample_id} 处理失败: {e}")
                    # 即使失败也要更新结果（空结果）
                    update_results(sample_id, "", [])

                # 更新进度条
                pbar.update(1)

    # 保存最终结果
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"推理完成，结果已保存到: {output_file}")
    print(f"总共处理了 {len(results['answer'])} 个样本")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="HotpotQA 推理脚本（多线程版本）")
    parser.add_argument("--model-url", type=str, default="https://api.siliconflow.cn/v1", help="vLLM 服务地址，例如: https://api.siliconflow.cn/v1")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-72B-Instruct", help="模型名称")
    parser.add_argument("--output-file", type=str, default="predictions.json", help="输出文件路径")
    parser.add_argument("--max-samples", type=int, default=None, help="最大处理样本数，用于测试")
    parser.add_argument("--num-threads", type=int, default=4, help="并行推理的线程数量，默认为 4")

    args = parser.parse_args()

    print("开始 HotpotQA 推理（多线程版本）...")
    print(f"模型地址: {args.model_url}")
    print(f"模型名称: {args.model_name}")
    print(f"输出文件: {args.output_file}")
    print(f"线程数量: {args.num_threads}")

    run_inference(args.model_url, args.model_name, args.output_file, args.max_samples, args.num_threads)


if __name__ == "__main__":
    main()
