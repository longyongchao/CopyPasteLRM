import argparse
import glob
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from matplotlib.lines import Line2D

# --- CopyPasteLRM Imports ---
from copypastelrm.utils.json_tools import read_json
from copypastelrm.metrics.utils import (
    extract_answer_and_facts,
    extract_answer_and_facts_old,
)
from copypastelrm.metrics.HotpotQA import compute_answer_em_hit_f1, update_sp
from copypastelrm.datasets import load as load_copypaste_qa, AvailableDataset

os.environ["HF_OFFLINE"] = "1"

# --- 全局缓存 ---
_DATASET_CACHE = {}

# --- Configuration ---
sns.set_theme(style="ticks", context="paper", font_scale=1.4)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['grid.alpha'] = 0.4
plt.rcParams['grid.linestyle'] = '--'

# --- 辅助函数 ---
def normalize_string(s):
    return str(s).lower().replace("-", "").replace("_", "")

def match_dataset_enum(dataset_name_str):
    target_str = normalize_string(dataset_name_str)
    for member in AvailableDataset:
        if isinstance(member.value, str):
            if normalize_string(member.value) == target_str:
                return member, member.value
        if normalize_string(member.name) == target_str:
            return member, member.value
    if "2wiki" in target_str and "multihop" not in target_str:
         for member in AvailableDataset:
             if "2wiki" in normalize_string(member.value):
                 return member, member.value
    return None, str(dataset_name_str)

def get_dataset_loader_with_cache(dataset_enum):
    global _DATASET_CACHE
    if dataset_enum in _DATASET_CACHE:
        return _DATASET_CACHE[dataset_enum]
    print(f"🐢 First time loading dataset: {dataset_enum.value} ...")
    loader = load_copypaste_qa(name=dataset_enum, split="test", distractor_docs=0, unanswerable=True, reload=False)
    _DATASET_CACHE[dataset_enum] = loader
    return loader

def calculate_log_visual_x(noise):
    """
    计算基于 Log2 的视觉坐标。
    映射规则:
    0 -> 0
    1 -> 1
    2 -> 2
    4 -> 3
    8 -> 4
    ...
    N -> log2(N) + 1 (for N >= 1)
    """
    if noise == 0:
        return 0
    return np.log2(noise) + 1

def process_single_file(file_path):
    try:
        infer_res = read_json(file_path)
    except Exception as e:
        print(f"Error reading {os.path.basename(file_path)}: {e}")
        return None

    info = infer_res.get("info", {})
    data = infer_res.get("data", {})

    raw_dataset_name = info.get("dataset", "Unknown")
    dataset_enum, display_name = match_dataset_enum(raw_dataset_name)
    if dataset_enum is None:
        return None
    
    try:
        dataset_loader = get_dataset_loader_with_cache(dataset_enum)
    except Exception as e:
        return None

    noise_level = info.get("噪音文档数量", None)
    is_gold_removed = info.get("是否剔除金标上下文", False)

    if noise_level is None:
        match = re.search(r'noise_(\d+)', os.path.basename(file_path))
        noise_level = int(match.group(1)) if match else 0
    else:
        noise_level = int(noise_level)

    # === 核心修改: Log2 坐标映射 ===
    base_x = calculate_log_visual_x(noise_level)
    
    # 左右分离逻辑：中间留出 1.0 的 Gap (-0.5 到 0.5)
    if is_gold_removed:
        visual_x = -(base_x + 0.5)
    else:
        visual_x = (base_x + 0.5)

    prompt_type = info.get('prompt_type', 'rag')
    if prompt_type in ['rag', 'ircot']:
        extract_func = extract_answer_and_facts
    elif prompt_type in ['copypaste', 'find_facts']:
        extract_func = extract_answer_and_facts_old
    else:
        extract_func = extract_answer_and_facts

    ems, f1s, sp_ems, sp_f1s = [], [], [], []
    valid_count = 0

    for _id, item in data.items():
        try:
            sample = dataset_loader.get_sample(_id)
            if not sample: continue
            
            p_ans, p_fact = extract_func(item["predict"])
            em, _, f1 = compute_answer_em_hit_f1(p_ans, sample['answers'])
            sp_em, sp_f1, _, _ = update_sp(p_fact, sample['facts'])

            ems.append(em)
            f1s.append(f1)
            sp_ems.append(sp_em)
            sp_f1s.append(sp_f1)
            valid_count += 1
        except:
            continue

    if valid_count == 0: return None

    return {
        "Dataset": display_name,
        "Visual X": visual_x,
        "Real Noise": noise_level,
        "Is Unanswerable": is_gold_removed,
        "Answer EM": np.mean(ems),
        "Answer F1": np.mean(f1s),
        "Facts EM": np.mean(sp_ems),
        "Facts F1": np.mean(sp_f1s)
    }

def plot_results(df, output_path):
    metrics_map = {
        "Answer EM": "Answer EM",
        "Answer F1": "Answer F1",
        "Facts EM": "Facts EM",
        "Facts F1": "Facts F1"
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 18), sharex=True)
    axes = axes.flatten()

    datasets = sorted(df['Dataset'].unique())
    palette = sns.color_palette("husl", len(datasets))
    color_map = dict(zip(datasets, palette))

    # 定义 Marker 列表
    markers_pool = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', 'X', 'd', 'P', '8']

    # === 生成 Log2 刻度 ===
    # 我们希望显示的刻度点: 0, 1, 2, 4, 8, 16, 32, 64
    target_ticks = [0, 1, 2, 4, 8, 16, 32, 64]
    
    # 计算这些刻度在 Visual X 轴上的位置
    right_tick_locs = [calculate_log_visual_x(t) + 0.5 for t in target_ticks]
    left_tick_locs = [-(calculate_log_visual_x(t) + 0.5) for t in target_ticks]
    
    # 合并左右刻度
    all_tick_locs = left_tick_locs + right_tick_locs
    all_tick_labels = [str(t) for t in target_ticks] + [str(t) for t in target_ticks]

    for i, (metric_col, metric_label) in enumerate(metrics_map.items()):
        ax = axes[i]
        
        # 1. 绘制左侧 (Visual X < 0)
        sns.lineplot(
            data=df[df['Visual X'] < 0],
            x="Visual X", y=metric_col,
            hue="Dataset", style="Dataset",
            palette=palette, hue_order=datasets, style_order=datasets,
            markers=markers_pool, markersize=9, linewidth=2.5,
            dashes=False, 
            legend=False, ax=ax
        )

        # 2. 绘制右侧 (Visual X > 0)
        sns.lineplot(
            data=df[df['Visual X'] > 0],
            x="Visual X", y=metric_col,
            hue="Dataset", style="Dataset",
            palette=palette, hue_order=datasets, style_order=datasets,
            markers=markers_pool, markersize=9, linewidth=2.5,
            dashes=False,
            legend=False, ax=ax
        )

        # === 装饰部分 ===
        # 分界线 (x=0)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
        
        # 区域背景色
        # 获取当前视图范围
        x_min_visual, x_max_visual = df['Visual X'].min(), df['Visual X'].max()
        padding = 0.5
        ax.set_xlim(x_min_visual - padding, x_max_visual + padding)
        
        # 左侧红底 (无金标)
        ax.axvspan(x_min_visual - padding, 0, facecolor='#fff0f0', alpha=0.3)
        # 右侧蓝底 (有金标)
        ax.axvspan(0, x_max_visual + padding, facecolor='#f0f8ff', alpha=0.3)

        if i < 2:
            ax.text(x_min_visual/2, 1.02, "Unanswerable\n(No Gold Doc)", 
                    transform=ax.get_xaxis_transform(), 
                    ha='center', va='bottom', fontsize=12, fontweight='bold', color='#8B0000')
            ax.text(x_max_visual/2, 1.02, "Answerable\n(With Gold Doc)", 
                    transform=ax.get_xaxis_transform(),
                    ha='center', va='bottom', fontsize=12, fontweight='bold', color='#00008B')

        # === 设置自定义 Log2 刻度 ===
        ax.set_xticks(all_tick_locs)
        ax.set_xticklabels(all_tick_labels)

        ax.set_ylabel(metric_label, fontweight='bold')
        ax.set_xlabel("")
        ax.grid(True, which='major', linestyle='--', alpha=0.6)
        ax.set_ylim(-0.02, 1.02)

    fig.text(0.5, 0.14, "Number of Noise Documents", ha='center', fontsize=16, fontweight='bold')
    fig.text(0.28, 0.14, "← Without Gold Context", ha='center', fontsize=12, color='#8B0000')
    fig.text(0.72, 0.14, "With Gold Context →", ha='center', fontsize=12, color='#00008B')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18, top=0.92)
    
    # 图例生成
    legend_elements = []
    for idx, ds in enumerate(datasets):
        marker = markers_pool[idx % len(markers_pool)]
        legend_elements.append(
            Line2D([0], [0], color=color_map[ds], lw=2.5, label=ds, 
                   marker=marker, markersize=9, linestyle='-') 
        )

    fig.legend(
        handles=legend_elements,
        loc='lower center', 
        bbox_to_anchor=(0.5, 0.95),
        ncol=min(len(datasets), 5), 
        frameon=False,
        fontsize=12, 
        title="Datasets"
    )

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Figure saved to: {os.path.abspath(output_path)}")

def main():
    parser = argparse.ArgumentParser(description="CopyPasteLRM Visualization: Log2 Scale.")
    parser.add_argument('folder_path', type=str, help="Result folder path.")
    parser.add_argument('--output', type=str, default=None, help="Output filename.")
    args = parser.parse_args()

    exclude_dataset = ['ConFiQA-MC-Original', 'ConFiQA-MR-Original', 'ConFiQA-QA-Original', 'Qasper']

    output_path = args.output if args.output else os.path.join(args.folder_path, "noise_scaling_log2.png")

    files = glob.glob(os.path.join(args.folder_path, "*.json"))
    if not files:
        print("No files found.")
        return

    files = [f for f in files if not any(exclude_ds in os.path.basename(f) for exclude_ds in exclude_dataset)]

    print(f"Found {len(files)} files. Processing with cache...")
    data_list = []
    
    total = len(files)
    for idx, f in enumerate(files):
        if idx % 10 == 0:
            print(f"[{idx}/{total}] Processing...", end="\r")
        res = process_single_file(f)
        if res:
            data_list.append(res)
    print(f"[{total}/{total}] Processing complete.    ")

    if not data_list:
        print("No valid data extracted.")
        return

    df = pd.DataFrame(data_list)
    df = df.sort_values(by=["Dataset", "Visual X"])

    print("Data loaded. Generating plots...")
    plot_results(df, output_path)

if __name__ == "__main__":
    main()