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
    loader = load_copypaste_qa(name=dataset_enum, split="test", distractor_docs=0, unanswerable=True)
    _DATASET_CACHE[dataset_enum] = loader
    return loader

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

    # 坐标映射
    if is_gold_removed:
        visual_x = -(noise_level + 1)
    else:
        visual_x = noise_level

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

def format_xaxis(x, pos):
    if x >= 0:
        val = int(x)
        return f"{val}"
    else:
        val = int(abs(x) - 1)
        return f"{val}"

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
    
    # === 修复 1: 使用 husl 色板 ===
    # husl 可以在色彩空间中均匀切分出 N 种颜色，保证 N 个数据集颜色均不相同
    # 避免了 bright/deep 等标准色板只有 10 种颜色的限制
    palette = sns.color_palette("husl", len(datasets))
    
    # 如果你觉得 husl 颜色太亮，可以使用 'tab20' (最多20种)
    # palette = sns.color_palette("tab20", len(datasets)) if len(datasets) <= 20 else sns.color_palette("husl", len(datasets))
    
    color_map = dict(zip(datasets, palette))

    # === 修复 2: 扩展 Marker 列表 ===
    # 定义更多种类的 Marker，防止形状重复
    # 优先级: 圆, 方, 三角, 菱形, 叉, 五边形, 六边形...
    markers_pool = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', 'X', 'd', 'P', '8']
    # 如果数据集比 marker 种类还多，循环使用，但因为颜色不同，组合依然是唯一的
    
    for i, (metric_col, metric_label) in enumerate(metrics_map.items()):
        ax = axes[i]
        
        # 1. 绘制左侧 (Visual X <= -1)
        sns.lineplot(
            data=df[df['Visual X'] <= -1],
            x="Visual X", y=metric_col,
            hue="Dataset", style="Dataset",
            palette=palette, hue_order=datasets, style_order=datasets,
            markers=markers_pool, markersize=9, linewidth=2.5,
            dashes=False, 
            legend=False, ax=ax
        )

        # 2. 绘制右侧 (Visual X >= 0)
        sns.lineplot(
            data=df[df['Visual X'] >= 0],
            x="Visual X", y=metric_col,
            hue="Dataset", style="Dataset",
            palette=palette, hue_order=datasets, style_order=datasets,
            markers=markers_pool, markersize=9, linewidth=2.5,
            dashes=False,
            legend=False, ax=ax
        )

        # 装饰部分
        ax.axvline(x=-0.5, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
        
        x_min_visual, x_max_visual = df['Visual X'].min(), df['Visual X'].max()
        padding = 1
        ax.set_xlim(x_min_visual - padding, x_max_visual + padding)
        
        ax.axvspan(x_min_visual - padding, -0.5, facecolor='#fff0f0', alpha=0.3)
        ax.axvspan(-0.5, x_max_visual + padding, facecolor='#f0f8ff', alpha=0.3)

        if i < 2:
            ax.text(x_min_visual/2 - 0.5, 1.02, "Unanswerable\n(No Gold Doc)", 
                    transform=ax.get_xaxis_transform(), 
                    ha='center', va='bottom', fontsize=12, fontweight='bold', color='#8B0000')
            ax.text(x_max_visual/2, 1.02, "Answerable\n(With Gold Doc)", 
                    transform=ax.get_xaxis_transform(),
                    ha='center', va='bottom', fontsize=12, fontweight='bold', color='#00008B')

        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_xaxis))
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        ax.set_ylabel(metric_label, fontweight='bold')
        ax.set_xlabel("")
        ax.grid(True, which='major', linestyle='--', alpha=0.6)

    fig.text(0.5, 0.14, "Number of Noise Documents", ha='center', fontsize=16, fontweight='bold')
    fig.text(0.28, 0.14, "← Without Gold Context", ha='center', fontsize=12, color='#8B0000')
    fig.text(0.72, 0.14, "With Gold Context →", ha='center', fontsize=12, color='#00008B')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18, top=0.92)
    
    # === 修复 3: 手动图例生成逻辑更新 ===
    legend_elements = []
    for idx, ds in enumerate(datasets):
        # 确保 marker 的索引与绘图时一致
        marker = markers_pool[idx % len(markers_pool)]
        legend_elements.append(
            Line2D([0], [0], color=color_map[ds], lw=2.5, label=ds, 
                   marker=marker, markersize=9, linestyle='-') 
        )

    fig.legend(
        handles=legend_elements,
        loc='lower center', 
        bbox_to_anchor=(0.5, 0.05),
        ncol=min(len(datasets), 6), # 自动调整列数
        frameon=False,
        fontsize=12, # 稍微调小一点以免太挤
        title="Datasets"
    )

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Figure saved to: {os.path.abspath(output_path)}")

def main():
    parser = argparse.ArgumentParser(description="CopyPasteLRM Visualization: Fix Colors.")
    parser.add_argument('folder_path', type=str, help="Result folder path.")
    parser.add_argument('--output', type=str, default="scaling_fixed_colors.png", help="Output filename.")
    args = parser.parse_args()

    files = glob.glob(os.path.join(args.folder_path, "*.json"))
    if not files:
        print("No files found.")
        return

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
    plot_results(df, args.output)

if __name__ == "__main__":
    main()