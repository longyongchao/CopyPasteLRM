from datasets import load_dataset
from statistics import median, mean, stdev
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def flatten_context(context):
    """将context字典平铺为字符串"""
    sentences = []
    for title, sent_list in zip(context['title'], context['sentences']):
        for sent in sent_list:
            sentences.append(sent)
    return ' '.join(sentences)

def get_sentence_length(context, sent_idx, title):
    """获取context中指定句子的长度"""
    title_to_sentences = dict(zip(context['title'], context['sentences']))
    if title in title_to_sentences:
        if 0 <= sent_idx < len(title_to_sentences[title]):
            return len(title_to_sentences[title][sent_idx])
    return 0

def analyze_hotpotqa():
    # 加载数据集
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
    
    # 按类型和难度分组存储数据
    stats = defaultdict(lambda: {
        'answer_lengths': [],
        'context_lengths': [],
        'supporting_fact_counts': [],
        'supporting_title_counts': [],
        'supporting_sentence_lengths': [],
        'context_title_counts': []
    })
    
    for item in dataset:
        qtype = item['type']
        level = item['level']
        key = f"{qtype}_{level}"
        
        # 1. 答案长度
        stats[key]['answer_lengths'].append(len(item['answer']))
        
        # 2. context长度
        flat_context = flatten_context(item['context'])
        stats[key]['context_lengths'].append(len(flat_context))
        
        # 3. supporting_facts句子数量
        stats[key]['supporting_fact_counts'].append(len(item['supporting_facts']['sent_id']))
        
        # 4. supporting_facts中title数量
        stats[key]['supporting_title_counts'].append(len(set(item['supporting_facts']['title'])))
        
        # 5. supporting_facts指向的句子长度
        for sent_idx, title in zip(item['supporting_facts']['sent_id'], item['supporting_facts']['title']):
            sent_length = get_sentence_length(item['context'], sent_idx, title)
            if sent_length > 0:
                stats[key]['supporting_sentence_lengths'].append(sent_length)
        
        # 6. context中title数量
        stats[key]['context_title_counts'].append(len(item['context']['title']))
    
    return stats

def create_bar_charts(stats):
    # 准备数据
    types = ['bridge', 'comparison']
    levels = ['easy', 'medium', 'hard']
    
    # 为每个指标准备数据
    metrics = {
        'Answer Length': {'mean': [], 'std': []},
        'Context Length': {'mean': [], 'std': []},
        'Supporting Facts Count': {'mean': [], 'std': []},
        'Unique Title Count': {'mean': [], 'std': []},
        'Supporting Sentence Length': {'mean': [], 'std': []},
        'Context Title Count': {'mean': [], 'std': []}
    }
    
    # 填充数据
    for qtype in types:
        for level in levels:
            key = f"{qtype}_{level}"
            data = stats[key]
            
            # Answer Length
            metrics['Answer Length']['mean'].append(mean(data['answer_lengths']))
            metrics['Answer Length']['std'].append(stdev(data['answer_lengths']) if len(data['answer_lengths']) > 1 else 0)
            
            # Context Length
            metrics['Context Length']['mean'].append(mean(data['context_lengths']))
            metrics['Context Length']['std'].append(stdev(data['context_lengths']) if len(data['context_lengths']) > 1 else 0)
            
            # Supporting Facts Count
            metrics['Supporting Facts Count']['mean'].append(mean(data['supporting_fact_counts']))
            metrics['Supporting Facts Count']['std'].append(stdev(data['supporting_fact_counts']) if len(data['supporting_fact_counts']) > 1 else 0)
            
            # Unique Title Count
            metrics['Unique Title Count']['mean'].append(mean(data['supporting_title_counts']))
            metrics['Unique Title Count']['std'].append(stdev(data['supporting_title_counts']) if len(data['supporting_title_counts']) > 1 else 0)
            
            # Supporting Sentence Length
            if data['supporting_sentence_lengths']:
                metrics['Supporting Sentence Length']['mean'].append(mean(data['supporting_sentence_lengths']))
                metrics['Supporting Sentence Length']['std'].append(stdev(data['supporting_sentence_lengths']) if len(data['supporting_sentence_lengths']) > 1 else 0)
            else:
                metrics['Supporting Sentence Length']['mean'].append(0)
                metrics['Supporting Sentence Length']['std'].append(0)
            
            # Context Title Count
            metrics['Context Title Count']['mean'].append(mean(data['context_title_counts']))
            metrics['Context Title Count']['std'].append(stdev(data['context_title_counts']) if len(data['context_title_counts']) > 1 else 0)
    
    # 设置图表样式
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(1, 6, figsize=(30, 6))
    fig.suptitle('HotpotQA Statistics by Type and Difficulty', fontsize=16, fontweight='bold')
    
    # 准备标签
    x = np.arange(len(levels))
    width = 0.35
    
    # 绘制每个指标的柱状图
    metric_names = list(metrics.keys())
    colors = {'Bridge': 'skyblue', 'Comparison': 'lightcoral'}
    
    for i, metric_name in enumerate(metric_names):
        ax = axes[i]
        
        # Bridge数据
        bridge_means = [metrics[metric_name]['mean'][j] for j in range(0, len(levels))]
        bridge_stds = [metrics[metric_name]['std'][j] for j in range(0, len(levels))]
        
        # Comparison数据
        comp_means = [metrics[metric_name]['mean'][j] for j in range(3, 6)]
        comp_stds = [metrics[metric_name]['std'][j] for j in range(3, 6)]
        
        # 绘制柱状图（带误差线）
        bars1 = ax.bar(x - width/2, bridge_means, width, label='Bridge', 
                      alpha=0.8, color=colors['Bridge'], 
                      yerr=bridge_stds, capsize=5)
        bars2 = ax.bar(x + width/2, comp_means, width, label='Comparison', 
                      alpha=0.8, color=colors['Comparison'], 
                      yerr=comp_stds, capsize=5)
        
        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
        
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Difficulty Level', fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(f'{metric_name}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(levels)
        ax.grid(True, alpha=0.3)
        
        # 只在第一个子图显示图例
        if i == 0:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=True)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # 为顶部图例留出空间
    plt.savefig('hotpotqa_statistics.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    stats = analyze_hotpotqa()
    create_bar_charts(stats)
