# 评估流程说明

本目录包含推理结果的评估和对比工具。

## 快速开始

### 1. 运行推理

```bash
bash scripts/infer/infer.sh
```

推理脚本会自动运行所有配置的数据集，结果保存在 `results/infer/` 目录下，每个样本一个 JSON 文件。

如需自定义参数，编辑 `scripts/infer/infer.sh` 中的配置。

### 2. 评估单个数据集结果

```bash
python -m copypastelrm.eval.eval_folder results/infer/test/Qwen2.5-7B-Instruct/.../timestamp_folder
```

输出三个指标：Answer EM、Answer F1、Supporting F1

### 2.5 批量评估多个数据集

```bash
python scripts/eval/batch_eval_compare.py --base-path results/infer/test/Qwen2.5-7B-Instruct/resamples_1000
```

自动评估指定路径下所有数据集的结果文件夹。

### 3. 对比两个方法的结果

```bash
python scripts/eval/compare_two_folders.py \
    results/infer/test/Qwen2.5-7B-Instruct/resamples_1000/seed_42/tpr_0.0/prompt_rag \
    results/infer/test/Qwen2.5-7B-Instruct/resamples_1000/seed_42/tpr_0.0/prompt_rag_rep_q \
    --baseline-name "cq" --compare-name "cqq"
```

输出对比表格，按 Δ EM 降序排序。

## 脚本说明

| 脚本 | 功能 |
|------|------|
| `copypastelrm/eval/eval_folder.py` | 评估单个结果文件夹 |
| `scripts/eval/compare_two_folders.py` | 对比两个方法的结果 |
| `scripts/eval/batch_eval_compare.py` | 批量评估多个结果文件夹 |

## 对比表格输出示例

| Dataset | RAG EM | RAG×2 EM | Δ EM | RAG F1 | RAG×2 F1 | Δ F1 |
|---------|--------|----------|------|--------|----------|------|
| PopQA | 0.3150 | 0.4120 | **+30.8%** | 0.3650 | 0.4659 | +27.7% |
| 2WikiMultiHopQA | 0.4230 | 0.5090 | **+20.3%** | 0.5052 | 0.5894 | +16.7% |
| **平均** | **0.4653** | **0.4852** | **+4.3%** | **0.5579** | **0.5805** | **+4.1%** |

## 常见参数

### compare_two_folders.py

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `baseline_folder` | 基线方法文件夹路径（必选） | - |
| `compare_folder` | 对比方法文件夹路径（必选） | - |
| `--baseline-name` | 基线方法显示名称 | "Baseline" |
| `--compare-name` | 对比方法显示名称 | "Method B" |
| `--output` | 输出文件路径 | 打印到终端 |
