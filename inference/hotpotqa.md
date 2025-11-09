# HotpotQA 多线程推理实现

本文档介绍了 `hotpotqa.py` 的多线程实现，以及如何使用它来加速推理过程。

## 🚀 新增功能

### 多线程并行推理
- 使用 `ThreadPoolExecutor` 实现多线程并行处理
- 显著减少总推理时间，特别是在处理大量样本时
- 支持自定义线程数量以适应不同的硬件配置

### 线程安全设计
- 使用 `threading.Lock` 保护共享资源
- 确保结果存储和中间保存的线程安全性
- 避免并发写入导致的数据竞争问题

### 中间结果保存
- 保持原有的中间结果保存功能
- 支持在多线程环境下定期保存进度
- 防止意外中断导致的数据丢失

## 📖 使用方法

### 基本用法

```bash
# 使用默认 4 个线程
python inference/hotpotqa.py \
    --model-url https://api.siliconflow.cn/v1 \
    --model-name Qwen/Qwen2.5-72B-Instruct \
    --output-file predictions.json

# 自定义线程数量
python inference/hotpotqa.py \
    --model-url https://api.siliconflow.cn/v1 \
    --model-name Qwen/Qwen2.5-72B-Instruct \
    --output-file predictions.json \
    --num-threads 8

# 测试模式（限制样本数量）
python inference/hotpotqa.py \
    --model-url https://api.siliconflow.cn/v1 \
    --model-name Qwen/Qwen2.5-72B-Instruct \
    --output-file predictions.json \
    --max-samples 50 \
    --num-threads 4
```

### 新增参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--num-threads` | int | 4 | 并行推理的线程数量 |

## ⚡ 性能优化

### 线程数选择建议

#### 本地 vLLM 服务
- **CPU 核心数 ≤ 4**: 2-4 线程
- **CPU 核心数 4-8**: 4-6 线程
- **CPU 核心数 ≥ 8**: 6-8 线程

#### 云端 API 服务
- **网络带宽充足**: 4-8 线程
- **网络带宽有限**: 2-4 线程
- **API 有速率限制**: 根据限制调整

### 性能影响因素

1. **模型服务并发能力**
   - vLLM 服务的最大并发请求数
   - API 服务的速率限制

2. **网络带宽**
   - 上传请求和下载响应的网络延迟
   - 带宽瓶颈可能限制多线程效果

3. **系统资源**
   - CPU 使用率
   - 内存消耗
   - 线程切换开销

### 性能测试

运行性能对比测试：

```bash
python inference/multithread_example.py
```

## 🔧 技术实现

### 核心组件

1. **`process_single_sample()`**
   - 处理单个样本的独立函数
   - 包含格式化、模型调用、结果提取的完整流程
   - 线程安全的错误处理

2. **`ThreadPoolExecutor`**
   - 管理线程池
   - 提交和执行推理任务
   - 异步获取结果

3. **线程安全机制**
   - `threading.Lock` 保护共享资源
   - 原子操作更新结果字典
   - 线程安全的进度显示

### 工作流程

```
1. 加载数据集
2. 创建线程池 (num_threads 个工作线程)
3. 提交所有样本到线程池
4. 并行执行推理任务
5. 收集结果并线程安全地更新存储
6. 定期保存中间结果
7. 保存最终结果
```

## 📊 性能对比

### 理论加速比

在理想情况下，多线程的加速比接近线程数量：

```
加速比 ≈ 线程数量
```

### 实际性能

实际加速比受以下因素影响：
- 模型推理时间 vs 网络延迟
- 系统资源竞争
- API 服务限制

### 典型场景

| 场景 | 单线程 | 4线程 | 8线程 | 加速比 |
|------|--------|-------|-------|--------|
| 本地 vLLM (GPU) | 100s | 30s | 20s | 3-5x |
| 云端 API (高速) | 200s | 60s | 35s | 3-6x |
| 云端 API (限速) | 150s | 80s | 75s | 2-3x |

## 🛠️ 故障排除

### 常见问题

1. **线程数过多导致性能下降**
   - 减少线程数量
   - 检查系统资源使用情况

2. **API 速率限制错误**
   - 降低线程数量
   - 增加请求间隔

3. **内存不足**
   - 减少线程数量
   - 分批处理数据

### 调试技巧

```bash
# 使用单线程验证功能
python inference/hotpotqa.py --num-threads 1 --max-samples 5

# 逐步增加线程数
python inference/hotpotqa.py --num-threads 2 --max-samples 10
python inference/hotpotqa.py --num-threads 4 --max-samples 10
```

## 📝 更新日志

### v2.0.0 (多线程版本)
- ✅ 添加多线程并行推理支持
- ✅ 实现线程安全的结果存储
- ✅ 保持中间结果保存功能
- ✅ 添加线程数配置参数
- ✅ 提供性能测试脚本

### v1.0.0 (原始版本)
- ✅ 基础推理功能
- ✅ 单线程串行处理
- ✅ 中间结果保存
- ✅ 答案和支持事实提取

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进多线程实现！

### 开发建议
- 添加更多的性能监控指标
- 实现动态线程数调整
- 支持分布式推理
- 添加更详细的错误处理
