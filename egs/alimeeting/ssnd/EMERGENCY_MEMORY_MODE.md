# 紧急内存管理模式

## 问题描述

如果仍然遇到内存不足的问题，可以使用以下紧急模式来确保训练能够进行。

## 紧急模式配置

### 1. 极保守配置（推荐用于内存严重不足的情况）

```bash
python train_accelerate_ddp.py \
    --memory-usage-ratio 0.3 \
    --fast-batch-size 1 \
    --fast-sub-batch-size 5 \
    --strict-memory-check True \
    --use-memory-safe True
```

### 2. 保守配置（适合内存受限的机器）

```bash
python train_accelerate_ddp.py \
    --memory-usage-ratio 0.4 \
    --fast-batch-size 1 \
    --fast-sub-batch-size 10 \
    --strict-memory-check True
```

### 3. 手动指定内存限制

```bash
python train_accelerate_ddp.py \
    --fast-max-memory-mb 8192 \
    --fast-batch-size 1 \
    --fast-sub-batch-size 10 \
    --strict-memory-check True
```

## 内存优化技巧

### 1. 系统级优化

```bash
# 清理系统缓存
sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches

# 检查内存使用
free -h

# 监控内存使用
watch -n 1 'free -h'
```

### 2. 进程级优化

```bash
# 限制进程内存使用
ulimit -v 8388608  # 8GB虚拟内存限制

# 使用cgroups限制内存
echo $$ > /sys/fs/cgroup/memory/mygroup/tasks
echo 8G > /sys/fs/cgroup/memory/mygroup/memory.limit_in_bytes
```

### 3. Python级优化

```python
# 在代码开头添加
import gc
import sys

# 设置垃圾回收阈值
gc.set_threshold(100, 5, 5)  # 更频繁的垃圾回收

# 禁用引用计数优化
sys.setcheckinterval(1)
```

## 故障排除

### 1. 如果仍然出现内存不足

1. **减少数据量**：
   ```bash
   --max-speakers-test 10  # 只处理10个说话人
   --max-files-per-speaker-test 5  # 每个说话人只处理5个文件
   ```

2. **使用更小的音频文件**：
   - 降低采样率到8kHz
   - 缩短音频长度
   - 使用更激进的VAD

3. **分批处理**：
   - 将数据分成多个小批次
   - 分别处理每个批次
   - 最后合并结果

### 2. 监控内存使用

```bash
# 实时监控内存使用
watch -n 1 'ps aux | grep python | grep train_accelerate_ddp'

# 监控系统内存
watch -n 1 'free -h && echo "---" && ps aux | grep python | head -5'
```

### 3. 日志分析

查看日志中的内存使用模式：
- 内存使用是否持续增长
- 垃圾回收是否有效
- 哪些操作消耗最多内存

## 性能权衡

使用紧急模式会带来以下性能影响：

1. **处理速度变慢**：更小的批次和更频繁的垃圾回收
2. **数据丢失风险**：严格的内存检查可能导致部分数据被跳过
3. **训练效果**：数据量减少可能影响模型性能

## 建议

1. **优先使用自动内存管理**：让系统自动检测和调整
2. **逐步调整参数**：从保守配置开始，逐步增加
3. **监控系统资源**：实时监控内存、CPU使用情况
4. **考虑硬件升级**：如果经常遇到内存问题，考虑增加内存 