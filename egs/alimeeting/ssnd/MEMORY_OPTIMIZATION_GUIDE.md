# 内存优化指南

## 问题描述

在训练过程中，由于音频数据量巨大，经常出现内存不足的问题，导致大量数据被跳过，影响训练效果。

## 解决方案

### 1. 新的内存管理参数

我们添加了以下参数来更好地控制内存使用：

- `--fast-max-memory-mb`: 最大内存限制（MB），默认0表示自动检测
- `--memory-usage-ratio`: 自动内存检测时使用的内存比例（0.0-1.0），默认0.6表示60%
- `--fast-batch-size`: 批处理大小（说话人数量），默认2
- `--fast-sub-batch-size`: 子批次大小（音频文件数量），默认20
- `--strict-memory-check`: 是否启用严格的内存检查，默认False

### 2. 智能内存管理

现在系统会自动检测可用内存并使用60%来处理任务，无需手动设置内存限制：

```bash
# 自动内存管理（推荐）
python train_accelerate_ddp.py \
    --fast-batch-size 5 \
    --fast-sub-batch-size 50 \
    --strict-memory-check False
```

系统会自动显示类似以下的日志：
```
INFO [train_accelerate_ddp.py:1408] 自动检测系统内存: 总计 32768 MB, 可用 24576 MB
INFO [train_accelerate_ddp.py:1412] 自动设置内存限制: 14745 MB (可用内存的60%)
```

#### 如果需要手动指定内存限制：
```bash
python train_accelerate_ddp.py \
    --fast-max-memory-mb 16384 \
    --fast-batch-size 5 \
    --fast-sub-batch-size 50 \
    --strict-memory-check False
```

### 3. 参数说明

#### `--fast-max-memory-mb`
- 设置内存使用上限
- 默认0表示自动检测并使用可用内存的指定百分比
- 如果手动设置，建议设置为总内存的60-80%
- 例如：16GB机器设置为10240MB（10GB）

#### `--memory-usage-ratio`
- 自动内存检测时使用的内存比例
- 范围：0.1-0.9（10%-90%）
- 默认：0.6（60%）
- 建议：0.5-0.7之间

#### `--fast-batch-size`
- 控制每次处理多少个说话人
- 较小的值可以减少内存峰值
- 建议从3-5开始调整

#### `--fast-sub-batch-size`
- 控制每个子批次处理多少个音频文件
- 较小的值可以更好地控制内存使用
- 建议从30-50开始调整

#### `--strict-memory-check`
- `False`（推荐）：即使内存超限也继续处理，只是发出警告
- `True`：内存超限时跳过处理，可能导致数据丢失

### 4. 进度监控和日志

#### 总体进度显示
系统会显示详细的处理进度，包括：
- 已处理说话人数量和总进度百分比
- 已用时间和预计剩余时间
- 当前批次和子批次的处理进度
- 实时内存使用情况

#### 进度日志示例
```
INFO [train_accelerate_ddp.py:1420] 总共需要处理 5994 个说话人
INFO [train_accelerate_ddp.py:1450] 进度: 10/5994 (0.2%) - 已用时间: 0.1分钟 - 预计剩余: 45.2分钟
INFO [train_accelerate_ddp.py:1521] 处理批次: 2 个说话人，156 个音频文件
INFO [train_accelerate_ddp.py:1529] 将 156 个任务分成 8 个子批次处理
INFO [train_accelerate_ddp.py:1577] 处理子批次 1/8: 20 个音频文件
INFO [train_accelerate_ddp.py:1599] 子批次 1/8 完成，耗时: 5.2秒，当前内存: 1845.3 MB
INFO [train_accelerate_ddp.py:1605] 批次进度: 5/8 (62.5%) - 预计剩余: 2.1分钟
```

#### 内存使用监控
在训练过程中，注意观察日志中的内存使用情况：

```
INFO [train_accelerate_ddp.py:1427] 处理批次: 3 个说话人，150 个音频文件
INFO [train_accelerate_ddp.py:1429] 将 150 个任务分成 3 个子批次处理
INFO [train_accelerate_ddp.py:1450] 处理子批次 1/3: 50 个音频文件
INFO [train_accelerate_ddp.py:1470] 子批次 1 完成，当前内存: 8192.5 MB
```

### 5. 故障排除

#### 如果仍然出现内存不足：
1. 减少 `--fast-batch-size`
2. 减少 `--fast-sub-batch-size`
3. 增加 `--fast-max-memory-mb`（如果机器有更多内存）

#### 如果处理速度太慢：
1. 增加 `--fast-batch-size`
2. 增加 `--fast-sub-batch-size`
3. 确保 `--strict-memory-check False`

### 6. 性能优化建议

1. **使用缓存**：首次运行会较慢，后续运行会使用缓存
2. **监控内存**：观察日志中的内存使用情况
3. **逐步调整**：从保守的参数开始，逐步增加直到找到最佳平衡点

### 7. 示例命令

```bash
# 自动内存管理（推荐，适合所有机器）
python train_accelerate_ddp.py \
    --fast-batch-size 2 \
    --fast-sub-batch-size 20 \
    --strict-memory-check False

# 使用70%内存（适合大内存机器）
python train_accelerate_ddp.py \
    --memory-usage-ratio 0.7 \
    --fast-batch-size 3 \
    --fast-sub-batch-size 30 \
    --strict-memory-check False

# 使用50%内存（适合内存受限的机器）
python train_accelerate_ddp.py \
    --memory-usage-ratio 0.5 \
    --fast-batch-size 1 \
    --fast-sub-batch-size 10 \
    --strict-memory-check False

# 保守配置（适合内存较小的机器）
python train_accelerate_ddp.py \
    --fast-max-memory-mb 8192 \
    --fast-batch-size 2 \
    --fast-sub-batch-size 20 \
    --strict-memory-check False

# 平衡配置（适合中等内存机器）
python train_accelerate_ddp.py \
    --fast-max-memory-mb 16384 \
    --fast-batch-size 5 \
    --fast-sub-batch-size 50 \
    --strict-memory-check False

# 激进配置（适合大内存机器）
python train_accelerate_ddp.py \
    --fast-max-memory-mb 32768 \
    --fast-batch-size 10 \
    --fast-sub-batch-size 100 \
    --strict-memory-check False
``` 