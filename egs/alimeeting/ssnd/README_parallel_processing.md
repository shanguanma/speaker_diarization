# 并行处理VoxCeleb2数据集

本目录包含了使用 `ThreadPoolExecutor` 进行并行计算的VoxCeleb2数据集处理脚本。

## 文件说明

### 1. `remove_silent_and_get_spk2chunks.py` (并行化版本)
- **功能**: 处理VoxCeleb2数据集，移除静音片段并生成说话人片段信息
- **主要改进**:
  - 使用 `ThreadPoolExecutor` 并行处理音频文件
  - 添加了进度条显示 (`tqdm`)
  - 增加了错误处理和日志记录
  - 支持自定义线程数
  - 添加了详细的处理统计信息

### 2. `performance_comparison.py`
- **功能**: 性能对比脚本，比较串行和并行处理的性能差异
- **输出**: 详细的性能报告和JSON格式的结果文件

### 3. `run_parallel_processing.sh`
- **功能**: 运行并行处理脚本的示例shell脚本

## 使用方法

### 基本使用

```bash
# 使用默认参数运行并行处理
python remove_silent_and_get_spk2chunks.py \
    --voxceleb2-dataset-dir /path/to/voxceleb2/dataset \
    --out-text /path/to/output/train.json

# 指定线程数
python remove_silent_and_get_spk2chunks.py \
    --voxceleb2-dataset-dir /path/to/voxceleb2/dataset \
    --out-text /path/to/output/train.json \
    --max-workers 32
```

### 性能对比

```bash
# 运行性能对比测试（处理100个文件）
python performance_comparison.py \
    --voxceleb2-dataset-dir /path/to/voxceleb2/dataset \
    --max-files 100 \
    --max-workers 16

# 处理更多文件进行更全面的测试
python performance_comparison.py \
    --voxceleb2-dataset-dir /path/to/voxceleb2/dataset \
    --max-files 1000 \
    --max-workers 32
```

### 使用shell脚本

```bash
# 给脚本执行权限
chmod +x run_parallel_processing.sh

# 运行脚本
./run_parallel_processing.sh
```

## 主要特性

### 1. 并行处理
- 使用 `ThreadPoolExecutor` 实现多线程并行处理
- 自动检测CPU核心数并设置合适的线程数
- 支持自定义最大线程数

### 2. 进度监控
- 使用 `tqdm` 显示实时处理进度
- 详细的日志输出，包括处理统计信息
- 错误文件统计和报告

### 3. 错误处理
- 完善的异常处理机制
- 单个文件处理失败不影响整体进度
- 详细的错误日志记录

### 4. 性能优化
- 智能的线程数设置
- 内存友好的处理方式
- 支持大规模数据集处理

## 性能提升

根据测试结果，并行处理相比串行处理通常可以获得：

- **2-8倍的速度提升**（取决于CPU核心数和I/O性能）
- **更好的资源利用率**
- **更稳定的处理性能**

## 系统要求

- Python 3.7+
- 依赖包：
  ```bash
  pip install funasr librosa soundfile tqdm numpy
  ```

## 注意事项

1. **内存使用**: 并行处理会增加内存使用量，请确保系统有足够的内存
2. **I/O瓶颈**: 如果存储设备I/O性能较低，并行处理的提升可能有限
3. **线程数设置**: 建议线程数设置为CPU核心数的1-2倍
4. **错误处理**: 处理失败的文件会被记录但不会中断整体处理流程

## 故障排除

### 常见问题

1. **内存不足**
   - 减少 `--max-workers` 参数值
   - 分批处理数据集

2. **处理速度慢**
   - 检查存储设备I/O性能
   - 增加线程数（但不要超过CPU核心数的4倍）

3. **文件读取错误**
   - 检查音频文件路径是否正确
   - 确认音频文件格式是否支持

### 日志分析

脚本会输出详细的日志信息，包括：
- 处理进度
- 成功/失败文件统计
- 性能指标
- 错误详情

## 扩展功能

可以根据需要进一步扩展功能：

1. **GPU加速**: 对于VAD模型，可以考虑使用GPU加速
2. **分布式处理**: 对于超大规模数据集，可以考虑分布式处理
3. **缓存机制**: 添加处理结果缓存，避免重复处理
4. **断点续传**: 支持中断后继续处理的功能 