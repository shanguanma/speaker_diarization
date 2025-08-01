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
3. **线程数设置**: 建议线程数设置为CPU核心数的1-2倍，**不要超过32个线程**
4. **错误处理**: 处理失败的文件会被记录但不会中断整体处理流程
5. **VAD模型限制**: VAD模型对音频长度和格式有特定要求，过短或过长的音频会被跳过

## 常见错误及解决方案

### 1. VAD处理失败: index out of bounds

**错误信息**: `ERROR:__main__:VAD处理失败: index 480444 is out of bounds for dimension 0 with size 179684`

**原因**: 
- 音频数据长度与VAD模型期望的输入长度不匹配
- 音频数据包含无效值（NaN、无穷大）
- 线程数设置过高导致资源竞争

**解决方案**:
1. **降低线程数**: 将 `--max-workers` 设置为16-24之间
2. **检查音频文件**: 确保音频文件完整且格式正确
3. **分批处理**: 如果数据集很大，考虑分批处理

**推荐设置**:
```bash
# 推荐使用16-24个线程
python remove_silent_and_get_spk2chunks.py \
    --voxceleb2-dataset-dir /path/to/dataset \
    --out-text /path/to/output.json \
    --max-workers 16
```

### 2. VAD处理失败: Sizes of tensors must match

**错误信息**: `ERROR:__main__:VAD处理失败: Sizes of tensors must match except in dimension 1. Expected size 4 but got size 1 for tensor number 1 in the list.`

**原因**: 
- VAD模型在处理音频时遇到张量尺寸不匹配问题
- 音频数据格式与模型期望的输入格式不一致
- 并发处理导致模型状态混乱

**解决方案**:
1. **使用串行处理**: 添加 `--serial` 参数避免所有并发问题
2. **使用进程池**: 添加 `--use-process-pool` 参数避免线程冲突
3. **降低并发度**: 使用较少的线程数

**推荐设置**:
```bash
# 最安全的串行处理
python remove_silent_and_get_spk2chunks.py \
    --voxceleb2-dataset-dir /path/to/dataset \
    --out-text /path/to/output.json \
    --serial

# 或使用进程池避免线程冲突
python remove_silent_and_get_spk2chunks.py \
    --voxceleb2-dataset-dir /path/to/dataset \
    --out-text /path/to/output.json \
    --max-workers 8 \
    --use-process-pool
```

### 2. 内存不足

**解决方案**:
- 减少线程数
- 分批处理数据集
- 增加系统内存

### 3. 音频文件读取失败

**解决方案**:
- 检查音频文件路径是否正确
- 确认音频文件格式是否支持
- 检查文件权限

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

## torchrun多进程分布式处理

为了解决GPU线程冲突问题，我们提供了基于`torchrun`的多进程分布式处理方案。

### 文件说明

- `remove_silent_and_get_spk2chunks_torchrun.py`: 基于torchrun的简化多进程版本（推荐）
- `torchrun_remove_silent_ref.py`: 参考文件，展示了简化的torchrun模式
- `run_torchrun.sh`: 自动检测GPU数量的运行脚本
- `run_torchrun_flexible.sh`: 支持自定义进程数的运行脚本
- `test_torchrun_simple.py`: 简化的torchrun测试版本
- `test_torchrun_basic.sh`: 测试torchrun基本功能的脚本
- `test_torchrun_simple_new.sh`: 测试新的简化torchrun版本的脚本

### 使用方法

#### 1. 基本使用（推荐）
```bash
# 测试新的简化版本
chmod +x test_torchrun_simple_new.sh
./test_torchrun_simple_new.sh

# 或者直接运行
torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    remove_silent_and_get_spk2chunks_torchrun.py \
    --voxceleb2-dataset-dir /path/to/dataset \
    --out-text /path/to/output.json
```

#### 2. 自动检测GPU数量
```bash
# 自动检测GPU数量
chmod +x run_torchrun.sh
./run_torchrun.sh
```

#### 2. 自定义进程数
```bash
# 交互式选择进程数
chmod +x run_torchrun_flexible.sh
./run_torchrun_flexible.sh
```

#### 3. 测试功能
```bash
# 先测试基本功能
chmod +x test_torchrun_basic.sh
./test_torchrun_basic.sh
```

#### 4. 手动运行
```bash
# 使用2个进程
torchrun \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    remove_silent_and_get_spk2chunks_torchrun.py \
    --voxceleb2-dataset-dir /path/to/dataset \
    --out-text /path/to/output.json
```

### 优势

1. **完全避免线程冲突**: 每个进程独立运行，不存在线程竞争
2. **GPU资源隔离**: 每个进程使用独立的GPU资源
3. **更好的稳定性**: 避免了Python线程池的复杂性
4. **原生PyTorch支持**: 使用PyTorch官方的分布式框架
5. **简化的架构**: 参考成熟的torchrun模式，代码更简洁可靠
6. **文件级并行**: 每个进程处理不同的文件，避免复杂的分布式通信

### 注意事项

1. **内存使用**: 每个进程都会加载独立的VAD模型，内存使用量会增加
2. **进程数限制**: 建议进程数不超过GPU数量
3. **端口冲突**: 确保使用的端口没有被其他程序占用
4. **环境要求**: 需要正确安装PyTorch和NCCL（GPU模式） 