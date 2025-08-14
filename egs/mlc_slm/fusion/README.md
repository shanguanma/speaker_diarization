# ASR Speaker Fusion Module

这个模块实现了ASR词级别时间戳与说话人标签的融合算法。

## 算法原理

融合算法按照以下步骤工作：

1. **重叠检测**: 如果词段与至少一个说话人段重叠，则将该词关联到与该词有最大时间重叠的说话人
2. **距离计算**: 如果词段不与任何说话人段重叠，则将该词关联到与该词在段边界上具有最小时间距离的说话人

## 核心类

### WordSegment
表示ASR输出的词段：
- `word`: 词文本
- `start_time`: 开始时间
- `end_time`: 结束时间
- `confidence`: 置信度（可选）

### SpeakerSegment
表示说话人分割输出的说话人段：
- `speaker_id`: 说话人ID
- `start_time`: 开始时间
- `end_time`: 结束时间
- `confidence`: 置信度（可选）

### FusedWord
表示融合后的词：
- `word`: 词文本
- `start_time`: 开始时间
- `end_time`: 结束时间
- `speaker_id`: 关联的说话人ID
- `confidence`: 置信度
- `fusion_method`: 融合方法（'overlap' 或 'distance'）

### ASRSpeakerFusion
主要的融合类，提供以下方法：
- `fuse()`: 主要的融合方法
- `fuse_words_sequential()`: 顺序处理融合
- `fuse_words_parallel()`: 并行处理融合
- `get_fusion_statistics()`: 获取融合统计信息

## 使用方法

### 基本用法

```python
from asr_speaker_fusion import ASRSpeakerFusion, WordSegment, SpeakerSegment

# 创建融合实例
fusion = ASRSpeakerFusion()

# 准备数据
words = [
    WordSegment("Hello", 0.0, 0.5, 0.95),
    WordSegment("world", 0.5, 1.0, 0.92),
]

speakers = [
    SpeakerSegment("speaker_1", 0.0, 1.2, 0.98),
    SpeakerSegment("speaker_2", 1.2, 2.5, 0.97),
]

# 执行融合
fused_words = fusion.fuse(words, speakers)

# 查看结果
for word in fused_words:
    print(f"'{word.word}' -> {word.speaker_id} ({word.fusion_method})")
```

### 并行处理

```python
# 使用线程并行处理
fused_words = fusion.fuse(words, speakers, parallel=True, use_processes=False)

# 使用进程并行处理
fused_words = fusion.fuse(words, speakers, parallel=True, use_processes=True)
```

### 获取统计信息

```python
stats = fusion.get_fusion_statistics(fused_words)
print(f"总词数: {stats['total_words']}")
print(f"重叠融合: {stats['overlap_fused']} ({stats['overlap_percentage']:.1f}%)")
print(f"距离融合: {stats['distance_fused']} ({stats['distance_percentage']:.1f}%)")
```

## 性能特性

- **并行处理**: 支持多线程和多进程并行处理
- **可扩展性**: 自动检测CPU核心数，优化工作进程数量
- **性能监控**: 内置性能计时和日志记录
- **结果验证**: 确保并行处理结果与顺序处理一致

## 运行测试

```bash
cd codebase/speaker_diarization/egs/mlc_slm/fusion
python asr_speaker_fusion.py
```

测试将运行：
1. 基本性能测试（小数据集）
2. 可扩展性测试（大数据集）
3. 并行vs顺序处理性能对比

## 依赖要求

- Python 3.7+
- numpy
- 标准库模块：concurrent.futures, multiprocessing, dataclasses, typing, logging

## 注意事项

1. 时间戳必须按时间顺序排列
2. 词段的开始时间必须小于结束时间
3. 说话人段的开始时间必须小于结束时间
4. 并行处理时，大数据集会有更好的性能提升
5. 进程并行处理适用于CPU密集型任务，线程并行处理适用于I/O密集型任务

## 扩展功能

该模块可以轻松扩展以支持：
- 不同的融合策略
- 置信度加权融合
- 时间窗口滑动融合
- 批量文件处理
- 导出为不同格式（JSON, CSV等）
