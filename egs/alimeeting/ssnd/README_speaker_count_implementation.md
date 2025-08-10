# SimuDiarMixer 说话人计数功能实现说明

## 概述

本文档说明了在 `SimuDiarMixer` 类中实现的 `_get_speaker_count()` 方法及其相关功能。该方法用于在懒加载模式下从VAD文件中获取说话人总数。

## 新增方法

### 1. `_get_speaker_count() -> int`

**功能**: 从VAD文件中获取说话人总数

**返回值**: 说话人数量

**使用场景**: 仅在懒加载模式下可用，用于计算数据集大小和限制说话人选择范围

**实现原理**: 
- 首先检查是否已加载说话人列表
- 如果未加载，则调用 `_load_speaker_list()` 方法
- 返回说话人列表的长度

### 2. `_load_speaker_list() -> List[str]`

**功能**: 从VAD文件中加载说话人列表

**返回值**: 说话人ID列表

**支持格式**: 
- 普通JSON文本文件
- Gzip压缩的JSON文件 (.json.gz)

**实现特点**:
- 逐行解析JSON数据
- 提取每行的 `spk_id` 字段
- 错误处理：跳过无效的JSON行
- 支持压缩文件自动检测

### 3. `_load_speaker_data(spk_idx: int) -> List[np.ndarray]`

**功能**: 加载指定索引的说话人数据

**参数**: 
- `spk_idx`: 说话人在列表中的索引

**返回值**: 音频片段列表

**实现特点**:
- 缓存机制：避免重复加载同一说话人数据
- 音频处理：自动重采样到目标采样率
- VAD片段提取：根据时间戳提取有效音频片段
- 内存管理：及时释放音频数据

### 4. `clear_cache()`

**功能**: 清理说话人数据缓存，释放内存

**使用场景**: 在内存受限的环境中，可以主动清理缓存

### 5. `get_cache_info() -> Dict[str, int]`

**功能**: 获取缓存信息

**返回值**: 包含缓存状态的字典
- `cached_speakers`: 已缓存的说话人数量
- `total_speakers`: 总说话人数量
- `lazy_mode`: 是否为懒加载模式

## VAD文件格式

VAD文件应该包含以下JSON格式的数据，每行一个说话人：

```json
{"spk_id": "speaker_id", "wav_paths": ["path1", "path2"], "results": [[[start1, end1], [start2, end2]]]}
```

**字段说明**:
- `spk_id`: 说话人唯一标识符
- `wav_paths`: 音频文件路径列表
- `results`: VAD结果，包含每个音频文件的时间戳列表

## 使用示例

### 基本用法

```python
from simu_diar_dataset import SimuDiarMixer

# 创建懒加载模式的实例
mixer = SimuDiarMixer(
    voxceleb2_spk2chunks_json="/path/to/vad_data.json",
    sample_rate=16000,
    max_mix_len=8.0,
    min_speakers=2,
    max_speakers=3
)

# 获取说话人总数
speaker_count = mixer._get_speaker_count()
print(f"总说话人数量: {speaker_count}")

# 获取缓存信息
cache_info = mixer.get_cache_info()
print(f"缓存信息: {cache_info}")
```

### 缓存管理

```python
# 清理缓存以释放内存
mixer.clear_cache()

# 检查缓存状态
cache_info = mixer.get_cache_info()
print(f"清理后缓存信息: {cache_info}")
```

### 错误处理

```python
try:
    # 在非懒加载模式下调用会抛出错误
    mixer = SimuDiarMixer(spk2chunks={})
    speaker_count = mixer._get_speaker_count()
except ValueError as e:
    print(f"错误: {e}")
```

## 性能优化

### 1. 懒加载机制
- 只在需要时加载说话人数据
- 避免一次性加载所有数据到内存

### 2. 缓存策略
- 说话人列表缓存：避免重复解析VAD文件
- 音频数据缓存：避免重复加载同一说话人的音频

### 3. 内存管理
- 及时释放音频数据
- 提供缓存清理接口
- 支持压缩文件以减少存储空间

## 注意事项

1. **模式限制**: `_get_speaker_count()` 方法仅在懒加载模式下可用
2. **文件格式**: 支持普通JSON和Gzip压缩格式
3. **错误处理**: 自动跳过无效的JSON行和音频文件
4. **内存使用**: 建议在内存受限的环境中定期调用 `clear_cache()`
5. **文件路径**: 确保VAD文件中引用的音频文件路径正确

## 测试

运行测试脚本验证功能：

```bash
cd /path/to/speaker_diarization/egs/alimeeting/ssnd
python test_simu_diar_dataset.py
```

测试覆盖：
- 说话人计数功能
- 说话人列表加载
- 缓存管理
- 错误处理
- 文件格式支持（普通JSON和Gzip）

## 更新日志

- **v1.0**: 初始实现，支持基本的说话人计数功能
- 支持普通JSON和Gzip压缩文件
- 实现缓存机制和内存管理
- 添加完整的错误处理
- 提供缓存管理接口
