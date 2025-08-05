# JSON压缩功能使用说明

## 问题分析

你遇到的错误是因为原代码生成的是JSONL格式（每行一个JSON对象），但你在读取时使用了`json.loads()`，这期望的是单个JSON对象。

## 解决方案

### 1. 修改后的主要功能

- **多种压缩格式支持**：gzip, bzip2, lzma, zstandard
- **两种JSON格式**：JSONL（每行一个对象）和单个JSON对象
- **可调节压缩级别**：1-9级，平衡压缩率和速度
- **自动格式检测**：读取时自动识别文件格式

### 2. 压缩格式对比

| 格式 | 压缩率 | 速度 | 兼容性 | 推荐场景 |
|------|--------|------|--------|----------|
| gzip | 中等 | 快 | 最好 | 通用场景 |
| bzip2 | 高 | 中等 | 好 | 高压缩率需求 |
| lzma | 最高 | 慢 | 好 | 最高压缩率 |
| zstandard | 高 | 很快 | 中等 | 现代系统 |

### 3. 使用方法

#### 生成压缩文件

```bash
# 使用默认的JSONL + gzip格式
python remove_silent_and_get_spk2chunks.py \
    --voxceleb2-dataset-dir /path/to/dataset \
    --out-text /path/to/output.json.gz

# 使用zstandard压缩（推荐）
python remove_silent_and_get_spk2chunks.py \
    --voxceleb2-dataset-dir /path/to/dataset \
    --out-text /path/to/output.json.zst \
    --format jsonl_zstd \
    --compression-level 6

# 使用单个JSON对象格式
python remove_silent_and_get_spk2chunks.py \
    --voxceleb2-dataset-dir /path/to/dataset \
    --out-text /path/to/output.json.gz \
    --format json_gzip
```

#### 读取压缩文件

```bash
# 自动检测格式
python read_compressed_json.py /path/to/output.json.gz

# 指定格式读取
python read_compressed_json.py /path/to/output.json.gz --format jsonl

# 显示样本数据
python read_compressed_json.py /path/to/output.json.gz --sample 3
```

#### 性能测试

```bash
# 运行压缩性能测试
python compression_benchmark.py --speakers 100 --files-per-speaker 50

# 测试不同压缩级别
python compression_benchmark.py --compression-level 9
```

### 4. 安装依赖

```bash
# 安装zstandard（推荐）
pip install zstandard

# 安装其他压缩库（可选）
# bzip2 和 lzma 通常已包含在Python标准库中
```

### 5. 读取示例

#### 读取JSONL格式（原格式）

```python
import gzip
import json

# 读取JSONL格式
with gzip.open("train.json.gz", 'rt', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            data = json.loads(line)
            print(f"说话人: {data['spk_id']}, 文件数: {len(data['wav_paths'])}")
```

#### 读取单个JSON对象格式

```python
import gzip
import json

# 读取单个JSON对象
with gzip.open("train.json.gz", 'rt', encoding='utf-8') as f:
    data = json.load(f)
    for spk_id, spk_data in data.items():
        print(f"说话人: {spk_id}, 文件数: {len(spk_data['wav_paths'])}")
```

### 6. 推荐配置

#### 对于大型数据集（推荐）
```bash
--format jsonl_zstd --compression-level 6
```
- **优点**：高压缩率，快速压缩/解压
- **缺点**：需要安装zstandard库

#### 对于兼容性要求高的场景
```bash
--format jsonl_gzip --compression-level 6
```
- **优点**：广泛兼容，无需额外依赖
- **缺点**：压缩率相对较低

#### 对于最高压缩率需求
```bash
--format jsonl_lzma --compression-level 9
```
- **优点**：最高压缩率
- **缺点**：压缩/解压速度较慢

### 7. 文件大小对比

以VoxCeleb2数据集为例，不同格式的文件大小对比：

| 格式 | 原始大小 | 压缩后大小 | 压缩率 |
|------|----------|------------|--------|
| 无压缩 | 1GB | 1GB | 0% |
| gzip | 1GB | ~300MB | ~70% |
| bzip2 | 1GB | ~250MB | ~75% |
| lzma | 1GB | ~200MB | ~80% |
| zstandard | 1GB | ~220MB | ~78% |

### 8. 故障排除

#### 读取错误
如果遇到读取错误，请检查：
1. 文件格式是否正确
2. 压缩类型是否匹配
3. 使用`read_compressed_json.py`进行自动检测

#### 压缩失败
如果压缩失败，请检查：
1. 磁盘空间是否充足
2. 是否有写入权限
3. 依赖库是否正确安装

### 9. 性能优化建议

1. **内存使用**：对于超大数据集，建议使用JSONL格式，可以逐行处理
2. **压缩级别**：平衡压缩率和速度，推荐使用6级
3. **并行处理**：可以考虑分片处理，然后合并结果
4. **存储选择**：SSD存储可以显著提升压缩/解压速度 