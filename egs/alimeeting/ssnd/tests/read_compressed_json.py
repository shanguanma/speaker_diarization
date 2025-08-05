#!/usr/bin/env python3
"""
读取压缩JSON文件的工具脚本
支持多种压缩格式：gzip, bz2, lzma, zstandard
支持两种JSON格式：JSONL和单个JSON对象
"""
import gzip
import json
import argparse
import logging
from collections import defaultdict

# 尝试导入更好的压缩库
try:
    import bz2
    HAS_BZ2 = True
except ImportError:
    HAS_BZ2 = False

try:
    import lzma
    HAS_LZMA = True
except ImportError:
    HAS_LZMA = False

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_format(file_path):
    """自动检测文件格式"""
    if file_path.endswith('.gz'):
        return 'gzip'
    elif file_path.endswith('.bz2'):
        return 'bz2'
    elif file_path.endswith('.xz') or file_path.endswith('.lzma'):
        return 'lzma'
    elif file_path.endswith('.zst'):
        return 'zstd'
    else:
        return 'gzip'  # 默认假设是gzip

def read_jsonl_file(file_path, compression_type=None):
    """读取JSONL格式文件（每行一个JSON对象）"""
    if compression_type is None:
        compression_type = detect_format(file_path)
    
    results = []
    
    if compression_type == "gzip":
        with gzip.open(file_path, "rt", encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # 跳过空行
                    try:
                        data = json.loads(line)
                        results.append(data)
                    except json.JSONDecodeError as e:
                        logger.warning(f"第{line_num}行JSON解析失败: {e}")
    
    elif compression_type == "bz2" and HAS_BZ2:
        with bz2.open(file_path, "rt", encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        results.append(data)
                    except json.JSONDecodeError as e:
                        logger.warning(f"第{line_num}行JSON解析失败: {e}")
    
    elif compression_type == "lzma" and HAS_LZMA:
        with lzma.open(file_path, "rt", encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        results.append(data)
                    except json.JSONDecodeError as e:
                        logger.warning(f"第{line_num}行JSON解析失败: {e}")
    
    elif compression_type == "zstd" and HAS_ZSTD:
        dctx = zstd.ZstdDecompressor()
        with open(file_path, "rb") as f:
            compressed_data = f.read()
            decompressed_data = dctx.decompress(compressed_data).decode('utf-8')
            for line_num, line in enumerate(decompressed_data.split('\n'), 1):
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        results.append(data)
                    except json.JSONDecodeError as e:
                        logger.warning(f"第{line_num}行JSON解析失败: {e}")
    
    else:
        raise ValueError(f"不支持的压缩类型: {compression_type}")
    
    return results

def read_single_json_file(file_path, compression_type=None):
    """读取单个JSON对象文件"""
    if compression_type is None:
        compression_type = detect_format(file_path)
    
    if compression_type == "gzip":
        with gzip.open(file_path, "rt", encoding='utf-8') as f:
            return json.load(f)
    
    elif compression_type == "bz2" and HAS_BZ2:
        with bz2.open(file_path, "rt", encoding='utf-8') as f:
            return json.load(f)
    
    elif compression_type == "lzma" and HAS_LZMA:
        with lzma.open(file_path, "rt", encoding='utf-8') as f:
            return json.load(f)
    
    elif compression_type == "zstd" and HAS_ZSTD:
        dctx = zstd.ZstdDecompressor()
        with open(file_path, "rb") as f:
            compressed_data = f.read()
            decompressed_data = dctx.decompress(compressed_data).decode('utf-8')
            return json.loads(decompressed_data)
    
    else:
        raise ValueError(f"不支持的压缩类型: {compression_type}")

def auto_detect_and_read(file_path):
    """自动检测格式并读取文件"""
    logger.info(f"读取文件: {file_path}")
    
    # 首先尝试作为单个JSON对象读取
    try:
        data = read_single_json_file(file_path)
        logger.info("检测到单个JSON对象格式")
        return data, "single_json"
    except (json.JSONDecodeError, UnicodeDecodeError):
        logger.info("尝试作为JSONL格式读取...")
    
    # 如果失败，尝试作为JSONL格式读取
    try:
        data = read_jsonl_file(file_path)
        logger.info("检测到JSONL格式")
        return data, "jsonl"
    except Exception as e:
        logger.error(f"读取文件失败: {e}")
        raise

def analyze_data(data, format_type):
    """分析数据内容"""
    if format_type == "single_json":
        logger.info(f"单个JSON对象包含 {len(data)} 个说话人")
        total_files = sum(len(spk_data['wav_paths']) for spk_data in data.values())
        logger.info(f"总文件数: {total_files}")
        
        # 显示前几个说话人的信息
        for i, (spk_id, spk_data) in enumerate(data.items()):
            if i >= 3:  # 只显示前3个
                break
            logger.info(f"说话人 {spk_id}: {len(spk_data['wav_paths'])} 个文件")
    
    elif format_type == "jsonl":
        logger.info(f"JSONL格式包含 {len(data)} 行数据")
        
        # 统计说话人数量
        spk_ids = set(item['spk_id'] for item in data)
        logger.info(f"包含 {len(spk_ids)} 个说话人")
        
        # 统计总文件数
        total_files = sum(len(item['wav_paths']) for item in data)
        logger.info(f"总文件数: {total_files}")
        
        # 显示前几行数据的信息
        for i, item in enumerate(data[:3]):
            logger.info(f"第{i+1}行 - 说话人 {item['spk_id']}: {len(item['wav_paths'])} 个文件")

def main():
    parser = argparse.ArgumentParser(description="读取压缩JSON文件")
    parser.add_argument("file_path", type=str, help="要读取的文件路径")
    parser.add_argument("--format", type=str, choices=["auto", "jsonl", "single_json"], 
                       default="auto", help="文件格式（auto为自动检测）")
    parser.add_argument("--compression", type=str, 
                       choices=["auto", "gzip", "bz2", "lzma", "zstd"], 
                       default="auto", help="压缩类型（auto为自动检测）")
    parser.add_argument("--sample", type=int, default=0, 
                       help="显示前N个样本（0表示不显示）")
    
    args = parser.parse_args()
    
    logger.info(f"可用压缩库: bz2={HAS_BZ2}, lzma={HAS_LZMA}, zstd={HAS_ZSTD}")
    
    try:
        if args.format == "auto":
            data, format_type = auto_detect_and_read(args.file_path)
        elif args.format == "jsonl":
            compression_type = None if args.compression == "auto" else args.compression
            data = read_jsonl_file(args.file_path, compression_type)
            format_type = "jsonl"
        elif args.format == "single_json":
            compression_type = None if args.compression == "auto" else args.compression
            data = read_single_json_file(args.file_path, compression_type)
            format_type = "single_json"
        
        # 分析数据
        analyze_data(data, format_type)
        
        # 显示样本数据
        if args.sample > 0:
            logger.info(f"\n显示前 {args.sample} 个样本:")
            if format_type == "single_json":
                for i, (spk_id, spk_data) in enumerate(data.items()):
                    if i >= args.sample:
                        break
                    print(f"\n说话人 {spk_id}:")
                    print(f"  文件数: {len(spk_data['wav_paths'])}")
                    print(f"  前3个文件: {spk_data['wav_paths'][:3]}")
                    print(f"  时间戳数量: {len(spk_data['results'])}")
            else:  # jsonl
                for i, item in enumerate(data[:args.sample]):
                    print(f"\n第{i+1}行 - 说话人 {item['spk_id']}:")
                    print(f"  文件数: {len(item['wav_paths'])}")
                    print(f"  前3个文件: {item['wav_paths'][:3]}")
                    print(f"  时间戳数量: {len(item['results'])}")
    
    except Exception as e:
        logger.error(f"读取文件失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 