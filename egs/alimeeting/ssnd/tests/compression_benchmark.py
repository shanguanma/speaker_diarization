#!/usr/bin/env python3
"""
压缩性能对比脚本
比较不同压缩算法的压缩率和速度
"""
import gzip
import json
import time
import os
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

def create_sample_data(num_speakers=100, files_per_speaker=50):
    """创建示例数据用于测试"""
    logger.info(f"创建示例数据: {num_speakers} 个说话人，每个 {files_per_speaker} 个文件")
    
    sample_data = {}
    for i in range(num_speakers):
        spk_id = f"id{i:06d}"
        wav_paths = [f"/path/to/audio/file_{i}_{j}.wav" for j in range(files_per_speaker)]
        time_stamps = [[[j*1000, (j+1)*1000] for j in range(10)] for _ in range(files_per_speaker)]
        
        sample_data[spk_id] = {
            'wav_paths': wav_paths,
            'results': time_stamps
        }
    
    return sample_data

def benchmark_compression(data, format_type, compression_level, output_dir):
    """测试压缩性能"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 准备数据
    if format_type.startswith("jsonl_"):
        # JSONL格式
        json_str = ""
        for spk_id, spk_data in data.items():
            res = {
                'spk_id': spk_id,
                'wav_paths': spk_data['wav_paths'],
                'results': spk_data['results'],
            }
            json_str += json.dumps(res, ensure_ascii=False, separators=(',', ':')) + "\n"
    else:
        # 单个JSON对象
        json_str = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
    
    original_size = len(json_str.encode('utf-8'))
    output_file = os.path.join(output_dir, f"test_{format_type}.tmp")
    
    # 测试压缩
    start_time = time.time()
    
    try:
        if format_type == "jsonl_gzip":
            with gzip.open(output_file, "wt", encoding='utf-8', compresslevel=compression_level) as f:
                f.write(json_str)
        elif format_type == "jsonl_bz2" and HAS_BZ2:
            with bz2.open(output_file, "wt", encoding='utf-8', compresslevel=compression_level) as f:
                f.write(json_str)
        elif format_type == "jsonl_lzma" and HAS_LZMA:
            with lzma.open(output_file, "wt", encoding='utf-8', preset=compression_level) as f:
                f.write(json_str)
        elif format_type == "jsonl_zstd" and HAS_ZSTD:
            cctx = zstd.ZstdCompressor(level=compression_level)
            compressed_data = cctx.compress(json_str.encode('utf-8'))
            with open(output_file, "wb") as f:
                f.write(compressed_data)
        elif format_type == "json_gzip":
            with gzip.open(output_file, "wt", encoding='utf-8', compresslevel=compression_level) as f:
                f.write(json_str)
        elif format_type == "json_bz2" and HAS_BZ2:
            with bz2.open(output_file, "wt", encoding='utf-8', compresslevel=compression_level) as f:
                f.write(json_str)
        elif format_type == "json_lzma" and HAS_LZMA:
            with lzma.open(output_file, "wt", encoding='utf-8', preset=compression_level) as f:
                f.write(json_str)
        elif format_type == "json_zstd" and HAS_ZSTD:
            cctx = zstd.ZstdCompressor(level=compression_level)
            compressed_data = cctx.compress(json_str.encode('utf-8'))
            with open(output_file, "wb") as f:
                f.write(compressed_data)
        else:
            return None
        
        compression_time = time.time() - start_time
        compressed_size = os.path.getsize(output_file)
        compression_ratio = (1 - compressed_size / original_size) * 100
        
        # 清理临时文件
        os.remove(output_file)
        
        return {
            'format': format_type,
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': compression_ratio,
            'compression_time': compression_time,
            'speed_mbps': (original_size / 1024 / 1024) / compression_time
        }
    
    except Exception as e:
        logger.error(f"压缩测试失败 {format_type}: {e}")
        if os.path.exists(output_file):
            os.remove(output_file)
        return None

def run_benchmark(num_speakers=100, files_per_speaker=50, compression_level=6):
    """运行完整的性能测试"""
    logger.info("开始压缩性能测试...")
    
    # 创建示例数据
    sample_data = create_sample_data(num_speakers, files_per_speaker)
    
    # 测试所有格式
    formats = [
        "jsonl_gzip", "jsonl_bz2", "jsonl_lzma", "jsonl_zstd",
        "json_gzip", "json_bz2", "json_lzma", "json_zstd"
    ]
    
    results = []
    temp_dir = "/tmp/compression_benchmark"
    
    for format_type in formats:
        logger.info(f"测试格式: {format_type}")
        result = benchmark_compression(sample_data, format_type, compression_level, temp_dir)
        if result:
            results.append(result)
            logger.info(f"  {format_type}: {result['compression_ratio']:.1f}% 压缩率, "
                       f"{result['speed_mbps']:.1f} MB/s")
    
    # 清理临时目录
    if os.path.exists(temp_dir):
        os.rmdir(temp_dir)
    
    return results

def print_results(results):
    """打印测试结果"""
    print("\n" + "="*80)
    print("压缩性能测试结果")
    print("="*80)
    
    # 按压缩率排序
    results_by_ratio = sorted(results, key=lambda x: x['compression_ratio'], reverse=True)
    
    print(f"{'格式':<15} {'原始大小(MB)':<12} {'压缩后(MB)':<12} {'压缩率(%)':<10} {'时间(s)':<8} {'速度(MB/s)':<10}")
    print("-" * 80)
    
    for result in results_by_ratio:
        print(f"{result['format']:<15} "
              f"{result['original_size']/1024/1024:<12.2f} "
              f"{result['compressed_size']/1024/1024:<12.2f} "
              f"{result['compression_ratio']:<10.1f} "
              f"{result['compression_time']:<8.3f} "
              f"{result['speed_mbps']:<10.1f}")
    
    print("\n" + "="*80)
    print("推荐方案:")
    print("="*80)
    
    # 最佳压缩率
    best_ratio = results_by_ratio[0]
    print(f"最佳压缩率: {best_ratio['format']} ({best_ratio['compression_ratio']:.1f}%)")
    
    # 最佳速度
    results_by_speed = sorted(results, key=lambda x: x['speed_mbps'], reverse=True)
    best_speed = results_by_speed[0]
    print(f"最佳速度: {best_speed['format']} ({best_speed['speed_mbps']:.1f} MB/s)")
    
    # 平衡方案（压缩率在前50%且速度在前50%）
    mid_ratio = len(results) // 2
    mid_speed = len(results) // 2
    
    balanced = []
    for result in results:
        ratio_rank = sum(1 for r in results if r['compression_ratio'] > result['compression_ratio'])
        speed_rank = sum(1 for r in results if r['speed_mbps'] > result['speed_mbps'])
        if ratio_rank <= mid_ratio and speed_rank <= mid_speed:
            balanced.append((result, ratio_rank + speed_rank))
    
    if balanced:
        balanced.sort(key=lambda x: x[1])
        best_balanced = balanced[0][0]
        print(f"平衡方案: {best_balanced['format']} "
              f"(压缩率: {best_balanced['compression_ratio']:.1f}%, "
              f"速度: {best_balanced['speed_mbps']:.1f} MB/s)")

def main():
    parser = argparse.ArgumentParser(description="压缩性能测试")
    parser.add_argument("--speakers", type=int, default=100, 
                       help="测试用的说话人数量")
    parser.add_argument("--files-per-speaker", type=int, default=50, 
                       help="每个说话人的文件数量")
    parser.add_argument("--compression-level", type=int, default=6, 
                       help="压缩级别 (1-9)")
    
    args = parser.parse_args()
    
    logger.info(f"可用压缩库: bz2={HAS_BZ2}, lzma={HAS_LZMA}, zstd={HAS_ZSTD}")
    
    # 运行测试
    results = run_benchmark(args.speakers, args.files_per_speaker, args.compression_level)
    
    if results:
        print_results(results)
    else:
        logger.error("没有成功的测试结果")

if __name__ == "__main__":
    main() 