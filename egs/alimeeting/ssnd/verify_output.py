#!/usr/bin/env python3
"""
验证输出文件格式和内容的脚本
"""
import os
import sys
import json
import gzip
import argparse
import logging
from collections import defaultdict

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

def verify_jsonl_file(file_path):
    """验证JSONL格式文件"""
    logger.info(f"验证JSONL文件: {file_path}")
    
    if not os.path.exists(file_path):
        logger.error(f"文件不存在: {file_path}")
        return False
    
    try:
        stats = {
            'total_lines': 0,
            'valid_lines': 0,
            'invalid_lines': 0,
            'speakers': set(),
            'total_files': 0,
            'total_timestamps': 0,
            'file_sizes': []
        }
        
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                stats['total_lines'] += 1
                line = line.strip()
                
                if not line:  # 跳过空行
                    continue
                
                try:
                    data = json.loads(line)
                    stats['valid_lines'] += 1
                    
                    # 验证数据结构
                    if 'spk_id' not in data:
                        logger.warning(f"第{line_num}行缺少spk_id字段")
                        continue
                    
                    if 'wav_paths' not in data:
                        logger.warning(f"第{line_num}行缺少wav_paths字段")
                        continue
                    
                    if 'results' not in data:
                        logger.warning(f"第{line_num}行缺少results字段")
                        continue
                    
                    # 统计信息
                    spk_id = data['spk_id']
                    wav_paths = data['wav_paths']
                    results = data['results']
                    
                    stats['speakers'].add(spk_id)
                    stats['total_files'] += len(wav_paths)
                    stats['total_timestamps'] += len(results)
                    
                    # 记录文件大小信息
                    if line_num <= 3:  # 只记录前3行的大小
                        stats['file_sizes'].append(len(line))
                    
                    # 显示前3行的详细信息
                    if line_num <= 3:
                        logger.info(f"第{line_num}行详情:")
                        logger.info(f"  说话人ID: {spk_id}")
                        logger.info(f"  音频文件数: {len(wav_paths)}")
                        logger.info(f"  时间戳数量: {len(results)}")
                        if wav_paths:
                            logger.info(f"  第一个音频文件: {wav_paths[0]}")
                        if results:
                            logger.info(f"  第一个时间戳: {results[0]}")
                        logger.info(f"  行大小: {len(line)} 字符")
                
                except json.JSONDecodeError as e:
                    stats['invalid_lines'] += 1
                    logger.error(f"第{line_num}行JSON解析失败: {e}")
                    logger.error(f"问题行内容: {line[:100]}...")
        
        # 输出统计信息
        logger.info("="*50)
        logger.info("文件验证结果:")
        logger.info("="*50)
        logger.info(f"总行数: {stats['total_lines']}")
        logger.info(f"有效行数: {stats['valid_lines']}")
        logger.info(f"无效行数: {stats['invalid_lines']}")
        logger.info(f"说话人数量: {len(stats['speakers'])}")
        logger.info(f"总文件数: {stats['total_files']}")
        logger.info(f"总时间戳数量: {stats['total_timestamps']}")
        
        if stats['file_sizes']:
            avg_size = sum(stats['file_sizes']) / len(stats['file_sizes'])
            logger.info(f"平均行大小: {avg_size:.1f} 字符")
        
        # 显示前几个说话人
        logger.info("前5个说话人:")
        for i, spk_id in enumerate(sorted(stats['speakers'])[:5]):
            logger.info(f"  {i+1}. {spk_id}")
        
        return stats['invalid_lines'] == 0
        
    except Exception as e:
        logger.error(f"验证文件时发生错误: {e}")
        return False

def verify_single_json_file(file_path):
    """验证单个JSON对象文件"""
    logger.info(f"验证单个JSON文件: {file_path}")
    
    if not os.path.exists(file_path):
        logger.error(f"文件不存在: {file_path}")
        return False
    
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            data = json.load(f)
        
        # 验证数据结构
        if not isinstance(data, dict):
            logger.error("根对象不是字典")
            return False
        
        stats = {
            'speakers': len(data),
            'total_files': 0,
            'total_timestamps': 0
        }
        
        # 统计信息
        for spk_id, spk_data in data.items():
            if not isinstance(spk_data, dict):
                logger.error(f"说话人 {spk_id} 的数据不是字典")
                continue
            
            if 'wav_paths' not in spk_data or 'results' not in spk_data:
                logger.error(f"说话人 {spk_id} 缺少必要字段")
                continue
            
            stats['total_files'] += len(spk_data['wav_paths'])
            stats['total_timestamps'] += len(spk_data['results'])
        
        # 输出统计信息
        logger.info("="*50)
        logger.info("文件验证结果:")
        logger.info("="*50)
        logger.info(f"说话人数量: {stats['speakers']}")
        logger.info(f"总文件数: {stats['total_files']}")
        logger.info(f"总时间戳数量: {stats['total_timestamps']}")
        
        # 显示前几个说话人
        logger.info("前5个说话人:")
        for i, (spk_id, spk_data) in enumerate(list(data.items())[:5]):
            logger.info(f"  {i+1}. {spk_id}: {len(spk_data['wav_paths'])} 个文件")
        
        return True
        
    except Exception as e:
        logger.error(f"验证文件时发生错误: {e}")
        return False

def auto_detect_and_verify(file_path):
    """自动检测格式并验证"""
    logger.info(f"自动检测并验证文件: {file_path}")
    
    # 首先尝试作为单个JSON对象读取
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            data = json.load(f)
        logger.info("检测到单个JSON对象格式")
        return verify_single_json_file(file_path)
    except (json.JSONDecodeError, UnicodeDecodeError):
        logger.info("尝试作为JSONL格式验证...")
    
    # 如果失败，尝试作为JSONL格式读取
    try:
        return verify_jsonl_file(file_path)
    except Exception as e:
        logger.error(f"验证文件失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="验证输出文件格式和内容")
    parser.add_argument("file_path", type=str, help="要验证的文件路径")
    parser.add_argument("--format", type=str, choices=["auto", "jsonl", "single_json"], 
                       default="auto", help="文件格式（auto为自动检测）")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file_path):
        logger.error(f"文件不存在: {args.file_path}")
        return 1
    
    # 获取文件信息
    file_size = os.path.getsize(args.file_path)
    logger.info(f"文件大小: {file_size} 字节 ({file_size/1024/1024:.2f} MB)")
    
    # 验证文件
    if args.format == "auto":
        success = auto_detect_and_verify(args.file_path)
    elif args.format == "jsonl":
        success = verify_jsonl_file(args.file_path)
    elif args.format == "single_json":
        success = verify_single_json_file(args.file_path)
    else:
        logger.error(f"不支持的格式: {args.format}")
        return 1
    
    if success:
        logger.info("文件验证成功!")
        return 0
    else:
        logger.error("文件验证失败!")
        return 1

if __name__ == "__main__":
    exit(main()) 