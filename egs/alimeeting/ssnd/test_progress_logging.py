#!/usr/bin/env python3
"""
测试spktochunks进度日志功能
"""

import sys
import os
import time
import logging
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_accelerate_ddp import spktochunks_fast, spktochunks_lazy, spktochunks_memory_safe

def setup_test_logging():
    """设置测试日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def test_progress_logging():
    """测试进度日志功能"""
    logger = setup_test_logging()
    
    # 模拟参数
    class MockArgs:
        def __init__(self):
            self.voxceleb2_spk2chunks_json = "/maduo/datasets/voxceleb2/vox2_dev/train.jsonl_gzip"
            self.compression_type = "gzip"
            self.fast_batch_size = 2
            self.fast_sub_batch_size = 10
            self.fast_max_memory_mb = 0
            self.memory_usage_ratio = 0.5
            self.strict_memory_check = False
            self.use_memory_safe = False
            self.use_lazy_loading = False
            self.disable_cache = True
    
    args = MockArgs()
    
    logger.info("=== 测试进度日志功能 ===")
    
    # 测试参数
    max_speakers = 5  # 只处理5个说话人用于测试
    max_files_per_speaker = 3  # 每个说话人只处理3个文件
    
    try:
        logger.info("1. 测试加速版本进度日志...")
        start_time = time.time()
        spk2chunks_fast = spktochunks_fast(
            args, 
            max_speakers=max_speakers, 
            max_files_per_speaker=max_files_per_speaker,
            use_cache=False
        )
        fast_time = time.time() - start_time
        logger.info(f"加速版本完成，处理了 {len(spk2chunks_fast)} 个说话人，耗时: {fast_time:.1f}秒")
        
        logger.info("2. 测试懒加载版本进度日志...")
        start_time = time.time()
        spk2chunks_lazy = spktochunks_lazy(
            args, 
            max_speakers=max_speakers, 
            max_files_per_speaker=max_files_per_speaker
        )
        lazy_time = time.time() - start_time
        logger.info(f"懒加载版本完成，处理了 {len(spk2chunks_lazy)} 个说话人，耗时: {lazy_time:.1f}秒")
        
        logger.info("3. 测试内存安全版本进度日志...")
        start_time = time.time()
        spk2chunks_safe = spktochunks_memory_safe(
            args, 
            max_speakers=max_speakers, 
            max_files_per_speaker=max_files_per_speaker
        )
        safe_time = time.time() - start_time
        logger.info(f"内存安全版本完成，处理了 {len(spk2chunks_safe)} 个说话人，耗时: {safe_time:.1f}秒")
        
        # 比较结果
        logger.info("=== 结果比较 ===")
        logger.info(f"加速版本: {len(spk2chunks_fast)} 个说话人，耗时: {fast_time:.1f}秒")
        logger.info(f"懒加载版本: {len(spk2chunks_lazy)} 个说话人，耗时: {lazy_time:.1f}秒")
        logger.info(f"内存安全版本: {len(spk2chunks_safe)} 个说话人，耗时: {safe_time:.1f}秒")
        
        # 验证数据一致性
        fast_speakers = set(spk2chunks_fast.keys())
        lazy_speakers = set(spk2chunks_lazy.keys())
        safe_speakers = set(spk2chunks_safe.keys())
        
        if fast_speakers == lazy_speakers == safe_speakers:
            logger.info("✓ 所有版本处理的说话人一致")
        else:
            logger.warning("✗ 不同版本处理的说话人不一致")
            logger.warning(f"加速版本: {fast_speakers}")
            logger.warning(f"懒加载版本: {lazy_speakers}")
            logger.warning(f"内存安全版本: {safe_speakers}")
        
        logger.info("=== 进度日志测试完成 ===")
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_progress_logging() 