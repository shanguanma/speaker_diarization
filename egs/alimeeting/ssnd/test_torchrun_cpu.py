#!/usr/bin/env python3
"""
CPU版本的torchrun测试
专门用于单GPU或CPU环境
"""

import os
import sys
import json
import argparse
import logging
from collections import defaultdict
import torch
import torch.distributed as dist
import numpy as np
import soundfile as sf
import librosa
from funasr import AutoModel
from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_data():
    """创建测试数据"""
    # 创建一些测试音频文件
    test_files = []
    for i in range(10):
        filename = f"test_audio_{i}.wav"
        # 生成1秒的测试音频
        t = np.linspace(0, 1, 16000, False)
        audio = 0.5 * np.sin(2 * np.pi * 1000 * t)
        sf.write(filename, audio, 16000)
        test_files.append(filename)
    
    return test_files

def process_single_file(wav_path, rank):
    """处理单个文件"""
    try:
        # 读取音频
        wav, sr = sf.read(wav_path)
        
        # 初始化VAD模型（强制使用CPU）
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 禁用GPU
        vad_model = AutoModel(model="fsmn-vad", model_revision="v2.0.4", disable_update=True)
        
        # 执行VAD检测
        if wav.dtype != np.int16:
            wav_int16 = (wav * 32767).astype(np.int16)
        else:
            wav_int16 = wav
        
        result = vad_model.generate(wav_int16, fs=sr)
        time_stamp = result[0]['value']
        
        return {
            'file': wav_path,
            'rank': rank,
            'success': True,
            'timestamps': len(time_stamp)
        }
        
    except Exception as e:
        return {
            'file': wav_path,
            'rank': rank,
            'success': False,
            'error': str(e)
        }

def main():
    parser = argparse.ArgumentParser(description="CPU版本的torchrun测试")
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    
    # 获取分布式环境信息
    rank = int(os.environ.get('RANK', args.local_rank))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    logger.info(f"进程信息: rank={rank}, world_size={world_size}")
    
    # 强制使用CPU
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.cuda.is_available = lambda: False
    
    # 初始化分布式环境（使用gloo后端）
    try:
        dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)
        logger.info(f"Rank {rank}: 分布式环境初始化成功，使用gloo后端")
    except Exception as e:
        logger.error(f"Rank {rank}: 分布式环境初始化失败: {e}")
        return
    
    # 创建测试数据
    if rank == 0:
        test_files = create_test_data()
        logger.info(f"创建了 {len(test_files)} 个测试文件")
    else:
        test_files = []
    
    # 广播测试文件列表
    dist.broadcast_object_list([test_files], src=0)
    test_files = test_files[0]  # 解包
    
    # 分配任务
    my_files = test_files[rank::world_size]
    logger.info(f"Rank {rank}: 分配了 {len(my_files)} 个文件")
    
    # 处理文件
    results = []
    for file_path in my_files:
        result = process_single_file(file_path, rank)
        results.append(result)
    
    # 收集所有结果
    all_results = [None] * world_size
    dist.all_gather_object(all_results, results)
    
    # 只在主进程中输出结果
    if rank == 0:
        logger.info("所有进程处理完成!")
        for i, proc_results in enumerate(all_results):
            logger.info(f"进程 {i}: 处理了 {len(proc_results)} 个文件")
            for result in proc_results:
                if result['success']:
                    logger.info(f"  {result['file']}: 成功，{result['timestamps']} 个时间戳")
                else:
                    logger.error(f"  {result['file']}: 失败 - {result['error']}")
    
    # 清理
    dist.destroy_process_group()
    
    # 清理测试文件
    if rank == 0:
        for file_path in test_files:
            if os.path.exists(file_path):
                os.remove(file_path)

if __name__ == "__main__":
    main() 