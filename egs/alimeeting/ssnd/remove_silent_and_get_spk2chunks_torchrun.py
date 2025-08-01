#!/usr/bin/env python3
"""
基于torchrun的多进程分布式VoxCeleb2数据集处理脚本
使用torchrun进行多进程分布式处理，避免GPU线程冲突
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from collections import defaultdict
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
import numpy as np
import soundfile as sf
import librosa
from funasr import AutoModel
from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VoxCeleb2Dataset(Dataset):
    """VoxCeleb2数据集类"""
    def __init__(self, spk2wav, rank=0, world_size=1):
        self.spk2wav = spk2wav
        self.rank = rank
        self.world_size = world_size
        
        # 准备任务列表
        self.tasks = []
        for spk_id, wav_paths in spk2wav.items():
            for wav_path in wav_paths:
                self.tasks.append((wav_path, spk_id))
        
        # 根据rank和world_size分配任务
        self.tasks = self.tasks[rank::world_size]
        
        logger.info(f"Rank {rank}: 分配了 {len(self.tasks)} 个任务")

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        return self.tasks[idx]

class VADProcessor:
    """VAD处理器，每个进程独立初始化"""
    def __init__(self, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self._init_model()
    
    def _init_model(self):
        """初始化VAD模型"""
        try:
            logger.info(f"初始化VAD模型，设备: {self.device}")
            self.model = AutoModel(model="fsmn-vad", model_revision="v2.0.4", disable_update=True)
            logger.info("VAD模型初始化成功")
        except Exception as e:
            logger.error(f"VAD模型初始化失败: {e}")
            self.model = None
    
    def process_audio(self, wav_path, spk_id):
        """处理单个音频文件"""
        try:
            # 检查文件是否存在
            if not os.path.exists(wav_path):
                return {
                    'spk_id': spk_id,
                    'wav_path': wav_path,
                    'time_stamp_list': [],
                    'success': False,
                    'error': f"文件不存在: {wav_path}"
                }
            
            # 检查文件大小
            file_size = os.path.getsize(wav_path)
            if file_size == 0:
                return {
                    'spk_id': spk_id,
                    'wav_path': wav_path,
                    'time_stamp_list': [],
                    'success': False,
                    'error': f"文件大小为0: {wav_path}"
                }
            
            # 读取音频文件
            try:
                wav, sr = sf.read(wav_path)
            except Exception as e:
                return {
                    'spk_id': spk_id,
                    'wav_path': wav_path,
                    'time_stamp_list': [],
                    'success': False,
                    'error': f"音频文件读取失败: {e}"
                }
            
            # 检查音频数据
            if wav is None or len(wav) == 0:
                return {
                    'spk_id': spk_id,
                    'wav_path': wav_path,
                    'time_stamp_list': [],
                    'success': False,
                    'error': f"音频数据为空: {wav_path}"
                }
            
            # 重采样到16kHz
            if sr != 16000:
                try:
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
                except Exception as e:
                    return {
                        'spk_id': spk_id,
                        'wav_path': wav_path,
                        'time_stamp_list': [],
                        'success': False,
                        'error': f"音频重采样失败: {e}"
                    }
            
            # 执行VAD检测
            time_stamp_list = self._vad_detect(wav, sr=16000)
            
            return {
                'spk_id': spk_id,
                'wav_path': wav_path,
                'time_stamp_list': time_stamp_list,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"处理音频文件失败 {wav_path}: {e}")
            return {
                'spk_id': spk_id,
                'wav_path': wav_path,
                'time_stamp_list': [],
                'success': False,
                'error': str(e)
            }
    
    def _vad_detect(self, wav, sr):
        """VAD检测函数"""
        try:
            if self.model is None:
                return []
            
            # 检查音频长度
            if len(wav) < sr * 0.1:  # 小于0.1秒的音频跳过
                return []
            
            # 确保音频是1D数组
            if wav.ndim > 1:
                wav = wav.flatten()
            
            # 检查音频数据是否包含NaN或无穷大值
            if np.any(np.isnan(wav)) or np.any(np.isinf(wav)):
                return []
            
            # 确保音频数据在合理范围内
            wav = np.clip(wav, -1.0, 1.0)
            
            # 更安全的数据类型转换
            if wav.dtype != np.int16:
                wav_normalized = np.clip(wav, -1.0, 1.0)
                wav_int16 = (wav_normalized * 32767).astype(np.int16)
            else:
                wav_int16 = wav
            
            # 检查转换后的数据是否有效
            if len(wav_int16) == 0:
                return []
            
            # 添加额外的安全检查
            if len(wav_int16) > sr * 3600:  # 超过1小时的音频跳过
                return []
            
            # 确保音频数据是连续的numpy数组
            wav_int16 = np.ascontiguousarray(wav_int16)
            
            # 确保输入格式正确
            if wav_int16.ndim == 1:
                wav_input = wav_int16.reshape(1, -1)  # 转换为2D数组 (1, samples)
            else:
                wav_input = wav_int16
            
            result = self.model.generate(wav_input, fs=sr)
            time_stamp = result[0]['value']
            return time_stamp  # in ms
            
        except Exception as e:
            logger.error(f"VAD检测失败: {e}")
            return []

def load_dataset_info(voxceleb2_dataset_dir):
    """加载数据集信息"""
    wavscp = f"{voxceleb2_dataset_dir}/wav.scp"
    spk2utt = f"{voxceleb2_dataset_dir}/spk2utt"
    
    if not os.path.exists(wavscp):
        raise FileNotFoundError(f"wav.scp文件不存在: {wavscp}")
    if not os.path.exists(spk2utt):
        raise FileNotFoundError(f"spk2utt文件不存在: {spk2utt}")
    
    spk2wav = defaultdict(list)
    wav2scp = {}
    
    # 读取wav.scp文件
    logger.info("读取wav.scp文件...")
    with open(wavscp, 'r') as fscp:
        for line in fscp:
            line = line.strip().split()
            key = line[0]
            wav2scp[key] = line[1]

    # 读取spk2utt文件
    logger.info("读取spk2utt文件...")
    with open(spk2utt, 'r') as fspk:
        for line in fspk:
            line = line.strip().split()
            key = line[0]
            paths = [wav2scp[i] for i in line[1:]]
            spk2wav[key].extend(paths)
    
    return spk2wav

def process_worker(rank, world_size, spk2wav, output_file):
    """工作进程函数"""
    try:
        # 初始化分布式环境
        # 对于单GPU环境，使用gloo后端避免NCCL冲突
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if num_gpus == 1 and world_size > 1:
                logger.info(f"Rank {rank}: 检测到单GPU环境，使用gloo后端")
                backend = 'gloo'
            else:
                backend = 'nccl'
        else:
            backend = 'gloo'
        
        logger.info(f"Rank {rank}: 初始化分布式环境，backend={backend}")
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        logger.info(f"Rank {rank}: 分布式环境初始化成功")
    except Exception as e:
        logger.error(f"Rank {rank}: 分布式环境初始化失败: {e}")
        return
    
    # 设置设备
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus == 1 and world_size > 1:
            # 单GPU多进程环境，所有进程共享同一个GPU
            device = torch.device('cuda:0')
            torch.cuda.set_device(device)
            logger.info(f"Rank {rank}: 单GPU环境，使用设备 {device}")
        else:
            # 多GPU环境，每个进程使用不同的GPU
            device = torch.device(f'cuda:{rank}')
            torch.cuda.set_device(device)
            logger.info(f"Rank {rank}: 多GPU环境，使用设备 {device}")
    else:
        device = torch.device('cpu')
        logger.info(f"Rank {rank}: 使用设备 {device}")
    
    # 创建数据集
    dataset = VoxCeleb2Dataset(spk2wav, rank, world_size)
    
    # 初始化VAD处理器
    vad_processor = VADProcessor(device)
    
    # 处理数据
    results = defaultdict(list)
    failed_files = []
    
    logger.info(f"Rank {rank}: 开始处理 {len(dataset)} 个文件")
    
    # 直接遍历数据集，避免DataLoader的复杂性
    for i in tqdm(range(len(dataset)), desc=f"Rank {rank}", disable=rank != 0):
        wav_path, spk_id = dataset[i]
        result = vad_processor.process_audio(wav_path, spk_id)
        
        if result['success']:
            results[spk_id].append({
                'wav_path': result['wav_path'],
                'time_stamp_list': result['time_stamp_list']
            })
        else:
            failed_files.append(result)
    
    # 收集所有进程的结果
    all_results = [None] * world_size
    all_failed = [None] * world_size
    
    # 使用all_gather收集所有进程的结果
    dist.all_gather_object(all_results, results)
    dist.all_gather_object(all_failed, failed_files)
    
    # 只在主进程中保存结果
    if rank == 0:
        # 合并所有进程的结果
        merged_results = defaultdict(list)
        merged_failed = []
        
        for proc_results in all_results:
            for spk_id, spk_results in proc_results.items():
                merged_results[spk_id].extend(spk_results)
        
        for proc_failed in all_failed:
            merged_failed.extend(proc_failed)
        
        # 保存结果
        logger.info(f"保存结果到 {output_file}")
        with open(output_file, "w") as f:
            for spk_id, spk_results in merged_results.items():
                spk2chunks = defaultdict(list)
                for item in spk_results:
                    spk2chunks[spk_id].append(item['time_stamp_list'])
                
                res = {
                    'spk_id': spk_id,
                    'results': spk2chunks,
                }
                json.dump(res, f)
                f.write("\n")
        
        # 输出统计信息
        total_processed = sum(len(spk_results) for spk_results in merged_results.values())
        logger.info(f"处理完成! 成功: {total_processed}, 失败: {len(merged_failed)}")
        
        if merged_failed:
            logger.warning(f"失败文件数: {len(merged_failed)}")
    
    # 清理分布式环境
    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description="基于torchrun的多进程VoxCeleb2数据集处理")
    parser.add_argument("--voxceleb2-dataset-dir", type=str, 
                       default="/maduo/datasets/voxceleb2/vox2_dev/", 
                       help="VoxCeleb2数据集Kaldi格式路径")
    parser.add_argument("--out-text", type=str, 
                       default="/maduo/datasets/voxceleb2/vox2_dev/train_torchrun.json", 
                       help="输出JSON文件路径")
    parser.add_argument("--local_rank", type=int, default=0,
                       help="本地进程rank（由torchrun自动设置）")
    
    args = parser.parse_args()
    
    # 获取分布式环境信息
    rank = int(os.environ.get('RANK', args.local_rank))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    logger.info(f"进程信息: rank={rank}, world_size={world_size}")
    
    # 加载数据集信息
    spk2wav = load_dataset_info(args.voxceleb2_dataset_dir)
    
    # 启动工作进程
    process_worker(rank, world_size, spk2wav, args.out_text)

if __name__ == "__main__":
    main() 