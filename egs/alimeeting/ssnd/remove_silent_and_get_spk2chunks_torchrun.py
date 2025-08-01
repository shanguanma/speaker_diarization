#!/usr/bin/env python3 -u
"""
基于torchrun的多进程VoxCeleb2数据集处理脚本
使用简化的多进程模式，参考torchrun_remove_silent_ref.py
"""

import os
import sys
import json
import argparse
import logging
from collections import defaultdict
import numpy as np
import soundfile as sf
import librosa
from funasr import AutoModel
from tqdm import tqdm

# 设置日志
formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
logging.basicConfig(format=formatter, level=logging.INFO)
logger = logging.getLogger(__name__)

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

def vad_detect(wav, sr):
    """VAD检测函数"""
    try:
#        # 检查音频长度
#        if len(wav) < sr * 0.1:  # 小于0.1秒的音频跳过
#            return []
#        
#        # 确保音频是1D数组
#        if wav.ndim > 1:
#            wav = wav.flatten()
#        
#        # 检查音频数据是否包含NaN或无穷大值
#        if np.any(np.isnan(wav)) or np.any(np.isinf(wav)):
#            return []
#        
#        # 确保音频数据在合理范围内
#        wav = np.clip(wav, -1.0, 1.0)
#        
#        # 初始化VAD模型
        vad_model = AutoModel(model="fsmn-vad", model_revision="v2.0.4", disable_update=True)
        
        # 更安全的数据类型转换
        if wav.dtype != np.int16:
            wav_normalized = np.clip(wav, -1.0, 1.0)
            wav_int16 = (wav_normalized * 32767).astype(np.int16)
        else:
            wav_int16 = wav
        
#        # 检查转换后的数据是否有效
#        if len(wav_int16) == 0:
#            return []
#        
#        # 添加额外的安全检查
#        if len(wav_int16) > sr * 3600:  # 超过1小时的音频跳过
#            return []
#        
#        # 确保音频数据是连续的numpy数组
#        wav_int16 = np.ascontiguousarray(wav_int16)
#        
#        # 确保输入格式正确
#        if wav_int16.ndim == 1:
#            wav_input = wav_int16.reshape(1, -1)  # 转换为2D数组 (1, samples)
#        else:
#            wav_input = wav_int16
#       
        wav_input=wav_int16
        result = vad_model.generate(wav_input, fs=sr)
        time_stamp = result[0]['value']
        return time_stamp  # in ms
        
    except Exception as e:
        logger.error(f"VAD检测失败: {e}")
        return []

def process_audio_file(wav_path, spk_id):
    """处理单个音频文件"""
    try:
        # 检查文件是否存在
        if not os.path.exists(wav_path):
            return {
                'spk_id': spk_id,
                'wav_path': wav_path,
                'speech_chunks': [],
                'success': False,
                'error': f"文件不存在: {wav_path}"
            }
        
        # 检查文件大小
        file_size = os.path.getsize(wav_path)
        if file_size == 0:
            return {
                'spk_id': spk_id,
                'wav_path': wav_path,
                'speech_chunks': [],
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
                'speech_chunks': [],
                'success': False,
                'error': f"音频文件读取失败: {e}"
            }
        
        # 检查音频数据
        if wav is None or len(wav) == 0:
            return {
                'spk_id': spk_id,
                'wav_path': wav_path,
                'speech_chunks': [],
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
                    'speech_chunks': [],
                    'success': False,
                    'error': f"音频重采样失败: {e}"
                }
        
        # 执行VAD检测
        speech_chunks = vad_detect(wav, sr=16000)
        # in ms ->(/1000) in second ->(*16000) in sample points
        speech_chunks = [wav[int(s*16):int(e*16)].tolist() for s, e in speech_chunks]        
        return {
            'spk_id': spk_id,
            'wav_path': wav_path,
            'speech_chunks': speech_chunks,
            'success': True
        }
        
    except Exception as e:
        logger.error(f"处理音频文件失败 {wav_path}: {e}")
        return {
            'spk_id': spk_id,
            'wav_path': wav_path,
            'speech_chunks': [],
            'success': False,
            'error': str(e)
        }

def process_local_tasks(local_tasks, output_file):
    """处理本地任务"""
    results = defaultdict(list)
    failed_files = []
    
    logger.info(f"开始处理 {len(local_tasks)} 个文件")
    
    for wav_path, spk_id in tqdm(local_tasks, desc="处理音频文件"):
        result = process_audio_file(wav_path, spk_id)
        
        if result['success']:
            results[spk_id].append({
                'wav_path': result['wav_path'],
                'speech_chunks': result['speech_chunks']
            })
        else:
            failed_files.append(result)
    
    # 保存本地结果到临时文件
    temp_output_file = f"{output_file}.rank_{os.environ.get('LOCAL_RANK', 0)}"
    with open(temp_output_file, "w") as f:
        for spk_id, spk_results in results.items():
            spk2chunks = defaultdict(list)
            #spk2wav_paths = defaultdict(list)
            for item in spk_results:
                spk2chunks[spk_id].append(item['speech_chunks'])
                #spk2wav_paths[spk_id].append(item['wav_path'])
            res = {
                'spk_id': spk_id,
                #'wav_paths': spk2wav_paths,
                'results': spk2chunks,
            }
            json.dump(res, f)
            f.write("\n")
    
    logger.info(f"本地处理完成，结果保存到: {temp_output_file}")
    logger.info(f"成功: {sum(len(spk_results) for spk_results in results.values())}, 失败: {len(failed_files)}")
    
    return temp_output_file

def merge_results(output_file, world_size):
    """合并所有进程的结果"""
    if int(os.environ.get('LOCAL_RANK', 0)) != 0:
        return  # 只在主进程中合并
    
    logger.info("合并所有进程的结果...")
    
    merged_results = defaultdict(list)
    
    # 读取所有临时文件
    for rank in range(world_size):
        temp_file = f"{output_file}.rank_{rank}"
        if os.path.exists(temp_file):
            with open(temp_file, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        spk_id = data['spk_id']
                        for item in data['results'][spk_id]:
                            merged_results[spk_id].append(item)
    
    # 保存合并后的结果
    with open(output_file, "w") as f:
        for spk_id, spk_results in merged_results.items():
            spk2chunks = defaultdict(list)
            for item in spk_results:
                spk2chunks[spk_id].append(item)
            
            res = {
                'spk_id': spk_id,
                'results': spk2chunks,
            }
            json.dump(res, f)
            f.write("\n")
    
    # 清理临时文件
    for rank in range(world_size):
        temp_file = f"{output_file}.rank_{rank}"
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    logger.info(f"结果合并完成，保存到: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="基于torchrun的多进程VoxCeleb2数据集处理")
    parser.add_argument("--voxceleb2-dataset-dir", type=str, 
                       default="/maduo/datasets/voxceleb2/vox2_dev/", 
                       help="VoxCeleb2数据集Kaldi格式路径")
    parser.add_argument("--out-text", type=str, 
                       default="/maduo/datasets/voxceleb2/vox2_dev/train_torchrun.json", 
                       help="输出JSON文件路径")
    
    args = parser.parse_args()
    
    # 获取分布式环境信息（参考torchrun_remove_silent_ref.py的方式）
    rank = int(os.environ['LOCAL_RANK'])        # 处理进程ID
    world_size = int(os.environ['WORLD_SIZE'])  # 进程总数
    
    logger.info(f"rank {rank}/{world_size}")
    
    # 加载数据集信息
    spk2wav = load_dataset_info(args.voxceleb2_dataset_dir)
    
    # 准备任务列表
    tasks = []
    for spk_id, wav_paths in spk2wav.items():
        for wav_path in wav_paths:
            tasks.append((wav_path, spk_id))
    
    # 按rank分配任务（参考torchrun_remove_silent_ref.py的方式）
    tasks.sort(key=lambda x: x[1])  # 按说话人ID排序
    local_tasks = tasks[rank::world_size]
    
    logger.info(f"本地任务数: {len(local_tasks)}")
    
    # 处理本地任务
    temp_output_file = process_local_tasks(local_tasks, args.out_text)
    
    # 合并结果
    merge_results(args.out_text, world_size)
    
    logger.info(f"rank {rank} 处理完成!")

if __name__ == "__main__":
    main() 
