#!/usr/bin/env python3 -u
"""
非并行的VoxCeleb2数据集处理脚本
移除torchrun依赖，顺序处理所有文件并保存为JSON格式
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

# 全局VAD模型，避免重复初始化
_vad_model = None

def get_vad_model():
    """获取VAD模型（单例模式）"""
    global _vad_model
    if _vad_model is None:
        try:
            logger.info("初始化VAD模型...")
            _vad_model = AutoModel(model="fsmn-vad", model_revision="v2.0.4", disable_update=True)
            logger.info("VAD模型初始化成功")
        except Exception as e:
            logger.error(f"VAD模型初始化失败: {e}")
            return None
    return _vad_model

def vad_detect(wav, sr):
    """VAD检测函数"""
    try:
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
        
        # 获取VAD模型
        vad_model = get_vad_model()
        if vad_model is None:
            return []
        
        # 更安全的数据类型转换
        if wav.dtype != np.int16:
            wav_normalized = np.clip(wav, -1.0, 1.0)
            wav_int16 = (wav_normalized * 32767).astype(np.int16)
        else:
            wav_int16 = wav

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
        time_stamp_list = vad_detect(wav, sr=16000)
        # in ms ->(/1000) in second ->(*16000) in sample points
        #time_stamp_list = [wav[int(s*16):int(e*16)].tolist() for s, e in time_stamp_list]        
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

def process_all_files(spk2wav, output_file):
    """处理所有文件"""
    results = defaultdict(list)
    failed_files = []
    
    # 准备所有任务
    all_tasks = []
    for spk_id, wav_paths in spk2wav.items():
        for wav_path in wav_paths:
            all_tasks.append((wav_path, spk_id))
    
    logger.info(f"开始处理 {len(all_tasks)} 个文件")
    
    # 分批处理，避免内存溢出
    batch_size = 50  # 每批处理50个文件
    for i in range(0, len(all_tasks), batch_size):
        batch_tasks = all_tasks[i:i+batch_size]
        logger.info(f"处理批次 {i//batch_size + 1}/{(len(all_tasks) + batch_size - 1)//batch_size}")
        
        for wav_path, spk_id in tqdm(batch_tasks, desc=f"批次 {i//batch_size + 1}"):
            try:
                result = process_audio_file(wav_path, spk_id)
                
                if result['success']:
                    results[spk_id].append({
                        'wav_path': result['wav_path'],
                        'time_stamp_list': result['time_stamp_list']
                    })
                else:
                    failed_files.append(result)
            except Exception as e:
                logger.error(f"处理文件失败 {wav_path}: {e}")
                failed_files.append({
                    'spk_id': spk_id,
                    'wav_path': wav_path,
                    'time_stamp_list': [],
                    'success': False,
                    'error': str(e)
                })
        
        # 每批处理后保存临时结果
        temp_output_file = f"{output_file}.batch_{i//batch_size}"
        save_results_to_json(results, temp_output_file)
        logger.info(f"批次 {i//batch_size + 1} 完成，临时结果保存到: {temp_output_file}")
        
        # 清理内存
        import gc
        gc.collect()
    
    # 保存最终结果
    save_results_to_json(results, output_file)
    
    # 保存失败文件信息
    if failed_files:
        failed_output_file = f"{output_file}.failed"
        with open(failed_output_file, "w") as f:
            json.dump(failed_files, f, indent=2, ensure_ascii=False)
        logger.info(f"失败文件信息保存到: {failed_output_file}")
    
    logger.info(f"处理完成，结果保存到: {output_file}")
    logger.info(f"成功: {sum(len(spk_results) for spk_results in results.values())}, 失败: {len(failed_files)}")
    
    return results

def save_results_to_json(results, output_file):
    """保存结果为JSON格式"""
    with open(output_file, "w") as f:
        for spk_id, spk_results in results.items():
            spk2chunks = defaultdict(list)
            spk2wav_paths = defaultdict(list)
            for item in spk_results:
                spk2chunks[spk_id].append(item['time_stamp_list'])
                spk2wav_paths[spk_id].append(item['wav_path'])
            
            res = {
                'spk_id': spk_id,
                'wav_paths': spk2wav_paths[spk_id],
                'results': spk2chunks[spk_id],
            }
            json.dump(res, f, ensure_ascii=False)
            f.write("\n")

def main():
    parser = argparse.ArgumentParser(description="非并行的VoxCeleb2数据集处理")
    parser.add_argument("--voxceleb2-dataset-dir", type=str, 
                       default="/maduo/datasets/voxceleb2/vox2_dev/", 
                       help="VoxCeleb2数据集Kaldi格式路径")
    parser.add_argument("--out-text", type=str, 
                       default="/maduo/datasets/voxceleb2/vox2_dev/train.json", 
                       help="输出JSON文件路径")
    
    args = parser.parse_args()
    
    logger.info("开始非并行处理VoxCeleb2数据集...")
    
    # 加载数据集信息
    spk2wav = load_dataset_info(args.voxceleb2_dataset_dir)
    logger.info(f"加载了 {len(spk2wav)} 个说话人的数据")
    
    # 处理所有文件
    results = process_all_files(spk2wav, args.out_text)
    
    logger.info("处理完成!")

if __name__ == "__main__":
    main() 
