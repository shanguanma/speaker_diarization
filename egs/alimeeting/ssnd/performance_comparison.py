#!/usr/bin/env python3
"""
性能对比脚本：比较串行和并行处理VoxCeleb2数据集的性能差异
"""

import time
import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from tqdm import tqdm
import json
from collections import defaultdict

# 导入原始的处理函数
from remove_silent_and_get_spk2chunks import vad_func, process_single_wav
import soundfile as sf
import librosa

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def serial_processing(spk2wav, max_files=None):
    """串行处理音频文件"""
    logger.info("开始串行处理...")
    start_time = time.time()
    
    results = defaultdict(list)
    failed_files = []
    
    # 限制处理的文件数量用于测试
    total_files = 0
    for spk_id, wav_paths in spk2wav.items():
        for wav_path in wav_paths:
            if max_files and total_files >= max_files:
                break
            try:
                # 读取音频文件
                wav, sr = sf.read(wav_path)
                if sr != 16000:
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
                
                # 执行VAD检测
                time_stamp_list = vad_func(wav, sr=16000)
                
                results[spk_id].append({
                    'wav_path': wav_path,
                    'time_stamp_list': time_stamp_list
                })
            except Exception as e:
                failed_files.append({
                    'wav_path': wav_path,
                    'error': str(e)
                })
            total_files += 1
        if max_files and total_files >= max_files:
            break
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    return results, failed_files, processing_time, total_files

def parallel_processing(spk2wav, max_workers=16, max_files=None):
    """并行处理音频文件"""
    logger.info(f"开始并行处理，使用 {max_workers} 个线程...")
    start_time = time.time()
    
    # 准备任务
    tasks = []
    total_files = 0
    for spk_id, wav_paths in spk2wav.items():
        for wav_path in wav_paths:
            if max_files and total_files >= max_files:
                break
            tasks.append((wav_path, spk_id))
            total_files += 1
        if max_files and total_files >= max_files:
            break
    
    results = defaultdict(list)
    failed_files = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(process_single_wav, task): task for task in tasks}
        
        with tqdm(total=len(tasks), desc="并行处理") as pbar:
            for future in as_completed(future_to_task):
                result = future.result()
                pbar.update(1)
                
                if result['success']:
                    spk_id = result['spk_id']
                    results[spk_id].append({
                        'wav_path': result['wav_path'],
                        'time_stamp_list': result['time_stamp_list']
                    })
                else:
                    failed_files.append(result)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    return results, failed_files, processing_time, total_files

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
    with open(wavscp, 'r') as fscp:
        for line in fscp:
            line = line.strip().split()
            key = line[0]
            wav2scp[key] = line[1]

    # 读取spk2utt文件
    with open(spk2utt, 'r') as fspk:
        for line in fspk:
            line = line.strip().split()
            key = line[0]
            paths = [wav2scp[i] for i in line[1:]]
            spk2wav[key].extend(paths)
    
    return spk2wav

def main():
    parser = argparse.ArgumentParser(description="性能对比：串行 vs 并行处理")
    parser.add_argument("--voxceleb2-dataset-dir", type=str, 
                       default="/maduo/datasets/voxceleb2/vox2_dev/", 
                       help="VoxCeleb2数据集路径")
    parser.add_argument("--max-files", type=int, default=100,
                       help="最大处理文件数（用于测试）")
    parser.add_argument("--max-workers", type=int, default=16,
                       help="并行处理的最大线程数")
    parser.add_argument("--output-dir", type=str, default="./performance_results",
                       help="结果输出目录")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据集信息
    logger.info("加载数据集信息...")
    spk2wav = load_dataset_info(args.voxceleb2_dataset_dir)
    
    total_files_in_dataset = sum(len(paths) for paths in spk2wav.values())
    logger.info(f"数据集总文件数: {total_files_in_dataset}")
    logger.info(f"测试文件数: {args.max_files}")
    
    # 串行处理
    logger.info("=" * 50)
    serial_results, serial_failed, serial_time, serial_processed = serial_processing(
        spk2wav, max_files=args.max_files
    )
    
    # 并行处理
    logger.info("=" * 50)
    parallel_results, parallel_failed, parallel_time, parallel_processed = parallel_processing(
        spk2wav, max_workers=args.max_workers, max_files=args.max_files
    )
    
    # 性能对比
    logger.info("=" * 50)
    logger.info("性能对比结果:")
    logger.info(f"串行处理时间: {serial_time:.2f} 秒")
    logger.info(f"并行处理时间: {parallel_time:.2f} 秒")
    logger.info(f"加速比: {serial_time / parallel_time:.2f}x")
    logger.info(f"效率提升: {((serial_time - parallel_time) / serial_time * 100):.1f}%")
    
    # 保存详细结果
    results = {
        'dataset_info': {
            'total_files': total_files_in_dataset,
            'test_files': args.max_files,
            'max_workers': args.max_workers
        },
        'serial_processing': {
            'processing_time': serial_time,
            'processed_files': serial_processed,
            'successful_files': serial_processed - len(serial_failed),
            'failed_files': len(serial_failed),
            'files_per_second': serial_processed / serial_time if serial_time > 0 else 0
        },
        'parallel_processing': {
            'processing_time': parallel_time,
            'processed_files': parallel_processed,
            'successful_files': parallel_processed - len(parallel_failed),
            'failed_files': len(parallel_failed),
            'files_per_second': parallel_processed / parallel_time if parallel_time > 0 else 0
        },
        'performance_improvement': {
            'speedup_ratio': serial_time / parallel_time if parallel_time > 0 else 0,
            'efficiency_improvement_percent': ((serial_time - parallel_time) / serial_time * 100) if serial_time > 0 else 0
        }
    }
    
    output_file = os.path.join(args.output_dir, f"performance_comparison_{int(time.time())}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"详细结果已保存到: {output_file}")
    
    # 打印失败文件统计
    if serial_failed:
        logger.warning(f"串行处理失败文件数: {len(serial_failed)}")
    if parallel_failed:
        logger.warning(f"并行处理失败文件数: {len(parallel_failed)}")

if __name__ == "__main__":
    main() 