#!/usr/bin/env python3
"""
测试懒加载SimuDiarMixer功能
这个脚本演示如何使用新的懒加载模式，避免预先处理整个spk2chunks数据
"""

import os
import sys
import json
import gzip
import time
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simu_diar_dataset import SimuDiarMixer

def create_test_vad_file(output_path, num_speakers=10, num_files_per_speaker=5):
    """
    创建测试用的VAD文件
    """
    print(f"创建测试VAD文件: {output_path}")
    
    test_data = []
    for spk_id in range(num_speakers):
        spk_data = {
            "spk_id": f"spk_{spk_id:03d}",
            "wav_paths": [],
            "results": []
        }
        
        for file_id in range(num_files_per_speaker):
            # 模拟音频文件路径
            wav_path = f"/fake/path/speaker_{spk_id:03d}/file_{file_id:03d}.wav"
            spk_data["wav_paths"].append(wav_path)
            
            # 模拟VAD时间戳（秒）
            timestamps = []
            current_time = 0.0
            for _ in range(np.random.randint(3, 8)):  # 3-7个语音片段
                start = current_time + np.random.uniform(0.1, 0.5)  # 0.1-0.5秒静音
                duration = np.random.uniform(0.5, 2.0)  # 0.5-2.0秒语音
                end = start + duration
                timestamps.append([start, end])
                current_time = end + np.random.uniform(0.1, 0.3)
            
            spk_data["results"].append(timestamps)
        
        test_data.append(spk_data)
    
    # 写入文件
    if output_path.endswith('.gz'):
        with gzip.open(output_path, 'wt', encoding='utf-8') as f:
            for item in test_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    else:
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in test_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"创建了 {len(test_data)} 个说话人的测试数据")

def create_test_audio_files(base_dir, num_speakers=10, num_files_per_speaker=5):
    """
    创建测试用的音频文件
    """
    print(f"创建测试音频文件目录: {base_dir}")
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    sample_rate = 16000
    duration = 10.0  # 10秒音频
    
    for spk_id in range(num_speakers):
        spk_dir = base_path / f"speaker_{spk_id:03d}"
        spk_dir.mkdir(exist_ok=True)
        
        for file_id in range(num_files_per_speaker):
            # 生成随机音频（模拟语音）
            t = np.linspace(0, duration, int(sample_rate * duration))
            # 使用正弦波模拟语音特征
            audio = np.sin(2 * np.pi * 200 * t) + 0.5 * np.sin(2 * np.pi * 400 * t)
            # 添加一些随机性
            audio += 0.1 * np.random.normal(0, 1, len(audio))
            
            # 保存音频文件
            output_path = spk_dir / f"file_{file_id:03d}.wav"
            sf.write(output_path, audio, sample_rate)
    
    print(f"创建了 {num_speakers} 个说话人的音频文件，每个说话人 {num_files_per_speaker} 个文件")

def test_traditional_mode():
    """
    测试传统模式（需要预先加载spk2chunks）
    """
    print("\n=== 测试传统模式 ===")
    
    # 创建模拟的spk2chunks数据
    spk2chunks = {}
    for spk_id in range(5):
        spk2chunks[f"spk_{spk_id:03d}"] = [
            np.random.normal(0, 0.1, 16000) for _ in range(3)  # 每个说话人3个1秒片段
        ]
    
    # 创建SimuDiarMixer实例
    mixer = SimuDiarMixer(
        spk2chunks=spk2chunks,
        sample_rate=16000,
        max_mix_len=8.0,
        min_speakers=2,
        max_speakers=3,
        target_overlap=0.2
    )
    
    print(f"传统模式数据集大小: {len(mixer)}")
    
    # 测试采样
    start_time = time.time()
    for i in range(3):
        mix, label, spk_ids = mixer.sample()
        print(f"样本 {i+1}: 音频长度={len(mix)/16000:.2f}s, 标签形状={label.shape}, 说话人={spk_ids}")
    
    elapsed = time.time() - start_time
    print(f"传统模式采样耗时: {elapsed:.3f}秒")

def test_lazy_mode(vad_file_path):
    """
    测试懒加载模式
    """
    print(f"\n=== 测试懒加载模式 ===")
    print(f"VAD文件: {vad_file_path}")
    
    # 创建SimuDiarMixer实例（懒加载模式）
    mixer = SimuDiarMixer(
        spk2chunks=None,  # 懒加载模式下不需要预先加载
        voxceleb2_spk2chunks_json=vad_file_path,
        sample_rate=16000,
        max_mix_len=8.0,
        min_speakers=2,
        max_speakers=3,
        target_overlap=0.2
    )
    
    print(f"懒加载模式数据集大小: {len(mixer)}")
    
    # 测试采样
    start_time = time.time()
    for i in range(3):
        try:
            mix, label, spk_ids = mixer.sample_lazy()
            print(f"样本 {i+1}: 音频长度={len(mix)/16000:.2f}s, 标签形状={label.shape}, 说话人={spk_ids}")
        except Exception as e:
            print(f"样本 {i+1} 失败: {e}")
    
    elapsed = time.time() - start_time
    print(f"懒加载模式采样耗时: {elapsed:.3f}秒")

def test_memory_usage():
    """
    测试内存使用情况
    """
    print("\n=== 测试内存使用情况 ===")
    
    import psutil
    import gc
    
    process = psutil.Process()
    
    # 测试传统模式内存使用
    print("传统模式内存使用:")
    gc.collect()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    spk2chunks = {}
    for spk_id in range(100):  # 100个说话人
        spk2chunks[f"spk_{spk_id:03d}"] = [
            np.random.normal(0, 0.1, 16000) for _ in range(10)  # 每个说话人10个1秒片段
        ]
    
    mixer = SimuDiarMixer(spk2chunks=spk2chunks)
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    
    print(f"  内存使用: {mem_before:.1f}MB -> {mem_after:.1f}MB (增加: {mem_after-mem_before:.1f}MB)")
    
    # 清理
    del spk2chunks, mixer
    gc.collect()
    
    # 测试懒加载模式内存使用
    print("懒加载模式内存使用:")
    gc.collect()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # 创建测试VAD文件
    test_vad_file = "test_vad.jsonl.gz"
    create_test_vad_file(test_vad_file, num_speakers=100, num_files_per_speaker=10)
    
    mixer = SimuDiarMixer(voxceleb2_spk2chunks_json=test_vad_file)
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    
    print(f"  内存使用: {mem_before:.1f}MB -> {mem_after:.1f}MB (增加: {mem_after-mem_before:.1f}MB)")
    
    # 清理
    del mixer
    gc.collect()
    
    # 删除测试文件
    if os.path.exists(test_vad_file):
        os.remove(test_vad_file)

def main():
    """
    主函数
    """
    print("=== SimuDiarMixer 懒加载模式测试 ===")
    
    # 创建测试目录
    test_dir = "test_data"
    test_vad_file = os.path.join(test_dir, "test_vad.jsonl.gz")
    
    # 创建测试数据
    if not os.path.exists(test_vad_file):
        os.makedirs(test_dir, exist_ok=True)
        create_test_vad_file(test_vad_file, num_speakers=20, num_files_per_speaker=5)
    
    # 测试传统模式
    test_traditional_mode()
    
    # 测试懒加载模式
    test_lazy_mode(test_vad_file)
    
    # 测试内存使用
    test_memory_usage()
    
    print("\n=== 测试完成 ===")
    print("懒加载模式的优势:")
    print("1. 避免预先加载整个spk2chunks数据")
    print("2. 内存使用更少")
    print("3. 启动时间更快")
    print("4. 支持更大的数据集")
    
    print("\n使用方法:")
    print("python train_accelerate_ddp.py --use-lazy-simu True --voxceleb2-spk2chunks-json /path/to/vad.jsonl.gz")

if __name__ == "__main__":
    main()
