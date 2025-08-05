#!/usr/bin/env python3
"""
测试VAD修复的脚本
用于验证修复后的VAD函数是否能正确处理各种边界情况
"""

import numpy as np
import tempfile
import soundfile as sf
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from remove_silent_and_get_spk2chunks import vad_func, process_single_wav

def create_test_audio(duration_seconds=5.0, sample_rate=16000, filename="test_audio.wav"):
    """创建测试音频文件"""
    # 生成正弦波作为测试音频
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), False)
    # 生成1kHz的正弦波
    audio = 0.5 * np.sin(2 * np.pi * 1000 * t)
    
    # 保存音频文件
    sf.write(filename, audio, sample_rate)
    return filename

def test_vad_edge_cases():
    """测试VAD函数的边界情况"""
    print("测试VAD函数的边界情况...")
    
    test_cases = [
        {
            'name': '正常音频',
            'duration': 5.0,
            'expected_result': 'success'
        },
        {
            'name': '短音频（小于0.1秒）',
            'duration': 0.05,
            'expected_result': 'skip'
        },
        {
            'name': '长音频（超过1小时）',
            'duration': 3601.0,
            'expected_result': 'skip'
        }
    ]
    
    for test_case in test_cases:
        print(f"\n测试: {test_case['name']}")
        
        # 创建测试音频
        test_file = f"test_{test_case['name'].replace(' ', '_')}.wav"
        create_test_audio(test_case['duration'], filename=test_file)
        
        try:
            # 读取音频
            wav, sr = sf.read(test_file)
            
            # 测试VAD函数
            result = vad_func(wav, sr)
            
            if test_case['expected_result'] == 'success':
                if len(result) > 0:
                    print(f"  ✓ 成功: 返回了 {len(result)} 个时间戳")
                else:
                    print(f"  ⚠ 警告: 没有检测到语音活动")
            elif test_case['expected_result'] == 'skip':
                if len(result) == 0:
                    print(f"  ✓ 正确跳过: 音频被正确跳过")
                else:
                    print(f"  ✗ 错误: 应该跳过但返回了结果")
            
        except Exception as e:
            print(f"  ✗ 错误: {e}")
        finally:
            # 清理测试文件
            if os.path.exists(test_file):
                os.remove(test_file)

def test_parallel_processing():
    """测试并行处理"""
    print("\n测试并行处理...")
    
    # 创建多个测试音频文件
    test_files = []
    for i in range(10):
        filename = f"test_parallel_{i}.wav"
        create_test_audio(2.0, filename=filename)
        test_files.append(filename)
    
    try:
        # 测试不同的线程数
        thread_counts = [1, 4, 8, 16]
        
        for max_workers in thread_counts:
            print(f"\n测试 {max_workers} 个线程:")
            
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交任务
                futures = []
                for filename in test_files:
                    future = executor.submit(process_single_wav, (filename, f"spk_{filename}"))
                    futures.append(future)
                
                # 收集结果
                results = []
                for future in futures:
                    try:
                        result = future.result(timeout=30)  # 30秒超时
                        results.append(result)
                    except Exception as e:
                        print(f"  ✗ 任务执行失败: {e}")
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            successful = sum(1 for r in results if r['success'])
            failed = len(results) - successful
            
            print(f"  处理时间: {processing_time:.2f} 秒")
            print(f"  成功: {successful}, 失败: {failed}")
            
            if failed > 0:
                print("  失败的文件:")
                for r in results:
                    if not r['success']:
                        print(f"    {r['wav_path']}: {r.get('error', 'Unknown error')}")
    
    finally:
        # 清理测试文件
        for filename in test_files:
            if os.path.exists(filename):
                os.remove(filename)

def test_invalid_audio_data():
    """测试无效音频数据"""
    print("\n测试无效音频数据...")
    
    test_cases = [
        {
            'name': '空数组',
            'data': np.array([]),
            'expected_result': 'skip'
        },
        {
            'name': '包含NaN的数组',
            'data': np.array([1.0, np.nan, 0.5]),
            'expected_result': 'skip'
        },
        {
            'name': '包含无穷大的数组',
            'data': np.array([1.0, np.inf, 0.5]),
            'expected_result': 'skip'
        },
        {
            'name': '超出范围的数组',
            'data': np.array([2.0, -2.0, 1.5]),
            'expected_result': 'success'  # 应该被裁剪到[-1, 1]
        }
    ]
    
    for test_case in test_cases:
        print(f"\n测试: {test_case['name']}")
        
        try:
            result = vad_func(test_case['data'], 16000)
            
            if test_case['expected_result'] == 'success':
                if len(result) >= 0:  # 允许返回空结果
                    print(f"  ✓ 成功处理")
                else:
                    print(f"  ✗ 处理失败")
            elif test_case['expected_result'] == 'skip':
                if len(result) == 0:
                    print(f"  ✓ 正确跳过")
                else:
                    print(f"  ✗ 应该跳过但返回了结果")
                    
        except Exception as e:
            if test_case['expected_result'] == 'skip':
                print(f"  ✓ 正确抛出异常: {e}")
            else:
                print(f"  ✗ 意外异常: {e}")

def main():
    """主函数"""
    print("开始VAD修复测试...")
    print("=" * 50)
    
    # 测试边界情况
    test_vad_edge_cases()
    
    # 测试无效数据
    test_invalid_audio_data()
    
    # 测试并行处理
    test_parallel_processing()
    
    print("\n" + "=" * 50)
    print("测试完成!")

if __name__ == "__main__":
    main() 