#!/usr/bin/env python3
"""
快速测试VAD修复的脚本
"""

import numpy as np
import soundfile as sf
import os
import sys
import time

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from remove_silent_and_get_spk2chunks import vad_func

def create_test_audio(duration_seconds=5.0, sample_rate=16000, filename="quick_test.wav"):
    """创建测试音频文件"""
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), False)
    # 生成1kHz的正弦波
    audio = 0.5 * np.sin(2 * np.pi * 1000 * t)
    sf.write(filename, audio, sample_rate)
    return filename

def test_vad_fix():
    """测试VAD修复"""
    print("测试VAD修复...")
    
    # 创建测试音频
    test_file = "quick_test.wav"
    create_test_audio(5.0, filename=test_file)
    
    try:
        # 读取音频
        wav, sr = sf.read(test_file)
        print(f"音频形状: {wav.shape}, 采样率: {sr}")
        
        # 测试VAD函数
        print("调用VAD函数...")
        start_time = time.time()
        result = vad_func(wav, sr)
        end_time = time.time()
        
        print(f"VAD处理时间: {end_time - start_time:.2f} 秒")
        print(f"VAD结果: {len(result)} 个时间戳")
        
        if len(result) > 0:
            print(f"第一个时间戳: {result[0]}")
        
        print("✓ VAD测试成功!")
        
    except Exception as e:
        print(f"✗ VAD测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理测试文件
        if os.path.exists(test_file):
            os.remove(test_file)

if __name__ == "__main__":
    test_vad_fix() 