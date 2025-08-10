#!/usr/bin/env python3
"""
测试SimuDiarMixer类的_get_speaker_count方法和相关功能
"""

import os
import json
import gzip
import tempfile
import numpy as np
from simu_diar_dataset import SimuDiarMixer

def create_test_vad_file(file_path, is_gzipped=False):
    """创建测试用的VAD文件"""
    test_data = [
        {"spk_id": "spk_001", "wav_paths": ["/path/to/spk_001_001.wav"], "results": [[[0.0, 2.0], [3.0, 5.0]]]},
        {"spk_id": "spk_002", "wav_paths": ["/path/to/spk_002_001.wav"], "results": [[[0.0, 1.5], [2.5, 4.0]]]},
        {"spk_id": "spk_003", "wav_paths": ["/path/to/spk_003_001.wav"], "results": [[[0.0, 3.0]]]},
        {"spk_id": "spk_004", "wav_paths": ["/path/to/spk_004_001.wav"], "results": [[[1.0, 2.5], [4.0, 6.0]]]},
        {"spk_id": "spk_005", "wav_paths": ["/path/to/spk_005_001.wav"], "results": [[[0.5, 2.0]]]},
    ]
    
    if is_gzipped:
        with gzip.open(file_path, "wt", encoding='utf-8') as f:
            for data in test_data:
                f.write(json.dumps(data) + '\n')
    else:
        with open(file_path, "w", encoding='utf-8') as f:
            for data in test_data:
                f.write(json.dumps(data) + '\n')
    
    return test_data

def test_speaker_count():
    """测试说话人计数功能"""
    print("=== 测试说话人计数功能 ===")
    
    # 测试普通文本文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        test_file = f.name
    
    try:
        test_data = create_test_vad_file(test_file, is_gzipped=False)
        
        # 创建SimuDiarMixer实例
        mixer = SimuDiarMixer(voxceleb2_spk2chunks_json=test_file)
        
        # 测试说话人计数
        speaker_count = mixer._get_speaker_count()
        print(f"说话人总数: {speaker_count}")
        assert speaker_count == 5, f"期望5个说话人，实际得到{speaker_count}个"
        
        # 测试缓存信息
        cache_info = mixer.get_cache_info()
        print(f"缓存信息: {cache_info}")
        assert cache_info['total_speakers'] == 5
        assert cache_info['lazy_mode'] == True
        
        print("✓ 普通文本文件测试通过")
        
    finally:
        os.unlink(test_file)
    
    # 测试gzip压缩文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json.gz', delete=False) as f:
        test_gz_file = f.name
    
    try:
        test_data = create_test_vad_file(test_gz_file, is_gzipped=True)
        
        # 创建SimuDiarMixer实例
        mixer = SimuDiarMixer(voxceleb2_spk2chunks_json=test_gz_file)
        
        # 测试说话人计数
        speaker_count = mixer._get_speaker_count()
        print(f"说话人总数: {speaker_count}")
        assert speaker_count == 5, f"期望5个说话人，实际得到{speaker_count}个")
        
        print("✓ Gzip压缩文件测试通过")
        
    finally:
        os.unlink(test_gz_file)

def test_speaker_list():
    """测试说话人列表加载功能"""
    print("\n=== 测试说话人列表加载功能 ===")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        test_file = f.name
    
    try:
        test_data = create_test_vad_file(test_file, is_gzipped=False)
        
        mixer = SimuDiarMixer(voxceleb2_spk2chunks_json=test_file)
        
        # 测试说话人列表
        speaker_list = mixer._load_speaker_list()
        print(f"说话人列表: {speaker_list}")
        
        expected_speakers = ["spk_001", "spk_002", "spk_003", "spk_004", "spk_005"]
        assert speaker_list == expected_speakers, f"说话人列表不匹配"
        
        print("✓ 说话人列表测试通过")
        
    finally:
        os.unlink(test_file)

def test_cache_management():
    """测试缓存管理功能"""
    print("\n=== 测试缓存管理功能 ===")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        test_file = f.name
    
    try:
        test_data = create_test_vad_file(test_file, is_gzipped=False)
        
        mixer = SimuDiarMixer(voxceleb2_spk2chunks_json=test_file)
        
        # 初始缓存信息
        initial_cache_info = mixer.get_cache_info()
        print(f"初始缓存信息: {initial_cache_info}")
        
        # 获取说话人数量（会触发缓存）
        speaker_count = mixer._get_speaker_count()
        print(f"说话人总数: {speaker_count}")
        
        # 缓存后的信息
        after_cache_info = mixer.get_cache_info()
        print(f"缓存后信息: {after_cache_info}")
        
        # 清理缓存
        mixer.clear_cache()
        after_clear_info = mixer.get_cache_info()
        print(f"清理后信息: {after_clear_info}")
        
        assert after_clear_info['cached_speakers'] == 0, "缓存应该被清理"
        
        print("✓ 缓存管理测试通过")
        
    finally:
        os.unlink(test_file)

def test_error_handling():
    """测试错误处理功能"""
    print("\n=== 测试错误处理功能 ===")
    
    # 测试非懒加载模式下的错误
    try:
        mixer = SimuDiarMixer(spk2chunks={})
        mixer._get_speaker_count()
        assert False, "应该在非懒加载模式下抛出错误"
    except ValueError as e:
        print(f"✓ 正确捕获错误: {e}")
    
    # 测试不存在的文件
    try:
        mixer = SimuDiarMixer(voxceleb2_spk2chunks_json="/nonexistent/file.json")
        speaker_count = mixer._get_speaker_count()
        print(f"说话人总数: {speaker_count}")
        assert speaker_count == 0, "不存在的文件应该返回0个说话人"
        print("✓ 文件不存在错误处理测试通过")
    except Exception as e:
        print(f"✓ 正确捕获错误: {e}")

def main():
    """主测试函数"""
    print("开始测试SimuDiarMixer类的说话人计数功能...")
    
    try:
        test_speaker_count()
        test_speaker_list()
        test_cache_management()
        test_error_handling()
        
        print("\n🎉 所有测试通过！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
