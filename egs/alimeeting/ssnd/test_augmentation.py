#!/usr/bin/env python3
"""
测试音频增强功能的脚本
"""

import numpy as np
import os
import sys
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simu_diar_dataset import SimuDiarMixer

def create_dummy_data():
    """创建测试用的虚拟数据"""
    # 创建一些虚拟的音频片段
    sr = 16000
    chunk_length = int(2.0 * sr)  # 2秒片段
    
    # 生成一些正弦波作为虚拟音频
    t = np.linspace(0, 2.0, chunk_length)
    audio_chunk1 = np.sin(2 * np.pi * 440 * t) * 0.1  # 440Hz
    audio_chunk2 = np.sin(2 * np.pi * 880 * t) * 0.1  # 880Hz
    audio_chunk3 = np.sin(2 * np.pi * 220 * t) * 0.1  # 220Hz
    
    # 创建说话人数据
    spk2chunks = {
        'spk1': [audio_chunk1, audio_chunk2],
        'spk2': [audio_chunk2, audio_chunk3],
        'spk3': [audio_chunk1, audio_chunk3]
    }
    
    return spk2chunks

def test_basic_functionality():
    """测试基本功能（无音频增强）"""
    logger.info("=== 测试基本功能（无音频增强） ===")
    
    spk2chunks = create_dummy_data()
    
    mixer = SimuDiarMixer(
        spk2chunks=spk2chunks,
        sample_rate=16000,
        max_mix_len=10.0,
        min_silence=0.5,
        max_silence=2.0,
        target_overlap=0.3
    )
    
    # 测试基本采样
    mix, label, spk_ids = mixer.sample()
    logger.info(f"基本采样 - 音频长度: {len(mix)/16000:.2f}s, 说话人数: {len(spk_ids)}")
    
    # 测试__getitem__
    sample = mixer[0]
    logger.info(f"__getitem__ - 返回数据类型: {type(sample)}, 长度: {len(sample)}")
    
    return True

def test_augmentation_without_resources():
    """测试音频增强功能（无实际资源文件）"""
    logger.info("=== 测试音频增强功能（无实际资源文件） ===")
    
    spk2chunks = create_dummy_data()
    
    # 设置不存在的路径
    mixer = SimuDiarMixer(
        spk2chunks=spk2chunks,
        sample_rate=16000,
        max_mix_len=10.0,
        musan_path="/nonexistent/musan/path",
        rir_path="/nonexistent/rir/path",
        noise_ratio=0.8
    )
    
    # 检查增强状态
    info = mixer.get_augmentation_info()
    logger.info(f"增强信息: {info}")
    
    # 测试采样（应该跳过增强）
    sample = mixer[0]
    logger.info(f"采样完成，音频增强被跳过")
    
    return True

def test_augmentation_methods():
    """测试音频增强相关方法"""
    logger.info("=== 测试音频增强相关方法 ===")
    
    spk2chunks = create_dummy_data()
    
    mixer = SimuDiarMixer(
        spk2chunks=spk2chunks,
        sample_rate=16000,
        max_mix_len=10.0
    )
    
    # 测试sample_post方法
    mix, label, spk_ids = mixer.sample()
    mix_post, label_post, spk_ids_post = mixer.sample_post(mix, label, spk_ids)
    
    logger.info(f"原始音频长度: {len(mix)}")
    logger.info(f"处理后音频长度: {len(mix_post)}")
    logger.info(f"音频是否相同: {np.array_equal(mix, mix_post)}")
    
    # 测试强制增强
    mix_forced, label_forced, spk_ids_forced = mixer.sample_post(mix, label, spk_ids, force_augment=True)
    logger.info(f"强制增强后音频长度: {len(mix_forced)}")
    
    return True

def test_lazy_mode():
    """测试懒加载模式"""
    logger.info("=== 测试懒加载模式 ===")
    
    # 创建一个不存在的VAD文件路径来测试懒加载模式
    mixer = SimuDiarMixer(
        voxceleb2_spk2chunks_json="/nonexistent/vad_data.json",
        sample_rate=16000,
        max_mix_len=10.0
    )
    
    logger.info(f"懒加载模式: {mixer.lazy_mode}")
    logger.info(f"数据集大小: {len(mixer)}")
    
    return True

def main():
    """主测试函数"""
    logger.info("开始测试音频增强功能...")
    
    try:
        # 运行所有测试
        test_basic_functionality()
        test_augmentation_without_resources()
        test_augmentation_methods()
        test_lazy_mode()
        
        logger.info("所有测试通过！✅")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
