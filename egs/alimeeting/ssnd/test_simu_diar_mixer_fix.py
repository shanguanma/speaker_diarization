#!/usr/bin/env python3
"""
测试SimuDiarMixer修复的脚本
"""
import sys
import os
import logging
import time

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

def test_simu_diar_mixer_basic():
    """测试SimuDiarMixer基本功能"""
    
    # 导入模块
    from simu_diar_dataset import SimuDiarMixer
    import numpy as np
    
    logger.info("测试SimuDiarMixer基本功能...")
    
    # 创建测试数据
    spk2chunks = {
        'spk1': [
            [np.random.randn(16000)],  # 1秒音频
            [np.random.randn(8000)]    # 0.5秒音频
        ],
        'spk2': [
            [np.random.randn(24000)],  # 1.5秒音频
            [np.random.randn(12000)]   # 0.75秒音频
        ]
    }
    
    try:
        # 创建SimuDiarMixer实例
        mixer = SimuDiarMixer(
            spk2chunks=spk2chunks,
            sample_rate=16000,
            max_mix_len=8.0,
            min_silence=0.0,
            max_silence=4.0,
            min_speakers=1,
            max_speakers=3,
            target_overlap=0.2,
            musan_path=None,
            rir_path=None,
            noise_ratio=0.0,
        )
        
        # 测试__len__方法
        dataset_size = len(mixer)
        logger.info(f"数据集大小: {dataset_size}")
        
        # 测试__getitem__方法
        sample = mixer[0]
        logger.info(f"样本类型: {type(sample)}")
        logger.info(f"样本长度: {len(sample)}")
        
        if len(sample) == 3:
            mix, label, spk_ids = sample
            logger.info(f"混合音频长度: {len(mix)}")
            logger.info(f"标签形状: {label.shape}")
            logger.info(f"说话人ID: {spk_ids}")
        
        logger.info("SimuDiarMixer基本功能测试成功!")
        return True
        
    except Exception as e:
        logger.error(f"SimuDiarMixer基本功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataloader_compatibility():
    """测试与PyTorch DataLoader的兼容性"""
    
    logger.info("\n测试与PyTorch DataLoader的兼容性...")
    
    try:
        from simu_diar_dataset import SimuDiarMixer
        from torch.utils.data import DataLoader
        import numpy as np
        
        # 创建测试数据
        spk2chunks = {
            'spk1': [
                [np.random.randn(16000)],
                [np.random.randn(8000)]
            ],
            'spk2': [
                [np.random.randn(24000)],
                [np.random.randn(12000)]
            ]
        }
        
        # 创建SimuDiarMixer实例
        dataset = SimuDiarMixer(
            spk2chunks=spk2chunks,
            sample_rate=16000,
            max_mix_len=8.0,
            min_silence=0.0,
            max_silence=4.0,
            min_speakers=1,
            max_speakers=3,
            target_overlap=0.2,
            musan_path=None,
            rir_path=None,
            noise_ratio=0.0,
        )
        
        # 简单的collate函数
        def simple_collate(batch):
            return dataset.collate_fn(batch, vad_out_len=200)
        
        # 创建DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            collate_fn=simple_collate,
            num_workers=0,  # 使用单线程避免复杂性
        )
        
        logger.info("DataLoader创建成功!")
        logger.info(f"DataLoader长度: {len(dataloader)}")
        
        # 测试迭代
        for i, batch in enumerate(dataloader):
            if i >= 2:  # 只测试前2个batch
                break
            
            wavs, labels, spk_ids_list, fbanks, labels_len = batch
            logger.info(f"Batch {i}: wavs={wavs.shape}, labels={labels.shape}, fbanks={fbanks.shape}")
        
        logger.info("DataLoader兼容性测试成功!")
        return True
        
    except Exception as e:
        logger.error(f"DataLoader兼容性测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_fast_spktochunks():
    """测试与快速spktochunks函数的集成"""
    
    logger.info("\n测试与快速spktochunks函数的集成...")
    
    try:
        import train_accelerate_ddp
        from simu_diar_dataset import SimuDiarMixer
        
        # 创建测试参数
        class TestArgs:
            def __init__(self):
                self.voxceleb2_spk2chunks_json = "/maduo/datasets/voxceleb2/vox2_dev/train.json.gz"
                self.compression_type = "gzip"
                self.disable_cache = True
                self.fast_batch_size = 5
                self.fast_max_memory_mb = 2048
                self.musan_path = None
                self.rir_path = None
                self.noise_ratio = 0.0
        
        args = TestArgs()
        
        # 使用快速版本获取spk2chunks
        spk2chunks = train_accelerate_ddp.spktochunks_fast(
            args, max_speakers=3, max_files_per_speaker=2
        )
        
        logger.info(f"获取到 {len(spk2chunks)} 个说话人的数据")
        
        # 创建SimuDiarMixer
        mixer = SimuDiarMixer(
            spk2chunks=spk2chunks,
            sample_rate=16000,
            max_mix_len=8.0,
            min_silence=0.0,
            max_silence=4.0,
            min_speakers=1,
            max_speakers=3,
            target_overlap=0.2,
            musan_path=None,
            rir_path=None,
            noise_ratio=0.0,
        )
        
        logger.info(f"SimuDiarMixer数据集大小: {len(mixer)}")
        
        # 测试生成样本
        sample = mixer[0]
        mix, label, spk_ids = sample
        logger.info(f"生成样本: 音频长度={len(mix)}, 标签形状={label.shape}, 说话人={spk_ids}")
        
        logger.info("与快速spktochunks函数的集成测试成功!")
        return True
        
    except Exception as e:
        logger.error(f"与快速spktochunks函数的集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    logger.info("开始测试SimuDiarMixer修复...")
    
    # 测试1: 基本功能
    success1 = test_simu_diar_mixer_basic()
    
    # 测试2: DataLoader兼容性
    success2 = test_dataloader_compatibility()
    
    # 测试3: 与快速spktochunks的集成
    success3 = test_with_fast_spktochunks()
    
    # 总结
    logger.info("\n" + "="*60)
    logger.info("测试结果总结")
    logger.info("="*60)
    
    if success1 and success2 and success3:
        logger.info("✅ 所有测试通过!")
        logger.info("SimuDiarMixer修复成功，现在可以与DataLoader正常配合使用")
        logger.info("\n🎯 修复内容:")
        logger.info("1. 添加了__len__()方法")
        logger.info("2. 添加了__getitem__()方法")
        logger.info("3. 修复了collate_fn中的变量定义问题")
        logger.info("4. 确保与PyTorch DataLoader完全兼容")
    else:
        logger.error("❌ 部分测试失败")
        if not success1:
            logger.error("基本功能测试失败")
        if not success2:
            logger.error("DataLoader兼容性测试失败")
        if not success3:
            logger.error("与快速spktochunks的集成测试失败")
    
    return success1 and success2 and success3

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)