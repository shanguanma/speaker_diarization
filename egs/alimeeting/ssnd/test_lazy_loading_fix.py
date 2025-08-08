#!/usr/bin/env python3
"""
测试懒加载版本修复的脚本
"""
import sys
import os
import logging

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

def test_lazy_loading_compatibility():
    """测试懒加载版本的兼容性"""
    
    # 导入主模块
    import train_accelerate_ddp
    
    # 创建测试参数
    class TestArgs:
        def __init__(self):
            self.voxceleb2_spk2chunks_json = "/maduo/datasets/voxceleb2/vox2_dev/train.json.gz"
            self.compression_type = "gzip"
            self.use_lazy_loading = True
            self.disable_cache = False
    
    args = TestArgs()
    
    # 测试参数（小数据集）
    max_speakers = 3
    max_files_per_speaker = 2
    
    logger.info("测试懒加载版本兼容性...")
    logger.info(f"测试参数: 最大说话人数={max_speakers}, 每个说话人最大文件数={max_files_per_speaker}")
    
    try:
        # 测试懒加载版本
        logger.info("测试懒加载版本...")
        spk2chunks_lazy = train_accelerate_ddp.spktochunks_lazy(
            args, max_speakers, max_files_per_speaker
        )
        
        # 检查返回的数据结构
        logger.info(f"懒加载版本返回数据类型: {type(spk2chunks_lazy)}")
        logger.info(f"说话人数量: {len(spk2chunks_lazy)}")
        logger.info(f"说话人列表: {list(spk2chunks_lazy.keys())}")
        
        # 测试数据访问
        for spk_id in spk2chunks_lazy.keys():
            chunks = spk2chunks_lazy[spk_id]
            logger.info(f"说话人 {spk_id}: {len(chunks)} 个音频文件")
            if chunks:
                logger.info(f"  第一个音频文件包含 {len(chunks[0])} 个语音片段")
        
        # 测试SimuDiarMixer兼容性
        logger.info("测试SimuDiarMixer兼容性...")
        try:
            from simu_diar_dataset import SimuDiarMixer
            
            # 创建SimuDiarMixer实例
            train_dataset = SimuDiarMixer(
                spk2chunks=spk2chunks_lazy,
                sample_rate=16000,
                frame_length=0.025,
                frame_shift=0.04,
                num_mel_bins=80,
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
            
            logger.info("SimuDiarMixer创建成功!")
            logger.info(f"数据集长度: {len(train_dataset)}")
            
            # 测试数据加载
            if len(train_dataset) > 0:
                sample = train_dataset[0]
                logger.info(f"样本数据类型: {type(sample)}")
                logger.info("数据加载测试成功!")
            
        except Exception as e:
            logger.error(f"SimuDiarMixer兼容性测试失败: {e}")
            return False
        
        logger.info("懒加载版本兼容性测试通过!")
        return True
        
    except Exception as e:
        logger.error(f"懒加载版本测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fallback_mechanism():
    """测试回退机制"""
    logger.info("\n测试回退机制...")
    
    import train_accelerate_ddp
    
    class TestArgs:
        def __init__(self):
            self.voxceleb2_spk2chunks_json = "/maduo/datasets/voxceleb2/vox2_dev/train.json.gz"
            self.compression_type = "gzip"
            self.use_lazy_loading = True
            self.disable_cache = False
    
    args = TestArgs()
    
    try:
        # 测试build_simu_data_train_dl函数
        spk2int = {"test_speaker": 0}  # 简单的spk2int
        
        train_dl = train_accelerate_ddp.build_simu_data_train_dl(
            args, spk2int, 
            use_fast_version=True,
            max_speakers=2,
            max_files_per_speaker=1
        )
        
        logger.info("回退机制测试成功!")
        return True
        
    except Exception as e:
        logger.error(f"回退机制测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    logger.info("开始测试懒加载版本修复...")
    
    # 测试1: 懒加载版本兼容性
    success1 = test_lazy_loading_compatibility()
    
    # 测试2: 回退机制
    success2 = test_fallback_mechanism()
    
    # 总结
    logger.info("\n" + "="*50)
    logger.info("测试结果总结")
    logger.info("="*50)
    
    if success1 and success2:
        logger.info("✅ 所有测试通过!")
        logger.info("懒加载版本修复成功，可以正常使用")
    else:
        logger.error("❌ 部分测试失败")
        if not success1:
            logger.error("懒加载版本兼容性测试失败")
        if not success2:
            logger.error("回退机制测试失败")
    
    return success1 and success2

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 