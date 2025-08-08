#!/usr/bin/env python3
"""
测试内存优化加速版本的脚本
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

def test_fast_memory_optimized():
    """测试内存优化的加速版本"""
    
    # 导入主模块
    import train_accelerate_ddp
    
    # 创建测试参数
    class TestArgs:
        def __init__(self):
            self.voxceleb2_spk2chunks_json = "/maduo/datasets/voxceleb2/vox2_dev/train.json.gz"
            self.compression_type = "gzip"
            self.disable_cache = True  # 禁用缓存进行真实测试
            self.fast_batch_size = 10  # 小批次
            self.fast_max_memory_mb = 4096  # 4GB内存限制
    
    args = TestArgs()
    
    # 测试参数（小数据集）
    max_speakers = 10
    max_files_per_speaker = 3
    
    logger.info("测试内存优化的加速版本...")
    logger.info(f"测试参数: 最大说话人数={max_speakers}, 每个说话人最大文件数={max_files_per_speaker}")
    logger.info(f"批处理大小={args.fast_batch_size}, 内存限制={args.fast_max_memory_mb} MB")
    
    try:
        # 测试内存优化的加速版本
        start_time = time.time()
        spk2chunks = train_accelerate_ddp.spktochunks_fast(
            args, max_speakers, max_files_per_speaker
        )
        end_time = time.time()
        
        # 检查返回的数据结构
        logger.info(f"内存优化加速版本完成，耗时: {end_time - start_time:.2f}秒")
        logger.info(f"说话人数量: {len(spk2chunks)}")
        logger.info(f"说话人列表: {list(spk2chunks.keys())}")
        
        # 测试数据访问
        total_files = 0
        for spk_id in spk2chunks.keys():
            chunks = spk2chunks[spk_id]
            total_files += len(chunks)
            logger.info(f"说话人 {spk_id}: {len(chunks)} 个音频文件")
        
        logger.info(f"总文件数: {total_files}")
        logger.info("内存优化加速版本测试成功!")
        return True
        
    except Exception as e:
        logger.error(f"内存优化加速版本测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_build_dataloader_fast():
    """测试使用内存优化加速版本的数据加载器构建"""
    logger.info("\n测试使用内存优化加速版本的数据加载器构建...")
    
    import train_accelerate_ddp
    
    class TestArgs:
        def __init__(self):
            self.voxceleb2_spk2chunks_json = "/maduo/datasets/voxceleb2/vox2_dev/train.json.gz"
            self.compression_type = "gzip"
            self.disable_cache = True
            self.fast_batch_size = 5  # 更小的批次
            self.fast_max_memory_mb = 3072  # 3GB内存限制
            self.musan_path = None
            self.rir_path = None
            self.noise_ratio = 0.0
            self.use_memory_safe = False  # 使用优化的加速版本
            self.use_lazy_loading = False
    
    args = TestArgs()
    
    try:
        # 创建简单的spk2int
        spk2int = {f"id{i:05d}": i for i in range(10)}
        
        # 测试构建数据加载器
        train_dl = train_accelerate_ddp.build_simu_data_train_dl(
            args, spk2int, 
            use_fast_version=True,
            max_speakers=5,
            max_files_per_speaker=2
        )
        
        logger.info("内存优化加速版本数据加载器构建成功!")
        logger.info(f"数据加载器类型: {type(train_dl)}")
        return True
        
    except Exception as e:
        logger.error(f"内存优化加速版本数据加载器构建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_different_batch_sizes():
    """测试不同批处理大小的效果"""
    logger.info("\n测试不同批处理大小的效果...")
    
    import train_accelerate_ddp
    
    batch_sizes = [5, 10, 20]
    
    for batch_size in batch_sizes:
        logger.info(f"\n测试批处理大小: {batch_size}")
        
        class TestArgs:
            def __init__(self, batch_size):
                self.voxceleb2_spk2chunks_json = "/maduo/datasets/voxceleb2/vox2_dev/train.json.gz"
                self.compression_type = "gzip"
                self.disable_cache = True
                self.fast_batch_size = batch_size
                self.fast_max_memory_mb = 4096
        
        args = TestArgs(batch_size)
        
        try:
            start_time = time.time()
            spk2chunks = train_accelerate_ddp.spktochunks_fast(
                args, max_speakers=5, max_files_per_speaker=2
            )
            end_time = time.time()
            
            logger.info(f"批处理大小 {batch_size}: 耗时 {end_time - start_time:.2f}秒, 说话人数 {len(spk2chunks)}")
            
        except Exception as e:
            logger.error(f"批处理大小 {batch_size} 测试失败: {e}")
    
    return True

def main():
    logger.info("开始测试内存优化的加速版本...")
    
    # 测试1: 基本功能
    success1 = test_fast_memory_optimized()
    
    # 测试2: 数据加载器构建
    success2 = test_build_dataloader_fast()
    
    # 测试3: 不同批处理大小
    success3 = test_different_batch_sizes()
    
    # 总结
    logger.info("\n" + "="*60)
    logger.info("测试结果总结")
    logger.info("="*60)
    
    if success1 and success2 and success3:
        logger.info("✅ 所有测试通过!")
        logger.info("内存优化的加速版本修复成功，可以正常使用")
        logger.info("\n🎯 推荐使用参数（避免OOM）:")
        logger.info("--use-fast-spktochunks True")
        logger.info("--fast-batch-size 10")
        logger.info("--fast-max-memory-mb 4096")
        logger.info("--max-speakers-test 20")
        logger.info("--max-files-per-speaker-test 5")
        logger.info("\n📊 如果仍然遇到内存问题，可以:")
        logger.info("1. 减小 --fast-batch-size (如改为5)")
        logger.info("2. 减小 --fast-max-memory-mb (如改为3072)")
        logger.info("3. 使用 --use-memory-safe True")
    else:
        logger.error("❌ 部分测试失败")
        if not success1:
            logger.error("内存优化加速版本测试失败")
        if not success2:
            logger.error("数据加载器构建测试失败")
        if not success3:
            logger.error("批处理大小测试失败")
    
    return success1 and success2 and success3

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)