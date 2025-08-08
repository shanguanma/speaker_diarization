#!/usr/bin/env python3
"""
测试内存安全版本修复的脚本
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

def test_memory_safe_version():
    """测试内存安全版本"""
    
    # 导入主模块
    import train_accelerate_ddp
    
    # 创建测试参数
    class TestArgs:
        def __init__(self):
            self.voxceleb2_spk2chunks_json = "/maduo/datasets/voxceleb2/vox2_dev/train.json.gz"
            self.compression_type = "gzip"
            self.use_memory_safe = True
            self.disable_cache = True  # 禁用缓存进行真实测试
    
    args = TestArgs()
    
    # 测试参数（小数据集）
    max_speakers = 5
    max_files_per_speaker = 2
    
    logger.info("测试内存安全版本...")
    logger.info(f"测试参数: 最大说话人数={max_speakers}, 每个说话人最大文件数={max_files_per_speaker}")
    
    try:
        # 测试内存安全版本
        start_time = time.time()
        spk2chunks = train_accelerate_ddp.spktochunks_memory_safe(
            args, max_speakers, max_files_per_speaker
        )
        end_time = time.time()
        
        # 检查返回的数据结构
        logger.info(f"内存安全版本完成，耗时: {end_time - start_time:.2f}秒")
        logger.info(f"说话人数量: {len(spk2chunks)}")
        logger.info(f"说话人列表: {list(spk2chunks.keys())}")
        
        # 测试数据访问
        total_files = 0
        for spk_id in spk2chunks.keys():
            chunks = spk2chunks[spk_id]
            total_files += len(chunks)
            logger.info(f"说话人 {spk_id}: {len(chunks)} 个音频文件")
        
        logger.info(f"总文件数: {total_files}")
        logger.info("内存安全版本测试成功!")
        return True
        
    except Exception as e:
        logger.error(f"内存安全版本测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_build_dataloader():
    """测试build_simu_data_train_dl函数"""
    logger.info("\n测试build_simu_data_train_dl函数...")
    
    import train_accelerate_ddp
    
    class TestArgs:
        def __init__(self):
            self.voxceleb2_spk2chunks_json = "/maduo/datasets/voxceleb2/vox2_dev/train.json.gz"
            self.compression_type = "gzip"
            self.use_memory_safe = True
            self.disable_cache = True
            self.musan_path = None
            self.rir_path = None
            self.noise_ratio = 0.0
    
    args = TestArgs()
    
    try:
        # 创建简单的spk2int
        spk2int = {f"id{i:05d}": i for i in range(10)}
        
        # 测试构建数据加载器
        train_dl = train_accelerate_ddp.build_simu_data_train_dl(
            args, spk2int, 
            use_fast_version=True,
            max_speakers=3,
            max_files_per_speaker=2
        )
        
        logger.info("build_simu_data_train_dl测试成功!")
        logger.info(f"数据加载器类型: {type(train_dl)}")
        return True
        
    except Exception as e:
        logger.error(f"build_simu_data_train_dl测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_monitoring():
    """测试内存监控功能"""
    logger.info("\n测试内存监控功能...")
    
    try:
        import psutil
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        logger.info(f"当前内存使用: {initial_memory:.1f} MB")
        
        # 创建一些数据来测试内存监控
        data = []
        for i in range(100):
            data.append([1.0] * 1000)  # 创建一些数据
            
            current_memory = process.memory_info().rss / 1024 / 1024
            if i % 20 == 0:
                logger.info(f"步骤 {i}: 内存使用 {current_memory:.1f} MB")
        
        final_memory = process.memory_info().rss / 1024 / 1024
        logger.info(f"最终内存使用: {final_memory:.1f} MB (增加: {final_memory - initial_memory:.1f} MB)")
        
        # 清理
        del data
        import gc
        gc.collect()
        
        after_gc_memory = process.memory_info().rss / 1024 / 1024
        logger.info(f"垃圾回收后内存: {after_gc_memory:.1f} MB")
        
        logger.info("内存监控功能测试成功!")
        return True
        
    except Exception as e:
        logger.error(f"内存监控功能测试失败: {e}")
        return False

def main():
    logger.info("开始测试内存安全版本修复...")
    
    # 测试1: 内存安全版本
    success1 = test_memory_safe_version()
    
    # 测试2: 数据加载器构建
    success2 = test_build_dataloader()
    
    # 测试3: 内存监控
    success3 = test_memory_monitoring()
    
    # 总结
    logger.info("\n" + "="*50)
    logger.info("测试结果总结")
    logger.info("="*50)
    
    if success1 and success2 and success3:
        logger.info("✅ 所有测试通过!")
        logger.info("内存安全版本修复成功，可以正常使用")
        logger.info("\n推荐使用参数:")
        logger.info("--use-memory-safe True")
        logger.info("--max-speakers-test 10")
        logger.info("--max-files-per-speaker-test 5")
    else:
        logger.error("❌ 部分测试失败")
        if not success1:
            logger.error("内存安全版本测试失败")
        if not success2:
            logger.error("数据加载器构建测试失败")
        if not success3:
            logger.error("内存监控功能测试失败")
    
    return success1 and success2 and success3

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)