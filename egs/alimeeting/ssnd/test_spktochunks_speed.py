#!/usr/bin/env python3
"""
测试spktochunks函数性能的脚本
比较原始版本、加速版本和懒加载版本的性能差异
"""
import time
import argparse
import logging
import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

def test_spktochunks_performance():
    """测试不同版本的spktochunks函数性能"""
    
    # 导入主模块
    import train_accelerate_ddp
    
    # 创建测试参数
    class TestArgs:
        def __init__(self):
            self.voxceleb2_spk2chunks_json = "/maduo/datasets/voxceleb2/vox2_dev/train.json.gz"
            self.compression_type = "gzip"
            self.use_lazy_loading = False
            self.disable_cache = False
    
    args = TestArgs()
    
    # 测试参数
    max_speakers = 10  # 只测试前10个说话人
    max_files_per_speaker = 5  # 每个说话人只测试5个文件
    
    logger.info("开始性能测试...")
    logger.info(f"测试参数: 最大说话人数={max_speakers}, 每个说话人最大文件数={max_files_per_speaker}")
    
    # 测试1: 原始版本
    logger.info("\n" + "="*50)
    logger.info("测试1: 原始版本")
    logger.info("="*50)
    
    start_time = time.time()
    try:
        spk2chunks_original = train_accelerate_ddp.spktochunks(args)
        original_time = time.time() - start_time
        original_count = sum(len(chunks) for chunks in spk2chunks_original.values())
        logger.info(f"原始版本完成，耗时: {original_time:.2f}秒，处理了 {original_count} 个音频文件")
    except Exception as e:
        logger.error(f"原始版本测试失败: {e}")
        original_time = float('inf')
        original_count = 0
    
    # 测试2: 加速版本
    logger.info("\n" + "="*50)
    logger.info("测试2: 加速版本")
    logger.info("="*50)
    
    start_time = time.time()
    try:
        spk2chunks_fast = train_accelerate_ddp.spktochunks_fast(
            args, max_speakers, max_files_per_speaker, use_cache=False
        )
        fast_time = time.time() - start_time
        fast_count = sum(len(chunks) for chunks in spk2chunks_fast.values())
        logger.info(f"加速版本完成，耗时: {fast_time:.2f}秒，处理了 {fast_count} 个音频文件")
    except Exception as e:
        logger.error(f"加速版本测试失败: {e}")
        fast_time = float('inf')
        fast_count = 0
    
    # 测试3: 懒加载版本
    logger.info("\n" + "="*50)
    logger.info("测试3: 懒加载版本")
    logger.info("="*50)
    
    start_time = time.time()
    try:
        spk2chunks_lazy = train_accelerate_ddp.spktochunks_lazy(
            args, max_speakers, max_files_per_speaker
        )
        lazy_init_time = time.time() - start_time
        logger.info(f"懒加载版本初始化完成，耗时: {lazy_init_time:.2f}秒")
        
        # 测试懒加载的实际访问时间
        start_time = time.time()
        total_chunks = 0
        for spk_id in list(spk2chunks_lazy.keys())[:3]:  # 只访问前3个说话人
            chunks = spk2chunks_lazy[spk_id]
            total_chunks += len(chunks)
        lazy_access_time = time.time() - start_time
        logger.info(f"懒加载版本访问完成，耗时: {lazy_access_time:.2f}秒，处理了 {total_chunks} 个音频文件")
        lazy_total_time = lazy_init_time + lazy_access_time
    except Exception as e:
        logger.error(f"懒加载版本测试失败: {e}")
        lazy_total_time = float('inf')
        total_chunks = 0
    
    # 性能对比
    logger.info("\n" + "="*50)
    logger.info("性能对比结果")
    logger.info("="*50)
    
    if original_time != float('inf'):
        logger.info(f"原始版本: {original_time:.2f}秒 ({original_count} 个文件)")
    
    if fast_time != float('inf'):
        speedup = original_time / fast_time if original_time != float('inf') else float('inf')
        logger.info(f"加速版本: {fast_time:.2f}秒 ({fast_count} 个文件), 加速比: {speedup:.2f}x")
    
    if lazy_total_time != float('inf'):
        speedup = original_time / lazy_total_time if original_time != float('inf') else float('inf')
        logger.info(f"懒加载版本: {lazy_total_time:.2f}秒 ({total_chunks} 个文件), 加速比: {speedup:.2f}x")
    
    # 推荐
    logger.info("\n" + "="*50)
    logger.info("推荐")
    logger.info("="*50)
    
    if fast_time < lazy_total_time and fast_time < original_time:
        logger.info("推荐使用加速版本 (spktochunks_fast)")
    elif lazy_total_time < fast_time and lazy_total_time < original_time:
        logger.info("推荐使用懒加载版本 (spktochunks_lazy)")
    else:
        logger.info("建议使用原始版本或检查数据文件")

def test_cache_performance():
    """测试缓存功能"""
    logger.info("\n" + "="*50)
    logger.info("测试缓存功能")
    logger.info("="*50)
    
    import train_accelerate_ddp
    
    class TestArgs:
        def __init__(self):
            self.voxceleb2_spk2chunks_json = "/maduo/datasets/voxceleb2/vox2_dev/train.json.gz"
            self.compression_type = "gzip"
            self.disable_cache = False
    
    args = TestArgs()
    max_speakers = 5
    max_files_per_speaker = 3
    
    # 第一次运行（无缓存）
    logger.info("第一次运行（无缓存）...")
    start_time = time.time()
    spk2chunks_1 = train_accelerate_ddp.spktochunks_fast(
        args, max_speakers, max_files_per_speaker, use_cache=True
    )
    first_run_time = time.time() - start_time
    logger.info(f"第一次运行耗时: {first_run_time:.2f}秒")
    
    # 第二次运行（有缓存）
    logger.info("第二次运行（有缓存）...")
    start_time = time.time()
    spk2chunks_2 = train_accelerate_ddp.spktochunks_fast(
        args, max_speakers, max_files_per_speaker, use_cache=True
    )
    second_run_time = time.time() - start_time
    logger.info(f"第二次运行耗时: {second_run_time:.2f}秒")
    
    # 计算缓存效果
    if first_run_time > 0:
        cache_speedup = first_run_time / second_run_time
        logger.info(f"缓存加速比: {cache_speedup:.2f}x")
    
    # 验证结果一致性
    if len(spk2chunks_1) == len(spk2chunks_2):
        logger.info("缓存结果验证: 通过")
    else:
        logger.warning("缓存结果验证: 失败")

def main():
    parser = argparse.ArgumentParser(description="测试spktochunks函数性能")
    parser.add_argument("--test-cache", action="store_true", help="测试缓存功能")
    
    args = parser.parse_args()
    
    if args.test_cache:
        test_cache_performance()
    else:
        test_spktochunks_performance()

if __name__ == "__main__":
    main() 