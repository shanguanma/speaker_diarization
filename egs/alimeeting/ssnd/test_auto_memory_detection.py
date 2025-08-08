#!/usr/bin/env python3
"""
测试自动内存检测功能
"""

import psutil
import argparse

def test_memory_detection():
    """测试内存检测功能"""
    print("=== 自动内存检测测试 ===\n")
    
    # 获取系统内存信息
    system_memory = psutil.virtual_memory()
    total_memory_mb = system_memory.total / 1024 / 1024
    available_memory_mb = system_memory.available / 1024 / 1024
    used_memory_mb = system_memory.used / 1024 / 1024
    
    print(f"系统内存信息:")
    print(f"  总内存: {total_memory_mb:.0f} MB ({total_memory_mb/1024:.1f} GB)")
    print(f"  已使用: {used_memory_mb:.0f} MB ({used_memory_mb/1024:.1f} GB)")
    print(f"  可用内存: {available_memory_mb:.0f} MB ({available_memory_mb/1024:.1f} GB)")
    print(f"  使用率: {system_memory.percent:.1f}%\n")
    
    # 测试不同的内存使用比例
    ratios = [0.5, 0.6, 0.7, 0.8]
    
    print("不同内存使用比例的计算结果:")
    for ratio in ratios:
        max_memory_mb = min(available_memory_mb * ratio, total_memory_mb * ratio)
        max_memory_mb = int(max_memory_mb)
        print(f"  {ratio*100:.0f}%: {max_memory_mb} MB ({max_memory_mb/1024:.1f} GB)")
    
    print("\n=== 推荐配置 ===")
    
    # 根据总内存推荐配置
    if total_memory_mb < 16384:  # < 16GB
        print("小内存机器 (< 16GB):")
        print("  --memory-usage-ratio 0.5")
        print("  --fast-batch-size 3")
        print("  --fast-sub-batch-size 30")
    elif total_memory_mb < 32768:  # < 32GB
        print("中等内存机器 (16-32GB):")
        print("  --memory-usage-ratio 0.6")
        print("  --fast-batch-size 5")
        print("  --fast-sub-batch-size 50")
    else:  # >= 32GB
        print("大内存机器 (>= 32GB):")
        print("  --memory-usage-ratio 0.7")
        print("  --fast-batch-size 8")
        print("  --fast-sub-batch-size 80")
    
    print("\n=== 测试命令 ===")
    print("python train_accelerate_ddp.py \\")
    print("    --fast-batch-size 5 \\")
    print("    --fast-sub-batch-size 50 \\")
    print("    --strict-memory-check False")

if __name__ == "__main__":
    test_memory_detection() 