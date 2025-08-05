#!/usr/bin/env python3
"""
测试日志功能的简单脚本
"""
import logging
import sys

def test_logging():
    """测试日志功能"""
    print("开始测试日志功能...")
    
    # 方法1: 使用basicConfig
    print("\n方法1: 使用basicConfig")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s',
        force=True  # 强制重新配置
    )
    logger1 = logging.getLogger(__name__)
    logger1.info("这是方法1的测试日志")
    
    # 方法2: 手动配置处理器
    print("\n方法2: 手动配置处理器")
    # 清除现有处理器
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    formatter = logging.Formatter('%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(console_handler)
    
    logger2 = logging.getLogger(__name__)
    logger2.info("这是方法2的测试日志")
    
    # 方法3: 直接打印
    print("\n方法3: 直接打印")
    print("2024-01-01 12:00:00 INFO [test_logging.py:45] 这是直接打印的测试日志")
    
    print("\n日志测试完成!")

if __name__ == "__main__":
    test_logging() 
