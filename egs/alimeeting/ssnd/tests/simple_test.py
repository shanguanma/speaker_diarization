#!/usr/bin/env python3
"""
简单的测试脚本，验证主脚本的日志功能
"""
import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_main_script():
    """测试主脚本的日志功能"""
    print("测试主脚本的日志功能...")
    
    try:
        # 导入主脚本
        import remove_silent_and_get_spk2chunks
        
        # 测试日志是否正常工作
        logger = remove_silent_and_get_spk2chunks.logger
        logger.info("测试日志功能正常工作!")
        logger.warning("这是一个警告信息")
        logger.error("这是一个错误信息")
        
        print("日志测试成功!")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_main_script() 