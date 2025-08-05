#!/usr/bin/env python3
"""
测试脚本：处理前三个batch的数据并验证结果
"""
import os
import sys
import json
import gzip
import argparse
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

def test_first_three_batches():
    """测试处理前三个batch的数据"""
    
    # 设置测试参数
    dataset_dir = "/maduo/datasets/voxceleb2/vox2_dev/"
    output_file = "/maduo/datasets/voxceleb2/vox2_dev/test_first_3_batches.json.gz"
    
    # 构建命令
    cmd = f"""python remove_silent_and_get_spk2chunks.py \\
        --voxceleb2-dataset-dir {dataset_dir} \\
        --out-text {output_file} \\
        --format jsonl_gzip \\
        --compression-level 6 \\
        --max-batches 3"""
    
    logger.info("开始测试前三个batch的数据处理...")
    logger.info(f"执行命令: {cmd}")
    
    # 执行命令
    import subprocess
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("命令执行成功!")
            logger.info("标准输出:")
            print(result.stdout)
        else:
            logger.error("命令执行失败!")
            logger.error("错误输出:")
            print(result.stderr)
            return False
            
    except Exception as e:
        logger.error(f"执行命令时发生错误: {e}")
        return False
    
    # 验证生成的文件
    logger.info("验证生成的文件...")
    
    # 检查主输出文件
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file)
        logger.info(f"主输出文件存在: {output_file} (大小: {file_size} 字节)")
        
        # 尝试读取文件
        try:
            with gzip.open(output_file, 'rt', encoding='utf-8') as f:
                line_count = 0
                for line in f:
                    line = line.strip()
                    if line:
                        line_count += 1
                        if line_count <= 3:  # 只显示前3行
                            data = json.loads(line)
                            logger.info(f"第{line_count}行: 说话人 {data['spk_id']}, 文件数 {len(data['wav_paths'])}")
                
                logger.info(f"总共读取了 {line_count} 行数据")
                
        except Exception as e:
            logger.error(f"读取文件失败: {e}")
            return False
    else:
        logger.error(f"主输出文件不存在: {output_file}")
        return False
    
    # 检查临时batch文件
    for batch_num in range(1, 4):
        temp_file = f"{output_file}.batch_{batch_num}"
        if os.path.exists(temp_file):
            file_size = os.path.getsize(temp_file)
            logger.info(f"批次{batch_num}临时文件存在: {temp_file} (大小: {file_size} 字节)")
        else:
            logger.warning(f"批次{batch_num}临时文件不存在: {temp_file}")
    
    # 检查失败文件
    failed_file = f"{output_file}.failed"
    if os.path.exists(failed_file):
        file_size = os.path.getsize(failed_file)
        logger.info(f"失败文件存在: {failed_file} (大小: {file_size} 字节)")
        
        # 读取失败信息
        try:
            with gzip.open(failed_file, 'rt', encoding='utf-8') as f:
                failed_data = json.load(f)
                logger.info(f"失败文件数量: {len(failed_data)}")
                if failed_data:
                    logger.info(f"第一个失败项: {failed_data[0]}")
        except Exception as e:
            logger.error(f"读取失败文件失败: {e}")
    else:
        logger.info("没有失败文件")
    
    logger.info("测试完成!")
    return True

def main():
    parser = argparse.ArgumentParser(description="测试前三个batch的数据处理")
    parser.add_argument("--dataset-dir", type=str, 
                       default="/maduo/datasets/voxceleb2/vox2_dev/",
                       help="数据集目录")
    parser.add_argument("--output-file", type=str,
                       default="/maduo/datasets/voxceleb2/vox2_dev/test_first_3_batches.json.gz",
                       help="输出文件路径")
    
    args = parser.parse_args()
    
    # 检查数据集目录是否存在
    if not os.path.exists(args.dataset_dir):
        logger.error(f"数据集目录不存在: {args.dataset_dir}")
        return 1
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"创建输出目录: {output_dir}")
    
    # 运行测试
    success = test_first_three_batches()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main()) 