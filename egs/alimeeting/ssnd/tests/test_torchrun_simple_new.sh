#!/bin/bash

# 测试新的简化torchrun版本
echo "测试新的简化torchrun版本"
echo "========================"

# 检测GPU数量
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
if [ $NUM_GPUS -eq 0 ]; then
    echo "未检测到GPU，使用CPU模式"
    NUM_PROCESSES=2
elif [ $NUM_GPUS -eq 1 ]; then
    echo "检测到1个GPU，使用单GPU多进程模式"
    NUM_PROCESSES=2  # 测试时使用较少的进程数
else
    echo "检测到 $NUM_GPUS 个GPU，使用多GPU模式"
    NUM_PROCESSES=2  # 测试时使用较少的进程数
fi

echo "使用进程数: $NUM_PROCESSES"
echo ""

# 检查torchrun是否可用
if ! command -v torchrun &> /dev/null; then
    echo "错误: torchrun 命令不可用"
    echo "请确保已安装PyTorch并正确配置环境"
    exit 1
fi

# 设置环境变量
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# 显示系统内存信息
echo "系统内存信息:"
free -h
echo ""

# 显示GPU内存信息（如果有GPU）
if command -v nvidia-smi &> /dev/null; then
    echo "GPU内存信息:"
    nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv
    echo ""
fi

# 运行新的简化版本
NUM_PROCESSES=4  # 减少进程数，避免内存不足
echo "运行新的简化torchrun版本（使用${NUM_PROCESSES}个进程）..."
torchrun \
    --nproc_per_node=$NUM_PROCESSES \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    ssnd/remove_silent_and_get_spk2chunks_torchrun.py \
    --voxceleb2-dataset-dir /maduo/datasets/voxceleb2/vox2_dev/ \
    --out-text /maduo/datasets/voxceleb2/vox2_dev/train_torchrun_simple.json

echo ""
echo "新的简化torchrun版本测试完成！"

# 检查输出文件
OUTPUT_FILE="/maduo/datasets/voxceleb2/vox2_dev/train_torchrun_simple.json"
if [ -f "$OUTPUT_FILE" ]; then
    echo ""
    echo "输出文件统计:"
    echo "文件大小: $(du -h $OUTPUT_FILE | cut -f1)"
    echo "行数: $(wc -l < $OUTPUT_FILE)"
else
    echo "警告: 输出文件未找到"
fi 
