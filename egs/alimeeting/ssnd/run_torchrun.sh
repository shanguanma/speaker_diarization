#!/bin/bash

# 使用torchrun进行多进程分布式处理
# 避免GPU线程冲突问题

echo "使用torchrun进行多进程分布式处理"
echo "=================================="

# 设置数据集路径
VOXCELEB2_DIR="/maduo/datasets/voxceleb2/vox2_dev/"
OUTPUT_FILE="/maduo/datasets/voxceleb2/vox2_dev/train_torchrun.json"

# 设置进程数（根据GPU数量调整）
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
if [ $NUM_GPUS -eq 0 ]; then
    echo "未检测到GPU，使用CPU模式"
    NUM_PROCESSES=4  # CPU模式使用4个进程
else
    echo "检测到 $NUM_GPUS 个GPU"
    NUM_PROCESSES=$NUM_GPUS
fi

echo "数据集路径: $VOXCELEB2_DIR"
echo "输出文件: $OUTPUT_FILE"
echo "进程数: $NUM_PROCESSES"
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

# 运行torchrun
echo "启动torchrun多进程处理..."
torchrun \
    --nproc_per_node=$NUM_PROCESSES \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    remove_silent_and_get_spk2chunks_torchrun.py \
    --voxceleb2-dataset-dir $VOXCELEB2_DIR \
    --out-text $OUTPUT_FILE

echo ""
echo "torchrun处理完成！"
echo "输出文件: $OUTPUT_FILE"

# 显示结果统计
if [ -f "$OUTPUT_FILE" ]; then
    echo ""
    echo "输出文件统计:"
    echo "文件大小: $(du -h $OUTPUT_FILE | cut -f1)"
    echo "行数: $(wc -l < $OUTPUT_FILE)"
fi 