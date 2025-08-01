#!/bin/bash

# 单GPU环境的torchrun运行脚本
# 使用gloo后端避免NCCL冲突

echo "单GPU环境的torchrun处理"
echo "======================="

# 设置数据集路径
VOXCELEB2_DIR="/maduo/datasets/voxceleb2/vox2_dev/"
OUTPUT_FILE="/maduo/datasets/voxceleb2/vox2_dev/train_torchrun_single_gpu.json"

# 检测GPU数量
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
if [ $NUM_GPUS -eq 0 ]; then
    echo "未检测到GPU，使用CPU模式"
    NUM_PROCESSES=4
    USE_CPU=true
elif [ $NUM_GPUS -eq 1 ]; then
    echo "检测到1个GPU，使用单GPU多进程模式"
    NUM_PROCESSES=4  # 单GPU环境建议使用4个进程
    USE_CPU=false
else
    echo "检测到 $NUM_GPUS 个GPU，建议使用多GPU模式"
    NUM_PROCESSES=$NUM_GPUS
    USE_CPU=false
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

# 如果是单GPU环境，强制使用gloo后端
if [ "$USE_CPU" = true ] || [ $NUM_GPUS -eq 1 ]; then
    echo "使用gloo后端避免NCCL冲突"
    export NCCL_P2P_DISABLE=1
    export NCCL_IB_DISABLE=1
fi

# 运行torchrun
echo "启动torchrun处理..."
if [ "$USE_CPU" = true ]; then
    echo "使用CPU模式"
    torchrun \
        --nproc_per_node=$NUM_PROCESSES \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=29500 \
        remove_silent_and_get_spk2chunks_torchrun.py \
        --voxceleb2-dataset-dir $VOXCELEB2_DIR \
        --out-text $OUTPUT_FILE
else
    echo "使用GPU模式"
    torchrun \
        --nproc_per_node=$NUM_PROCESSES \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=29500 \
        remove_silent_and_get_spk2chunks_torchrun.py \
        --voxceleb2-dataset-dir $VOXCELEB2_DIR \
        --out-text $OUTPUT_FILE
fi

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