#!/bin/bash

# 灵活的torchrun多进程处理脚本
# 支持自定义进程数和GPU配置

echo "灵活的torchrun多进程处理"
echo "=========================="

# 设置数据集路径
VOXCELEB2_DIR="/maduo/datasets/voxceleb2/vox2_dev/"
OUTPUT_FILE="/maduo/datasets/voxceleb2/vox2_dev/train_torchrun.json"

# 显示GPU信息
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
if [ $NUM_GPUS -eq 0 ]; then
    echo "未检测到GPU，将使用CPU模式"
    DEFAULT_PROCESSES=4
else
    echo "检测到 $NUM_GPUS 个GPU"
    DEFAULT_PROCESSES=$NUM_GPUS
fi

echo "数据集路径: $VOXCELEB2_DIR"
echo "输出文件: $OUTPUT_FILE"
echo ""

# 询问用户选择进程数
echo "请选择进程数:"
echo "1) 自动检测 (GPU数量: $DEFAULT_PROCESSES)"
echo "2) 自定义进程数"
echo "3) 单进程模式 (调试用)"

read -p "请输入选择 (1-3): " choice

case $choice in
    1)
        NUM_PROCESSES=$DEFAULT_PROCESSES
        echo "使用自动检测的进程数: $NUM_PROCESSES"
        ;;
    2)
        read -p "请输入进程数: " NUM_PROCESSES
        if ! [[ "$NUM_PROCESSES" =~ ^[0-9]+$ ]] || [ $NUM_PROCESSES -lt 1 ]; then
            echo "无效的进程数，使用默认值: $DEFAULT_PROCESSES"
            NUM_PROCESSES=$DEFAULT_PROCESSES
        fi
        ;;
    3)
        NUM_PROCESSES=1
        echo "使用单进程模式"
        ;;
    *)
        echo "无效选择，使用默认值: $DEFAULT_PROCESSES"
        NUM_PROCESSES=$DEFAULT_PROCESSES
        ;;
esac

echo "最终进程数: $NUM_PROCESSES"
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

# 如果使用GPU，设置CUDA相关环境变量
if [ $NUM_GPUS -gt 0 ]; then
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
    echo "设置CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
fi

# 运行torchrun
echo "启动torchrun多进程处理..."
echo "命令: torchrun --nproc_per_node=$NUM_PROCESSES --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=29500 remove_silent_and_get_spk2chunks_torchrun.py --voxceleb2-dataset-dir $VOXCELEB2_DIR --out-text $OUTPUT_FILE"
echo ""

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