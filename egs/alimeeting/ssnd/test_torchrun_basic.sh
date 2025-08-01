#!/bin/bash

# 测试torchrun基本功能
echo "测试torchrun基本功能"
echo "===================="

# 检查torchrun是否可用
if ! command -v torchrun &> /dev/null; then
    echo "错误: torchrun 命令不可用"
    echo "请确保已安装PyTorch并正确配置环境"
    exit 1
fi

# 设置环境变量
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# 检测GPU数量
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
if [ $NUM_GPUS -eq 0 ]; then
    echo "未检测到GPU，使用CPU模式"
    NUM_PROCESSES=2
else
    echo "检测到 $NUM_GPUS 个GPU"
    NUM_PROCESSES=$NUM_GPUS
fi

echo "使用进程数: $NUM_PROCESSES"
echo ""

# 运行简化测试
echo "运行简化测试..."
torchrun \
    --nproc_per_node=$NUM_PROCESSES \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    test_torchrun_simple.py

echo ""
echo "简化测试完成！"

# 如果简化测试成功，询问是否运行完整测试
echo ""
read -p "简化测试是否成功？是否运行完整的torchrun测试？(y/n): " choice

if [[ $choice == "y" || $choice == "Y" ]]; then
    echo ""
    echo "运行完整的torchrun测试..."
    torchrun \
        --nproc_per_node=$NUM_PROCESSES \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=29501 \
        remove_silent_and_get_spk2chunks_torchrun.py \
        --voxceleb2-dataset-dir /maduo/datasets/voxceleb2/vox2_dev/ \
        --out-text /maduo/datasets/voxceleb2/vox2_dev/train_torchrun_test.json
fi 