#!/bin/bash

# 测试单GPU环境的torchrun
echo "测试单GPU环境的torchrun"
echo "======================="

# 检测GPU数量
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
if [ $NUM_GPUS -eq 0 ]; then
    echo "未检测到GPU，使用CPU模式"
    NUM_PROCESSES=2
    USE_CPU=true
elif [ $NUM_GPUS -eq 1 ]; then
    echo "检测到1个GPU，使用单GPU多进程模式"
    NUM_PROCESSES=2  # 测试时使用较少的进程数
    USE_CPU=false
else
    echo "检测到 $NUM_GPUS 个GPU，使用多GPU模式"
    NUM_PROCESSES=2  # 测试时使用较少的进程数
    USE_CPU=false
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

# 如果是单GPU环境，强制使用gloo后端
if [ "$USE_CPU" = true ] || [ $NUM_GPUS -eq 1 ]; then
    echo "使用gloo后端避免NCCL冲突"
    export NCCL_P2P_DISABLE=1
    export NCCL_IB_DISABLE=1
fi

# 运行CPU测试
echo "运行CPU版本测试..."
torchrun \
    --nproc_per_node=$NUM_PROCESSES \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    test_torchrun_cpu.py

echo ""
echo "CPU版本测试完成！"

# 如果CPU测试成功，询问是否运行GPU测试
if [ "$USE_CPU" = false ]; then
    echo ""
    read -p "CPU测试是否成功？是否运行GPU版本测试？(y/n): " choice
    
    if [[ $choice == "y" || $choice == "Y" ]]; then
        echo ""
        echo "运行GPU版本测试..."
        torchrun \
            --nproc_per_node=$NUM_PROCESSES \
            --nnodes=1 \
            --node_rank=0 \
            --master_addr=localhost \
            --master_port=29501 \
            test_torchrun_simple.py
    fi
fi 