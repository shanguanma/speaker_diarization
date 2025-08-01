#!/bin/bash

# 保守的torchrun测试脚本
# 使用更少的进程和更小的数据集进行测试

echo "保守的torchrun测试"
echo "=================="

# 检测GPU数量
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
if [ $NUM_GPUS -eq 0 ]; then
    echo "未检测到GPU，使用CPU模式"
    NUM_PROCESSES=2
elif [ $NUM_GPUS -eq 1 ]; then
    echo "检测到1个GPU，使用单GPU模式"
    NUM_PROCESSES=2  # 保守设置
else
    echo "检测到 $NUM_GPUS 个GPU，使用多GPU模式"
    NUM_PROCESSES=2  # 保守设置
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

# 显示系统资源信息
echo "系统资源信息:"
echo "CPU核心数: $(nproc)"
echo "内存信息:"
free -h
echo ""

if command -v nvidia-smi &> /dev/null; then
    echo "GPU信息:"
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv
    echo ""
fi

# 创建小规模测试数据集
echo "创建小规模测试数据集..."
TEST_DIR="/tmp/voxceleb2_test"
mkdir -p $TEST_DIR

# 创建测试的wav.scp和spk2utt文件
cat > $TEST_DIR/wav.scp << EOF
test1 /tmp/test_audio_1.wav
test2 /tmp/test_audio_2.wav
test3 /tmp/test_audio_3.wav
test4 /tmp/test_audio_4.wav
test5 /tmp/test_audio_5.wav
EOF

cat > $TEST_DIR/spk2utt << EOF
spk1 test1 test2
spk2 test3 test4 test5
EOF

# 创建测试音频文件
echo "创建测试音频文件..."
for i in {1..5}; do
    python3 -c "
import numpy as np
import soundfile as sf
t = np.linspace(0, 5, 80000, False)
audio = 0.5 * np.sin(2 * np.pi * 1000 * t)
sf.write(f'/tmp/test_audio_{i}.wav', audio, 16000)
print(f'创建测试音频文件: /tmp/test_audio_{i}.wav')
"
done

echo "测试数据集创建完成"
echo ""

# 运行保守测试
echo "运行保守测试..."
torchrun \
    --nproc_per_node=$NUM_PROCESSES \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    remove_silent_and_get_spk2chunks_torchrun.py \
    --voxceleb2-dataset-dir $TEST_DIR \
    --out-text /tmp/test_output.json

echo ""
echo "保守测试完成！"

# 检查输出文件
if [ -f "/tmp/test_output.json" ]; then
    echo ""
    echo "测试输出文件统计:"
    echo "文件大小: $(du -h /tmp/test_output.json | cut -f1)"
    echo "行数: $(wc -l < /tmp/test_output.json)"
    echo ""
    echo "输出文件内容预览:"
    head -5 /tmp/test_output.json
else
    echo "警告: 测试输出文件未找到"
fi

# 清理测试文件
echo ""
echo "清理测试文件..."
rm -rf $TEST_DIR
rm -f /tmp/test_audio_*.wav
rm -f /tmp/test_output.json

echo "测试完成！" 