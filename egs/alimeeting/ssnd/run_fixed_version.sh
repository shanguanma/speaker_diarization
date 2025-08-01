#!/bin/bash

# 运行修复后的并行处理版本
# 使用推荐的线程数设置，避免VAD模型冲突

echo "运行修复后的并行处理版本..."
echo "使用推荐的线程数设置 (16个线程)"

# 设置数据集路径
VOXCELEB2_DIR="/maduo/datasets/voxceleb2/vox2_dev/"
OUTPUT_FILE="/maduo/datasets/voxceleb2/vox2_dev/train_fixed.json"

# 使用推荐的线程数
MAX_WORKERS=16

echo "数据集路径: $VOXCELEB2_DIR"
echo "输出文件: $OUTPUT_FILE"
echo "线程数: $MAX_WORKERS"
echo ""

# 运行修复后的脚本
python remove_silent_and_get_spk2chunks.py \
    --voxceleb2-dataset-dir $VOXCELEB2_DIR \
    --out-text $OUTPUT_FILE \
    --max-workers $MAX_WORKERS

echo ""
echo "处理完成！"
echo "输出文件: $OUTPUT_FILE"

# 如果处理成功，显示一些统计信息
if [ -f "$OUTPUT_FILE" ]; then
    echo ""
    echo "输出文件统计:"
    echo "文件大小: $(du -h $OUTPUT_FILE | cut -f1)"
    echo "行数: $(wc -l < $OUTPUT_FILE)"
fi 