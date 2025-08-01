#!/bin/bash

# 解决VAD张量尺寸不匹配问题的运行脚本
# 提供多种处理模式选择

echo "VAD问题修复版本 - 多种处理模式"
echo "=================================="

# 设置数据集路径
VOXCELEB2_DIR="/maduo/datasets/voxceleb2/vox2_dev/"
OUTPUT_FILE="/maduo/datasets/voxceleb2/vox2_dev/train_vad_fixed.json"

echo "数据集路径: $VOXCELEB2_DIR"
echo "输出文件: $OUTPUT_FILE"
echo ""

# 显示处理模式选项
echo "请选择处理模式:"
echo "1) 串行处理 (最安全，避免所有并发问题)"
echo "2) 线程池处理 (8个线程，较低并发)"
echo "3) 线程池处理 (16个线程，中等并发)"
echo "4) 进程池处理 (8个进程，避免线程冲突)"
echo "5) 进程池处理 (16个进程，较高并发)"

read -p "请输入选择 (1-5): " choice

case $choice in
    1)
        echo "选择: 串行处理"
        python remove_silent_and_get_spk2chunks.py \
            --voxceleb2-dataset-dir $VOXCELEB2_DIR \
            --out-text $OUTPUT_FILE \
            --serial
        ;;
    2)
        echo "选择: 线程池处理 (8个线程)"
        python remove_silent_and_get_spk2chunks.py \
            --voxceleb2-dataset-dir $VOXCELEB2_DIR \
            --out-text $OUTPUT_FILE \
            --max-workers 8
        ;;
    3)
        echo "选择: 线程池处理 (16个线程)"
        python remove_silent_and_get_spk2chunks.py \
            --voxceleb2-dataset-dir $VOXCELEB2_DIR \
            --out-text $OUTPUT_FILE \
            --max-workers 16
        ;;
    4)
        echo "选择: 进程池处理 (8个进程)"
        python remove_silent_and_get_spk2chunks.py \
            --voxceleb2-dataset-dir $VOXCELEB2_DIR \
            --out-text $OUTPUT_FILE \
            --max-workers 8 \
            --use-process-pool
        ;;
    5)
        echo "选择: 进程池处理 (16个进程)"
        python remove_silent_and_get_spk2chunks.py \
            --voxceleb2-dataset-dir $VOXCELEB2_DIR \
            --out-text $OUTPUT_FILE \
            --max-workers 16 \
            --use-process-pool
        ;;
    *)
        echo "无效选择，使用默认串行处理"
        python remove_silent_and_get_spk2chunks.py \
            --voxceleb2-dataset-dir $VOXCELEB2_DIR \
            --out-text $OUTPUT_FILE \
            --serial
        ;;
esac

echo ""
echo "处理完成！"
echo "输出文件: $OUTPUT_FILE"

# 显示结果统计
if [ -f "$OUTPUT_FILE" ]; then
    echo ""
    echo "输出文件统计:"
    echo "文件大小: $(du -h $OUTPUT_FILE | cut -f1)"
    echo "行数: $(wc -l < $OUTPUT_FILE)"
fi 