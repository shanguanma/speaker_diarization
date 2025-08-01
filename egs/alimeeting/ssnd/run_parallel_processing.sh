#!/bin/bash

# 并行处理VoxCeleb2数据集的示例脚本
# 使用ThreadPoolExecutor进行并行计算

echo "开始并行处理VoxCeleb2数据集..."

# 设置数据集路径
VOXCELEB2_DIR="/maduo/datasets/voxceleb2/vox2_dev/"
OUTPUT_FILE="/maduo/datasets/voxceleb2/vox2_dev/train_parallel.json"

# 设置并行线程数（可以根据系统配置调整）
MAX_WORKERS=16

# 运行并行处理脚本
python remove_silent_and_get_spk2chunks.py \
    --voxceleb2-dataset-dir $VOXCELEB2_DIR \
    --out-text $OUTPUT_FILE \
    --max-workers $MAX_WORKERS

echo "并行处理完成！"
echo "输出文件: $OUTPUT_FILE" 