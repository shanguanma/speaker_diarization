#!/bin/bash

# 使用focal BCE loss + 增强正则化的训练脚本
# 这个脚本展示了如何在使用focal loss的同时降低过拟合风险

# 基础配置
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=/path/to/your/project:$PYTHONPATH

# 数据路径
TRAIN_WAV_DIR="/path/to/train/wav"
TRAIN_TEXTGRID_DIR="/path/to/train/textgrid"
VALID_WAV_DIR="/path/to/valid/wav"
VALID_TEXTGRID_DIR="/path/to/valid/textgrid"
SPEAKER_PRETRAIN_MODEL_PATH="/path/to/speaker/model.pth"

# 实验配置
EXP_DIR="exp/ssnd_focal_regularized"
NUM_EPOCHS=30
BATCH_SIZE=32  # 减小batch size来增加正则化效果
MAX_UPDATES=40000
WARMUP_UPDATES=400

# ========== 正则化配置 ==========

# 1. 学习率和优化器配置
LR=1e-4  # 降低学习率
WEIGHT_DECAY=0.01  # 增加权重衰减
GRADIENT_CLIP=1.0  # 梯度裁剪

# 2. Focal Loss参数（更保守的设置）
FOCAL_ALPHA=0.6  # 降低alpha，减少对困难样本的过度关注
FOCAL_GAMMA=1.5  # 降低gamma，减少focal effect

# 3. 其他正则化参数
DROPOUT=0.15  # 增加dropout
LABEL_SMOOTHING=0.1  # 标签平滑
EARLY_STOPPING_PATIENCE=5  # 早停耐心值
EARLY_STOPPING_MIN_DELTA=0.001  # 早停最小改进阈值

# 4. 模型配置
EXTRACTOR_MODEL_TYPE="CAM++_wo_gsp"
FEAT_DIM=80
EMB_DIM=256

echo "Starting training with enhanced regularization..."
echo "Focal Loss + Regularization Configuration:"
echo "  - Learning Rate: $LR"
echo "  - Weight Decay: $WEIGHT_DECAY"
echo "  - Gradient Clip: $GRADIENT_CLIP"
echo "  - Focal Alpha: $FOCAL_ALPHA"
echo "  - Focal Gamma: $FOCAL_GAMMA"
echo "  - Dropout: $DROPOUT"
echo "  - Label Smoothing: $LABEL_SMOOTHING"
echo "  - Early Stopping Patience: $EARLY_STOPPING_PATIENCE"

# 训练命令
python train_accelerate_ddp.py \
    --num-epochs $NUM_EPOCHS \
    --batch-size $BATCH_SIZE \
    --max-updates $MAX_UPDATES \
    --warmup-updates $WARMUP_UPDATES \
    --lr $LR \
    --weight-decay $WEIGHT_DECAY \
    --gradient-clip $GRADIENT_CLIP \
    --dropout $DROPOUT \
    --label-smoothing $LABEL_SMOOTHING \
    --early-stopping-patience $EARLY_STOPPING_PATIENCE \
    --early-stopping-min-delta $EARLY_STOPPING_MIN_DELTA \
    --focal-alpha $FOCAL_ALPHA \
    --focal-gamma $FOCAL_GAMMA \
    --use-standard-bce False \
    --lr-type "PolynomialDecayLR" \
    --exp-dir $EXP_DIR \
    --train_wav_dir $TRAIN_WAV_DIR \
    --train_textgrid_dir $TRAIN_TEXTGRID_DIR \
    --valid_wav_dir $VALID_WAV_DIR \
    --valid_textgrid_dir $VALID_TEXTGRID_DIR \
    --speaker_pretrain_model_path $SPEAKER_PRETRAIN_MODEL_PATH \
    --extractor_model_type $EXTRACTOR_MODEL_TYPE \
    --feat-dim $FEAT_DIM \
    --emb-dim $EMB_DIM \
    --tensorboard True \
    --save-every-n 1500 \
    --keep-last-k 20 \
    --keep-last-epoch 10 \
    --world-size 4 \
    --grad-clip True \
    --debug False

echo "Training completed!"
echo "Check the logs in $EXP_DIR for detailed training information."
echo "Monitor TensorBoard for training curves and overfitting detection." 