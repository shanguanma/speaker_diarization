#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
python train_accelerate_ddp_overfit.py \
  --wav-dir /maduo/datasets/alimeeting/Train_Ali_far/audio_dir \
  --textgrid-dir  /maduo/datasets/alimeeting/Train_Ali_far/textgrid_dir \
  --speaker-pretrain-model-path /maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin \
  --batch-size 2 \
  --max-speakers 4 \
  --vad-out-len 200 \
  --num-steps 1000