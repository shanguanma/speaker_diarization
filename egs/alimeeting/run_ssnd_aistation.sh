#!/usr/bin/env bash

stage=0
stop_stage=1000
. utils/parse_options.sh
. path_for_dia_cuda11.8_py3111_aistation.sh
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
   export NCCL_DEBUG=INFO
   export PYTHONFAULTHANDLER=1
   musan_path=/maduo/datasets/musan
   rir_path=/maduo/datasets/RIRS_NOISES  
   train_wav_dir=/maduo/datasets/alimeeting/Train_Ali_far/audio_dir
   train_textgrid_dir=/maduo/datasets/alimeeting/Train_Ali_far/textgrid_dir
   valid_wav_dir=/maduo/datasets/alimeeting/Eval_Ali/Eval_Ali_far/audio_dir
   valid_textgrid_dir=/maduo/datasets/alimeeting/Eval_Ali/Eval_Ali_far/textgrid_dir
   CUDA_VISIABLE_DEVICES=0,1\
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15915 \
   ssnd/train_accelerate_ddp.py\
    --debug false\
    --world-size 2 \
    --num-epochs 20\
    --batch-size 64 \
    --start-epoch 1\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --grad-clip false\
    --lr 1e-4\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --train_wav_dir $train_wav_dir\
    --train_textgrid_dir $train_textgrid_dir\
    --valid_wav_dir $valid_wav_dir\
    --valid_textgrid_dir $valid_textgrid_dir
     
fi
