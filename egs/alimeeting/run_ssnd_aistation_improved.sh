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
   speaker_pretrain_model_path=/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin
   extractor_model_type='CAM++_wo_gsp'
   out_bias=-3.0
   exp_dir=/maduo/exp/speaker_diarization/ssnd/ssnd_alimeeting_improved_lr1e-4_batch64_out_bias${out_bias}
   CUDA_VISIABLE_DEVICES=0,1\
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15915 \
   ssnd/train_accelerate_ddp.py\
    --debug true\
    --world-size 2 \
    --num-epochs 30\
    --batch-size 64 \
    --start-epoch 1\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --grad-clip true\
    --lr 1e-4\
    --exp-dir $exp_dir\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --train_wav_dir $train_wav_dir\
    --train_textgrid_dir $train_textgrid_dir\
    --valid_wav_dir $valid_wav_dir\
    --valid_textgrid_dir $valid_textgrid_dir\
    --arcface-margin 0.2\
    --arcface-scale 32.0\
    --mask-prob 0.5\
    --speaker_pretrain_model_path $speaker_pretrain_model_path\
    --extractor_model_type $extractor_model_type\
    --out-bias $out_bias
     
fi 
