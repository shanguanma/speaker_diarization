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
   extractor_model_type='CAM++_gsp'
   #out_bias=-0.5
   mask_prob=0.5
   arcface_weight=0.01
   arcface_margin=0.2
   arcface_scale=32.0
   weight_decay=0.001
   bce_alpha=0.75
   bce_gamma=2.0
   max_speakers=4
   exp_dir=/maduo/exp/speaker_diarization/ssnd/ssnd_alimeeting_improved_lr1e-4_batch64_mask_prob_${mask_prob}_arcface_weight_${arcface_weight}_arcface_margin${arcface_margin}_arcface_scale${arcface_scale}
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
    --train_wav_dir $train_wav_dir\
    --train_textgrid_dir $train_textgrid_dir\
    --valid_wav_dir $valid_wav_dir\
    --valid_textgrid_dir $valid_textgrid_dir\
    --arcface-margin $arcface_margin\
    --arcface-scale $arcface_scale\
    --mask-prob $mask_prob\
    --speaker_pretrain_model_path $speaker_pretrain_model_path\
    --extractor_model_type $extractor_model_type\
    --warmup-updates 3000\
    --arcface-weight $arcface_weight\
    --bce-alpha $bce_alpha\
    --bce-gamma $bce_gamma\
    --weight-decay $weight_decay\
    --max-speakers $max_speakers

fi
# the above setting is very fast overfit. about 400 steps is overfit.
# it uses fn build_valid_dl_with_local_spk2int and uild_train_dl_with_local_spk2int
# log is logs/run_ssnd_aistation_improved_stage0_debug27_max_speaker4_local_spk2int_weights_arcface_0.01_mask_prob0.5.log

# compared with stage0, stage1 will use fn build_valid_dl and fn build_train_dl
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
   export NCCL_DEBUG=INFO
   export PYTHONFAULTHANDLER=1
   musan_path=/maduo/datasets/musan
   rir_path=/maduo/datasets/RIRS_NOISES
   train_wav_dir=/maduo/datasets/alimeeting/Train_Ali_far/audio_dir
   train_textgrid_dir=/maduo/datasets/alimeeting/Train_Ali_far/textgrid_dir
   valid_wav_dir=/maduo/datasets/alimeeting/Eval_Ali/Eval_Ali_far/audio_dir
   valid_textgrid_dir=/maduo/datasets/alimeeting/Eval_Ali/Eval_Ali_far/textgrid_dir
   speaker_pretrain_model_path=/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin
   extractor_model_type='CAM++_gsp'
   #out_bias=-0.5
   mask_prob=0.5
   arcface_weight=0.01
   arcface_margin=0.2
   arcface_scale=32.0
   weight_decay=0.001
   bce_alpha=0.75
   bce_gamma=2.0
   max_speakers=4
   exp_dir=/maduo/exp/speaker_diarization/ssnd/ssnd_alimeeting_improved_lr1e-4_batch64_mask_prob_${mask_prob}_arcface_weight_${arcface_weight}_arcface_margin${arcface_margin}_arcface_scale${arcface_scale}_with_global_spk2int
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
    --train_wav_dir $train_wav_dir\
    --train_textgrid_dir $train_textgrid_dir\
    --valid_wav_dir $valid_wav_dir\
    --valid_textgrid_dir $valid_textgrid_dir\
    --arcface-margin $arcface_margin\
    --arcface-scale $arcface_scale\
    --mask-prob $mask_prob\
    --speaker_pretrain_model_path $speaker_pretrain_model_path\
    --extractor_model_type $extractor_model_type\
    --warmup-updates 3000\
    --arcface-weight $arcface_weight\
    --bce-alpha $bce_alpha\
    --bce-gamma $bce_gamma\
    --weight-decay $weight_decay\
    --max-speakers $max_speakers
    #--rir-path $rir_path\
    #--musan-path $musan_path

fi
# this above setting  is also overfit, I think max_speakers is much too small, 
# representation decoder use groundth label to train

# compared with stage1, stage2 will increase number of  speaker 
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
   export NCCL_DEBUG=INFO
   export PYTHONFAULTHANDLER=1
   musan_path=/maduo/datasets/musan
   rir_path=/maduo/datasets/RIRS_NOISES
   train_wav_dir=/maduo/datasets/alimeeting/Train_Ali_far/audio_dir
   train_textgrid_dir=/maduo/datasets/alimeeting/Train_Ali_far/textgrid_dir
   valid_wav_dir=/maduo/datasets/alimeeting/Eval_Ali/Eval_Ali_far/audio_dir
   valid_textgrid_dir=/maduo/datasets/alimeeting/Eval_Ali/Eval_Ali_far/textgrid_dir
   speaker_pretrain_model_path=/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin
   extractor_model_type='CAM++_gsp'
   #out_bias=-0.5
   mask_prob=0.5
   arcface_weight=0.01
   arcface_margin=0.2
   arcface_scale=32.0
   weight_decay=0.001
   bce_alpha=0.75
   bce_gamma=2.0
   max_speakers=10
   exp_dir=/maduo/exp/speaker_diarization/ssnd/ssnd_alimeeting_improved_lr1e-4_batch64_mask_prob_${mask_prob}_arcface_weight_${arcface_weight}_arcface_margin${arcface_margin}_arcface_scale${arcface_scale}_with_global_spk2int
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
    --train_wav_dir $train_wav_dir\
    --train_textgrid_dir $train_textgrid_dir\
    --valid_wav_dir $valid_wav_dir\
    --valid_textgrid_dir $valid_textgrid_dir\
    --arcface-margin $arcface_margin\
    --arcface-scale $arcface_scale\
    --mask-prob $mask_prob\
    --speaker_pretrain_model_path $speaker_pretrain_model_path\
    --extractor_model_type $extractor_model_type\
    --warmup-updates 3000\
    --arcface-weight $arcface_weight\
    --bce-alpha $bce_alpha\
    --bce-gamma $bce_gamma\
    --weight-decay $weight_decay\
    --max-speakers $max_speakers
    #--rir-path $rir_path\
    #--musan-path $musan_path

fi
