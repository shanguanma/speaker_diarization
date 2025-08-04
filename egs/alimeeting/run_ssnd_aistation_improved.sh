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
   max_speakers=6 # 
   exp_dir=/maduo/exp/speaker_diarization/ssnd/ssnd_alimeeting_improved_lr1e-4_batch64_mask_prob_${mask_prob}_arcface_weight_${arcface_weight}_arcface_margin${arcface_margin}_arcface_scale${arcface_scale}_with_global_spk2int_max_speakers${max_speakers}
   CUDA_VISIABLE_DEVICES=0,1\
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15815 \
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
# this abive setting is also overfit, 
# running log:  logs/run_ssnd_aistation_improved_stage2_max_speaker6_weights_arcface_0.01_mask_prob0.5_with_global_spk2int.log
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
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
   max_speakers=10 #
   exp_dir=/maduo/exp/speaker_diarization/ssnd/ssnd_alimeeting_improved_lr1e-4_batch64_mask_prob_${mask_prob}_arcface_weight_${arcface_weight}_arcface_margin${arcface_margin}_arcface_scale${arcface_scale}_with_global_spk2int_max_speakers${max_speakers}
   mkdir -p $exp_dir
   CUDA_VISIABLE_DEVICES=0,1\
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15815 \
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

# compared with stage3, I will increase max_speaker from 10 to 30
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
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
   max_speakers=30 #
   exp_dir=/maduo/exp/speaker_diarization/ssnd/ssnd_alimeeting_improved_lr1e-4_batch64_mask_prob_${mask_prob}_arcface_weight_${arcface_weight}_arcface_margin${arcface_margin}_arcface_scale${arcface_scale}_with_global_spk2int_max_speakers${max_speakers}
   
   mkdir -p $exp_dir
   
   CUDA_VISIABLE_DEVICES=0,1\
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15715 \
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

# compared with stage4, I will increase arcface_weight from 0.01 to 0.1
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ];then
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
   arcface_weight=0.1
   arcface_margin=0.2
   arcface_scale=32.0
   weight_decay=0.001
   bce_alpha=0.75
   bce_gamma=2.0
   max_speakers=30 #
   exp_dir=/maduo/exp/speaker_diarization/ssnd/ssnd_alimeeting_improved_lr1e-4_batch64_mask_prob_${mask_prob}_arcface_weight_${arcface_weight}_arcface_margin${arcface_margin}_arcface_scale${arcface_scale}_with_global_spk2int_max_speakers${max_speakers}


   mkdir -p $exp_dir

   CUDA_VISIABLE_DEVICES=0,1\
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15615 \
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

# compared with stage5, I will increase arcface_weight from 0.1 to 1
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ];then
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
   arcface_weight=1
   arcface_margin=0.2
   arcface_scale=32.0
   weight_decay=0.001
   bce_alpha=0.75
   bce_gamma=2.0
   max_speakers=30 #
   exp_dir=/maduo/exp/speaker_diarization/ssnd/ssnd_alimeeting_improved_lr1e-4_batch64_mask_prob_${mask_prob}_arcface_weight_${arcface_weight}_arcface_margin${arcface_margin}_arcface_scale${arcface_scale}_with_global_spk2int_max_speakers${max_speakers} 
   mkdir -p $exp_dir

   CUDA_VISIABLE_DEVICES=0,1\
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15515 \
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


if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ];then
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
   arcface_weight=1
   arcface_margin=0.2
   arcface_scale=32.0
   weight_decay=0.001
   bce_alpha=0.75
   bce_gamma=2.0
   max_speakers=30 #
   exp_dir=/maduo/exp/speaker_diarization/ssnd/ssnd_alimeeting_improved_lr1e-4_batch64_mask_prob_${mask_prob}_arcface_weight_${arcface_weight}_arcface_margin${arcface_margin}_arcface_scale${arcface_scale}_with_global_spk2int_max_speakers${max_speakers}_with_musan_rir

   mkdir -p $exp_dir

   CUDA_VISIABLE_DEVICES=0,1\
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15415 \
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
    --max-speakers $max_speakers\
    --rir-path $rir_path\
    --musan-path $musan_path
fi
# stage1-7, all setting is overfit

# compared with stage7, stage8 will use standard bce loss
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ];then
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
   arcface_weight=1
   arcface_margin=0.2
   arcface_scale=32.0
   weight_decay=0.001
   bce_alpha=0.75
   bce_gamma=2.0
   max_speakers=30 #
   exp_dir=/maduo/exp/speaker_diarization/ssnd/ssnd_alimeeting_improved_lr1e-4_batch64_mask_prob_${mask_prob}_arcface_weight_${arcface_weight}_arcface_margin${arcface_margin}_arcface_scale${arcface_scale}_with_global_spk2int_max_speakers${max_speakers}_with_musan_rir_standard_bce_loss

   mkdir -p $exp_dir

   CUDA_VISIABLE_DEVICES=0,1\
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15415 \
   ssnd/train_accelerate_ddp.py\
    --debug true\
    --use-standard-bce true\
    --world-size 2 \
    --num-epochs 30\
    --batch-size 64 \
    --start-epoch 1\
    --start-batch 16500\
    --keep-last-k 10\
    --keep-last-epoch 10\
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
    --max-speakers $max_speakers\
    --rir-path $rir_path\
    --musan-path $musan_path
fi

# compared with stage7, I will add dropout into decoder and extractor 
if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ];then
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
   arcface_weight=1
   arcface_margin=0.2
   arcface_scale=32.0
   weight_decay=0.001
   bce_alpha=0.75
   bce_gamma=2.0
   max_speakers=30 #
   decoder_dropout=0.1
   extractor_dropout=0.1
   exp_dir=/maduo/exp/speaker_diarization/ssnd/ssnd_alimeeting_improved_lr1e-4_batch64_mask_prob_${mask_prob}_arcface_weight_${arcface_weight}_arcface_margin${arcface_margin}_arcface_scale${arcface_scale}_with_global_spk2int_max_speakers${max_speakers}_with_musan_rir_standard_bce_loss_dropout0.1

   mkdir -p $exp_dir

   CUDA_VISIABLE_DEVICES=0,1\
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15515 \
   ssnd/train_accelerate_ddp.py\
    --debug false\
    --use-standard-bce false\
    --world-size 2 \
    --num-epochs 30\
    --batch-size 64 \
    --start-epoch 1\
    --start-batch 18000\
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
    --max-speakers $max_speakers\
    --rir-path $rir_path\
    --musan-path $musan_path\
    --decoder-dropout $decoder_dropout\
    --extractor-dropout $extractor_dropout
fi


# compared with stage7, stage10, I will add dropout into decoder and extractor and increase weight_decay on optimizer from 0.001 to 0.01
if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ];then
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
   arcface_weight=1
   arcface_margin=0.2
   arcface_scale=32.0
   weight_decay=0.01
   bce_alpha=0.75
   bce_gamma=2.0
   max_speakers=30 #
   decoder_dropout=0.1
   extractor_dropout=0.1
   exp_dir=/maduo/exp/speaker_diarization/ssnd/ssnd_alimeeting_improved_lr1e-4_batch64_mask_prob_${mask_prob}_arcface_weight_${arcface_weight}_arcface_margin${arcface_margin}_arcface_scale${arcface_scale}_with_global_spk2int_max_speakers${max_speakers}_with_musan_rir_standard_bce_loss_dropout0.1_weight_decay0.01

   mkdir -p $exp_dir

   CUDA_VISIABLE_DEVICES=0,1\
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15615 \
   ssnd/train_accelerate_ddp.py\
    --debug false\
    --use-standard-bce false\
    --world-size 2 \
    --num-epochs 30\
    --batch-size 64 \
    --start-epoch 1\
    --start-batch 18000\
    --keep-last-k 10\
    --keep-last-epoch 10\
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
    --max-speakers $max_speakers\
    --rir-path $rir_path\
    --musan-path $musan_path\
    --decoder-dropout $decoder_dropout\
    --extractor-dropout $extractor_dropout
fi

# compared with stage7, I will add dropout into decoder and extractor and increase weight_decay on optimizer from 0.001 to 0.01, add label_smoothing from 0.0 to 0.01
if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ];then
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
   arcface_weight=1
   arcface_margin=0.2
   arcface_scale=32.0
   weight_decay=0.01
   bce_alpha=0.75
   bce_gamma=2.0
   max_speakers=30 #
   decoder_dropout=0.1
   extractor_dropout=0.1
   label_smoothing=0.01
   exp_dir=/maduo/exp/speaker_diarization/ssnd/ssnd_alimeeting_improved_lr1e-4_batch64_mask_prob_${mask_prob}_arcface_weight_${arcface_weight}_arcface_margin${arcface_margin}_arcface_scale${arcface_scale}_with_global_spk2int_max_speakers${max_speakers}_with_musan_rir_standard_bce_loss_dropout0.1_weight_decay${weight_decay}_label_smoothing_${label_smoothing}

   mkdir -p $exp_dir

   CUDA_VISIABLE_DEVICES=0,1\
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15715 \
   ssnd/train_accelerate_ddp.py\
    --debug true\
    --use-standard-bce false\
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
    --max-speakers $max_speakers\
    --rir-path $rir_path\
    --musan-path $musan_path\
    --decoder-dropout $decoder_dropout\
    --extractor-dropout $extractor_dropout\
    --label-smoothing $label_smoothing
fi

# compared with stage8, I will add dropout into decoder and extractor and increase weight_decay on optimizer from 0.001 to 0.01, add label_smoothing from 0.0 to 0.01
if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ];then
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
   arcface_weight=1
   arcface_margin=0.2
   arcface_scale=32.0
   weight_decay=0.01
   bce_alpha=0.75
   bce_gamma=2.0
   max_speakers=30 #
   decoder_dropout=0.1
   extractor_dropout=0.1
   label_smoothing=0.01
   standard_bce_loss=true
   exp_dir=/maduo/exp/speaker_diarization/ssnd/ssnd_alimeeting_improved_lr1e-4_batch64_mask_prob_${mask_prob}_arcface_weight_${arcface_weight}_arcface_margin${arcface_margin}_arcface_scale${arcface_scale}_with_global_spk2int_max_speakers${max_speakers}_with_musan_rir_standard_bce_loss_${standard_bce_loss}_dropout0.1_weight_decay${weight_decay}_label_smoothing_${label_smoothing}

   mkdir -p $exp_dir

   CUDA_VISIABLE_DEVICES=0,1\
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15815 \
   ssnd/train_accelerate_ddp.py\
    --debug false\
    --use-standard-bce $standard_bce_loss\
    --world-size 2 \
    --num-epochs 30\
    --batch-size 64 \
    --start-epoch 1\
    --start-batch 18000\
    --keep-last-k 10\
    --keep-last-epoch 10\
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
    --max-speakers $max_speakers\
    --rir-path $rir_path\
    --musan-path $musan_path\
    --decoder-dropout $decoder_dropout\
    --extractor-dropout $extractor_dropout\
    --label-smoothing $label_smoothing
fi



# compared with stage12, remove label_smoothing
if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ];then
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
   arcface_weight=1
   arcface_margin=0.2
   arcface_scale=32.0
   weight_decay=0.01
   bce_alpha=0.75
   bce_gamma=2.0
   max_speakers=30 #
   decoder_dropout=0.1
   extractor_dropout=0.1
   label_smoothing=0.0
   standard_bce_loss=true
   exp_dir=/maduo/exp/speaker_diarization/ssnd/ssnd_alimeeting_improved_lr1e-4_batch64_mask_prob_${mask_prob}_arcface_weight_${arcface_weight}_arcface_margin${arcface_margin}_arcface_scale${arcface_scale}_with_global_spk2int_max_speakers${max_speakers}_with_musan_rir_standard_bce_loss_${standard_bce_loss}_dropout0.1_weight_decay${weight_decay}_label_smoothing_${label_smoothing}

   mkdir -p $exp_dir

   CUDA_VISIABLE_DEVICES=0,1\
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15915 \
   ssnd/train_accelerate_ddp.py\
    --debug false\
    --use-standard-bce $standard_bce_loss\
    --world-size 2 \
    --num-epochs 30\
    --batch-size 64 \
    --start-epoch 1\
    --start-batch 30000\
    --keep-last-k 10\
    --keep-last-epoch 10\
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
    --max-speakers $max_speakers\
    --rir-path $rir_path\
    --musan-path $musan_path\
    --decoder-dropout $decoder_dropout\
    --extractor-dropout $extractor_dropout\
    --label-smoothing $label_smoothing
fi


# compared with stage13, stage14 is same as stage13, will use debug=False
if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ];then
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
   arcface_weight=1
   arcface_margin=0.2
   arcface_scale=32.0
   weight_decay=0.01
   bce_alpha=0.75
   bce_gamma=2.0
   max_speakers=30 #
   decoder_dropout=0.1
   extractor_dropout=0.1
   label_smoothing=0.0
   standard_bce_loss=true
   exp_dir=/maduo/exp/speaker_diarization/ssnd/ssnd_alimeeting_improved_lr1e-4_batch64_mask_prob_${mask_prob}_arcface_weight_${arcface_weight}_arcface_margin${arcface_margin}_arcface_scale${arcface_scale}_with_global_spk2int_max_speakers${max_speakers}_with_musan_rir_standard_bce_loss_${standard_bce_loss}_dropout0.1_weight_decay${weight_decay}_label_smoothing_${label_smoothing}_no_debug

   mkdir -p $exp_dir

   CUDA_VISIABLE_DEVICES=0,1\
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 16915 \
   ssnd/train_accelerate_ddp.py\
    --debug false\
    --use-standard-bce $standard_bce_loss\
    --world-size 2 \
    --num-epochs 30\
    --batch-size 64 \
    --start-epoch 1\
    --start-batch 30000\
    --keep-last-k 10\
    --keep-last-epoch 10\
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
    --max-speakers $max_speakers\
    --rir-path $rir_path\
    --musan-path $musan_path\
    --decoder-dropout $decoder_dropout\
    --extractor-dropout $extractor_dropout\
    --label-smoothing $label_smoothing
fi

# compared with stage14, stage15 will reduce wight_decay from 0.01 to 0.001, dropout from 0.1 to 0.05
if [ ${stage} -le 15 ] && [ ${stop_stage} -ge 15 ];then
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
   arcface_weight=1
   arcface_margin=0.2
   arcface_scale=32.0
   weight_decay=0.001
   bce_alpha=0.75
   bce_gamma=2.0
   max_speakers=30 #
   decoder_dropout=0.05
   extractor_dropout=0.05
   label_smoothing=0.0
   standard_bce_loss=true
   exp_dir=/maduo/exp/speaker_diarization/ssnd/ssnd_alimeeting_improved_lr1e-4_batch64_mask_prob_${mask_prob}_arcface_weight_${arcface_weight}_arcface_margin${arcface_margin}_arcface_scale${arcface_scale}_with_global_spk2int_max_speakers${max_speakers}_with_musan_rir_standard_bce_loss_${standard_bce_loss}_dropout0.05_weight_decay${weight_decay}_label_smoothing_${label_smoothing}_no_debug

   mkdir -p $exp_dir

   CUDA_VISIABLE_DEVICES=0,1\
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 16815 \
   ssnd/train_accelerate_ddp.py\
    --debug false\
    --use-standard-bce $standard_bce_loss\
    --world-size 2 \
    --num-epochs 30\
    --batch-size 64 \
    --start-epoch 1\
    --start-batch 22500\
    --keep-last-k 10\
    --keep-last-epoch 10\
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
    --max-speakers $max_speakers\
    --rir-path $rir_path\
    --musan-path $musan_path\
    --decoder-dropout $decoder_dropout\
    --extractor-dropout $extractor_dropout\
    --label-smoothing $label_smoothing
fi


# compared with stage10, stage16 will increase arcface_weight from 1 to 1.5
if [ ${stage} -le 16 ] && [ ${stop_stage} -ge 16 ];then
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
   arcface_weight=1.5
   arcface_margin=0.2
   arcface_scale=32.0
   weight_decay=0.01
   bce_alpha=0.75
   bce_gamma=2.0
   max_speakers=30 #
   decoder_dropout=0.1
   extractor_dropout=0.1
   exp_dir=/maduo/exp/speaker_diarization/ssnd/ssnd_alimeeting_improved_lr1e-4_batch64_mask_prob_${mask_prob}_arcface_weight_${arcface_weight}_arcface_margin${arcface_margin}_arcface_scale${arcface_scale}_with_global_spk2int_max_speakers${max_speakers}_with_musan_rir_standard_bce_loss_dropout0.1_weight_decay0.01

   mkdir -p $exp_dir

   CUDA_VISIABLE_DEVICES=0,1\
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 16615 \
   ssnd/train_accelerate_ddp.py\
    --debug false\
    --use-standard-bce false\
    --world-size 2 \
    --num-epochs 30\
    --batch-size 64 \
    --start-epoch 1\
    --start-batch 4500\
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
    --max-speakers $max_speakers\
    --rir-path $rir_path\
    --musan-path $musan_path\
    --decoder-dropout $decoder_dropout\
    --extractor-dropout $extractor_dropout
fi

# compared with stage 15, stage17 will reduce lr from 1e-4 to 5e-5, reduce epoch from 30 to 20(because it effects lr rate,) increase warmup from 3000 to 4000
if [ ${stage} -le 17 ] && [ ${stop_stage} -ge 17 ];then
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
   arcface_weight=1
   arcface_margin=0.2
   arcface_scale=32.0
   weight_decay=0.001
   bce_alpha=0.75
   bce_gamma=2.0
   max_speakers=30 #
   decoder_dropout=0.05
   extractor_dropout=0.05
   label_smoothing=0.0
   standard_bce_loss=true
   exp_dir=/maduo/exp/speaker_diarization/ssnd/ssnd_alimeeting_improved_lr5e-5_batch64_mask_prob_${mask_prob}_arcface_weight_${arcface_weight}_arcface_margin${arcface_margin}_arcface_scale${arcface_scale}_with_global_spk2int_max_speakers${max_speakers}_with_musan_rir_standard_bce_loss_${standard_bce_loss}_dropout0.05_weight_decay${weight_decay}_label_smoothing_${label_smoothing}_warmup4000_epoch20

   mkdir -p $exp_dir

   CUDA_VISIABLE_DEVICES=0,1\
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 17815 \
   ssnd/train_accelerate_ddp.py\
    --debug false\
    --use-standard-bce $standard_bce_loss\
    --world-size 2 \
    --num-epochs 20\
    --batch-size 64 \
    --start-epoch 1\
    --start-batch 4500\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --grad-clip true\
    --lr 5e-5\
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
    --warmup-updates 4000\
    --arcface-weight $arcface_weight\
    --bce-alpha $bce_alpha\
    --bce-gamma $bce_gamma\
    --weight-decay $weight_decay\
    --max-speakers $max_speakers\
    --rir-path $rir_path\
    --musan-path $musan_path\
    --decoder-dropout $decoder_dropout\
    --extractor-dropout $extractor_dropout\
    --label-smoothing $label_smoothing
fi

if [ ${stage} -le 18 ] && [ ${stop_stage} -ge 18 ];then
   # prepare voxceleb2 kaldi format
   rawdata_dir=/F00120240032/VoxCeleb/vox2/dev/
   data=/maduo/datasets/voxceleb2
   find ${rawdata_dir} -name "*.wav" | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF,$0}' | sort >${data}/vox2_dev/wav.scp
   head ${data}/vox2_dev/wav.scp
    awk '{print $1}' ${data}/vox2_dev/wav.scp | awk -F "/" '{print $0,$1}' >${data}/vox2_dev/utt2spk
  ./utils/utt2spk_to_spk2utt.pl ${data}/vox2_dev/utt2spk >${data}/vox2_dev/spk2utt
fi
if [ ${stage} -le 19 ] && [ ${stop_stage} -ge 19 ];then
  # prepare spk2chunks files, it is removed silent clips from raw voxceleb2 audio
  # 设置环境变量
#   export NCCL_DEBUG=INFO
#   export PYTHONFAULTHANDLER=1
#
#  # 运行新的简化版本
#	NUM_PROCESSES=4
#	echo "运行新的简化torchrun版本..."
#	torchrun \
#	    --nproc_per_node=$NUM_PROCESSES \
#	    --nnodes=1 \
#	    --node_rank=0 \
#	    --master_addr=localhost \
#	    --master_port=29500 \
#	    ssnd/remove_silent_and_get_spk2chunks_torchrun.py \
#	    --voxceleb2-dataset-dir /maduo/datasets/voxceleb2/vox2_dev/ \
#	    --out-text /maduo/datasets/voxceleb2/vox2_dev/train_torchrun_simple.json
#

 python ssnd/remove_silent_and_get_spk2chunks.py\
	 --voxceleb2-dataset-dir  /maduo/datasets/voxceleb2/vox2_dev/\
	 --out-text /maduo/datasets/voxceleb2/vox2_dev/train.json
fi
# 
if [ ${stage} -le 20 ] && [ ${stop_stage} -ge 20 ];then
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
   arcface_weight=1
   arcface_margin=0.2
   arcface_scale=32.0
   weight_decay=0.001
   bce_alpha=0.75
   bce_gamma=2.0
   max_speakers=30 #
   decoder_dropout=0.05
   extractor_dropout=0.05
   label_smoothing=0.0
   standard_bce_loss=true
   train_stage=1
   voxceleb2_dataset_dir=/maduo/datasets/voxceleb2/vox2_dev
   voxceleb2_spk2chunks_json=$voxceleb2_dataset_dir/train.json
   #exp_dir=/maduo/exp/speaker_diarization/ssnd/ssnd_alimeeting_improved_lr1e-4_batch64_mask_prob_${mask_prob}_arcface_weight_${arcface_weight}_arcface_margin${arcface_margin}_arcface_scale${arcface_scale}_with_global_spk2int_max_speakers${max_speakers}_with_musan_rir_standard_bce_loss_${standard_bce_loss}_dropout0.05_weight_decay${weight_decay}_label_smoothing_${label_smoothing}_warmup4000_epoch20_train_stage1_voxceleb2
   exp_dir=/maduo/exp/speaker_diarization/ssnd/cam++_gsp_lr1e-4_batch64_standard_bce_loss_${standard_bce_loss}_train_stage1_voxceleb2
   mkdir -p $exp_dir

   CUDA_VISIABLE_DEVICES=0,1\
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 17815 \
   ssnd/train_accelerate_ddp.py\
    --debug false\
    --use-standard-bce $standard_bce_loss\
    --world-size 2 \
    --num-epochs 20\
    --batch-size 64 \
    --start-epoch 1\
    --keep-last-k 3\
    --keep-last-epoch 3\
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
    --warmup-updates 4000\
    --arcface-weight $arcface_weight\
    --bce-alpha $bce_alpha\
    --bce-gamma $bce_gamma\
    --weight-decay $weight_decay\
    --max-speakers $max_speakers\
    --rir-path $rir_path\
    --musan-path $musan_path\
    --decoder-dropout $decoder_dropout\
    --extractor-dropout $extractor_dropout\
    --label-smoothing $label_smoothing\
    --train-stage $train_stage\
    --voxceleb2-dataset-dir $voxceleb2_dataset_dir\
    --voxceleb2-spk2chunks-json $voxceleb2_spk2chunks_json
fi

