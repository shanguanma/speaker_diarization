#!/usr/bin/env bash

stage=0
stop_stage=1000

. utils/parse_options.sh
. path_for_speaker_diarization.sh
# compared with stage104-105 of run_ts_vad2.sh, it is sota util 2024-10-22.
# 1. I will put speaker encoder model put into our tsvad model and freeze its parameters, and get speaker utterance embedding
# and I use cut_target_speech_v2() fn and ts_len=6
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is wav-bert2.0 (only using position embedding and first 6 layers conformer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this w2v-bert2.0 is trained Languages: 143+ , Size: 4.5M hours (it is from this paper https://arxiv.org/pdf/2312.05187)
    # checkpoint is from https://huggingface.co/facebook/w2v-bert-2.0/tree/main
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="w2v-bert2"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
    speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    #spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    #speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_w2v-bert2.0_first6layer_epoch40_front_fix_seed_lr1e4_freeze_speaker_encoder_model_online_utt_speaker_embedding
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 12815 \
   ts_vad3/train_accelerate_ddp.py \
    --verbose 1 \
    --world-size 2 \
    --num-epochs 40\
    --start-epoch 1\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-speech-encoder-updates 4000\
    --freeze-speaker-encoder-updates 62600\
    --fuse-fbank-feat false\
    --fuse-speaker-embedding-feat false\
    --grad-clip true\
    --lr 1e-4\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --speech-encoder-config $speech_encoder_config\
    --select-encoder-layer-nums 6\
    --exp-dir $exp_dir
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_w2v-bert2.0_first6layer_epoch40_front_fix_seed_lr1e4_freeze_speaker_encoder_model_online_utt_speaker_embedding
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 rs_segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="w2v-bert2"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 #spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 #speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

 for name in $infer_sets;do
    results_path=$exp_dir
  python3 ts_vad3/infer.py \
    --model-file $model_file\
    --rs-len $rs_len\
    --rs-segment-shift $rs_segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name alimeeting_${name}.rttm\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --speech-encoder-config $speech_encoder_config\
    --wavlm-fuse-feat-post-norm false \
    --fuse-fbank-feat false\
    --fuse-speaker-embedding-feat false\
    --data-dir $data_dir
done
## cat  logs/run_ts_vad3_stage1-2_v2.log
# Eval set
## Model DER:  0.16575875471132806
#Model ACC:  0.937980771709068
#100%|██████████| 25/25 [00:22<00:00,  1.10it/s]
#Eval for threshold 0.20: DER 10.23%, MS 1.62%, FA 6.21%, SC 2.40%
#
#Eval for threshold 0.30: DER 8.92%, MS 2.68%, FA 3.39%, SC 2.85%
#
#Eval for threshold 0.35: DER 8.77%, MS 3.26%, FA 2.58%, SC 2.93%
#
#Eval for threshold 0.40: DER 8.88%, MS 3.98%, FA 2.02%, SC 2.88%
#
#Eval for threshold 0.45: DER 9.07%, MS 4.87%, FA 1.44%, SC 2.76%
#
#Eval for threshold 0.50: DER 9.52%, MS 5.90%, FA 1.08%, SC 2.54%
#
#Eval for threshold 0.55: DER 10.19%, MS 7.08%, FA 0.88%, SC 2.23%
#
#Eval for threshold 0.60: DER 10.85%, MS 8.18%, FA 0.72%, SC 1.96%
#
#Eval for threshold 0.70: DER 12.68%, MS 10.79%, FA 0.48%, SC 1.41%
#
#Eval for threshold 0.80: DER 15.56%, MS 14.28%, FA 0.38%, SC 0.90%
# Test set
#Model DER:  0.21497034829120273
#Model ACC:  0.9097504294392752
#100%|██████████| 60/60 [00:55<00:00,  1.09it/s]
#Eval for threshold 0.20: DER 16.15%, MS 2.03%, FA 8.44%, SC 5.69%
#
#Eval for threshold 0.30: DER 14.54%, MS 3.18%, FA 5.04%, SC 6.32%
#
#Eval for threshold 0.35: DER 14.22%, MS 3.82%, FA 3.87%, SC 6.53%
#
#Eval for threshold 0.40: DER 14.10%, MS 4.56%, FA 2.90%, SC 6.64%
#
#Eval for threshold 0.45: DER 14.24%, MS 5.50%, FA 2.04%, SC 6.70%
#
#Eval for threshold 0.50: DER 14.58%, MS 6.59%, FA 1.38%, SC 6.61%
#
#Eval for threshold 0.55: DER 15.24%, MS 7.94%, FA 0.97%, SC 6.34%
#
#Eval for threshold 0.60: DER 16.17%, MS 9.47%, FA 0.68%, SC 6.03%
#
#Eval for threshold 0.70: DER 18.56%, MS 12.98%, FA 0.38%, SC 5.21%
#
#Eval for threshold 0.80: DER 22.25%, MS 17.88%, FA 0.19%, SC 4.18%

fi

# compared with stage1-2, stage3-4 will increase ts_len from 6 to 10.
# and I  use cut_target_speech() fn
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is wav-bert2.0 (only using position embedding and first 6 layers conformer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this w2v-bert2.0 is trained Languages: 143+ , Size: 4.5M hours (it is from this paper https://arxiv.org/pdf/2312.05187)
    # checkpoint is from https://huggingface.co/facebook/w2v-bert-2.0/tree/main
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="w2v-bert2"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
    speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    #spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    #speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_w2v-bert2.0_first6layer_epoch40_front_fix_seed_lr1e4_freeze_speaker_encoder_model_online_utt_speaker_embedding_ts_len10
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 12815 \
   ts_vad3/train_accelerate_ddp2.py \
    --verbose 1 \
    --world-size 2 \
    --num-epochs 40\
    --start-epoch 1\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-speech-encoder-updates 4000\
    --freeze-speaker-encoder-updates 62600\
    --fuse-fbank-feat false\
    --fuse-speaker-embedding-feat false\
    --ts-len 10\
    --grad-clip true\
    --lr 1e-4\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --speech-encoder-config $speech_encoder_config\
    --select-encoder-layer-nums 6\
    --exp-dir $exp_dir
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_w2v-bert2.0_first6layer_epoch40_front_fix_seed_lr1e4_freeze_speaker_encoder_model_online_utt_speaker_embedding_ts_len10
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 rs_segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="w2v-bert2"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 #spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 #speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

 for name in $infer_sets;do
    results_path=$exp_dir
  python3 ts_vad3/infer2.py \
    --model-file $model_file\
    --ts-len 10\
    --rs-len $rs_len\
    --rs-segment-shift $rs_segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name alimeeting_${name}.rttm\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --speech-encoder-config $speech_encoder_config\
    --wavlm-fuse-feat-post-norm false \
    --fuse-fbank-feat false\
    --fuse-speaker-embedding-feat false\
    --data-dir $data_dir
done
## Eval set
#Model DER:  0.1319398490152673
#Model ACC:  0.9526697683133564
#100%|██████████| 25/25 [00:24<00:00,  1.04it/s]
#Eval for threshold 0.20: DER 7.62%, MS 0.94%, FA 6.13%, SC 0.55%
#
#Eval for threshold 0.30: DER 5.76%, MS 1.49%, FA 3.55%, SC 0.72%
#
#Eval for threshold 0.35: DER 5.35%, MS 1.79%, FA 2.78%, SC 0.78%
#
#Eval for threshold 0.40: DER 5.15%, MS 2.19%, FA 2.16%, SC 0.80%
#
#Eval for threshold 0.45: DER 5.07%, MS 2.58%, FA 1.68%, SC 0.81%
#
#Eval for threshold 0.50: DER 5.13%, MS 3.07%, FA 1.29%, SC 0.77%
#
#Eval for threshold 0.55: DER 5.39%, MS 3.65%, FA 1.01%, SC 0.72%
#
#Eval for threshold 0.60: DER 5.76%, MS 4.25%, FA 0.82%, SC 0.68%
#
#Eval for threshold 0.70: DER 7.00%, MS 5.90%, FA 0.58%, SC 0.52%
#
#Eval for threshold 0.80: DER 9.34%, MS 8.58%, FA 0.41%, SC 0.35%
#
## Test set
#Model DER:  0.16503781123805386
#Model ACC:  0.9329548984718543
#100%|██████████| 60/60 [00:59<00:00,  1.01it/s]
#Eval for threshold 0.20: DER 13.44%, MS 1.31%, FA 8.57%, SC 3.56%
#
#Eval for threshold 0.30: DER 11.41%, MS 2.01%, FA 4.91%, SC 4.50%
#
#Eval for threshold 0.35: DER 10.86%, MS 2.47%, FA 3.57%, SC 4.82%
#
#Eval for threshold 0.40: DER 10.56%, MS 3.05%, FA 2.44%, SC 5.07%
#
#Eval for threshold 0.45: DER 10.52%, MS 3.78%, FA 1.62%, SC 5.11%
#
#Eval for threshold 0.50: DER 10.87%, MS 4.83%, FA 1.10%, SC 4.94%
#
#Eval for threshold 0.55: DER 11.53%, MS 6.12%, FA 0.84%, SC 4.58%
#
#Eval for threshold 0.60: DER 12.24%, MS 7.42%, FA 0.62%, SC 4.20%
#
#Eval for threshold 0.70: DER 14.17%, MS 10.44%, FA 0.36%, SC 3.38%
#
#Eval for threshold 0.80: DER 17.27%, MS 14.61%, FA 0.19%, SC 2.47%
fi
## stage5-6, I use fn cut_target_speech() and ts_len=6, this is my first experiment,
## But I don't store the chckpoint, it is rewrited on stage1-2
# if I have time and will run it again.
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is wav-bert2.0 (only using position embedding and first 6 layers conformer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this w2v-bert2.0 is trained Languages: 143+ , Size: 4.5M hours (it is from this paper https://arxiv.org/pdf/2312.05187)
    # checkpoint is from https://huggingface.co/facebook/w2v-bert-2.0/tree/main
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="w2v-bert2"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
    speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    #spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    #speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_w2v-bert2.0_first6layer_epoch40_front_fix_seed_lr1e4_freeze_speaker_encoder_model_online_utt_speaker_embedding_ts_len6
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 12515 \
   ts_vad3/train_accelerate_ddp2.py \
    --verbose 1 \
    --world-size 2 \
    --num-epochs 40\
    --start-epoch 1\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-speech-encoder-updates 4000\
    --freeze-speaker-encoder-updates 62600\
    --fuse-fbank-feat false\
    --fuse-speaker-embedding-feat false\
    --ts-len 6\
    --grad-clip true\
    --lr 1e-4\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --speech-encoder-config $speech_encoder_config\
    --select-encoder-layer-nums 6\
    --exp-dir $exp_dir
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_w2v-bert2.0_first6layer_epoch40_front_fix_seed_lr1e4_freeze_speaker_encoder_model_online_utt_speaker_embedding_ts_len6
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 rs_segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="w2v-bert2"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 #spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 #speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

 for name in $infer_sets;do
    results_path=$exp_dir
  python3 ts_vad3/infer2.py \
    --model-file $model_file\
    --ts-len 6\
    --rs-len $rs_len\
    --rs-segment-shift $rs_segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name alimeeting_${name}.rttm\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --speech-encoder-config $speech_encoder_config\
    --wavlm-fuse-feat-post-norm false \
    --fuse-fbank-feat false\
    --fuse-speaker-embedding-feat false\
    --data-dir $data_dir
done
# cat logs/run_ts_vad3_stage2.log this is correct. because it actually runs it on stage1-2.
#Eval set
#Model DER:  0.13010483938454873
#Model ACC:  0.953414945685124
#100%|██████████| 25/25 [00:23<00:00,  1.05it/s]
#Eval for threshold 0.20: DER 8.37%, MS 0.90%, FA 6.99%, SC 0.48%
#
#Eval for threshold 0.30: DER 5.98%, MS 1.36%, FA 4.01%, SC 0.61%
#
#Eval for threshold 0.35: DER 5.37%, MS 1.67%, FA 3.00%, SC 0.70%
#
#Eval for threshold 0.40: DER 5.05%, MS 2.04%, FA 2.25%, SC 0.76%
#
#Eval for threshold 0.45: DER 4.89%, MS 2.41%, FA 1.72%, SC 0.75%
#
#Eval for threshold 0.50: DER 4.89%, MS 2.87%, FA 1.25%, SC 0.76%
#
#Eval for threshold 0.55: DER 5.10%, MS 3.44%, FA 0.95%, SC 0.72%
#
#Eval for threshold 0.60: DER 5.52%, MS 4.14%, FA 0.77%, SC 0.62%
#
#Eval for threshold 0.70: DER 6.79%, MS 5.81%, FA 0.52%, SC 0.46%
#
#Eval for threshold 0.80: DER 9.08%, MS 8.40%, FA 0.40%, SC 0.27%

#Test set
#Model DER:  0.14844079727276555
#Model ACC:  0.941778676462181
#100%|██████████| 60/60 [00:57<00:00,  1.05it/s]
#Eval for threshold 0.20: DER 11.58%, MS 1.13%, FA 8.20%, SC 2.26%
#
#Eval for threshold 0.30: DER 9.19%, MS 1.87%, FA 4.60%, SC 2.72%
#
#Eval for threshold 0.35: DER 8.52%, MS 2.33%, FA 3.30%, SC 2.88%
#
#Eval for threshold 0.40: DER 8.21%, MS 2.89%, FA 2.32%, SC 2.99%
#
#Eval for threshold 0.45: DER 8.19%, MS 3.58%, FA 1.56%, SC 3.05%
#
#Eval for threshold 0.50: DER 8.47%, MS 4.50%, FA 1.06%, SC 2.91%
#
#Eval for threshold 0.55: DER 9.05%, MS 5.65%, FA 0.79%, SC 2.60%
#
#Eval for threshold 0.60: DER 9.79%, MS 6.87%, FA 0.59%, SC 2.33%
#
#Eval for threshold 0.70: DER 11.81%, MS 9.69%, FA 0.34%, SC 1.78%
#
#Eval for threshold 0.80: DER 15.00%, MS 13.58%, FA 0.18%, SC 1.24%
fi

# compared with stage3-4, stage7-8 will increase ts_len from 10 to 15
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is wav-bert2.0 (only using position embedding and first 6 layers conformer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this w2v-bert2.0 is trained Languages: 143+ , Size: 4.5M hours (it is from this paper https://arxiv.org/pdf/2312.05187)
    # checkpoint is from https://huggingface.co/facebook/w2v-bert-2.0/tree/main
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="w2v-bert2"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
    speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    #spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    #speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_w2v-bert2.0_first6layer_epoch40_front_fix_seed_lr1e4_freeze_speaker_encoder_model_online_utt_speaker_embedding_ts_len15
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 12715 \
   ts_vad3/train_accelerate_ddp2.py \
    --verbose 1 \
    --world-size 2 \
    --num-epochs 40\
    --start-epoch 38\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-speech-encoder-updates 4000\
    --freeze-speaker-encoder-updates 62600\
    --fuse-fbank-feat false\
    --fuse-speaker-embedding-feat false\
    --ts-len 15\
    --grad-clip true\
    --lr 1e-4\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --speech-encoder-config $speech_encoder_config\
    --select-encoder-layer-nums 6\
    --exp-dir $exp_dir
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_w2v-bert2.0_first6layer_epoch40_front_fix_seed_lr1e4_freeze_speaker_encoder_model_online_utt_speaker_embedding_ts_len15
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 rs_segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="w2v-bert2"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 #spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 #speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

 for name in $infer_sets;do
    results_path=$exp_dir
  python3 ts_vad3/infer2.py \
    --model-file $model_file\
    --ts-len 15\
    --rs-len $rs_len\
    --rs-segment-shift $rs_segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name alimeeting_${name}.rttm\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --speech-encoder-config $speech_encoder_config\
    --wavlm-fuse-feat-post-norm false \
    --fuse-fbank-feat false\
    --fuse-speaker-embedding-feat false\
    --data-dir $data_dir
done
#  cat logs/run_ts_vad3_stage7-8_v1_A100_3.log
# Eval set
# Model DER:  0.13682189409692933
#Model ACC:  0.9523727912838599
#100%|██████████| 25/25 [00:33<00:00,  1.34s/it]
#Eval for threshold 0.20: DER 6.97%, MS 1.28%, FA 5.07%, SC 0.61%
#
#Eval for threshold 0.30: DER 5.47%, MS 1.93%, FA 2.79%, SC 0.74%
#
#Eval for threshold 0.35: DER 5.30%, MS 2.38%, FA 2.13%, SC 0.79%
#
#Eval for threshold 0.40: DER 5.26%, MS 2.82%, FA 1.63%, SC 0.81%
#
#Eval for threshold 0.45: DER 5.33%, MS 3.28%, FA 1.27%, SC 0.78%
#
#Eval for threshold 0.50: DER 5.61%, MS 3.91%, FA 1.00%, SC 0.70%
#
#Eval for threshold 0.55: DER 6.06%, MS 4.63%, FA 0.84%, SC 0.58%
#
#Eval for threshold 0.60: DER 6.59%, MS 5.40%, FA 0.71%, SC 0.48%
#
#Eval for threshold 0.70: DER 8.23%, MS 7.37%, FA 0.55%, SC 0.31%
#
#Eval for threshold 0.80: DER 10.84%, MS 10.22%, FA 0.40%, SC 0.22%
#
#Test set
#Model DER:  0.16605163939917514
#Model ACC:  0.9336183643664164
#100%|██████████| 60/60 [01:23<00:00,  1.39s/it]
#Eval for threshold 0.20: DER 12.76%, MS 1.58%, FA 7.77%, SC 3.41%
#
#Eval for threshold 0.30: DER 10.76%, MS 2.43%, FA 4.34%, SC 3.99%
#
#Eval for threshold 0.35: DER 10.30%, MS 2.89%, FA 3.23%, SC 4.18%
#
#Eval for threshold 0.40: DER 10.18%, MS 3.52%, FA 2.32%, SC 4.35%
#
#Eval for threshold 0.45: DER 10.21%, MS 4.22%, FA 1.56%, SC 4.43%
#
#Eval for threshold 0.50: DER 10.55%, MS 5.20%, FA 1.00%, SC 4.35%
#
#Eval for threshold 0.55: DER 11.22%, MS 6.46%, FA 0.75%, SC 4.01%
#
#Eval for threshold 0.60: DER 12.08%, MS 7.73%, FA 0.59%, SC 3.75%
#
#Eval for threshold 0.70: DER 14.14%, MS 10.70%, FA 0.36%, SC 3.08%
#
#Eval for threshold 0.80: DER 17.46%, MS 14.84%, FA 0.22%, SC 2.41%

fi
# compared with stage3-4, stage9-10 will add fuse_fbank
if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is wav-bert2.0 (only using position embedding and first 6 layers conformer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this w2v-bert2.0 is trained Languages: 143+ , Size: 4.5M hours (it is from this paper https://arxiv.org/pdf/2312.05187)
    # checkpoint is from https://huggingface.co/facebook/w2v-bert-2.0/tree/main
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="w2v-bert2"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
    speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    #spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    #speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_w2v-bert2.0_first6layer_epoch40_front_fix_seed_lr1e4_freeze_speaker_encoder_model_online_utt_speaker_embedding_ts_len10_fuse_fbank
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 12615 \
   ts_vad3/train_accelerate_ddp2.py \
    --verbose 1 \
    --world-size 2 \
    --num-epochs 40\
    --start-epoch 29\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-speech-encoder-updates 4000\
    --freeze-speaker-encoder-updates 62600\
    --fuse-fbank-feat true\
    --fuse-speaker-embedding-feat false\
    --ts-len 10\
    --grad-clip true\
    --lr 1e-4\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --speech-encoder-config $speech_encoder_config\
    --select-encoder-layer-nums 6\
    --exp-dir $exp_dir
fi

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_w2v-bert2.0_first6layer_epoch40_front_fix_seed_lr1e4_freeze_speaker_encoder_model_online_utt_speaker_embedding_ts_len10_fuse_fbank
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 rs_segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="w2v-bert2"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 #spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 #speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

 for name in $infer_sets;do
    results_path=$exp_dir
  python3 ts_vad3/infer2.py \
    --model-file $model_file\
    --ts-len 10\
    --rs-len $rs_len\
    --rs-segment-shift $rs_segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name alimeeting_${name}.rttm\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --speech-encoder-config $speech_encoder_config\
    --wavlm-fuse-feat-post-norm false \
    --fuse-fbank-feat true\
    --fuse-speaker-embedding-feat false\
    --data-dir $data_dir
done
fi

# compared with stage3-4, stage11-12 will reduce ts_len from 10 to 4
if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is wav-bert2.0 (only using position embedding and first 6 layers conformer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this w2v-bert2.0 is trained Languages: 143+ , Size: 4.5M hours (it is from this paper https://arxiv.org/pdf/2312.05187)
    # checkpoint is from https://huggingface.co/facebook/w2v-bert-2.0/tree/main
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="w2v-bert2"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
    speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    #spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    #speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_w2v-bert2.0_first6layer_epoch40_front_fix_seed_lr1e4_freeze_speaker_encoder_model_online_utt_speaker_embedding_ts_len4
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 12115 \
   ts_vad3/train_accelerate_ddp2.py \
    --verbose 1 \
    --world-size 2 \
    --num-epochs 40\
    --start-epoch 1\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-speech-encoder-updates 4000\
    --freeze-speaker-encoder-updates 62600\
    --fuse-fbank-feat false\
    --fuse-speaker-embedding-feat false\
    --ts-len 4\
    --grad-clip true\
    --lr 1e-4\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --speech-encoder-config $speech_encoder_config\
    --select-encoder-layer-nums 6\
    --exp-dir $exp_dir
fi

if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_w2v-bert2.0_first6layer_epoch40_front_fix_seed_lr1e4_freeze_speaker_encoder_model_online_utt_speaker_embedding_ts_len4
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 rs_segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="w2v-bert2"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 #spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 #speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

 for name in $infer_sets;do
    results_path=$exp_dir
  python3 ts_vad3/infer2.py \
    --model-file $model_file\
    --ts-len 4\
    --rs-len $rs_len\
    --rs-segment-shift $rs_segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name alimeeting_${name}.rttm\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --speech-encoder-config $speech_encoder_config\
    --wavlm-fuse-feat-post-norm false \
    --fuse-fbank-feat false\
    --fuse-speaker-embedding-feat false\
    --data-dir $data_dir
done

# Eval set
## Model DER:  0.1264568620906494
#Model ACC:  0.9553069648463788
#100%|██████████| 25/25 [00:34<00:00,  1.36s/it]
#Eval for threshold 0.20: DER 8.19%, MS 0.86%, FA 6.98%, SC 0.36%
#
#Eval for threshold 0.30: DER 5.79%, MS 1.29%, FA 3.96%, SC 0.54%
#
#Eval for threshold 0.35: DER 5.17%, MS 1.55%, FA 3.02%, SC 0.59%
#
#Eval for threshold 0.40: DER 4.77%, MS 1.84%, FA 2.29%, SC 0.64%
#
#Eval for threshold 0.45: DER 4.60%, MS 2.18%, FA 1.75%, SC 0.66% as report
#
#Eval for threshold 0.50: DER 4.64%, MS 2.63%, FA 1.31%, SC 0.69%
#
#Eval for threshold 0.55: DER 4.83%, MS 3.20%, FA 1.02%, SC 0.61%
#
#Eval for threshold 0.60: DER 5.14%, MS 3.82%, FA 0.82%, SC 0.50%
#
#Eval for threshold 0.70: DER 6.31%, MS 5.39%, FA 0.55%, SC 0.36%
#
#Eval for threshold 0.80: DER 8.57%, MS 7.92%, FA 0.41%, SC 0.23%
#
#Test set
#Model DER:  0.1446648540670526
#Model ACC:  0.9426719281147777
#100%|██████████| 60/60 [01:24<00:00,  1.41s/it]
#Eval for threshold 0.20: DER 12.11%, MS 1.08%, FA 9.13%, SC 1.89%
#
#Eval for threshold 0.30: DER 9.42%, MS 1.70%, FA 5.22%, SC 2.51%
#
#Eval for threshold 0.35: DER 8.74%, MS 2.11%, FA 3.88%, SC 2.74%
#
#Eval for threshold 0.40: DER 8.29%, MS 2.57%, FA 2.78%, SC 2.95%
#
#Eval for threshold 0.45: DER 8.12%, MS 3.18%, FA 1.92%, SC 3.02% as report
#
#Eval for threshold 0.50: DER 8.30%, MS 4.03%, FA 1.32%, SC 2.95%
#
#Eval for threshold 0.55: DER 8.76%, MS 5.15%, FA 0.94%, SC 2.68%
#
#Eval for threshold 0.60: DER 9.42%, MS 6.34%, FA 0.69%, SC 2.39%
#
#Eval for threshold 0.70: DER 11.34%, MS 9.13%, FA 0.37%, SC 1.84%
#
#Eval for threshold 0.80: DER 14.41%, MS 12.97%, FA 0.20%, SC 1.24%

fi

# compared with stage3-4, stage13-14 will add fuse_speaker_embedding_feat
if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is wav-bert2.0 (only using position embedding and first 6 layers conformer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this w2v-bert2.0 is trained Languages: 143+ , Size: 4.5M hours (it is from this paper https://arxiv.org/pdf/2312.05187)
    # checkpoint is from https://huggingface.co/facebook/w2v-bert-2.0/tree/main
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="w2v-bert2"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
    speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    #spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    #speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_w2v-bert2.0_first6layer_epoch40_front_fix_seed_lr1e4_freeze_speaker_encoder_model_online_utt_speaker_embedding_ts_len10_fuse_speaker_embedding_feat
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 12695 \
   ts_vad3/train_accelerate_ddp2.py \
    --verbose 1 \
    --world-size 2 \
    --num-epochs 40\
    --start-epoch 1\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-speech-encoder-updates 4000\
    --freeze-speaker-encoder-updates 62600\
    --fuse-fbank-feat false\
    --fuse-speaker-embedding-feat true\
    --ts-len 10\
    --grad-clip true\
    --lr 1e-4\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --speech-encoder-config $speech_encoder_config\
    --select-encoder-layer-nums 6\
    --exp-dir $exp_dir
fi

if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_w2v-bert2.0_first6layer_epoch40_front_fix_seed_lr1e4_freeze_speaker_encoder_model_online_utt_speaker_embedding_ts_len10_fuse_speaker_embedding_feat
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 rs_segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="w2v-bert2"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 #spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 #speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

 for name in $infer_sets;do
    results_path=$exp_dir
  python3 ts_vad3/infer2.py \
    --model-file $model_file\
    --ts-len 10\
    --rs-len $rs_len\
    --rs-segment-shift $rs_segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name alimeeting_${name}.rttm\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --speech-encoder-config $speech_encoder_config\
    --wavlm-fuse-feat-post-norm false \
    --fuse-fbank-feat false\
    --fuse-speaker-embedding-feat true\
    --data-dir $data_dir
done

#cat logs/run_ts_vad3_stage13-14_v1_A100.log
## Eval set
## Model DER:  0.13917269943227759
#Model ACC:  0.9494997142962723
#100%|██████████| 25/25 [00:35<00:00,  1.40s/it]
#Eval for threshold 0.20: DER 10.07%, MS 0.85%, FA 8.54%, SC 0.69%
#
#Eval for threshold 0.30: DER 7.59%, MS 1.31%, FA 5.31%, SC 0.96%
#
#Eval for threshold 0.35: DER 6.88%, MS 1.57%, FA 4.21%, SC 1.09%
#
#Eval for threshold 0.40: DER 6.26%, MS 1.83%, FA 3.19%, SC 1.24%
#
#Eval for threshold 0.45: DER 5.89%, MS 2.15%, FA 2.37%, SC 1.37%
#
#Eval for threshold 0.50: DER 5.77%, MS 2.62%, FA 1.72%, SC 1.43%
#
#Eval for threshold 0.55: DER 5.93%, MS 3.26%, FA 1.31%, SC 1.36%
#
#Eval for threshold 0.60: DER 6.30%, MS 4.01%, FA 1.04%, SC 1.25%
#
#Eval for threshold 0.70: DER 7.57%, MS 5.91%, FA 0.68%, SC 0.98%
#
#Eval for threshold 0.80: DER 9.84%, MS 8.67%, FA 0.47%, SC 0.71%
#
#Test set
#Model DER:  0.1585302792089608
#Model ACC:  0.9340861177047974
#100%|██████████| 60/60 [01:22<00:00,  1.37s/it]
#Eval for threshold 0.20: DER 12.62%, MS 1.12%, FA 9.02%, SC 2.48%
#
#Eval for threshold 0.30: DER 10.06%, MS 1.83%, FA 5.13%, SC 3.10%
#
#Eval for threshold 0.35: DER 9.37%, MS 2.24%, FA 3.76%, SC 3.37%
#
#Eval for threshold 0.40: DER 8.98%, MS 2.73%, FA 2.69%, SC 3.56%
#
#Eval for threshold 0.45: DER 8.88%, MS 3.39%, FA 1.84%, SC 3.64%
#
#Eval for threshold 0.50: DER 9.12%, MS 4.30%, FA 1.28%, SC 3.55%
#
#Eval for threshold 0.55: DER 9.74%, MS 5.46%, FA 0.97%, SC 3.31%
#
#Eval for threshold 0.60: DER 10.40%, MS 6.65%, FA 0.72%, SC 3.03%
#
#Eval for threshold 0.70: DER 12.33%, MS 9.49%, FA 0.39%, SC 2.45%
#
#Eval for threshold 0.80: DER 15.30%, MS 13.21%, FA 0.20%, SC 1.89%
fi

# compared with stage 9-10, stage15-16 will use offline extract speaker frame embedding
if [ ${stage} -le 15 ] && [ ${stop_stage} -ge 15 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is wav-bert2.0 (only using position embedding and first 6 layers conformer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this w2v-bert2.0 is trained Languages: 143+ , Size: 4.5M hours (it is from this paper https://arxiv.org/pdf/2312.05187)
    # checkpoint is from https://huggingface.co/facebook/w2v-bert-2.0/tree/main
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="w2v-bert2"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
    speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker utt level embedding directory
    speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
    frame_spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/frame_featSpeakerEmbedding # store speaker frame level embedding directory
    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_w2v-bert2.0_first6layer_epoch40_front_fix_seed_lr1e4_offline_utt_speaker_embedding_ts_len10_fuse_speaker_frame_embedding_feat
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 13695 \
   ts_vad3/train_accelerate_ddp3.py \
    --verbose 1 \
    --world-size 2 \
    --num-epochs 40\
    --start-epoch 2\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --frame-spk-path $frame_spk_path\
    --freeze-speech-encoder-updates 4000\
    --freeze-speaker-encoder-updates 62600\
    --fuse-fbank-feat false\
    --fuse-speaker-embedding-feat true\
    --ts-len 10\
    --grad-clip true\
    --lr 1e-4\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --speech-encoder-config $speech_encoder_config\
    --select-encoder-layer-nums 6\
    --exp-dir $exp_dir
fi

if [ ${stage} -le 16 ] && [ ${stop_stage} -ge 16 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_w2v-bert2.0_first6layer_epoch40_front_fix_seed_lr1e4_offline_utt_speaker_embedding_ts_len10_fuse_speaker_frame_embedding_feat
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 rs_segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="w2v-bert2"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker utt level embedding directory
 speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 frame_spk_path=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/ts_vad/spk_embed/alimeeting/frame_featSpeakerEmbedding # store speaker frame level embedding directory
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

 for name in $infer_sets;do
    results_path=$exp_dir
  python3 ts_vad3/infer3.py \
    --model-file $model_file\
    --ts-len 10\
    --rs-len $rs_len\
    --rs-segment-shift $rs_segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name alimeeting_${name}.rttm\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --speech-encoder-config $speech_encoder_config\
    --wavlm-fuse-feat-post-norm false \
    --fuse-fbank-feat false\
    --fuse-speaker-embedding-feat true\
    --spk-path $spk_path\
    --frame-spk-path $frame_spk_path\
    --data-dir $data_dir
 done
fi
# compared with stage 13-14 ,stage 17-18 ts_len is decreased from 10 to 6
if [ ${stage} -le 17 ] && [ ${stop_stage} -ge 17 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is wav-bert2.0 (only using position embedding and first 6 layers conformer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this w2v-bert2.0 is trained Languages: 143+ , Size: 4.5M hours (it is from this paper https://arxiv.org/pdf/2312.05187)
    # checkpoint is from https://huggingface.co/facebook/w2v-bert-2.0/tree/main
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="w2v-bert2"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
    speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    #spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    #speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_w2v-bert2.0_first6layer_epoch40_front_fix_seed_lr1e4_freeze_speaker_encoder_model_online_utt_speaker_embedding_ts_len6_fuse_speaker_embedding_feat
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 12695 \
   ts_vad3/train_accelerate_ddp2.py \
    --verbose 1 \
    --world-size 2 \
    --num-epochs 40\
    --start-epoch 1\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-speech-encoder-updates 4000\
    --freeze-speaker-encoder-updates 62600\
    --fuse-fbank-feat false\
    --fuse-speaker-embedding-feat true\
    --ts-len 6\
     --grad-clip true\
    --lr 1e-4\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --speech-encoder-config $speech_encoder_config\
    --select-encoder-layer-nums 6\
    --exp-dir $exp_dir
fi

if [ ${stage} -le 18 ] && [ ${stop_stage} -ge 18 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_w2v-bert2.0_first6layer_epoch40_front_fix_seed_lr1e4_freeze_speaker_encoder_model_online_utt_speaker_embedding_ts_len6_fuse_speaker_embedding_feat
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 rs_segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="w2v-bert2"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 #spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 #speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

 for name in $infer_sets;do
    results_path=$exp_dir
  python3 ts_vad3/infer2.py \
    --model-file $model_file\
    --ts-len 6\
    --rs-len $rs_len\
    --rs-segment-shift $rs_segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name alimeeting_${name}.rttm\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --speech-encoder-config $speech_encoder_config\
    --wavlm-fuse-feat-post-norm false \
    --fuse-fbank-feat false\
    --fuse-speaker-embedding-feat true\
    --data-dir $data_dir
done
#cat logs/run_ts_vad3_stage17-18_v1_A100.log
# Eval set
#Model DER:  0.15450913990128695
#Model ACC:  0.9409164655453593
#100%|██████████| 25/25 [00:33<00:00,  1.36s/it]
#Eval for threshold 0.20: DER 12.96%, MS 0.84%, FA 11.13%, SC 0.99%
#
#Eval for threshold 0.30: DER 10.01%, MS 1.31%, FA 7.12%, SC 1.58%
#
#Eval for threshold 0.35: DER 9.02%, MS 1.59%, FA 5.55%, SC 1.88%
#
#Eval for threshold 0.40: DER 8.36%, MS 1.93%, FA 4.26%, SC 2.16%
#
#Eval for threshold 0.45: DER 7.99%, MS 2.37%, FA 3.12%, SC 2.49%
#
#Eval for threshold 0.50: DER 7.65%, MS 2.89%, FA 2.05%, SC 2.70%
#
#Eval for threshold 0.55: DER 7.74%, MS 3.66%, FA 1.35%, SC 2.73%
#
#Eval for threshold 0.60: DER 8.35%, MS 4.88%, FA 1.03%, SC 2.43%
#
#Eval for threshold 0.70: DER 10.08%, MS 7.72%, FA 0.62%, SC 1.73%
#
#Eval for threshold 0.80: DER 12.77%, MS 11.21%, FA 0.44%, SC 1.12%
#
#Test set
#Model DER:  0.15437002089330631
#Model ACC:  0.9377302809385323
#100%|██████████| 60/60 [01:23<00:00,  1.39s/it]
#Eval for threshold 0.20: DER 13.09%, MS 1.02%, FA 9.98%, SC 2.08%
#
#Eval for threshold 0.30: DER 10.00%, MS 1.66%, FA 5.64%, SC 2.70%
#
#Eval for threshold 0.35: DER 9.15%, MS 2.07%, FA 4.15%, SC 2.93%
#
#Eval for threshold 0.40: DER 8.72%, MS 2.58%, FA 2.97%, SC 3.18%
#
#Eval for threshold 0.45: DER 8.49%, MS 3.17%, FA 1.97%, SC 3.35%
#
#Eval for threshold 0.50: DER 8.68%, MS 4.10%, FA 1.29%, SC 3.30%
#
#Eval for threshold 0.55: DER 9.14%, MS 5.21%, FA 0.90%, SC 3.04%
#
#Eval for threshold 0.60: DER 9.89%, MS 6.47%, FA 0.64%, SC 2.78%
#
#Eval for threshold 0.70: DER 11.92%, MS 9.37%, FA 0.32%, SC 2.23%
#
#Eval for threshold 0.80: DER 15.11%, MS 13.29%, FA 0.15%, SC 1.67%


fi

# compared with stage 11-12 ,stage 19-20 will add fuse frame speaker embedding
if [ ${stage} -le 19 ] && [ ${stop_stage} -ge 19 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is wav-bert2.0 (only using position embedding and first 6 layers conformer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this w2v-bert2.0 is trained Languages: 143+ , Size: 4.5M hours (it is from this paper https://arxiv.org/pdf/2312.05187)
    # checkpoint is from https://huggingface.co/facebook/w2v-bert-2.0/tree/main
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="w2v-bert2"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
    speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    #spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    #speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_w2v-bert2.0_first6layer_epoch40_front_fix_seed_lr1e4_freeze_speaker_encoder_model_online_utt_speaker_embedding_ts_len4_fuse_speaker_embedding_feat
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 12495 \
   ts_vad3/train_accelerate_ddp2.py \
    --verbose 1 \
    --world-size 2 \
    --num-epochs 40\
    --start-epoch 1\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-speech-encoder-updates 4000\
    --freeze-speaker-encoder-updates 62600\
    --fuse-fbank-feat false\
    --fuse-speaker-embedding-feat true\
    --ts-len 4\
     --grad-clip true\
    --lr 1e-4\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --speech-encoder-config $speech_encoder_config\
    --select-encoder-layer-nums 6\
    --exp-dir $exp_dir
fi

if [ ${stage} -le 20 ] && [ ${stop_stage} -ge 20 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_w2v-bert2.0_first6layer_epoch40_front_fix_seed_lr1e4_freeze_speaker_encoder_model_online_utt_speaker_embedding_ts_len4_fuse_speaker_embedding_feat
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 rs_segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="w2v-bert2"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 #spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 #speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

 for name in $infer_sets;do
    results_path=$exp_dir
  python3 ts_vad3/infer2.py \
    --model-file $model_file\
    --ts-len 4\
    --rs-len $rs_len\
    --rs-len $rs_len\
    --rs-segment-shift $rs_segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name alimeeting_${name}.rttm\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --speech-encoder-config $speech_encoder_config\
    --wavlm-fuse-feat-post-norm false \
    --fuse-fbank-feat false\
    --fuse-speaker-embedding-feat true\
    --data-dir $data_dir
done
# sbatch --nodes 1 --gres=gpu:2  --cpus-per-gpu=8  --ntasks-per-node 2  -p p-A100 -A t00120220002 -o logs/run_ts_vad3_stage19-20_v1_A100.log run_ts_vad3.sh --stage 19 --stop-stage 20
# cat logs/run_ts_vad3_stage19-20_v1_A100.log
# Eval set
#Model DER:  0.12619006790261547
#Model ACC:  0.9559363256472596
#100%|██████████| 25/25 [00:33<00:00,  1.32s/it]
#Eval for threshold 0.20: DER 8.79%, MS 0.80%, FA 7.70%, SC 0.29%
#
#Eval for threshold 0.30: DER 5.97%, MS 1.18%, FA 4.41%, SC 0.39%
#
#Eval for threshold 0.35: DER 5.21%, MS 1.42%, FA 3.37%, SC 0.42%
#
#Eval for threshold 0.40: DER 4.77%, MS 1.74%, FA 2.55%, SC 0.48%
#
#Eval for threshold 0.45: DER 4.55%, MS 2.05%, FA 1.99%, SC 0.52% as report
#
#Eval for threshold 0.50: DER 4.53%, MS 2.47%, FA 1.53%, SC 0.53%
#
#Eval for threshold 0.55: DER 4.59%, MS 2.91%, FA 1.17%, SC 0.50%
#
#Eval for threshold 0.60: DER 4.81%, MS 3.43%, FA 0.92%, SC 0.47%
#
#Eval for threshold 0.70: DER 5.98%, MS 5.00%, FA 0.64%, SC 0.34%
#
#Eval for threshold 0.80: DER 8.21%, MS 7.56%, FA 0.45%, SC 0.20%
#
#
#Test set
#Model DER:  0.13966956175149237
#Model ACC:  0.9448512829182962
#100%|██████████| 60/60 [01:22<00:00,  1.37s/it]
#Eval for threshold 0.20: DER 11.40%, MS 1.01%, FA 8.54%, SC 1.85%
#
#Eval for threshold 0.30: DER 8.72%, MS 1.62%, FA 4.87%, SC 2.23%
#
#Eval for threshold 0.35: DER 8.01%, MS 2.02%, FA 3.62%, SC 2.37%
#
#Eval for threshold 0.40: DER 7.60%, MS 2.50%, FA 2.65%, SC 2.46%
#
#Eval for threshold 0.45: DER 7.46%, MS 3.06%, FA 1.89%, SC 2.51% as report
#
#Eval for threshold 0.50: DER 7.60%, MS 3.80%, FA 1.35%, SC 2.45%
#
#Eval for threshold 0.55: DER 8.03%, MS 4.76%, FA 1.00%, SC 2.26%
#
#Eval for threshold 0.60: DER 8.60%, MS 5.80%, FA 0.70%, SC 2.09%
#
#Eval for threshold 0.70: DER 10.22%, MS 8.25%, FA 0.35%, SC 1.62%
#
#Eval for threshold 0.80: DER 12.91%, MS 11.53%, FA 0.19%, SC 1.18%
fi

# compared with stage 11-12 ,stage 21-22 will ts_len is decreased from 4 to 3(cam++ is trained on sv task, its wav_len is 3s)
# reference: https://github.com/modelscope/3D-Speaker/blob/main/egs/3dspeaker/sv-cam%2B%2B/conf/cam%2B%2B.yaml#L16
if [ ${stage} -le 21 ] && [ ${stop_stage} -ge 21 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is wav-bert2.0 (only using position embedding and first 6 layers conformer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this w2v-bert2.0 is trained Languages: 143+ , Size: 4.5M hours (it is from this paper https://arxiv.org/pdf/2312.05187)
    # checkpoint is from https://huggingface.co/facebook/w2v-bert-2.0/tree/main
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="w2v-bert2"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
    speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    #spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    #speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_w2v-bert2.0_first6layer_epoch40_front_fix_seed_lr1e4_freeze_speaker_encoder_model_online_utt_speaker_embedding_ts_len3
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 12795 \
   ts_vad3/train_accelerate_ddp2.py \
    --verbose 1 \
    --world-size 2 \
    --num-epochs 40\
    --start-epoch 39\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-speech-encoder-updates 4000\
    --freeze-speaker-encoder-updates 62600\
    --fuse-fbank-feat false\
    --fuse-speaker-embedding-feat false\
    --ts-len 3\
    --grad-clip true\
    --lr 1e-4\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --speech-encoder-config $speech_encoder_config\
    --select-encoder-layer-nums 6\
    --exp-dir $exp_dir
fi

if [ ${stage} -le 22 ] && [ ${stop_stage} -ge 22 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_w2v-bert2.0_first6layer_epoch40_front_fix_seed_lr1e4_freeze_speaker_encoder_model_online_utt_speaker_embedding_ts_len3
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 rs_segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="w2v-bert2"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 #spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 #speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

 for name in $infer_sets;do
    results_path=$exp_dir
  python3 ts_vad3/infer2.py \
    --model-file $model_file\
    --ts-len 3\
    --rs-len $rs_len\
    --rs-len $rs_len\
    --rs-segment-shift $rs_segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name alimeeting_${name}.rttm\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --speech-encoder-config $speech_encoder_config\
    --wavlm-fuse-feat-post-norm false \
    --fuse-fbank-feat false\
    --fuse-speaker-embedding-feat false\
    --data-dir $data_dir
done
fi

##compared with stage21-22, stage23-24 will add fuse frame level speaker embedding
if [ ${stage} -le 23 ] && [ ${stop_stage} -ge 23 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is wav-bert2.0 (only using position embedding and first 6 layers conformer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this w2v-bert2.0 is trained Languages: 143+ , Size: 4.5M hours (it is from this paper https://arxiv.org/pdf/2312.05187)
    # checkpoint is from https://huggingface.co/facebook/w2v-bert-2.0/tree/main
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="w2v-bert2"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
    speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    #spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    #speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_w2v-bert2.0_first6layer_epoch40_front_fix_seed_lr1e4_freeze_speaker_encoder_model_online_utt_speaker_embedding_ts_len3_fuse_speaker_embedding_feat
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 12595 \
   ts_vad3/train_accelerate_ddp2.py \
    --verbose 1 \
    --world-size 2 \
    --num-epochs 40\
    --start-epoch 1\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-speech-encoder-updates 4000\
    --freeze-speaker-encoder-updates 62600\
    --fuse-fbank-feat false\
    --fuse-speaker-embedding-feat true\
    --ts-len 3\
    --grad-clip true\
    --lr 1e-4\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --speech-encoder-config $speech_encoder_config\
    --select-encoder-layer-nums 6\
    --exp-dir $exp_dir
fi

if [ ${stage} -le 24 ] && [ ${stop_stage} -ge 24 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_w2v-bert2.0_first6layer_epoch40_front_fix_seed_lr1e4_freeze_speaker_encoder_model_online_utt_speaker_embedding_ts_len3_fuse_speaker_embedding_feat
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 rs_segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="w2v-bert2"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 #spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 #speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

 for name in $infer_sets;do
    results_path=$exp_dir
  python3 ts_vad3/infer2.py \
    --model-file $model_file\
    --ts-len 3\
    --rs-len $rs_len\
    --rs-segment-shift $rs_segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name alimeeting_${name}.rttm\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --speech-encoder-config $speech_encoder_config\
    --wavlm-fuse-feat-post-norm false \
    --fuse-fbank-feat false\
    --fuse-speaker-embedding-feat true\
    --data-dir $data_dir
done
fi
# compared with stage19-20,stage25-26 will unfreeze after 40000steps
if [ ${stage} -le 25 ] && [ ${stop_stage} -ge 25 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is wav-bert2.0 (only using position embedding and first 6 layers conformer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this w2v-bert2.0 is trained Languages: 143+ , Size: 4.5M hours (it is from this paper https://arxiv.org/pdf/2312.05187)
    # checkpoint is from https://huggingface.co/facebook/w2v-bert-2.0/tree/main
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="w2v-bert2"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
    speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    #spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    #speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_w2v-bert2.0_first6layer_epoch40_front_fix_seed_lr1e4_freeze_speaker_encoder_model_online_utt_speaker_embedding_ts_len4_fuse_speaker_embedding_feat_with_freeze_40000_steps
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 14595 \
   ts_vad3/train_accelerate_ddp2.py \
    --verbose 1 \
    --world-size 2 \
    --num-epochs 40\
    --start-epoch 1\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-speech-encoder-updates 4000\
    --freeze-speaker-encoder-updates 40000\
    --fuse-fbank-feat false\
    --fuse-speaker-embedding-feat true\
    --ts-len 4\
    --grad-clip true\
    --lr 1e-4\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --speech-encoder-config $speech_encoder_config\
    --select-encoder-layer-nums 6\
    --exp-dir $exp_dir
fi

if [ ${stage} -le 26 ] && [ ${stop_stage} -ge 26 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_w2v-bert2.0_first6layer_epoch40_front_fix_seed_lr1e4_freeze_speaker_encoder_model_online_utt_speaker_embedding_ts_len4_fuse_speaker_embedding_feat_with_freeze_40000_steps
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 rs_segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="w2v-bert2"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 #spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 #speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

 for name in $infer_sets;do
    results_path=$exp_dir
  python3 ts_vad3/infer2.py \
    --model-file $model_file\
    --ts-len 4\
    --rs-len $rs_len\
    --rs-segment-shift $rs_segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name alimeeting_${name}.rttm\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --speech-encoder-config $speech_encoder_config\
    --wavlm-fuse-feat-post-norm false \
    --fuse-fbank-feat false\
    --fuse-speaker-embedding-feat true\
    --data-dir $data_dir
done
fi

## compared with stage19-20, stage27-28 will  add fuse_fbank_feat and fuse_frame_speaker_feat=False
if [ ${stage} -le 27 ] && [ ${stop_stage} -ge 27 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is wav-bert2.0 (only using position embedding and first 6 layers conformer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this w2v-bert2.0 is trained Languages: 143+ , Size: 4.5M hours (it is from this paper https://arxiv.org/pdf/2312.05187)
    # checkpoint is from https://huggingface.co/facebook/w2v-bert-2.0/tree/main
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="w2v-bert2"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
    speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    #spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    #speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_w2v-bert2.0_first6layer_epoch40_front_fix_seed_lr1e4_freeze_speaker_encoder_model_online_utt_speaker_embedding_ts_len4_fuse_fbank_feat
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 12395 \
   ts_vad3/train_accelerate_ddp2.py \
    --verbose 1 \
    --world-size 2 \
    --num-epochs 40\
    --start-epoch 1\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-speech-encoder-updates 4000\
    --freeze-speaker-encoder-updates 62600\
    --fuse-fbank-feat true\
    --fuse-speaker-embedding-feat false\
    --ts-len 4\
    --grad-clip true\
    --lr 1e-4\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --speech-encoder-config $speech_encoder_config\
    --select-encoder-layer-nums 6\
    --exp-dir $exp_dir
fi

if [ ${stage} -le 28 ] && [ ${stop_stage} -ge 28 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_w2v-bert2.0_first6layer_epoch40_front_fix_seed_lr1e4_freeze_speaker_encoder_model_online_utt_speaker_embedding_ts_len4_fuse_fbank_feat
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 rs_segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="w2v-bert2"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 #spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 #speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

 for name in $infer_sets;do
     results_path=$exp_dir
  python3 ts_vad3/infer2.py \
    --model-file $model_file\
    --ts-len 4\
    --rs-len $rs_len\
    --rs-len $rs_len\
    --rs-segment-shift $rs_segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name alimeeting_${name}.rttm\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --speech-encoder-config $speech_encoder_config\
    --wavlm-fuse-feat-post-norm false \
    --fuse-fbank-feat true\
    --fuse-speaker-embedding-feat false\
    --data-dir $data_dir
done
fi
