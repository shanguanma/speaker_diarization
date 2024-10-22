#!/usr/bin/env bash

stage=0
stop_stage=1000

. utils/parse_options.sh
. path_for_speaker_diarization.sh
# compared with stage104-105 of run_ts_vad2.sh, it is sota util 2024-10-22.
# 1. I will put speaker encoder model put into our tsvad model and freeze its parameters
# 2. I will use attention method on target speaker fbank feat and mix speech fbank.
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
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_w2v-bert2.0_first6layer_epoch40_front_fix_seed_lr1e4_freeze_speaker_encoder_model_fuse_fbank
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 12815 \
   ts_vad3/train_accelerate_ddp2_debug.py \
    --world-size 2 \
    --num-epochs 40\
    --start-epoch 1\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-speech-encoder-updates 4000\
    --freeze-speaker-encoder-updates 62600\
    --fuse-fbank-feat true\
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
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_w2v-bert2.0_first6layer_epoch40_front_fix_seed_lr1e4_freeze_speaker_encoder_model_fuse_fbank
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
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
    --rs-len $rs_len\
    --segment-shift $segment_shift\
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
