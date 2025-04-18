#!/usr/bin/env bash
stage=0
stop_stage=1000

. utils/parse_options.sh
. path_for_dia_pt2.4.sh


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
    dataset_name="alimeeting_ami_aishell_4" # dataset name
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    # cam++ 200k
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    # cam++ en_zh
    #speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    spk_path=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/data/alimeeting_ami_aishell_4 # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/alimeeting_ami_aishell_4_ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_mamba2_multi_backend_transformer_d_state256_rs_len10
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/data/alimeeting_ami_aishell_4" # oracle target audio , mix audio and labels path
   rs_len=10
   segment_shift=2
   single_backend_type="mamba2"
   multi_backend_type="transformer"
   d_state=256
   num_transformer_layer=2
   max_num_speaker=7
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 17115 \
   ts_vad2/train_accelerate_ddp.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-updates 4000\
    --grad-clip true\
    --lr 1e-4\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --exp-dir $exp_dir\
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state\
    --max-num-speaker $max_num_speaker
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/alimeeting_ami_aishell_4_ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_mamba2_multi_backend_transformer_d_state256_rs_len10
 model_file=$exp_dir/best-valid-der.pt
 rs_len=10
 segment_shift=1
 single_backend_type="mamba2"
 multi_backend_type="transformer"
 d_state=256
 num_transformer_layer=2
 max_num_speaker=7
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0

 dataset_name="alimeeting" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_path=/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_collar${c}
   python3 ts_vad2/infer.py \
    --model-file $model_file\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name alimeeting_${name}.rttm\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $c\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state\
    --max-num-speaker $max_num_speaker
 done
done
fi
#grep -r Eval logs/run_ts_vad2_stage2_infer_cam++_zh_200k_rs_len10.log
# Eval of alimeeting, collar=0.0
#2025-02-06 09:16:28,306 (infer:257) INFO: currently, it will infer Eval set.
#Eval for threshold 0.2 DER=22.48, miss=2.65, falarm=18.67, confusion=1.16
#Eval for threshold 0.3 DER=17.76, miss=4.08, falarm=12.28, confusion=1.40
#Eval for threshold 0.35 DER=16.33, miss=4.89, falarm=10.05, confusion=1.39
#Eval for threshold 0.4 DER=15.53, miss=5.78, falarm=8.37, confusion=1.39
#Eval for threshold 0.45 DER=14.90, miss=6.69, falarm=6.79, confusion=1.41
#Eval for threshold 0.5 DER=14.60, miss=7.70, falarm=5.50, confusion=1.40
#Eval for threshold 0.55 DER=14.60, miss=8.87, falarm=4.45, confusion=1.29
#Eval for threshold 0.6 DER=14.82, miss=10.10, falarm=3.53, confusion=1.19
#Eval for threshold 0.7 DER=16.29, miss=13.15, falarm=2.24, confusion=0.90
#Eval for threshold 0.8 DER=19.33, miss=17.43, falarm=1.32, confusion=0.58

# Test of alimeeting, collar=0.0
#Eval for threshold 0.2 DER=21.39, miss=2.60, falarm=17.44, confusion=1.35
#Eval for threshold 0.3 DER=17.04, miss=4.12, falarm=11.37, confusion=1.55
#Eval for threshold 0.35 DER=15.80, miss=5.04, falarm=9.13, confusion=1.63
#Eval for threshold 0.4 DER=15.07, miss=6.03, falarm=7.40, confusion=1.64
#Eval for threshold 0.45 DER=14.70, miss=7.10, falarm=5.97, confusion=1.64
#Eval for threshold 0.5 DER=14.60, miss=8.22, falarm=4.82, confusion=1.56
#Eval for threshold 0.55 DER=14.77, miss=9.43, falarm=3.88, confusion=1.47
#Eval for threshold 0.6 DER=15.23, miss=10.79, falarm=3.09, confusion=1.35
#Eval for threshold 0.7 DER=17.14, miss=14.17, falarm=1.93, confusion=1.04
#Eval for threshold 0.8 DER=20.57, miss=18.78, falarm=1.10, confusion=0.69

# Eval of alimeeting, collar=0.25
#2025-02-06 09:30:13,561 (infer:257) INFO: currently, it will infer Eval set.
#Eval for threshold 0.2 DER=10.94, miss=0.88, falarm=9.71, confusion=0.35
#Eval for threshold 0.3 DER=7.74, miss=1.50, falarm=5.80, confusion=0.44
#Eval for threshold 0.35 DER=6.80, miss=1.84, falarm=4.50, confusion=0.45
#Eval for threshold 0.4 DER=6.27, miss=2.25, falarm=3.56, confusion=0.46
#Eval for threshold 0.45 DER=5.96, miss=2.74, falarm=2.72, confusion=0.49
#Eval for threshold 0.5 DER=5.87, miss=3.32, falarm=2.08, confusion=0.47
#Eval for threshold 0.55 DER=5.96, miss=3.97, falarm=1.58, confusion=0.41
#Eval for threshold 0.6 DER=6.17, miss=4.65, falarm=1.15, confusion=0.37
#Eval for threshold 0.7 DER=7.42, miss=6.50, falarm=0.65, confusion=0.27
#Eval for threshold 0.8 DER=9.86, miss=9.30, falarm=0.39, confusion=0.16

# Test of alimeeting, collar=0.25
#Eval for threshold 0.2 DER=10.40, miss=1.07, falarm=8.81, confusion=0.51
#Eval for threshold 0.3 DER=7.40, miss=1.80, falarm=4.99, confusion=0.62
#Eval for threshold 0.35 DER=6.64, miss=2.30, falarm=3.68, confusion=0.66
#Eval for threshold 0.4 DER=6.19, miss=2.82, falarm=2.70, confusion=0.68
#Eval for threshold 0.45 DER=6.05, miss=3.39, falarm=1.96, confusion=0.70
#Eval for threshold 0.5 DER=6.11, miss=4.03, falarm=1.42, confusion=0.66
#Eval for threshold 0.55 DER=6.31, miss=4.68, falarm=1.01, confusion=0.62
#Eval for threshold 0.6 DER=6.74, miss=5.45, falarm=0.71, confusion=0.57
#Eval for threshold 0.7 DER=8.28, miss=7.52, falarm=0.35, confusion=0.40
#Eval for threshold 0.8 DER=10.94, miss=10.56, falarm=0.14, confusion=0.24

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/alimeeting_ami_aishell_4_ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_mamba2_multi_backend_transformer_d_state256_rs_len10
 model_file=$exp_dir/best-valid-der.pt
 rs_len=10
 segment_shift=1
 single_backend_type="mamba2"
 multi_backend_type="transformer"
 d_state=256
 num_transformer_layer=2
 max_num_speaker=7
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/datasets/aishell-4/data_processed/
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0

 dataset_name="aishell_4" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_path=/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding
 spk_path=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/model_hub/ts_vad/spk_embed/aishell_4/SpeakerEmbedding/ # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/datasets/aishell-4/data_processed" # oracle target audio , mix audio and labels path
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_collar${c}
   python3 ts_vad2/infer.py \
    --model-file $model_file\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name ${name}/${name}.rttm\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $c\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state\
    --max-num-speaker $max_num_speaker
 done
done
fi
# grep -r Eval logs/run_ts_vad2_stage3_infer_cam++_zh_200k_rs_len10.log
# Test of aishell-4, collar=0.0
#Eval for threshold 0.2 DER=20.24, miss=2.80, falarm=13.78, confusion=3.66
#Eval for threshold 0.3 DER=15.86, miss=5.35, falarm=6.95, confusion=3.55
#Eval for threshold 0.35 DER=15.40, miss=6.81, falarm=5.23, confusion=3.36
#Eval for threshold 0.4 DER=15.40, miss=8.44, falarm=3.90, confusion=3.06
#Eval for threshold 0.45 DER=15.82, miss=10.13, falarm=2.93, confusion=2.76
#Eval for threshold 0.5 DER=16.65, miss=11.90, falarm=2.31, confusion=2.44
#Eval for threshold 0.55 DER=17.75, miss=13.79, falarm=1.84, confusion=2.11
#Eval for threshold 0.6 DER=19.08, miss=15.81, falarm=1.48, confusion=1.79
#Eval for threshold 0.7 DER=22.54, miss=20.28, falarm=0.99, confusion=1.27
#Eval for threshold 0.8 DER=27.71, miss=26.27, falarm=0.65, confusion=0.79

# Test of aishell-4, collar=0.25
#Eval for threshold 0.2 DER=14.21, miss=1.54, falarm=9.51, confusion=3.15
#Eval for threshold 0.3 DER=10.53, miss=3.50, falarm=3.99, confusion=3.04
#Eval for threshold 0.35 DER=10.27, miss=4.69, falarm=2.72, confusion=2.85
#Eval for threshold 0.4 DER=10.29, miss=6.02, falarm=1.69, confusion=2.57
#Eval for threshold 0.45 DER=10.76, miss=7.43, falarm=1.03, confusion=2.30
#Eval for threshold 0.5 DER=11.57, miss=8.91, falarm=0.68, confusion=1.98
#Eval for threshold 0.55 DER=12.68, miss=10.54, falarm=0.46, confusion=1.68
#Eval for threshold 0.6 DER=13.93, miss=12.21, falarm=0.31, confusion=1.41
#Eval for threshold 0.7 DER=17.16, miss=16.04, falarm=0.14, confusion=0.98
#Eval for threshold 0.8 DER=21.98, miss=21.29, falarm=0.07, confusion=0.62

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/alimeeting_ami_aishell_4_ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_mamba2_multi_backend_transformer_d_state256_rs_len10
 model_file=$exp_dir/best-valid-der.pt
 rs_len=10
 segment_shift=1
 single_backend_type="mamba2"
 multi_backend_type="transformer"
 d_state=256
 num_transformer_layer=2
 max_num_speaker=7
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/datasets/ami/ami_version1.6.2/data_processed/data/ami
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0

 dataset_name="ami" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_path=/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding
 spk_path=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/model_hub/ts_vad/spk_embed/ami/SpeakerEmbedding/ # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/datasets/ami/ami_version1.6.2/data_processed/data/ami" # oracle target audio , mix audio and labels path
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_collar${c}
   python3 ts_vad2/infer.py \
    --model-file $model_file\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name ${name}/${name}.rttm\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $c\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state\
    --max-num-speaker $max_num_speaker
 done
done
fi
#grep -r Eval logs/run_ts_vad2_stage4_infer_cam++_zh_200k_rs_len10.log
# test of ami, collar=0.0
#Eval for threshold 0.2 DER=28.60, miss=3.59, falarm=22.93, confusion=2.07
#Eval for threshold 0.3 DER=22.20, miss=6.08, falarm=13.49, confusion=2.62
#Eval for threshold 0.35 DER=20.90, miss=7.57, falarm=10.59, confusion=2.73
#Eval for threshold 0.4 DER=20.15, miss=9.19, falarm=8.22, confusion=2.74
#Eval for threshold 0.45 DER=19.91, miss=10.88, falarm=6.35, confusion=2.68
#Eval for threshold 0.5 DER=20.07, miss=12.67, falarm=4.84, confusion=2.56
#Eval for threshold 0.55 DER=20.69, miss=14.68, falarm=3.69, confusion=2.32
#Eval for threshold 0.6 DER=21.65, miss=16.87, falarm=2.74, confusion=2.04
#Eval for threshold 0.7 DER=24.53, miss=21.62, falarm=1.50, confusion=1.41
#Eval for threshold 0.8 DER=29.18, miss=27.58, falarm=0.78, confusion=0.82

# test of ami, collar=0.25
#Eval for threshold 0.2 DER=18.23, miss=2.51, falarm=14.60, confusion=1.11
#Eval for threshold 0.3 DER=13.63, miss=4.25, flarm=7.91, confusion=1.48
#Eval for threshold 0.35 DER=12.88, miss=5.31, falarm=5.99, confusion=1.59
#Eval for threshold 0.4 DER=12.56, miss=6.46, falarm=4.50, confusion=1.60
#Eval for threshold 0.45 DER=12.60, miss=7.68, falarm=3.32, confusion=1.60
#Eval for threshold 0.5 DER=12.89, miss=8.98, falarm=2.36, confusion=1.55
#Eval for threshold 0.55 DER=13.55, miss=10.45, falarm=1.68, confusion=1.41
#Eval for threshold 0.6 DER=14.44, miss=12.05, falarm=1.12, confusion=1.27
#Eval for threshold 0.7 DER=17.09, miss=15.73, falarm=0.49, confusion=0.87
#Eval for threshold 0.8 DER=21.27, miss=20.56, falarm=0.22, confusion=0.50


## lr=(1e-4)/2
#if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ];then
#    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
#    # # speech encoder is wav-bert2.0 (only using position embedding and first 6 layers conformer) ,
#    #  oracle target speaker embedding is from cam++ pretrain model
#    # this w2v-bert2.0 is trained Languages: 143+ , Size: 4.5M hours (it is from this paper https://arxiv.org/pdf/2312.05187)
#    # checkpoint is from https://huggingface.co/facebook/w2v-bert-2.0/tree/main
#    # how to look for port ?
#    # netstat -tuln
#    export NCCL_DEBUG=INFO
#    export PYTHONFAULTHANDLER=1
#    musan_path=/mntcephfs/lee_dataset/asr/musan
#    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
#    dataset_name="alimeeting_ami_aishell_4" # dataset name
#    # for loading pretrain model weigt
#    #speech_encoder_type="CAM++"
#    # cam++ 200k
#    #speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
#    # cam++ en_zh
#    #speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt"
#    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
#
#    speech_encoder_type="w2v-bert2"
#    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
#    speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
#    # for loading speaker embedding file
#    spk_path=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/data/alimeeting_ami_aishell_4 # store speaker embedding directory
#    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
#
#    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
#    exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/alimeeting_ami_aishell_4_ts_vad2_two_gpus_freeze_with_musan_rirs_w2v-bert2_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_mamba2_multi_backend_transformer_d_state256_rs_len10
#   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
#  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
#   data_dir="/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/data/alimeeting_ami_aishell_4" # oracle target audio , mix audio and labels path
#   rs_len=10
#   segment_shift=2
#   single_backend_type="mamba2"
#   multi_backend_type="transformer"
#   d_state=256
#   num_transformer_layer=2
#   max_num_speaker=7
#  CUDA_VISIABLE_DEVICES=0,1 \
#  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 17215 \
#   ts_vad2/train_accelerate_ddp.py \
#    --world-size 2 \
#    --num-epochs 20\
#    --start-epoch 1\
#    --keep-last-k 1\
#    --keep-last-epoch 1\
#    --freeze-updates 4000\
#    --grad-clip true\
#    --lr 5e-5\
#    --musan-path $musan_path \
#    --rir-path $rir_path \
#    --rs-len $rs_len\
#    --segment-shift $segment_shift\
#    --speech-encoder-type $speech_encoder_type\
#    --speech-encoder-path $speech_encoder_path\
#    --speech-encoder-config $speech_encoder_config\
#    --select-encoder-layer-nums 6\
#    --spk-path $spk_path\
#    --speaker-embedding-name-dir $speaker_embedding_name_dir\
#    --exp-dir $exp_dir\
#    --data-dir $data_dir\
#    --dataset-name $dataset_name\
#    --single-backend-type $single_backend_type\
#    --multi-backend-type $multi_backend_type\
#    --num-transformer-layer $num_transformer_layer\
#    --d-state $d_state\
#    --max-num-speaker $max_num_speaker
#fi
# epoch=3, loss is nan


if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ];then
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/alimeeting_ami_aishell_4_ts_vad2_two_gpus_freeze_with_musan_rirs_w2v-bert2_cam++_200k_zh_cn_epoch40_front_fix_seed_lr2e5_single_backend_2layer_mamba2_multi_backend_transformer_d_state256_rs_len10
 model_file=$exp_dir/best-valid-der.pt
 rs_len=10
 segment_shift=1
 single_backend_type="mamba2"
 multi_backend_type="transformer"
 d_state=256
 num_transformer_layer=2
 max_num_speaker=7
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0

 dataset_name="alimeeting" # dataset name

 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.

 #speech_encoder_type="CAM++"
 #speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_path=/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 speech_encoder_type="w2v-bert2"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_collar${c}
   python3 ts_vad2/infer.py \
    --model-file $model_file\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name alimeeting_${name}.rttm\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $c\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --speech-encoder-config $speech_encoder_config\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state\
    --max-num-speaker $max_num_speaker
 done
done
fi

if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ];then
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/alimeeting_ami_aishell_4_ts_vad2_two_gpus_freeze_with_musan_rirs_w2v-bert2_cam++_200k_zh_cn_epoch40_front_fix_seed_lr2e5_single_backend_2layer_mamba2_multi_backend_transformer_d_state256_rs_len10
 model_file=$exp_dir/best-valid-der.pt
 rs_len=10
 segment_shift=1
 single_backend_type="mamba2"
 multi_backend_type="transformer"
 d_state=256
 num_transformer_layer=2
 max_num_speaker=7
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/datasets/aishell-4/data_processed/
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0

 dataset_name="aishell_4" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 #speech_encoder_type="CAM++"
 #speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_path=/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 speech_encoder_type="w2v-bert2"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
 # for loading speaker embedding
 spk_path=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/model_hub/ts_vad/spk_embed/aishell_4/SpeakerEmbedding/ # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/datasets/aishell-4/data_processed" # oracle target audio , mix audio and labels path
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_collar${c}
   python3 ts_vad2/infer.py \
    --model-file $model_file\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name ${name}/${name}.rttm\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $c\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --speech-encoder-config $speech_encoder_config\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state\
    --max-num-speaker $max_num_speaker
 done
done
fi

if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ];then
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/alimeeting_ami_aishell_4_ts_vad2_two_gpus_freeze_with_musan_rirs_w2v-bert2_cam++_200k_zh_cn_epoch40_front_fix_seed_lr2e5_single_backend_2layer_mamba2_multi_backend_transformer_d_state256_rs_len10
 model_file=$exp_dir/best-valid-der.pt
 rs_len=10
 segment_shift=1
 single_backend_type="mamba2"
 multi_backend_type="transformer"
 d_state=256
 num_transformer_layer=2
 max_num_speaker=7
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/datasets/ami/ami_version1.6.2/data_processed/data/ami
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0

 dataset_name="ami" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 #speech_encoder_type="CAM++"
 #speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_path=/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 speech_encoder_type="w2v-bert2"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
 # for loading speaker embedding
 spk_path=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/model_hub/ts_vad/spk_embed/ami/SpeakerEmbedding/ # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/datasets/ami/ami_version1.6.2/data_processed/data/ami" # oracle target audio , mix audio and labels path
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_collar${c}
   python3 ts_vad2/infer.py \
    --model-file $model_file\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name ${name}/${name}.rttm\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $c\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --speech-encoder-config $speech_encoder_config\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state\
    --max-num-speaker $max_num_speaker
 done
done
fi

# lr=(1e-4)/2/2
if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ];then
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
    dataset_name="alimeeting_ami_aishell_4" # dataset name
    # for loading pretrain model weigt
    #speech_encoder_type="CAM++"
    # cam++ 200k
    #speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    # cam++ en_zh
    #speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    speech_encoder_type="w2v-bert2"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
    speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
    # for loading speaker embedding file
    spk_path=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/data/alimeeting_ami_aishell_4 # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/alimeeting_ami_aishell_4_ts_vad2_two_gpus_freeze_with_musan_rirs_w2v-bert2_cam++_200k_zh_cn_epoch40_front_fix_seed_lr2e5_single_backend_2layer_mamba2_multi_backend_transformer_d_state256_rs_len10
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/data/alimeeting_ami_aishell_4" # oracle target audio , mix audio and labels path
   rs_len=10
   segment_shift=2
   single_backend_type="mamba2"
   multi_backend_type="transformer"
   d_state=256
   num_transformer_layer=2
   max_num_speaker=7
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 17215 \
   ts_vad2/train_accelerate_ddp.py \
    --world-size 2 \
    --num-epochs 40\
    --start-epoch 1\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-updates 4000\
    --grad-clip true\
    --lr 2e-5\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --speech-encoder-config $speech_encoder_config\
    --select-encoder-layer-nums 6\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --exp-dir $exp_dir\
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state\
    --max-num-speaker $max_num_speaker
fi



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
    dataset_name="alimeeting_ami_aishell_4" # dataset name
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    # cam++ 200k
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    # cam++ en_zh
    #speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    spk_path=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/data/alimeeting_ami_aishell_4 # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/alimeeting_ami_aishell_4_ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_single_backend_2layer_transformer_multi_backend_transformer_rs_len8
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/data/alimeeting_ami_aishell_4" # oracle target audio , mix audio and labels path
   rs_len=8
   segment_shift=2
   single_backend_type="transformer"
   multi_backend_type="transformer"
   d_state=256
   num_transformer_layer=2
   max_num_speaker=7
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 17115 \
   ts_vad2/train_accelerate_ddp.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-updates 4000\
    --grad-clip true\
    --lr 2e-4\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --exp-dir $exp_dir\
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state\
    --max-num-speaker $max_num_speaker
fi

if [ ${stage} -le 22 ] && [ ${stop_stage} -ge 22 ];then
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/alimeeting_ami_aishell_4_ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_single_backend_2layer_transformer_multi_backend_transformer_rs_len8
 model_file=$exp_dir/best-valid-der.pt
 rs_len=8
 segment_shift=1
 single_backend_type="transformer"
 multi_backend_type="transformer"
 d_state=256
 num_transformer_layer=2
 max_num_speaker=7
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0

 dataset_name="alimeeting" # dataset name

 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.

 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_path=/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 #speech_encoder_type="w2v-bert2"
 #speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_collar${c}
   python3 ts_vad2/infer.py \
    --model-file $model_file\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name alimeeting_${name}.rttm\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $c\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state\
    --max-num-speaker $max_num_speaker
 done
done
fi
#grep -r Eval logs/run_ts_vad2_stage22_infer_cam++_zh_200k_transformer_rs_len8.log
# Eval of alimeeting, collar=0.0
#2025-02-08 09:06:41,681 (infer:257) INFO: currently, it will infer Eval set.
#Eval for threshold 0.2 DER=19.67, miss=2.45, falarm=15.89, confusion=1.32
#Eval for threshold 0.3 DER=15.99, miss=3.70, falarm=10.81, confusion=1.49
#Eval for threshold 0.35 DER=15.00, miss=4.45, falarm=9.03, confusion=1.52
#Eval for threshold 0.4 DER=14.30, miss=5.29, falarm=7.51, confusion=1.50
#Eval for threshold 0.45 DER=13.87, miss=6.13, falarm=6.27, confusion=1.48
#Eval for threshold 0.5 DER=13.64, miss=6.97, falarm=5.20, confusion=1.47
#Eval for threshold 0.55 DER=13.71, miss=7.91, falarm=4.40, confusion=1.41
#Eval for threshold 0.6 DER=13.91, miss=8.97, falarm=3.61, confusion=1.32
#Eval for threshold 0.7 DER=15.05, miss=11.64, falarm=2.36, confusion=1.05
#Eval for threshold 0.8 DER=17.33, miss=15.15, falarm=1.47, confusion=0.70
# Test of alimeeting, collar=0.0
#Eval for threshold 0.2 DER=17.96, miss=2.63, falarm=14.15, confusion=1.18
#Eval for threshold 0.3 DER=14.85, miss=4.02, falarm=9.48, confusion=1.34
#Eval for threshold 0.35 DER=14.03, miss=4.81, falarm=7.84, confusion=1.38
#Eval for threshold 0.4 DER=13.51, miss=5.67, falarm=6.47, confusion=1.37
#Eval for threshold 0.45 DER=13.30, miss=6.61, falarm=5.35, confusion=1.35
#Eval for threshold 0.5 DER=13.29, miss=7.57, falarm=4.44, confusion=1.27
#Eval for threshold 0.55 DER=13.42, miss=8.60, falarm=3.62, confusion=1.20
#Eval for threshold 0.6 DER=13.83, miss=9.73, falarm=2.99, confusion=1.10
#Eval for threshold 0.7 DER=15.22, miss=12.40, falarm=1.94, confusion=0.88
#Eval for threshold 0.8 DER=17.78, miss=16.05, falarm=1.13, confusion=0.60
# Eval of alimeeting, collar=0.25
#2025-02-08 09:17:27,866 (infer:257) INFO: currently, it will infer Eval set.
#Eval for threshold 0.2 DER=8.97, miss=0.85, falarm=7.64, confusion=0.48
#Eval for threshold 0.3 DER=6.68, miss=1.36, falarm=4.76, confusion=0.56
#Eval for threshold 0.35 DER=6.13, miss=1.75, falarm=3.80, confusion=0.58
#Eval for threshold 0.4 DER=5.71, miss=2.13, falarm=2.98, confusion=0.60
#Eval for threshold 0.45 DER=5.47, miss=2.54, falarm=2.34, confusion=0.60
#Eval for threshold 0.5 DER=5.33, miss=2.88, falarm=1.84, confusion=0.61
#Eval for threshold 0.55 DER=5.46, miss=3.35, falarm=1.53, confusion=0.59
#Eval for threshold 0.6 DER=5.69, miss=3.92, falarm=1.21, confusion=0.56
#Eval for threshold 0.7 DER=6.54, miss=5.41, falarm=0.71, confusion=0.43
#Eval for threshold 0.8 DER=8.29, miss=7.55, falarm=0.45, confusion=0.29

# Test of alimeeting, collar=0.25
#Eval for threshold 0.2 DER=7.80, miss=1.13, falarm=6.27, confusion=0.39
#Eval for threshold 0.3 DER=5.93, miss=1.81, falarm=3.66, confusion=0.46
#Eval for threshold 0.35 DER=5.48, miss=2.18, falarm=2.82, confusion=0.47
#Eval for threshold 0.4 DER=5.19, miss=2.60, falarm=2.13, confusion=0.46
#Eval for threshold 0.45 DER=5.14, miss=3.08, falarm=1.62, confusion=0.45
#Eval for threshold 0.5 DER=5.22, miss=3.58, falarm=1.22, confusion=0.42
#Eval for threshold 0.55 DER=5.42, miss=4.13, falarm=0.91, confusion=0.38
#Eval for threshold 0.6 DER=5.80, miss=4.75, falarm=0.70, confusion=0.34
#Eval for threshold 0.7 DER=6.89, miss=6.26, falarm=0.38, confusion=0.25
#Eval for threshold 0.8 DER=8.71, miss=8.38, falarm=0.18, confusion=0.16

if [ ${stage} -le 23 ] && [ ${stop_stage} -ge 23 ];then
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/alimeeting_ami_aishell_4_ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_single_backend_2layer_transformer_multi_backend_transformer_rs_len8
 model_file=$exp_dir/best-valid-der.pt
 rs_len=8
 segment_shift=1
 single_backend_type="transformer"
 multi_backend_type="transformer"
 d_state=256
 num_transformer_layer=2
 max_num_speaker=7
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/datasets/aishell-4/data_processed/
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0

 dataset_name="aishell_4" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_path=/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 #speech_encoder_type="w2v-bert2"
 #speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
 # for loading speaker embedding
 spk_path=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/model_hub/ts_vad/spk_embed/aishell_4/SpeakerEmbedding/ # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/datasets/aishell-4/data_processed" # oracle target audio , mix audio and labels path
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_collar${c}
   python3 ts_vad2/infer.py \
    --model-file $model_file\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name ${name}/${name}.rttm\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $c\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state\
    --max-num-speaker $max_num_speaker
 done
done
fi

#grep -r Eval logs/run_ts_vad2_stage23_infer_cam++_zh_200k_transformer_rs_len8.log

# collar=0.0
#Eval for threshold 0.2 DER=18.50, miss=1.07, falarm=15.87, confusion=1.55
#Eval for threshold 0.3 DER=12.63, miss=2.06, falarm=8.76, confusion=1.81
#Eval for threshold 0.35 DER=11.38, miss=2.72, falarm=6.78, confusion=1.88
#Eval for threshold 0.4 DER=10.60, miss=3.50, falarm=5.23, confusion=1.87
#Eval for threshold 0.45 DER=10.45, miss=4.48, falarm=4.17, confusion=1.80
#Eval for threshold 0.5 DER=10.69, miss=5.64, falarm=3.36, confusion=1.69
#Eval for threshold 0.55 DER=11.30, miss=7.00, falarm=2.73, confusion=1.57
#Eval for threshold 0.6 DER=12.34, miss=8.69, falarm=2.21, confusion=1.44
#Eval for threshold 0.7 DER=15.71, miss=13.10, falarm=1.47, confusion=1.14
#Eval for threshold 0.8 DER=21.32, miss=19.57, falarm=0.90, confusion=0.85

# collar=0.25
#Eval for threshold 0.2 DER=11.27, miss=0.47, falarm=9.83, confusion=0.98
#Eval for threshold 0.3 DER=6.57, miss=0.94, falarm=4.45, confusion=1.18
#Eval for threshold 0.35 DER=5.64, miss=1.27, falarm=3.12, confusion=1.24
#Eval for threshold 0.4 DER=5.04, miss=1.75, falarm=2.06, confusion=1.23
#Eval for threshold 0.45 DER=5.02, miss=2.41, falarm=1.44, confusion=1.17
#Eval for threshold 0.5 DER=5.31, miss=3.18, falarm=1.01, confusion=1.11
#Eval for threshold 0.55 DER=5.89, miss=4.14, falarm=0.70, confusion=1.04
#Eval for threshold 0.6 DER=6.90, miss=5.43, falarm=0.50, confusion=0.97
#Eval for threshold 0.7 DER=10.15, miss=9.12, falarm=0.26, confusion=0.77
#Eval for threshold 0.8 DER=15.54, miss=14.84, falarm=0.10, confusion=0.59


if [ ${stage} -le 24 ] && [ ${stop_stage} -ge 24 ];then
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/alimeeting_ami_aishell_4_ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_single_backend_2layer_transformer_multi_backend_transformer_rs_len8
 model_file=$exp_dir/best-valid-der.pt
 rs_len=8
 segment_shift=1
 single_backend_type="transformer"
 multi_backend_type="transformer"
 d_state=256
 num_transformer_layer=2
 max_num_speaker=7
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/datasets/ami/ami_version1.6.2/data_processed/data/ami
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0

 dataset_name="ami" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_path=/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 #speech_encoder_type="w2v-bert2"
 #speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
 # for loading speaker embedding
 spk_path=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/model_hub/ts_vad/spk_embed/ami/SpeakerEmbedding/ # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/datasets/ami/ami_version1.6.2/data_processed/data/ami" # oracle target audio , mix audio and labels path
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_collar${c}
   python3 ts_vad2/infer.py \
    --model-file $model_file\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name ${name}/${name}.rttm\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $c\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state\
    --max-num-speaker $max_num_speaker
 done
done
fi

#grep -r Eval logs/run_ts_vad2_stage24_infer_cam++_zh_200k_transformer_rs_len8.log
# collar=0.0
#Eval for threshold 0.2 DER=27.81, miss=3.09, falarm=22.77, confusion=1.95
#Eval for threshold 0.3 DER=21.96, miss=5.34, falarm=14.19, confusion=2.43
#Eval for threshold 0.35 DER=20.37, miss=6.60, falarm=11.21, confusion=2.57
#Eval for threshold 0.4 DER=19.31, miss=7.89, falarm=8.82, confusion=2.60
#Eval for threshold 0.45 DER=18.81, miss=9.33, falarm=6.92, confusion=2.56
#Eval for threshold 0.5 DER=18.85, miss=10.94, falarm=5.45, confusion=2.46
#Eval for threshold 0.55 DER=19.22, miss=12.66, falarm=4.29, confusion=2.27
#Eval for threshold 0.6 DER=19.93, miss=14.57, falarm=3.36, confusion=2.00
#Eval for threshold 0.7 DER=22.24, miss=18.83, falarm=1.98, confusion=1.43
#Eval for threshold 0.8 DER=26.03, miss=24.13, falarm=1.07, confusion=0.83

# collar=0.25
#Eval for threshold 0.2 DER=17.31, miss=2.00, falarm=14.34, confusion=0.97
#Eval for threshold 0.3 DER=13.40, miss=3.52, falarm=8.57, confusion=1.31
#Eval for threshold 0.35 DER=12.44, miss=4.37, falarm=6.62, confusion=1.45
#Eval for threshold 0.4 DER=11.80, miss=5.27, falarm=5.04, confusion=1.50
#Eval for threshold 0.45 DER=11.51, miss=6.24, falarm=3.76, confusion=1.51
#Eval for threshold 0.5 DER=11.69, miss=7.36, falarm=2.86, confusion=1.48
#Eval for threshold 0.55 DER=12.12, miss=8.63, falarm=2.12, confusion=1.37
#Eval for threshold 0.6 DER=12.82, miss=10.04, falarm=1.60, confusion=1.18
#Eval for threshold 0.7 DER=14.89, miss=13.27, falarm=0.80, confusion=0.83
#Eval for threshold 0.8 DER=18.25, miss=17.44, falarm=0.36, confusion=0.45



if [ ${stage} -le 31 ] && [ ${stop_stage} -ge 31 ];then
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
    dataset_name="alimeeting_ami_aishell_4" # dataset name
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    # cam++ 200k
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    # cam++ en_zh
    #speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    spk_path=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/data/alimeeting_ami_aishell_4 # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/alimeeting_ami_aishell_4_ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_single_backend_2layer_conformer_multi_backend_conformer_rs_len8
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/data/alimeeting_ami_aishell_4" # oracle target audio , mix audio and labels path
   rs_len=8
   segment_shift=2
   single_backend_type="conformer"
   multi_backend_type="conformer"
   d_state=256
   num_transformer_layer=2
   max_num_speaker=7
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 17115 \
   ts_vad2/train_accelerate_ddp.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-updates 4000\
    --grad-clip true\
    --lr 2e-4\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --exp-dir $exp_dir\
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state\
    --max-num-speaker $max_num_speaker
fi


if [ ${stage} -le 32 ] && [ ${stop_stage} -ge 32 ];then
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/alimeeting_ami_aishell_4_ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_single_backend_2layer_conformer_multi_backend_conformer_rs_len8
 model_file=$exp_dir/best-valid-der.pt
 rs_len=8
 segment_shift=1
 single_backend_type="conformer"
 multi_backend_type="conformer"
 d_state=256
 num_transformer_layer=2
 max_num_speaker=7
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0

 dataset_name="alimeeting" # dataset name

 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.

 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_path=/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 #speech_encoder_type="w2v-bert2"
 #speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_collar${c}
   python3 ts_vad2/infer.py \
    --model-file $model_file\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name alimeeting_${name}.rttm\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $c\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state\
    --max-num-speaker $max_num_speaker
 done
done
fi

# grep -r Eval logs/run_ts_vad2_stage32_infer_cam++_zh_200k_conformer_rs_len8.log
# Eval of alimeeting,collar=0.0
#2025-02-10 09:12:05,513 (infer:257) INFO: currently, it will infer Eval set.
#Eval for threshold 0.2 DER=21.39, miss=2.32, falarm=17.82, confusion=1.24
#Eval for threshold 0.3 DER=16.72, miss=3.78, falarm=11.55, confusion=1.39
#Eval for threshold 0.35 DER=15.43, miss=4.58, falarm=9.44, confusion=1.41
#Eval for threshold 0.4 DER=14.61, miss=5.49, falarm=7.72, confusion=1.40
#Eval for threshold 0.45 DER=14.15, miss=6.49, falarm=6.27, confusion=1.39
#Eval for threshold 0.5 DER=13.97, miss=7.58, falarm=5.07, confusion=1.31
#Eval for threshold 0.55 DER=14.09, miss=8.80, falarm=4.07, confusion=1.21
#Eval for threshold 0.6 DER=14.47, miss=10.16, falarm=3.21, confusion=1.10
#Eval for threshold 0.7 DER=16.05, miss=13.22, falarm=1.99, confusion=0.83
#Eval for threshold 0.8 DER=19.25, miss=17.58, falarm=1.15, confusion=0.53

# Test of alimeeting, collar=0.0
#Eval for threshold 0.2 DER=20.89, miss=2.23, falarm=17.42, confusion=1.24
#Eval for threshold 0.3 DER=16.42, miss=3.72, falarm=11.24, confusion=1.46
#Eval for threshold 0.35 DER=15.21, miss=4.57, falarm=9.13, confusion=1.51
#Eval for threshold 0.4 DER=14.38, miss=5.51, falarm=7.33, confusion=1.54
#Eval for threshold 0.45 DER=13.95, miss=6.55, falarm=5.85, confusion=1.55
#Eval for threshold 0.5 DER=13.85, miss=7.71, falarm=4.64, confusion=1.50
#Eval for threshold 0.55 DER=14.17, miss=9.12, falarm=3.68, confusion=1.37
#Eval for threshold 0.6 DER=14.75, miss=10.60, falarm=2.92, confusion=1.23
#Eval for threshold 0.7 DER=16.65, miss=14.06, falarm=1.69, confusion=0.90
#Eval for threshold 0.8 DER=20.16, miss=18.70, falarm=0.85, confusion=0.60

# Eval of alimeeting, collar=0.25
#2025-02-10 09:24:17,571 (infer:257) INFO: currently, it will infer Eval set.
#Eval for threshold 0.2 DER=10.01, miss=0.83, falarm=8.78, confusion=0.41
#Eval for threshold 0.3 DER=7.01, miss=1.43, falarm=5.12, confusion=0.46
#Eval for threshold 0.35 DER=6.20, miss=1.81, falarm=3.90, confusion=0.50
#Eval for threshold 0.4 DER=5.72, miss=2.20, falarm=2.99, confusion=0.52
#Eval for threshold 0.45 DER=5.45, miss=2.64, falarm=2.29, confusion=0.52
#Eval for threshold 0.5 DER=5.41, miss=3.25, falarm=1.69, confusion=0.48
#Eval for threshold 0.55 DER=5.61, miss=3.90, falarm=1.25, confusion=0.45
#Eval for threshold 0.6 DER=6.01, miss=4.67, falarm=0.93, confusion=0.41
#Eval for threshold 0.7 DER=7.30, miss=6.42, falarm=0.57, confusion=0.31
#Eval for threshold 0.8 DER=9.68, miss=9.10, falarm=0.39, confusion=0.18

# Test of alimeeting, collar=0.25
#Eval for threshold 0.2 DER=9.92, miss=0.96, falarm=8.51, confusion=0.44
#Eval for threshold 0.3 DER=6.99, miss=1.67, falarm=4.78, confusion=0.54
#Eval for threshold 0.35 DER=6.28, miss=2.09, falarm=3.59, confusion=0.60
#Eval for threshold 0.4 DER=5.82, miss=2.54, falarm=2.64, confusion=0.64
#Eval for threshold 0.45 DER=5.62, miss=3.07, falarm=1.88, confusion=0.67
#Eval for threshold 0.5 DER=5.67, miss=3.69, falarm=1.33, confusion=0.65
#Eval for threshold 0.55 DER=6.01, miss=4.48, falarm=0.96, confusion=0.56
#Eval for threshold 0.6 DER=6.53, miss=5.34, falarm=0.70, confusion=0.49
#Eval for threshold 0.7 DER=7.99, miss=7.37, falarm=0.32, confusion=0.30
#Eval for threshold 0.8 DER=10.50, miss=10.19, falarm=0.12, confusion=0.19





if [ ${stage} -le 33 ] && [ ${stop_stage} -ge 33 ];then

 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/alimeeting_ami_aishell_4_ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_single_backend_2layer_conformer_multi_backend_conformer_rs_len8
 model_file=$exp_dir/best-valid-der.pt
 rs_len=8
 segment_shift=1
 single_backend_type="conformer"
 multi_backend_type="conformer"
 d_state=256
 num_transformer_layer=2
 max_num_speaker=7
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/datasets/aishell-4/data_processed/
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0

 dataset_name="aishell_4" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_path=/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 #speech_encoder_type="w2v-bert2"
 #speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
 # for loading speaker embedding
 spk_path=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/model_hub/ts_vad/spk_embed/aishell_4/SpeakerEmbedding/ # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/datasets/aishell-4/data_processed" # oracle target audio , mix audio and labels path
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_collar${c}
   python3 ts_vad2/infer.py \
    --model-file $model_file\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name ${name}/${name}.rttm\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $c\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state\
    --max-num-speaker $max_num_speaker
 done
done
fi
#grep -r Eval  logs/run_ts_vad2_stage33_infer_cam++_zh_200k_conformer_rs_len8.log

# test of aishell-4, collar=0.0
#Eval for threshold 0.2 DER=22.88, miss=1.54, falarm=19.38, confusion=1.96
#Eval for threshold 0.3 DER=14.47, miss=2.99, falarm=8.81, confusion=2.68
#Eval for threshold 0.35 DER=12.86, miss=4.02, falarm=5.98, confusion=2.85
#Eval for threshold 0.4 DER=12.26, miss=5.42, falarm=4.01, confusion=2.83
#Eval for threshold 0.45 DER=12.67, miss=7.19, falarm=2.88, confusion=2.61
#Eval for threshold 0.5 DER=13.77, miss=9.37, falarm=2.19, confusion=2.21
#Eval for threshold 0.55 DER=15.31, miss=11.72, falarm=1.74, confusion=1.84
#Eval for threshold 0.6 DER=17.31, miss=14.36, falarm=1.41, confusion=1.54
#Eval for threshold 0.7 DER=22.57, miss=20.62, falarm=0.93, confusion=1.02
#Eval for threshold 0.8 DER=30.31, miss=29.17, falarm=0.57, confusion=0.57

# test of aishell-4, collar=0.25
#Eval for threshold 0.2 DER=16.16, miss=0.69, falarm=14.09, confusion=1.39
#Eval for threshold 0.3 DER=8.89, miss=1.40, falarm=5.42, confusion=2.07
#Eval for threshold 0.35 DER=7.49, miss=2.02, falarm=3.17, confusion=2.30
#Eval for threshold 0.4 DER=6.97, miss=2.98, falarm=1.67, confusion=2.32
#Eval for threshold 0.45 DER=7.36, miss=4.35, falarm=0.88, confusion=2.13
#Eval for threshold 0.5 DER=8.39, miss=6.11, falarm=0.51, confusion=1.77
#Eval for threshold 0.55 DER=9.84, miss=8.07, falarm=0.32, confusion=1.44
#Eval for threshold 0.6 DER=11.71, miss=10.31, falarm=0.21, confusion=1.19
#Eval for threshold 0.7 DER=16.69, miss=15.85, falarm=0.10, confusion=0.74
#Eval for threshold 0.8 DER=24.30, miss=23.84, falarm=0.05, confusion=0.41


if [ ${stage} -le 34 ] && [ ${stop_stage} -ge 34 ];then
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/alimeeting_ami_aishell_4_ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_single_backend_2layer_conformer_multi_backend_conformer_rs_len8
 model_file=$exp_dir/best-valid-der.pt
 rs_len=8
 segment_shift=1
 single_backend_type="conformer"
 multi_backend_type="conformer"
 d_state=256
 num_transformer_layer=2
 max_num_speaker=7
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/datasets/ami/ami_version1.6.2/data_processed/data/ami
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0

 dataset_name="ami" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_path=/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 #speech_encoder_type="w2v-bert2"
 #speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
 # for loading speaker embedding
 spk_path=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/model_hub/ts_vad/spk_embed/ami/SpeakerEmbedding/ # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/datasets/ami/ami_version1.6.2/data_processed/data/ami" # oracle target audio , mix audio and labels path
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_collar${c}
   python3 ts_vad2/infer.py \
    --model-file $model_file\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name ${name}/${name}.rttm\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $c\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state\
    --max-num-speaker $max_num_speaker
 done
done
fi

# grep -r Eval  logs/run_ts_vad2_stage34_infer_cam++_zh_200k_conformer_rs_len8.log
# test of ami. collar=0.0
#Eval for threshold 0.2 DER=32.18, miss=2.83, falarm=27.18, confusion=2.17
#Eval for threshold 0.3 DER=23.82, miss=5.30, falarm=15.67, confusion=2.86
#Eval for threshold 0.35 DER=21.77, miss=6.81, falarm=11.88, confusion=3.08
#Eval for threshold 0.4 DER=20.61, miss=8.49, falarm=8.96, confusion=3.16
#Eval for threshold 0.45 DER=20.12, miss=10.32, falarm=6.67, confusion=3.12
#Eval for threshold 0.5 DER=20.15, miss=12.34, falarm=4.87, confusion=2.94
#Eval for threshold 0.55 DER=20.84, miss=14.54, falarm=3.59, confusion=2.71
#Eval for threshold 0.6 DER=22.08, miss=17.10, falarm=2.67, confusion=2.32
#Eval for threshold 0.7 DER=25.58, miss=22.70, falarm=1.39, confusion=1.50
#Eval for threshold 0.8 DER=30.96, miss=29.57, falarm=0.65, confusion=0.73

# test of ami. collar=0.25
#Eval for threshold 0.2 DER=21.64, miss=2.07, falarm=18.34, confusion=1.23
#Eval for threshold 0.3 DER=15.57, miss=3.71, falarm=10.09, confusion=1.76
#Eval for threshold 0.35 DER=14.21, miss=4.74, falarm=7.51, confusion=1.96
#Eval for threshold 0.4 DER=13.43, miss=5.93, falarm=5.44, confusion=2.06
#Eval for threshold 0.45 DER=13.15, miss=7.21, falarm=3.87, confusion=2.08
#Eval for threshold 0.5 DER=13.29, miss=8.63, falarm=2.63, confusion=2.03
#Eval for threshold 0.55 DER=13.94, miss=10.25, falarm=1.81, confusion=1.88
#Eval for threshold 0.6 DER=15.10, miss=12.22, falarm=1.28, confusion=1.59
#Eval for threshold 0.7 DER=18.22, miss=16.61, falarm=0.57, confusion=1.03
#Eval for threshold 0.8 DER=23.07, miss=22.35, falarm=0.23, confusion=0.50



if [ ${stage} -le 41 ] && [ ${stop_stage} -ge 41 ];then
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
    dataset_name="alimeeting_ami_aishell_4" # dataset name
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    # cam++ 200k
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    # cam++ en_zh
    #speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    spk_path=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/data/alimeeting_ami_aishell_4 # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/alimeeting_ami_aishell_4_ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_single_backend_2layer_conformer_multi_backend_transformer_rs_len8
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/data/alimeeting_ami_aishell_4" # oracle target audio , mix audio and labels path
   rs_len=8
   segment_shift=2
   single_backend_type="conformer"
   multi_backend_type="transformer"
   d_state=256
   num_transformer_layer=2
   max_num_speaker=7
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 17115 \
   ts_vad2/train_accelerate_ddp.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 14\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-updates 4000\
    --grad-clip true\
    --lr 2e-4\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --exp-dir $exp_dir\
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state\
    --max-num-speaker $max_num_speaker
fi


if [ ${stage} -le 42 ] && [ ${stop_stage} -ge 42 ];then
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/alimeeting_ami_aishell_4_ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_single_backend_2layer_conformer_multi_backend_transformer_rs_len8
 model_file=$exp_dir/best-valid-der.pt
 rs_len=8
 segment_shift=1
 single_backend_type="conformer"
 multi_backend_type="transformer"
 d_state=256
 num_transformer_layer=2
 max_num_speaker=7
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0

 dataset_name="alimeeting" # dataset name

 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.

 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_path=/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 #speech_encoder_type="w2v-bert2"
 #speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_collar${c}
   python3 ts_vad2/infer.py \
    --model-file $model_file\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name alimeeting_${name}.rttm\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $c\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state\
    --max-num-speaker $max_num_speaker
 done
done
fi
#grep -r Eval logs/run_ts_vad2_stage42_infer_cam++_zh_200k_conformer_rs_len8_1.log
# Eval of alimeeting, collar=0.0
#2025-02-14 09:19:36,637 (infer:257) INFO: currently, it will infer Eval set.
#Eval for threshold 0.2 DER=18.27, miss=2.62, falarm=14.53, confusion=1.12
#Eval for threshold 0.3 DER=15.07, miss=3.80, falarm=9.96, confusion=1.31
#Eval for threshold 0.35 DER=14.13, miss=4.44, falarm=8.36, confusion=1.34
#Eval for threshold 0.4 DER=13.52, miss=5.12, falarm=7.06, confusion=1.34
#Eval for threshold 0.45 DER=13.09, miss=5.85, falarm=5.94, confusion=1.30
#Eval for threshold 0.5 DER=12.85, miss=6.64, falarm=4.97, confusion=1.24
#Eval for threshold 0.55 DER=12.83, miss=7.47, falarm=4.18, confusion=1.19
#Eval for threshold 0.6 DER=13.06, miss=8.48, falarm=3.47, confusion=1.11
#Eval for threshold 0.7 DER=14.07, miss=10.76, falarm=2.41, confusion=0.90
#Eval for threshold 0.8 DER=16.23, miss=14.07, falarm=1.54, confusion=0.63

# Test of alimeeting, collar=0.0
#Eval for threshold 0.2 DER=18.38, miss=2.56, falarm=14.65, confusion=1.17
#Eval for threshold 0.3 DER=15.06, miss=3.79, falarm=9.92, confusion=1.35
#Eval for threshold 0.35 DER=14.15, miss=4.43, falarm=8.31, confusion=1.42
#Eval for threshold 0.4 DER=13.52, miss=5.17, falarm=6.89, confusion=1.46
#Eval for threshold 0.45 DER=13.20, miss=5.95, falarm=5.78, confusion=1.46
#Eval for threshold 0.5 DER=13.09, miss=6.82, falarm=4.83, confusion=1.45
#Eval for threshold 0.55 DER=13.16, miss=7.77, falarm=4.01, confusion=1.39
#Eval for threshold 0.6 DER=13.44, miss=8.87, falarm=3.28, confusion=1.29
#Eval for threshold 0.7 DER=14.69, miss=11.45, falarm=2.19, confusion=1.06
#Eval for threshold 0.8 DER=17.13, miss=15.02, falarm=1.31, confusion=0.80

# Eval of alimeeting, collar=0.25
##2025-02-14 09:30:23,404 (infer:257) INFO: currently, it will infer Eval set.
#Eval for threshold 0.2 DER=7.95, miss=0.92, falarm=6.66, confusion=0.38
#Eval for threshold 0.3 DER=5.94, miss=1.39, falarm=4.05, confusion=0.51
#Eval for threshold 0.35 DER=5.41, miss=1.65, falarm=3.22, confusion=0.54
#Eval for threshold 0.4 DER=5.12, miss=1.95, falarm=2.64, confusion=0.53
#Eval for threshold 0.45 DER=4.88, miss=2.28, falarm=2.08, confusion=0.53
#Eval for threshold 0.5 DER=4.80, miss=2.68, falarm=1.61, confusion=0.51
#Eval for threshold 0.55 DER=4.87, miss=3.08, falarm=1.31, confusion=0.48
#Eval for threshold 0.6 DER=5.10, miss=3.59, falarm=1.05, confusion=0.45
#Eval for threshold 0.7 DER=5.89, miss=4.82, falarm=0.71, confusion=0.36
#Eval for threshold 0.8 DER=7.43, miss=6.74, falarm=0.46, confusion=0.23

# Test of alimeeting, collar=0.25
#Eval for threshold 0.2 DER=8.46, miss=1.11, falarm=6.92, confusion=0.42
#Eval for threshold 0.3 DER=6.27, miss=1.69, falarm=4.06, confusion=0.53
#Eval for threshold 0.35 DER=5.72, miss=1.98, falarm=3.17, confusion=0.57
#Eval for threshold 0.4 DER=5.39, miss=2.34, falarm=2.46, confusion=0.60
#Eval for threshold 0.45 DER=5.25, miss=2.72, falarm=1.90, confusion=0.62
#Eval for threshold 0.5 DER=5.26, miss=3.14, falarm=1.48, confusion=0.64
#Eval for threshold 0.55 DER=5.36, miss=3.61, falarm=1.13, confusion=0.62
#Eval for threshold 0.6 DER=5.63, miss=4.21, falarm=0.85, confusion=0.57
#Eval for threshold 0.7 DER=6.61, miss=5.70, falarm=0.48, confusion=0.43
#Eval for threshold 0.8 DER=8.42, miss=7.88, falarm=0.24, confusion=0.30



if [ ${stage} -le 43 ] && [ ${stop_stage} -ge 43 ];then
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/alimeeting_ami_aishell_4_ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_single_backend_2layer_conformer_multi_backend_transformer_rs_len8
 model_file=$exp_dir/best-valid-der.pt
 rs_len=8
 segment_shift=1
 single_backend_type="conformer"
 multi_backend_type="transformer"
 d_state=256
 num_transformer_layer=2
 max_num_speaker=7
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/datasets/aishell-4/data_processed/
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0

 dataset_name="aishell_4" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_path=/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 #speech_encoder_type="w2v-bert2"
 #speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
 # for loading speaker embedding
 spk_path=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/model_hub/ts_vad/spk_embed/aishell_4/SpeakerEmbedding/ # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/datasets/aishell-4/data_processed" # oracle target audio , mix audio and labels path
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_collar${c}
   python3 ts_vad2/infer.py \
    --model-file $model_file\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name ${name}/${name}.rttm\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $c\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state\
    --max-num-speaker $max_num_speaker
 done
done
fi
# grep -r Eval logs/run_ts_vad2_stage43_infer_cam++_zh_200k_conformer_rs_len8_1.log
# test of aishell-4, collar=0.0
#Eval for threshold 0.2 DER=19.54, miss=1.20, falarm=16.71, confusion=1.64
#Eval for threshold 0.3 DER=13.34, miss=2.50, falarm=8.76, confusion=2.08
#Eval for threshold 0.35 DER=12.14, miss=3.49, falarm=6.52, confusion=2.13
#Eval for threshold 0.4 DER=11.50, miss=4.69, falarm=4.73, confusion=2.07
#Eval for threshold 0.45 DER=11.74, miss=6.06, falarm=3.76, confusion=1.91
#Eval for threshold 0.5 DER=12.44, miss=7.64, falarm=3.04, confusion=1.76
#Eval for threshold 0.55 DER=13.52, miss=9.47, falarm=2.47, confusion=1.58
#Eval for threshold 0.6 DER=15.00, miss=11.61, falarm=2.01, confusion=1.39
#Eval for threshold 0.7 DER=18.93, miss=16.56, falarm=1.37, confusion=1.00
#Eval for threshold 0.8 DER=24.91, miss=23.36, falarm=0.88, confusion=0.67

# test of aishell-4, collar=0.25
#Eval for threshold 0.2 DER=12.14, miss=0.55, falarm=10.51, confusion=1.08
#Eval for threshold 0.3 DER=7.42, miss=1.25, falarm=4.76, confusion=1.41
#Eval for threshold 0.35 DER=6.52, miss=1.89, falarm=3.17, confusion=1.46
#Eval for threshold 0.4 DER=6.04, miss=2.77, falarm=1.84, confusion=1.42
#Eval for threshold 0.45 DER=6.37, miss=3.79, falarm=1.27, confusion=1.31
#Eval for threshold 0.5 DER=7.09, miss=5.01, falarm=0.88, confusion=1.20
#Eval for threshold 0.55 DER=8.11, miss=6.44, falarm=0.61, confusion=1.07
#Eval for threshold 0.6 DER=9.53, miss=8.19, falarm=0.41, confusion=0.93
#Eval for threshold 0.7 DER=13.32, miss=12.43, falarm=0.22, confusion=0.67
#Eval for threshold 0.8 DER=19.10, miss=18.52, falarm=0.11, confusion=0.46


if [ ${stage} -le 44 ] && [ ${stop_stage} -ge 44 ];then
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/alimeeting_ami_aishell_4_ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_single_backend_2layer_conformer_multi_backend_transformer_rs_len8
 model_file=$exp_dir/best-valid-der.pt
 rs_len=8
 segment_shift=1
 single_backend_type="conformer"
 multi_backend_type="transformer"
 d_state=256
 num_transformer_layer=2
 max_num_speaker=7
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/datasets/ami/ami_version1.6.2/data_processed/data/ami
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0

 dataset_name="ami" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_path=/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 #speech_encoder_type="w2v-bert2"
 #speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
 # for loading speaker embedding
 spk_path=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/model_hub/ts_vad/spk_embed/ami/SpeakerEmbedding/ # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/datasets/ami/ami_version1.6.2/data_processed/data/ami" # oracle target audio , mix audio and labels path
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_collar${c}
   python3 ts_vad2/infer.py \
    --model-file $model_file\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name ${name}/${name}.rttm\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $c\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state\
    --max-num-speaker $max_num_speaker
 done
done
fi

if [ ${stage} -le 51 ] && [ ${stop_stage} -ge 51 ];then
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
    dataset_name="alimeeting_ami_aishell_4" # dataset name
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    # cam++ 200k
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    # cam++ en_zh
    #speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    spk_path=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/data/alimeeting_ami_aishell_4 # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/alimeeting_ami_aishell_4_ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_mamba2_multi_backend_transformer_d_state256_rs_len4
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/data/alimeeting_ami_aishell_4" # oracle target audio , mix audio and labels path
   rs_len=4
   segment_shift=2
   single_backend_type="mamba2"
   multi_backend_type="transformer"
   d_state=256
   num_transformer_layer=2
   max_num_speaker=7
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 17115 \
   ts_vad2/train_accelerate_ddp.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-updates 4000\
    --grad-clip true\
    --lr 1e-4\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --exp-dir $exp_dir\
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state\
    --max-num-speaker $max_num_speaker
fi

if [ ${stage} -le 52 ] && [ ${stop_stage} -ge 52 ];then
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/alimeeting_ami_aishell_4_ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_mamba2_multi_backend_transformer_d_state256_rs_len4
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 single_backend_type="mamba2"
 multi_backend_type="transformer"
 d_state=256
 num_transformer_layer=2
 max_num_speaker=7
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0

 dataset_name="alimeeting" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_path=/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_collar${c}
   python3 ts_vad2/infer.py \
    --model-file $model_file\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name alimeeting_${name}.rttm\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $c\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state\
    --max-num-speaker $max_num_speaker
 done
done
fi


if [ ${stage} -le 53 ] && [ ${stop_stage} -ge 53 ];then
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/alimeeting_ami_aishell_4_ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_mamba2_multi_backend_transformer_d_state256_rs_len4
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 single_backend_type="mamba2"
 multi_backend_type="transformer"
 d_state=256
 num_transformer_layer=2
 max_num_speaker=7
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/datasets/aishell-4/data_processed/
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0

 dataset_name="aishell_4" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_path=/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding
 spk_path=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/model_hub/ts_vad/spk_embed/aishell_4/SpeakerEmbedding/ # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/datasets/aishell-4/data_processed" # oracle target audio , mix audio and labels path
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_collar${c}
   python3 ts_vad2/infer.py \
    --model-file $model_file\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name ${name}/${name}.rttm\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $c\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state\
    --max-num-speaker $max_num_speaker
 done
done
fi

if [ ${stage} -le 54 ] && [ ${stop_stage} -ge 54 ];then
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/alimeeting_ami_aishell_4_ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_mamba2_multi_backend_transformer_d_state256_rs_len4
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 single_backend_type="mamba2"
 multi_backend_type="transformer"
 d_state=256
 num_transformer_layer=2
 max_num_speaker=7
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/datasets/ami/ami_version1.6.2/data_processed/data/ami
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0

 dataset_name="ami" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_path=/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding
 spk_path=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/model_hub/ts_vad/spk_embed/ami/SpeakerEmbedding/ # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/datasets/ami/ami_version1.6.2/data_processed/data/ami" # oracle target audio , mix audio and labels path
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_collar${c}
   python3 ts_vad2/infer.py \
    --model-file $model_file\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name ${name}/${name}.rttm\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $c\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state\
    --max-num-speaker $max_num_speaker
 done
done
fi


if [ ${stage} -le 61 ] && [ ${stop_stage} -ge 61 ];then
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
    dataset_name="alimeeting_ami_aishell_4" # dataset name
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    # cam++ 200k
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    # cam++ en_zh
    #speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    spk_path=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/data/alimeeting_ami_aishell_4 # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/alimeeting_ami_aishell_4_ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_single_backend_2layer_conformer_multi_backend_transformer_rs_len4
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/data/alimeeting_ami_aishell_4" # oracle target audio , mix audio and labels path
   rs_len=4
   segment_shift=2
   single_backend_type="conformer"
   multi_backend_type="transformer"
   d_state=256
   num_transformer_layer=2
   max_num_speaker=7
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 17215 \
   ts_vad2/train_accelerate_ddp.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-updates 4000\
    --grad-clip true\
    --lr 2e-4\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --exp-dir $exp_dir\
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state\
    --max-num-speaker $max_num_speaker
fi


if [ ${stage} -le 62 ] && [ ${stop_stage} -ge 62 ];then
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/alimeeting_ami_aishell_4_ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_single_backend_2layer_conformer_multi_backend_transformer_rs_len4
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 single_backend_type="conformer"
 multi_backend_type="transformer"
 d_state=256
 num_transformer_layer=2
 max_num_speaker=7
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0

 dataset_name="alimeeting" # dataset name

 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.

 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_path=/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 #speech_encoder_type="w2v-bert2"
 #speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_collar${c}
   python3 ts_vad2/infer.py \
    --model-file $model_file\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name alimeeting_${name}.rttm\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $c\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state\
    --max-num-speaker $max_num_speaker
 done
done
fi


if [ ${stage} -le 63 ] && [ ${stop_stage} -ge 63 ];then
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/alimeeting_ami_aishell_4_ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_single_backend_2layer_conformer_multi_backend_transformer_rs_len4
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 single_backend_type="conformer"
 multi_backend_type="transformer"
 d_state=256
 num_transformer_layer=2
 max_num_speaker=7
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/datasets/aishell-4/data_processed/
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0

 dataset_name="aishell_4" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_path=/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 #speech_encoder_type="w2v-bert2"
 #speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
 # for loading speaker embedding
 spk_path=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/model_hub/ts_vad/spk_embed/aishell_4/SpeakerEmbedding/ # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/datasets/aishell-4/data_processed" # oracle target audio , mix audio and labels path
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_collar${c}
   python3 ts_vad2/infer.py \
    --model-file $model_file\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name ${name}/${name}.rttm\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $c\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state\
    --max-num-speaker $max_num_speaker
 done
done
fi


if [ ${stage} -le 64 ] && [ ${stop_stage} -ge 64 ];then
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/alimeeting_ami_aishell_4_ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_single_backend_2layer_conformer_multi_backend_transformer_rs_len4
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 single_backend_type="conformer"
 multi_backend_type="transformer"
 d_state=256
 num_transformer_layer=2
 max_num_speaker=7
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/datasets/ami/ami_version1.6.2/data_processed/data/ami
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0

 dataset_name="ami" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_path=/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 #speech_encoder_type="w2v-bert2"
 #speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
 # for loading speaker embedding
 spk_path=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/model_hub/ts_vad/spk_embed/ami/SpeakerEmbedding/ # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/datasets/ami/ami_version1.6.2/data_processed/data/ami" # oracle target audio , mix audio and labels path
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_collar${c}
   python3 ts_vad2/infer.py \
    --model-file $model_file\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name ${name}/${name}.rttm\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $c\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state\
    --max-num-speaker $max_num_speaker
 done
done
fi





if [ ${stage} -le 71 ] && [ ${stop_stage} -ge 71 ];then
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
    dataset_name="alimeeting_ami_aishell_4" # dataset name
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    # cam++ 200k
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    # cam++ en_zh
    #speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    spk_path=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/data/alimeeting_ami_aishell_4 # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/alimeeting_ami_aishell_4_ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_single_backend_2layer_transformer_multi_backend_transformer_rs_len4
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/data/alimeeting_ami_aishell_4" # oracle target audio , mix audio and labels path
   rs_len=4
   segment_shift=2
   single_backend_type="transformer"
   multi_backend_type="transformer"
   d_state=256
   num_transformer_layer=2
   max_num_speaker=7
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 17315 \
   ts_vad2/train_accelerate_ddp.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-updates 4000\
    --grad-clip true\
    --lr 2e-4\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --exp-dir $exp_dir\
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state\
    --max-num-speaker $max_num_speaker
fi

if [ ${stage} -le 72 ] && [ ${stop_stage} -ge 72 ];then
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/alimeeting_ami_aishell_4_ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_single_backend_2layer_transformer_multi_backend_transformer_rs_len4
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 single_backend_type="transformer"
 multi_backend_type="transformer"
 d_state=256
 num_transformer_layer=2
 max_num_speaker=7
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0

 dataset_name="alimeeting" # dataset name

 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.

 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_path=/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 #speech_encoder_type="w2v-bert2"
 #speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_collar${c}
   python3 ts_vad2/infer.py \
    --model-file $model_file\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name alimeeting_${name}.rttm\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $c\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state\
    --max-num-speaker $max_num_speaker
 done
done
fi

if [ ${stage} -le 73 ] && [ ${stop_stage} -ge 73 ];then
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/alimeeting_ami_aishell_4_ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_single_backend_2layer_transformer_multi_backend_transformer_rs_len4
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 single_backend_type="transformer"
 multi_backend_type="transformer"
 d_state=256
 num_transformer_layer=2
 max_num_speaker=7
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/datasets/aishell-4/data_processed/
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0

 dataset_name="aishell_4" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_path=/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 #speech_encoder_type="w2v-bert2"
 #speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
 # for loading speaker embedding
 spk_path=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/model_hub/ts_vad/spk_embed/aishell_4/SpeakerEmbedding/ # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/datasets/aishell-4/data_processed" # oracle target audio , mix audio and labels path
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_collar${c}
   python3 ts_vad2/infer.py \
    --model-file $model_file\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name ${name}/${name}.rttm\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $c\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state\
    --max-num-speaker $max_num_speaker
 done
done
fi


if [ ${stage} -le 74 ] && [ ${stop_stage} -ge 74 ];then
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/alimeeting_ami_aishell_4_ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_single_backend_2layer_transformer_multi_backend_transformer_rs_len4
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 single_backend_type="transformer"
 multi_backend_type="transformer"
 d_state=256
 num_transformer_layer=2
 max_num_speaker=7
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/datasets/ami/ami_version1.6.2/data_processed/data/ami
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0

 dataset_name="ami" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_path=/data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 #speech_encoder_type="w2v-bert2"
 #speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
 # for loading speaker embedding
 spk_path=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/model_hub/ts_vad/spk_embed/ami/SpeakerEmbedding/ # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/datasets/ami/ami_version1.6.2/data_processed/data/ami" # oracle target audio , mix audio and labels path
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_collar${c}
   python3 ts_vad2/infer.py \
    --model-file $model_file\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name ${name}/${name}.rttm\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $c\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state\
    --max-num-speaker $max_num_speaker
 done
done
fi
