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
#Eval for threshold 0.3 DER=13.63, miss=4.25, falarm=7.91, confusion=1.48
#Eval for threshold 0.35 DER=12.88, miss=5.31, falarm=5.99, confusion=1.59
#Eval for threshold 0.4 DER=12.56, miss=6.46, falarm=4.50, confusion=1.60
#Eval for threshold 0.45 DER=12.60, miss=7.68, falarm=3.32, confusion=1.60
#Eval for threshold 0.5 DER=12.89, miss=8.98, falarm=2.36, confusion=1.55
#Eval for threshold 0.55 DER=13.55, miss=10.45, falarm=1.68, confusion=1.41
#Eval for threshold 0.6 DER=14.44, miss=12.05, falarm=1.12, confusion=1.27
#Eval for threshold 0.7 DER=17.09, miss=15.73, falarm=0.49, confusion=0.87
#Eval for threshold 0.8 DER=21.27, miss=20.56, falarm=0.22, confusion=0.50


# lr=(1e-4)/2
if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ];then
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
    exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/alimeeting_ami_aishell_4_ts_vad2_two_gpus_freeze_with_musan_rirs_w2v-bert2_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_mamba2_multi_backend_transformer_d_state256_rs_len10
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
    --num-epochs 20\
    --start-epoch 1\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-updates 4000\
    --grad-clip true\
    --lr 5e-5\
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



if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ];then
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/alimeeting_ami_aishell_4_ts_vad2_two_gpus_freeze_with_musan_rirs_w2v-bert2_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_mamba2_multi_backend_transformer_d_state256_rs_len10
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
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/alimeeting_ami_aishell_4_ts_vad2_two_gpus_freeze_with_musan_rirs_w2v-bert2_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_mamba2_multi_backend_transformer_d_state256_rs_len10
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
 exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/alimeeting_ami_aishell_4_ts_vad2_two_gpus_freeze_with_musan_rirs_w2v-bert2_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2layer_mamba2_multi_backend_transformer_d_state256_rs_len10
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

# lr=(1e-4)/2
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
    exp_dir=/mntcephfs/data/haizhouli/Lab-projects/maduo/huawei_diarization/exp/speaker_diarization/ts_vad2/alimeeting_ami_aishell_4_ts_vad2_two_gpus_freeze_with_musan_rirs_w2v-bert2_cam++_200k_zh_cn_epoch40_front_fix_seed_lr5e5_single_backend_2layer_mamba2_multi_backend_transformer_d_state256_rs_len10
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
    --lr 5e-5\
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

