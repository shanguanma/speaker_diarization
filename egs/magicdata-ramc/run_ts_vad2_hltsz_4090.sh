#!/usr/bin/env bash
stage=0
stop_stage=1000

. utils/parse_options.sh
. path_for_speaker_diarization_hltsz_4090.sh


if [ ${stage} -le 128 ] && [ ${stop_stage} -ge 128 ];then
    
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is wav-bert2.0 (only using position embedding and first 6 layers conformer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this w2v-bert2.0 is trained Languages: 143+ , Size: 4.5M hours (it is from this paper https://arxiv.org/pdf/2312.05187)
    # checkpoint is from https://huggingface.co/facebook/w2v-bert-2.0/tree/main
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/data2/shared_datasets/speechdata/14_musan
    rir_path=/data1/home/maduo/datasets/RIRS_NOISES
    dataset_name="magicdata-ramc" # dataset name
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/data1/home/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    spk_path=/data1/home/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/data1/home/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_single_backend_2layer_transformer_multi_backend_transformer_rs_len6_shift0.8
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/data1/home/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format" # oracle target audio , mix audio and labels path
   rs_len=6
   segment_shift=0.8
   single_backend_type="transformer"
   multi_backend_type="transformer"
   num_transformer_layer=2
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 18815 \
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

fi

if [ ${stage} -le 129 ] && [ ${stop_stage} -ge 129 ];then
 exp_dir=/data1/home/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_single_backend_2layer_transformer_multi_backend_transformer_rs_len6_shift0.8
 model_file=$exp_dir/best-valid-der.pt
 rs_len=6
 segment_shift=0.8
 single_backend_type="transformer"
 multi_backend_type="transformer"
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="dev test cssd_testset"
 #infer_sets="Test"
 rttm_dir=/data1/home/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0

 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/data1/home/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 spk_path=/data1/home/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

 #data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
 data_dir="/data1/home/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format"
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_${name}_collar${c}
  python3 ts_vad2/infer.py \
    --model-file $model_file\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name ${name}/rttm_debug_nog0\
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
    --num-transformer-layer $num_transformer_layer
 done
done
fi

# grep -r Eval logs/run_ts_vad2_hltsz_4090_stage128-130.log
# dev of magicdata-ramc, collar=0.0
# Eval for threshold 0.2 DER=15.38, miss=0.30, falarm=12.30, confusion=2.79
#Eval for threshold 0.3 DER=12.90, miss=0.54, falarm=8.94, confusion=3.42
#Eval for threshold 0.35 DER=12.27, miss=0.70, falarm=8.01, confusion=3.57
#Eval for threshold 0.4 DER=11.84, miss=0.89, falarm=7.30, confusion=3.65
#Eval for threshold 0.45 DER=11.52, miss=1.11, falarm=6.72, confusion=3.68
#Eval for threshold 0.5 DER=11.34, miss=1.39, falarm=6.26, confusion=3.68
#Eval for threshold 0.55 DER=11.24, miss=1.80, falarm=5.84, confusion=3.60
#Eval for threshold 0.6 DER=11.23, miss=2.29, falarm=5.44, confusion=3.50
#Eval for threshold 0.7 DER=11.59, miss=3.72, falarm=4.70, confusion=3.16
#Eval for threshold 0.8 DER=13.28, miss=6.97, falarm=3.84, confusion=2.47
#Eval for threshold 0.9 DER=18.47, miss=14.45, falarm=2.71, confusion=1.31


# test of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=17.93, miss=0.37, falarm=15.90, confusion=1.66
#Eval for threshold 0.3 DER=15.53, miss=0.73, falarm=12.77, confusion=2.04
#Eval for threshold 0.35 DER=14.79, miss=0.97, falarm=11.67, confusion=2.16
#Eval for threshold 0.4 DER=14.24, miss=1.27, falarm=10.72, confusion=2.24
#Eval for threshold 0.45 DER=13.75, miss=1.66, falarm=9.79, confusion=2.31
#Eval for threshold 0.5 DER=12.11, miss=2.22, falarm=6.62, confusion=3.26
#Eval for threshold 0.55 DER=13.24, miss=5.16, falarm=5.96, confusion=2.12
#Eval for threshold 0.6 DER=13.47, miss=5.99, falarm=5.52, confusion=1.96
#Eval for threshold 0.7 DER=14.32, miss=8.08, falarm=4.64, confusion=1.60
#Eval for threshold 0.8 DER=16.38, miss=11.49, falarm=3.74, confusion=1.15
#Eval for threshold 0.9 DER=21.79, miss=18.54, falarm=2.63, confusion=0.62

# cssd_testset of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=29.13, miss=3.23, falarm=24.08, confusion=1.81
#Eval for threshold 0.3 DER=24.44, miss=4.49, falarm=17.16, confusion=2.79
#Eval for threshold 0.35 DER=22.81, miss=5.21, falarm=14.31, confusion=3.29
#Eval for threshold 0.4 DER=21.57, miss=6.03, falarm=11.80, confusion=3.74
#Eval for threshold 0.45 DER=20.78, miss=7.10, falarm=9.70, confusion=3.99
#Eval for threshold 0.5 DER=20.72, miss=8.71, falarm=8.19, confusion=3.83
#Eval for threshold 0.55 DER=21.42, miss=10.89, falarm=7.33, confusion=3.20
#Eval for threshold 0.6 DER=22.39, miss=13.23, falarm=6.57, confusion=2.59
#Eval for threshold 0.7 DER=24.96, miss=18.28, falarm=5.10, confusion=1.58
#Eval for threshold 0.8 DER=29.21, miss=24.79, falarm=3.60, confusion=0.83
#Eval for threshold 0.9 DER=37.56, miss=35.23, falarm=2.00, confusion=0.33
# dev of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=7.37, miss=0.09, falarm=4.76, confusion=2.52
#Eval for threshold 0.3 DER=5.70, miss=0.17, falarm=2.42, confusion=3.11
#Eval for threshold 0.35 DER=5.40, miss=0.22, falarm=1.94, confusion=3.24
#Eval for threshold 0.4 DER=5.24, miss=0.28, falarm=1.65, confusion=3.31
#Eval for threshold 0.45 DER=5.13, miss=0.35, falarm=1.45, confusion=3.33
#Eval for threshold 0.5 DER=5.11, miss=0.46, falarm=1.32, confusion=3.34
#Eval for threshold 0.55 DER=5.14, miss=0.62, falarm=1.22, confusion=3.30
#Eval for threshold 0.6 DER=5.22, miss=0.81, falarm=1.15, confusion=3.26
#Eval for threshold 0.7 DER=5.61, miss=1.54, falarm=1.04, confusion=3.03
#Eval for threshold 0.8 DER=7.25, miss=3.92, falarm=0.92, confusion=2.42
#Eval for threshold 0.9 DER=12.38, miss=10.35, falarm=0.74, confusion=1.29

# test of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=9.66, miss=0.12, falarm=8.30, confusion=1.25
#Eval for threshold 0.3 DER=8.16, miss=0.26, falarm=6.39, confusion=1.50
#Eval for threshold 0.35 DER=7.76, miss=0.37, falarm=5.81, confusion=1.58
#Eval for threshold 0.4 DER=7.51, miss=0.51, falarm=5.38, confusion=1.62
#Eval for threshold 0.45 DER=7.32, miss=0.66, falarm=4.99, confusion=1.67
#Eval for threshold 0.5 DER=5.92, miss=0.94, falarm=2.31, confusion=2.67
#Eval for threshold 0.55 DER=7.26, miss=3.58, falarm=2.11, confusion=1.56
#Eval for threshold 0.6 DER=7.52, miss=4.05, falarm=2.00, confusion=1.48
#Eval for threshold 0.7 DER=8.29, miss=5.28, falarm=1.73, confusion=1.27
#Eval for threshold 0.8 DER=10.13, miss=7.70, falarm=1.45, confusion=0.97
#Eval for threshold 0.9 DER=15.15, miss=13.51, falarm=1.09, confusion=0.56

# cssd_testset of magicdata-ramc,collar=0.25
#Eval for threshold 0.2 DER=11.03, miss=1.00, falarm=9.37, confusion=0.65
#Eval for threshold 0.3 DER=8.05, miss=1.45, falarm=5.37, confusion=1.23
#Eval for threshold 0.35 DER=7.09, miss=1.71, falarm=3.77, confusion=1.61
#Eval for threshold 0.4 DER=6.37, miss=2.00, falarm=2.38, confusion=1.98
#Eval for threshold 0.45 DER=5.91, miss=2.45, falarm=1.18, confusion=2.28
#Eval for threshold 0.5 DER=6.05, miss=3.31, falarm=0.48, confusion=2.27
#Eval for threshold 0.55 DER=6.91, miss=4.74, falarm=0.36, confusion=1.81
#Eval for threshold 0.6 DER=8.03, miss=6.37, falarm=0.30, confusion=1.36
#Eval for threshold 0.7 DER=10.68, miss=9.73, falarm=0.20, confusion=0.76
#Eval for threshold 0.8 DER=15.03, miss=14.56, falarm=0.10, confusion=0.37
#Eval for threshold 0.9 DER=23.46, miss=23.25, falarm=0.04, confusion=0.17

if [ ${stage} -le 130 ] && [ ${stop_stage} -ge 130 ];then
   echo "compute CDER for magicdata-ramc"
   threshold="0.2 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.7 0.8 0.9" # total 11
   predict_rttm_dir=/data1/home/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr2e4_single_backend_2layer_transformer_multi_backend_transformer_rs_len6_shift0.8
   oracle_rttm_dir=/data1/home/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
   infer_sets="dev test cssd_testset"
   c=0.0
   dataset_name="magicdata-ramc"
   for name in $infer_sets;do
    for thres in $threshold;do
     echo "currently, compute $name set in $thres threshold mode"
     python3 cder/score.py -s $predict_rttm_dir/${dataset_name}_${name}_collar${c}/${name}/res_rttm_${thres}  -r $oracle_rttm_dir/$name/rttm_debug_nog0
    done
   done
fi
#  grep -r Avg logs/run_ts_vad2_hltsz_4090_stage128-130.log
# dev of magicdata-ramc
#Avg CDER : 0.714
#Avg CDER : 0.427
#Avg CDER : 0.343
#Avg CDER : 0.280
#Avg CDER : 0.219
#Avg CDER : 0.181
#Avg CDER : 0.139
#Avg CDER : 0.121
#Avg CDER : 0.104
#Avg CDER : 0.103
#Avg CDER : 0.099
# test of magicdata-ramc
#Avg CDER : 0.479
#Avg CDER : 0.361
#Avg CDER : 0.291
#Avg CDER : 0.262
#Avg CDER : 0.212
#Avg CDER : 0.131
#Avg CDER : 0.126
#Avg CDER : Error!
#Avg CDER : Error!
#Avg CDER : Error!
#Avg CDER : Error!
# cssd_testset of magicdata-ramc
#Avg CDER : 0.290
#Avg CDER : 0.224
#Avg CDER : 0.195
#Avg CDER : 0.166
#Avg CDER : 0.141
#Avg CDER : 0.115
#Avg CDER : 0.104
#Avg CDER : 0.099
#Avg CDER : 0.092
#Avg CDER : 0.086
#Avg CDER : 0.143




if [ ${stage} -le 131 ] && [ ${stop_stage} -ge 131 ];then

    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is wav-bert2.0 (only using position embedding and first 6 layers conformer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this w2v-bert2.0 is trained Languages: 143+ , Size: 4.5M hours (it is from this paper https://arxiv.org/pdf/2312.05187)
    # checkpoint is from https://huggingface.co/facebook/w2v-bert-2.0/tree/main
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/data2/shared_datasets/speechdata/14_musan
    rir_path=/data1/home/maduo/datasets/RIRS_NOISES
    dataset_name="magicdata-ramc" # dataset name
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/data1/home/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    spk_path=/data1/home/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/data1/home/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2mamba2_multi_backend_transformer_rs_len6_shift0.8
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/data1/home/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format" # oracle target audio , mix audio and labels path
   rs_len=6
   segment_shift=0.8
   single_backend_type="mamba2"
   d_state=128
   multi_backend_type="transformer"
   num_transformer_layer=2
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 18915 \
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
    --d-state $d_state

fi

if [ ${stage} -le 132 ] && [ ${stop_stage} -ge 132 ];then
 exp_dir=/share/workspace/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2mamba2_multi_backend_transformer_rs_len6_shift0.8
 model_file=$exp_dir/best-valid-der.pt
 rs_len=6
 segment_shift=0.8
 single_backend_type="mamba2"
 d_state=128
 multi_backend_type="transformer"
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="dev test cssd_testset"
 #infer_sets="Test"
 rttm_dir=/share/workspace/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0

 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/share/workspace/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 spk_path=/share/workspace/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

 #data_dir="/share/workspace/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
 data_dir="/share/workspace/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format"
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_${name}_collar${c}_again
  python3 ts_vad2/infer.py \
    --model-file $model_file\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name ${name}/rttm_debug_nog0\
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
    --d-state $d_state
 done
done
fi
#grep -r Eval logs/run_ts_vad2_hltsz_4090_stage132-133.log
# dev of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=14.72, miss=0.25, falarm=11.13, confusion=3.34
#Eval for threshold 0.3 DER=13.10, miss=0.44, falarm=8.99, confusion=3.67
#Eval for threshold 0.35 DER=12.55, miss=0.60, falarm=8.22, confusion=3.73
#Eval for threshold 0.4 DER=12.12, miss=0.82, falarm=7.54, confusion=3.76
#Eval for threshold 0.45 DER=11.78, miss=1.08, falarm=6.93, confusion=3.77
#Eval for threshold 0.5 DER=11.58, miss=1.45, falarm=6.41, confusion=3.71 as report
#Eval for threshold 0.55 DER=11.44, miss=1.88, falarm=5.95, confusion=3.61
#Eval for threshold 0.6 DER=11.43, miss=2.41, falarm=5.51, confusion=3.50
#Eval for threshold 0.7 DER=11.67, miss=3.77, falarm=4.62, confusion=3.29
#Eval for threshold 0.8 DER=12.57, miss=5.96, falarm=3.66, confusion=2.95
#Eval for threshold 0.9 DER=16.01, miss=11.11, falarm=2.56, confusion=2.34
# test of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=17.36, miss=0.32, falarm=15.23, confusion=1.82
#Eval for threshold 0.3 DER=15.37, miss=0.68, falarm=12.51, confusion=2.18
#Eval for threshold 0.35 DER=14.69, miss=0.95, falarm=11.45, confusion=2.29
#Eval for threshold 0.4 DER=14.19, miss=1.30, falarm=10.53, confusion=2.36
#Eval for threshold 0.45 DER=13.17, miss=1.72, falarm=8.80, confusion=2.65
#Eval for threshold 0.5 DER=12.25, miss=2.34, falarm=6.64, confusion=3.27 as report
#Eval for threshold 0.55 DER=12.74, miss=4.26, falarm=5.93, confusion=2.55
#Eval for threshold 0.6 DER=13.49, miss=6.08, falarm=5.40, confusion=2.00
#Eval for threshold 0.7 DER=14.28, miss=8.16, falarm=4.46, confusion=1.66
#Eval for threshold 0.8 DER=16.00, miss=11.19, falarm=3.50, confusion=1.31
#Eval for threshold 0.9 DER=20.30, miss=17.06, falarm=2.41, confusion=0.84
# cssd_testset of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=27.68, miss=3.52, falarm=22.06, confusion=2.10
#Eval for threshold 0.3 DER=23.66, miss=4.90, falarm=15.80, confusion=2.96
#Eval for threshold 0.35 DER=22.39, miss=5.74, falarm=13.35, confusion=3.29
#Eval for threshold 0.4 DER=21.51, miss=6.77, falarm=11.21, confusion=3.53 
#Eval for threshold 0.45 DER=21.08, miss=8.02, falarm=9.48, confusion=3.58 as report
#Eval for threshold 0.5 DER=21.21, miss=9.71, falarm=8.19, confusion=3.31
#Eval for threshold 0.55 DER=21.93, miss=11.80, falarm=7.36, confusion=2.77
#Eval for threshold 0.6 DER=22.81, miss=13.99, falarm=6.56, confusion=2.25
#Eval for threshold 0.7 DER=25.10, miss=18.73, falarm=4.95, confusion=1.42
#Eval for threshold 0.8 DER=29.04, miss=24.99, falarm=3.31, confusion=0.74
#Eval for threshold 0.9 DER=38.04, miss=36.02, falarm=1.73, confusion=0.30

# dev of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=6.35, miss=0.05, falarm=3.23, confusion=3.07
#Eval for threshold 0.3 DER=5.56, miss=0.10, falarm=2.15, confusion=3.30
#Eval for threshold 0.35 DER=5.33, miss=0.15, falarm=1.85, confusion=3.34
#Eval for threshold 0.4 DER=5.20, miss=0.24, falarm=1.61, confusion=3.35
#Eval for threshold 0.45 DER=5.11, miss=0.32, falarm=1.44, confusion=3.35
#Eval for threshold 0.5 DER=5.07, miss=0.44, falarm=1.30, confusion=3.33 as report
#Eval for threshold 0.55 DER=5.10, miss=0.61, falarm=1.21, confusion=3.28
#Eval for threshold 0.6 DER=5.18, miss=0.81, falarm=1.13, confusion=3.24
#Eval for threshold 0.7 DER=5.57, miss=1.41, falarm=1.01, confusion=3.15
#Eval for threshold 0.8 DER=6.39, miss=2.57, falarm=0.88, confusion=2.94
#Eval for threshold 0.9 DER=9.45, miss=6.28, falarm=0.74, confusion=2.44

# test of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=8.84, miss=0.07, falarm=7.41, confusion=1.36
#Eval for threshold 0.3 DER=7.81, miss=0.21, falarm=6.04, confusion=1.56
#Eval for threshold 0.35 DER=7.52, miss=0.33, falarm=5.57, confusion=1.62
#Eval for threshold 0.4 DER=7.34, miss=0.48, falarm=5.19, confusion=1.66
#Eval for threshold 0.45 DER=6.61, miss=0.65, falarm=4.02, confusion=1.94
#Eval for threshold 0.5 DER=5.93, miss=0.95, falarm=2.32, confusion=2.65 as report
#Eval for threshold 0.55 DER=6.54, miss=2.52, falarm=2.02, confusion=2.00
#Eval for threshold 0.6 DER=7.40, miss=4.02, falarm=1.88, confusion=1.51
#Eval for threshold 0.7 DER=8.16, miss=5.19, falarm=1.65, confusion=1.33
#Eval for threshold 0.8 DER=9.62, miss=7.09, falarm=1.40, confusion=1.13
#Eval for threshold 0.9 DER=13.28, miss=11.45, falarm=1.05, confusion=0.78

# cssd_testset of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=9.34, miss=1.07, falarm=7.61, confusion=0.65
#Eval for threshold 0.3 DER=6.93, miss=1.57, falarm=4.13, confusion=1.22
#Eval for threshold 0.35 DER=6.28, miss=1.91, falarm=2.86, confusion=1.51
#Eval for threshold 0.4 DER=5.80, miss=2.31, falarm=1.73, confusion=1.76
#Eval for threshold 0.45 DER=5.65, miss=2.90, falarm=0.86, confusion=1.89 as report
#Eval for threshold 0.5 DER=6.01, miss=3.82, falarm=0.37, confusion=1.82
#Eval for threshold 0.55 DER=6.88, miss=5.17, falarm=0.28, confusion=1.43
#Eval for threshold 0.6 DER=7.91, miss=6.60, falarm=0.23, confusion=1.09
#Eval for threshold 0.7 DER=10.49, miss=9.73, falarm=0.14, confusion=0.62
#Eval for threshold 0.8 DER=14.48, miss=14.13, falarm=0.06, confusion=0.29
#Eval for threshold 0.9 DER=23.37, miss=23.20, falarm=0.03, confusion=0.14
if [ ${stage} -le 133 ] && [ ${stop_stage} -ge 133 ];then
   echo "compute CDER for magicdata-ramc"
   threshold="0.2 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.7 0.8 0.9" # total 11
   predict_rttm_dir=/data1/home/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2mamba2_multi_backend_transformer_rs_len6_shift0.8/
   oracle_rttm_dir=/data1/home/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
   infer_sets="dev test cssd_testset"
   c=0.0
   dataset_name="magicdata-ramc"
   for name in $infer_sets;do
    for thres in $threshold;do
     echo "currently, compute $name set in $thres threshold mode"
     python3 cder/score.py -s $predict_rttm_dir/${dataset_name}_${name}_collar${c}/${name}/res_rttm_${thres}  -r $oracle_rttm_dir/$name/rttm_debug_nog0
    done
   done
fi

#grep -r Avg  logs/run_ts_vad2_hltsz_4090_stage132-133.log
# dev of magicdata-ramc
#Avg CDER : 0.453
#Avg CDER : 0.309
#Avg CDER : 0.253
#Avg CDER : 0.207
#Avg CDER : 0.174
#Avg CDER : 0.141
#Avg CDER : 0.123
#Avg CDER : 0.113
#Avg CDER : 0.105
#Avg CDER : 0.096
#Avg CDER : 0.092
#test of magicdata-ramc
#Avg CDER : 0.351
#Avg CDER : 0.288
#Avg CDER : 0.246
#Avg CDER : 0.225
#Avg CDER : 0.207
#Avg CDER : 0.141
#Avg CDER : 0.107
#Avg CDER : 0.123
#Avg CDER : Error!
#Avg CDER : Error!
#Avg CDER : Error!
#cssd_testset of magicdata-ramc
#Avg CDER : 0.249
#Avg CDER : 0.189
#Avg CDER : 0.166
#Avg CDER : 0.143
#Avg CDER : 0.123
#Avg CDER : 0.105
#Avg CDER : 0.098
#Avg CDER : 0.093
#Avg CDER : 0.087
#Avg CDER : 0.094
#Avg CDER : 0.136i


if [ ${stage} -le 134 ] && [ ${stop_stage} -ge 134 ];then

    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is wav-bert2.0 (only using position embedding and first 6 layers conformer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this w2v-bert2.0 is trained Languages: 143+ , Size: 4.5M hours (it is from this paper https://arxiv.org/pdf/2312.05187)
    # checkpoint is from https://huggingface.co/facebook/w2v-bert-2.0/tree/main
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/share/workspace/shared_datasets/speechdata/14_musan
    rir_path=/share/workspace/shared_datasets/speechdata/21_RIRS_NOISES/RIRS_NOISES
    dataset_name="magicdata-ramc" # dataset name
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/share/workspace/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    spk_path=/share/workspace/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/share/workspace/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2mamba2_multi_backend_transformer_rs_len8_shift0.8
    mkdir -p $exp_dir
    #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/share/workspace/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format" # oracle target audio , mix audio and labels path
   rs_len=8
   segment_shift=0.8
   single_backend_type="mamba2"
   d_state=128
   multi_backend_type="transformer"
   num_transformer_layer=2
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 18915 \
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
    --d-state $d_state

fi

if [ ${stage} -le 135 ] && [ ${stop_stage} -ge 135 ];then
 exp_dir=/share/workspace/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2mamba2_multi_backend_transformer_rs_len8_shift0.8
 model_file=$exp_dir/best-valid-der.pt
 rs_len=8
 segment_shift=0.8
 single_backend_type="mamba2"
 d_state=128
 multi_backend_type="transformer"
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="dev test cssd_testset"
 #infer_sets="Test"
 rttm_dir=/share/workspace/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0

 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/share/workspace/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 spk_path=/share/workspace/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

 #data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
 data_dir="/share/workspace/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format"
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_${name}_collar${c}
  python3 ts_vad2/infer.py \
    --model-file $model_file\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name ${name}/rttm_debug_nog0\
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
    --d-state $d_state
 done
done
fi

#grep -r Eval logs/run_ts_vad2_hltsz_4090_stage134-136.log
# dev of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=14.51, miss=0.27, falarm=10.84, confusion=3.40
#Eval for threshold 0.3 DER=13.02, miss=0.47, falarm=8.88, confusion=3.66
#Eval for threshold 0.35 DER=12.52, miss=0.64, falarm=8.17, confusion=3.71
#Eval for threshold 0.4 DER=12.13, miss=0.84, falarm=7.54, confusion=3.75
#Eval for threshold 0.45 DER=11.82, miss=1.13, falarm=6.95, confusion=3.74
#Eval for threshold 0.5 DER=11.63, miss=1.46, falarm=6.45, confusion=3.71
#Eval for threshold 0.55 DER=11.52, miss=1.87, falarm=6.02, confusion=3.62
#Eval for threshold 0.6 DER=11.46, miss=2.33, falarm=5.61, confusion=3.53
#Eval for threshold 0.7 DER=11.70, miss=3.65, falarm=4.78, confusion=3.28
#Eval for threshold 0.8 DER=12.57, miss=5.77, falarm=3.85, confusion=2.95
#Eval for threshold 0.9 DER=15.40, miss=10.42, falarm=2.70, confusion=2.27

# test of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=17.02, miss=0.39, falarm=14.73, confusion=1.90
#Eval for threshold 0.3 DER=15.23, miss=0.82, falarm=12.21, confusion=2.20
#Eval for threshold 0.35 DER=14.64, miss=1.10, falarm=11.26, confusion=2.28
#Eval for threshold 0.4 DER=14.14, miss=1.46, falarm=10.34, confusion=2.33
#Eval for threshold 0.45 DER=13.44, miss=1.91, falarm=9.05, confusion=2.48
#Eval for threshold 0.5 DER=12.25, miss=2.58, falarm=6.48, confusion=3.19
#Eval for threshold 0.55 DER=13.08, miss=4.96, falarm=5.81, confusion=2.31
#Eval for threshold 0.6 DER=13.63, miss=6.33, falarm=5.35, confusion=1.96
#Eval for threshold 0.7 DER=14.45, miss=8.37, falarm=4.45, confusion=1.64
#Eval for threshold 0.8 DER=16.18, miss=11.38, falarm=3.51, confusion=1.29
#Eval for threshold 0.9 DER=20.81, miss=17.54, falarm=2.42, confusion=0.85

# cssd_testset of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=27.97, miss=3.80, falarm=21.71, confusion=2.45
#Eval for threshold 0.3 DER=23.99, miss=5.29, falarm=15.42, confusion=3.28
#Eval for threshold 0.35 DER=22.76, miss=6.19, falarm=13.00, confusion=3.58
#Eval for threshold 0.4 DER=22.01, miss=7.21, falarm=11.02, confusion=3.78
#Eval for threshold 0.45 DER=21.76, miss=8.59, falarm=9.40, confusion=3.78
#Eval for threshold 0.5 DER=22.06, miss=10.35, falarm=8.18, confusion=3.54
#Eval for threshold 0.55 DER=22.83, miss=12.54, falarm=7.33, confusion=2.96
#Eval for threshold 0.6 DER=23.80, miss=14.83, falarm=6.57, confusion=2.40
#Eval for threshold 0.7 DER=26.41, miss=19.87, falarm=5.05, confusion=1.49
#Eval for threshold 0.8 DER=30.71, miss=26.50, falarm=3.42, confusion=0.80
#Eval for threshold 0.9 DER=40.17, miss=38.18, falarm=1.70, confusion=0.29

# dev of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=6.22, miss=0.07, falarm=3.06, confusion=3.09
#Eval for threshold 0.3 DER=5.52, miss=0.12, falarm=2.12, confusion=3.27
#Eval for threshold 0.35 DER=5.32, miss=0.18, falarm=1.83, confusion=3.31
#Eval for threshold 0.4 DER=5.18, miss=0.25, falarm=1.61, confusion=3.33
#Eval for threshold 0.45 DER=5.10, miss=0.35, falarm=1.43, confusion=3.33
#Eval for threshold 0.5 DER=5.10, miss=0.48, falarm=1.30, confusion=3.32
#Eval for threshold 0.55 DER=5.15, miss=0.64, falarm=1.22, confusion=3.28
#Eval for threshold 0.6 DER=5.22, miss=0.82, falarm=1.16, confusion=3.24
#Eval for threshold 0.7 DER=5.57, miss=1.41, falarm=1.03, confusion=3.13
#Eval for threshold 0.8 DER=6.38, miss=2.55, falarm=0.89, confusion=2.94
#Eval for threshold 0.9 DER=8.88, miss=5.79, falarm=0.74, confusion=2.35

# test of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=8.60, miss=0.12, falarm=7.06, confusion=1.42
#Eval for threshold 0.3 DER=7.74, miss=0.31, falarm=5.86, confusion=1.56
#Eval for threshold 0.35 DER=7.50, miss=0.43, falarm=5.46, confusion=1.61
#Eval for threshold 0.4 DER=7.32, miss=0.59, falarm=5.10, confusion=1.64
#Eval for threshold 0.45 DER=6.91, miss=0.78, falarm=4.35, confusion=1.78
#Eval for threshold 0.5 DER=5.94, miss=1.12, falarm=2.23, confusion=2.59
#Eval for threshold 0.55 DER=6.91, miss=3.19, falarm=1.96, confusion=1.76
#Eval for threshold 0.6 DER=7.53, miss=4.22, falarm=1.84, confusion=1.47
#Eval for threshold 0.7 DER=8.33, miss=5.40, falarm=1.61, confusion=1.32
#Eval for threshold 0.8 DER=9.79, miss=7.31, falarm=1.36, confusion=1.13
#Eval for threshold 0.9 DER=13.83, miss=12.00, falarm=1.03, confusion=0.80

# cssd_testset of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=9.98, miss=1.24, falarm=7.80, confusion=0.94
#Eval for threshold 0.3 DER=7.53, miss=1.84, falarm=4.16, confusion=1.52
#Eval for threshold 0.35 DER=6.80, miss=2.21, falarm=2.82, confusion=1.78
#Eval for threshold 0.4 DER=6.43, miss=2.66, falarm=1.76, confusion=2.00
#Eval for threshold 0.45 DER=6.44, miss=3.35, falarm=0.95, confusion=2.14
#Eval for threshold 0.5 DER=6.89, miss=4.36, falarm=0.45, confusion=2.07
#Eval for threshold 0.55 DER=7.78, miss=5.79, falarm=0.33, confusion=1.66
#Eval for threshold 0.6 DER=8.86, miss=7.30, falarm=0.26, confusion=1.31
#Eval for threshold 0.7 DER=11.67, miss=10.78, falarm=0.16, confusion=0.74
#Eval for threshold 0.8 DER=16.13, miss=15.67, falarm=0.08, confusion=0.38
#Eval for threshold 0.9 DER=25.92, miss=25.72, falarm=0.03, confusion=0.17

if [ ${stage} -le 136 ] && [ ${stop_stage} -ge 136 ];then
   echo "compute CDER for magicdata-ramc"
   threshold="0.2 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.7 0.8 0.9" # total 11
   predict_rttm_dir=/share/workspace/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2mamba2_multi_backend_transformer_rs_len8_shift0.8/
   oracle_rttm_dir=/share/workspace/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
   infer_sets="dev test cssd_testset"
   c=0.0
   dataset_name="magicdata-ramc"
   for name in $infer_sets;do
    for thres in $threshold;do
     echo "currently, compute $name set in $thres threshold mode"
     python3 cder/score.py -s $predict_rttm_dir/${dataset_name}_${name}_collar${c}/${name}/res_rttm_${thres}  -r $oracle_rttm_dir/$name/rttm_debug_nog0
    done
   done
fi

# grep -r Avg logs/run_ts_vad2_hltsz_4090_stage136.log
# dev of magicdata-ramc
#Avg CDER : 0.485
#Avg CDER : 0.326
#Avg CDER : 0.267
#Avg CDER : 0.215
#Avg CDER : 0.171
#Avg CDER : 0.143
#Avg CDER : 0.122
#Avg CDER : 0.113
#Avg CDER : 0.102
#Avg CDER : 0.097
#Avg CDER : 0.091

# test of magicdata-ramc
#Avg CDER : 0.308
#Avg CDER : 0.254
#Avg CDER : 0.232
#Avg CDER : 0.216
#Avg CDER : 0.201
#Avg CDER : 0.140
#Avg CDER : 0.130
#Avg CDER : 0.122
#Avg CDER : Error!
#Avg CDER : Error!
#Avg CDER : Error!

# cssd_test of magicdata-ramc
#Avg CDER : 0.249
#Avg CDER : 0.189
#Avg CDER : 0.165
#Avg CDER : 0.144
#Avg CDER : 0.126
#Avg CDER : 0.109
#Avg CDER : 0.100
#Avg CDER : 0.095
#Avg CDER : 0.088
#Avg CDER : 0.081
#Avg CDER : 0.108




if [ ${stage} -le 137 ] && [ ${stop_stage} -ge 137 ];then

    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is wav-bert2.0 (only using position embedding and first 6 layers conformer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this w2v-bert2.0 is trained Languages: 143+ , Size: 4.5M hours (it is from this paper https://arxiv.org/pdf/2312.05187)
    # checkpoint is from https://huggingface.co/facebook/w2v-bert-2.0/tree/main
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/share/workspace/shared_datasets/speechdata/14_musan
    rir_path=/share/workspace/shared_datasets/speechdata/21_RIRS_NOISES/RIRS_NOISES
    dataset_name="magicdata-ramc" # dataset name
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/share/workspace/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    spk_path=/share/workspace/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/share/workspace/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2mamba2_multi_backend_transformer_rs_len16_shift0.8
    mkdir -p $exp_dir
    #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
   data_dir="/share/workspace/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format" # oracle target audio , mix audio and labels path
   rs_len=16
   segment_shift=0.8
   single_backend_type="mamba2"
   d_state=128
   multi_backend_type="transformer"
   num_transformer_layer=2
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 18915 \
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
    --d-state $d_state

fi

if [ ${stage} -le 138 ] && [ ${stop_stage} -ge 138 ];then
 exp_dir=/share/workspace/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2mamba2_multi_backend_transformer_rs_len16_shift0.8
 model_file=$exp_dir/best-valid-der.pt
 rs_len=16
 segment_shift=0.8
 single_backend_type="mamba2"
 d_state=128
 multi_backend_type="transformer"
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="dev test cssd_testset"
 #infer_sets="Test"
 rttm_dir=/share/workspace/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0

 dataset_name="magicdata-ramc" # dataset name
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/share/workspace/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 spk_path=/shar/workspace/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

 #data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
 data_dir="/share/workspace/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format"
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_${name}_collar${c}
  python3 ts_vad2/infer.py \
    --model-file $model_file\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name ${name}/rttm_debug_nog0\
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
    --d-state $d_state
 done
done
fi
#grep -r Eval logs/run_ts_vad2_hltsz_4090_stage138-139.log

# dev of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=14.14, miss=0.28, falarm=10.38, confusion=3.48
#Eval for threshold 0.3 DER=12.78, miss=0.52, falarm=8.60, confusion=3.65
#Eval for threshold 0.35 DER=12.36, miss=0.70, falarm=7.97, confusion=3.70
#Eval for threshold 0.4 DER=11.99, miss=0.91, falarm=7.36, confusion=3.71
#Eval for threshold 0.45 DER=11.69, miss=1.16, falarm=6.82, confusion=3.71
#Eval for threshold 0.5 DER=11.50, miss=1.47, falarm=6.35, confusion=3.68
#Eval for threshold 0.55 DER=11.43, miss=1.88, falarm=5.95, confusion=3.60
#Eval for threshold 0.6 DER=11.44, miss=2.36, falarm=5.58, confusion=3.51
#Eval for threshold 0.7 DER=11.69, miss=3.55, falarm=4.82, confusion=3.33
#Eval for threshold 0.8 DER=12.54, miss=5.56, falarm=3.92, confusion=3.06
#Eval for threshold 0.9 DER=15.66, miss=10.43, falarm=2.85, confusion=2.37

# test of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=17.18, miss=0.50, falarm=14.77, confusion=1.91
#Eval for threshold 0.3 DER=15.41, miss=0.95, falarm=12.25, confusion=2.21
#Eval for threshold 0.35 DER=14.78, miss=1.28, falarm=11.21, confusion=2.29
#Eval for threshold 0.4 DER=14.37, miss=1.69, falarm=10.30, confusion=2.37
#Eval for threshold 0.45 DER=13.84, miss=2.24, falarm=9.07, confusion=2.53
#Eval for threshold 0.5 DER=12.83, miss=3.37, falarm=6.32, confusion=3.14
#Eval for threshold 0.55 DER=13.61, miss=5.62, falarm=5.68, confusion=2.31
#Eval for threshold 0.6 DER=14.09, miss=6.86, falarm=5.26, confusion=1.97
#Eval for threshold 0.7 DER=15.09, miss=9.07, falarm=4.42, confusion=1.60
#Eval for threshold 0.8 DER=16.92, miss=12.14, falarm=3.53, confusion=1.25
#Eval for threshold 0.9 DER=21.52, miss=18.25, falarm=2.47, confusion=0.80

# cssd_testset of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=29.41, miss=3.95, falarm=22.66, confusion=2.80
#Eval for threshold 0.3 DER=25.28, miss=5.64, falarm=15.89, confusion=3.75
#Eval for threshold 0.35 DER=23.96, miss=6.65, falarm=13.15, confusion=4.16
#Eval for threshold 0.4 DER=23.12, miss=7.90, falarm=10.86, confusion=4.35
#Eval for threshold 0.45 DER=22.80, miss=9.41, falarm=9.04, confusion=4.35
#Eval for threshold 0.5 DER=23.10, miss=11.35, falarm=7.69, confusion=4.05
#Eval for threshold 0.55 DER=24.05, miss=13.74, falarm=6.84, confusion=3.48
#Eval for threshold 0.6 DER=25.20, miss=16.20, falarm=6.11, confusion=2.90
#Eval for threshold 0.7 DER=28.21, miss=21.59, falarm=4.71, confusion=1.91
#Eval for threshold 0.8 DER=32.73, miss=28.41, falarm=3.20, confusion=1.12
#Eval for threshold 0.9 DER=42.67, miss=40.59, falarm=1.64, confusion=0.44

# dev of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=6.01, miss=0.07, falarm=2.74, confusion=3.20
#Eval for threshold 0.3 DER=5.40, miss=0.14, falarm=1.97, confusion=3.29
#Eval for threshold 0.35 DER=5.26, miss=0.21, falarm=1.74, confusion=3.31
#Eval for threshold 0.4 DER=5.15, miss=0.27, falarm=1.57, confusion=3.32
#Eval for threshold 0.45 DER=5.08, miss=0.35, falarm=1.42, confusion=3.32
#Eval for threshold 0.5 DER=5.05, miss=0.46, falarm=1.29, confusion=3.30
#Eval for threshold 0.55 DER=5.10, miss=0.62, falarm=1.21, confusion=3.27
#Eval for threshold 0.6 DER=5.18, miss=0.81, falarm=1.15, confusion=3.23
#Eval for threshold 0.7 DER=5.50, miss=1.30, falarm=1.03, confusion=3.16
#Eval for threshold 0.8 DER=6.31, miss=2.39, falarm=0.90, confusion=3.02
#Eval for threshold 0.9 DER=9.19, miss=5.98, falarm=0.76, confusion=2.44

# test of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=8.94, miss=0.17, falarm=7.40, confusion=1.37
#Eval for threshold 0.3 DER=8.05, miss=0.35, falarm=6.15, confusion=1.54
#Eval for threshold 0.35 DER=7.76, miss=0.48, falarm=5.67, confusion=1.61
#Eval for threshold 0.4 DER=7.62, miss=0.68, falarm=5.26, confusion=1.67
#Eval for threshold 0.45 DER=7.37, miss=0.97, falarm=4.57, confusion=1.83
#Eval for threshold 0.5 DER=6.49, miss=1.72, falarm=2.21, confusion=2.56
#Eval for threshold 0.55 DER=7.42, miss=3.73, falarm=1.92, confusion=1.77
#Eval for threshold 0.6 DER=7.95, miss=4.68, falarm=1.81, confusion=1.47
#Eval for threshold 0.7 DER=8.91, miss=6.06, falarm=1.59, confusion=1.26
#Eval for threshold 0.8 DER=10.53, miss=8.12, falarm=1.34, confusion=1.07
#Eval for threshold 0.9 DER=14.67, miss=12.92, falarm=1.02, confusion=0.73

# cssd_testset of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=11.99, miss=1.30, falarm=9.45, confusion=1.24
#Eval for threshold 0.3 DER=9.27, miss=2.00, falarm=5.33, confusion=1.95
#Eval for threshold 0.35 DER=8.41, miss=2.46, falarm=3.60, confusion=2.35
#Eval for threshold 0.4 DER=7.86, miss=3.07, falarm=2.18, confusion=2.61
#Eval for threshold 0.45 DER=7.78, miss=3.91, falarm=1.12, confusion=2.75 as report
#Eval for threshold 0.5 DER=8.27, miss=5.16, falarm=0.44, confusion=2.67
#Eval for threshold 0.55 DER=9.40, miss=6.87, falarm=0.27, confusion=2.26
#Eval for threshold 0.6 DER=10.72, miss=8.67, falarm=0.21, confusion=1.84
#Eval for threshold 0.7 DER=14.02, miss=12.75, falarm=0.13, confusion=1.14
#Eval for threshold 0.8 DER=18.94, miss=18.23, falarm=0.07, confusion=0.65
#Eval for threshold 0.9 DER=29.45, miss=29.16, falarm=0.03, confusion=0.26


if [ ${stage} -le 139 ] && [ ${stop_stage} -ge 139 ];then
   echo "compute CDER for magicdata-ramc"
   threshold="0.2 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.7 0.8 0.9" # total 11
   predict_rttm_dir=/share/workspace/maduo/exp/speaker_diarization/ts_vad2/magicdata-ramc-ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_200k_zh_cn_epoch20_front_fix_seed_lr1e4_single_backend_2mamba2_multi_backend_transformer_rs_len16_shift0.8/
   oracle_rttm_dir=/share/workspace/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
   infer_sets="dev test cssd_testset"
   c=0.0
   dataset_name="magicdata-ramc"
   for name in $infer_sets;do
    for thres in $threshold;do
     echo "currently, compute $name set in $thres threshold mode"
     python3 cder/score.py -s $predict_rttm_dir/${dataset_name}_${name}_collar${c}/${name}/res_rttm_${thres}  -r $oracle_rttm_dir/$name/rttm_debug_nog0
    done
   done
fi

# grep -r Avg logs/run_ts_vad2_hltsz_4090_stage138-139.log
# dev 
#Avg CDER : 0.343
#Avg CDER : 0.256
#Avg CDER : 0.222
#Avg CDER : 0.193
#Avg CDER : 0.167
#Avg CDER : 0.134
#Avg CDER : 0.116
#Avg CDER : 0.104
#Avg CDER : 0.099
#Avg CDER : 0.095
#Avg CDER : 0.096

# test
#Avg CDER : 0.295
#Avg CDER : 0.241
#Avg CDER : 0.219
#Avg CDER : 0.199
#Avg CDER : 0.193
#Avg CDER : 0.123
#Avg CDER : 0.113
#Avg CDER : 0.107
#Avg CDER : Error!
#Avg CDER : Error!
#Avg CDER : Error!

# cssd_testset
#Avg CDER : 0.253
#Avg CDER : 0.198
#Avg CDER : 0.172
#Avg CDER : 0.145
#Avg CDER : 0.125
#Avg CDER : 0.108
#Avg CDER : 0.093
#Avg CDER : 0.088
#Avg CDER : 0.093
#Avg CDER : 0.109
#Avg CDER : 0.189
