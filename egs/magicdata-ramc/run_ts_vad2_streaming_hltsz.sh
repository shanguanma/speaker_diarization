#!/usr/bin/env bash
stage=0
stop_stage=1000

. utils/parse_options.sh
. path_for_speaker_diarization_hltsz.sh
#. path_for_dia_pt2.4.sh

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is cam++ 200k speaker model
    #  oracle target speaker embedding is from cam++ pretrain model
    # checkpoint is from https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/files
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/data/maduo/datasets/musan
    rir_path=/data/maduo/datasets/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
    dataset_name="magicdata-ramc" # dataset name

    # for loading speaker embedding file
    spk_path=/data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2_streaming/magicdata-ramc_ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr1e4_single_backend_transformer_multi_backend_transformer_rs_len4_streaming
    data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
    rs_len=4
    segment_shift=2
    num_transformer_layer=2
    CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15115 \
   ts_vad2_streaming/train_accelerate_ddp.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --grad-clip false\
    --lr 1e-4\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --exp-dir $exp_dir\
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --num-transformer-layer $num_transformer_layer

fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
 #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_single_backend_transformer_multi_backend_transformer_ts_len10_streaming
 exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2_streaming/magicdata-ramc_ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr1e4_single_backend_transformer_multi_backend_transformer_rs_len4_streaming
 model_file=$exp_dir/best-valid-der.pt
 #model_file=$exp_dir/epoch-1.pt
 rs_len=4
 segment_shift=1
 decoding_chunk_size=25
 num_decoding_left_chunks=-1
 simulate_streaming=false
 batch_size=1
 if $simulate_streaming;then
   fn_name="self.forward_chunk_by_chunk_temp"
 else
   fn_name=""
 fi

 #single_backend_type="mamba2"
 #multi_backend_type="transformer"
 #d_state=256
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 #infer_sets="Eval Test"
 #infer_sets="Test"
 infer_sets="dev test cssd_testset"
 rttm_dir=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0
 #collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 dataset_name="magicdata-ramc" # dataset name
 # for loading speaker embedding file
 spk_path=/data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 #data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
 data_dir="/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format"
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_collar${c}_decoding_chunk_size${decoding_chunk_size}_num_decoding_left_chunks${num_decoding_left_chunks}_simulate_streaming${simulate_streaming}_${fn_name}
  python3 ts_vad2_streaming/infer.py \
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
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --num-transformer-layer $num_transformer_layer\
    --decoding-chunk-size $decoding_chunk_size\
    --num-decoding-left-chunks $num_decoding_left_chunks\
    --simulate-streaming $simulate_streaming\
    --batch-size $batch_size

 done
done
fi

# grep -r Eval logs/run_ts_vad2_streaming_hltsz_stage2.log
# dev of magicdat-ramc, collar=0.0
#Eval for threshold 0.2 DER=20.16, miss=0.26, falarm=17.76, confusion=2.14
#Eval for threshold 0.3 DER=15.73, miss=0.55, falarm=12.12, confusion=3.05
#Eval for threshold 0.35 DER=14.08, miss=0.80, falarm=9.75, confusion=3.52
#Eval for threshold 0.4 DER=12.97, miss=1.22, falarm=7.87, confusion=3.87
#Eval for threshold 0.45 DER=12.36, miss=1.82, falarm=6.53, confusion=4.00
#Eval for threshold 0.5 DER=12.24, miss=2.67, falarm=5.66, confusion=3.91
#Eval for threshold 0.55 DER=12.51, miss=3.80, falarm=5.10, confusion=3.61
#Eval for threshold 0.6 DER=13.06, miss=5.25, falarm=4.61, confusion=3.20
#Eval for threshold 0.7 DER=15.19, miss=9.13, falarm=3.78, confusion=2.28
#Eval for threshold 0.8 DER=18.74, miss=14.30, falarm=2.99, confusion=1.45

# test of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=22.39, miss=0.30, falarm=20.63, confusion=1.46
#Eval for threshold 0.3 DER=18.10, miss=0.67, falarm=15.33, confusion=2.10
#Eval for threshold 0.35 DER=16.42, miss=0.98, falarm=12.94, confusion=2.50
#Eval for threshold 0.4 DER=14.94, miss=1.49, falarm=10.51, confusion=2.94
#Eval for threshold 0.45 DER=13.69, miss=2.38, falarm=7.89, confusion=3.42
#Eval for threshold 0.5 DER=13.53, miss=4.18, falarm=6.14, confusion=3.21
#Eval for threshold 0.55 DER=14.05, miss=6.05, falarm=5.37, confusion=2.63
#Eval for threshold 0.6 DER=14.85, miss=7.79, falarm=4.86, confusion=2.20
#Eval for threshold 0.7 DER=16.81, miss=11.32, falarm=4.01, confusion=1.48
#Eval for threshold 0.8 DER=19.84, miss=15.71, falarm=3.18, confusion=0.95

# cssd_testset of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=35.69, miss=2.47, falarm=31.65, confusion=1.56
#Eval for threshold 0.3 DER=27.72, miss=3.77, falarm=21.43, confusion=2.52
#Eval for threshold 0.35 DER=24.47, miss=4.79, falarm=16.51, confusion=3.17
#Eval for threshold 0.4 DER=22.12, miss=6.33, falarm=12.09, confusion=3.71
#Eval for threshold 0.45 DER=21.36, miss=8.68, falarm=8.96, confusion=3.72
#Eval for threshold 0.5 DER=22.06, miss=11.81, falarm=7.12, confusion=3.12
#Eval for threshold 0.55 DER=23.38, miss=15.00, falarm=5.93, confusion=2.45
#Eval for threshold 0.6 DER=24.98, miss=18.10, falarm=5.00, confusion=1.88
#Eval for threshold 0.7 DER=28.83, miss=24.28, falarm=3.49, confusion=1.06
#Eval for threshold 0.8 DER=34.14, miss=31.33, falarm=2.27, confusion=0.55

# dev of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=11.00, miss=0.04, falarm=9.24, confusion=1.72
#Eval for threshold 0.3 DER=7.99, miss=0.07, falarm=5.47, confusion=2.45
#Eval for threshold 0.35 DER=6.77, miss=0.11, falarm=3.78, confusion=2.88
#Eval for threshold 0.4 DER=6.01, miss=0.24, falarm=2.57, confusion=3.21
#Eval for threshold 0.45 DER=5.66, miss=0.49, falarm=1.77, confusion=3.40
#Eval for threshold 0.5 DER=5.62, miss=0.89, falarm=1.33, confusion=3.40
#Eval for threshold 0.55 DER=5.96, miss=1.57, falarm=1.16, confusion=3.23
#Eval for threshold 0.6 DER=6.54, miss=2.56, falarm=1.07, confusion=2.91
#Eval for threshold 0.7 DER=8.62, miss=5.56, falarm=0.95, confusion=2.11
#Eval for threshold 0.8 DER=12.03, miss=9.80, falarm=0.84, confusion=1.39

# test of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=12.94, miss=0.04, falarm=11.92, confusion=0.98
#Eval for threshold 0.3 DER=10.20, miss=0.12, falarm=8.66, confusion=1.42
#Eval for threshold 0.35 DER=9.11, miss=0.22, falarm=7.16, confusion=1.74
#Eval for threshold 0.4 DER=8.13, miss=0.43, falarm=5.58, confusion=2.12
#Eval for threshold 0.45 DER=7.20, miss=0.89, falarm=3.71, confusion=2.61
#Eval for threshold 0.5 DER=7.13, miss=2.20, falarm=2.38, confusion=2.54
#Eval for threshold 0.55 DER=7.64, miss=3.58, falarm=2.00, confusion=2.06
#Eval for threshold 0.6 DER=8.37, miss=4.83, falarm=1.80, confusion=1.75
#Eval for threshold 0.7 DER=10.12, miss=7.38, falarm=1.55, confusion=1.19
#Eval for threshold 0.8 DER=12.81, miss=10.71, falarm=1.31, confusion=0.80

# cssd_test of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=15.11, miss=0.58, falarm=14.22, confusion=0.31
#Eval for threshold 0.3 DER=10.35, miss=1.01, falarm=8.63, confusion=0.71
#Eval for threshold 0.35 DER=8.38, miss=1.34, falarm=5.91, confusion=1.13
#Eval for threshold 0.4 DER=6.90, miss=1.89, falarm=3.39, confusion=1.62
#Eval for threshold 0.45 DER=6.44, miss=2.93, falarm=1.61, confusion=1.91
#Eval for threshold 0.5 DER=7.25, miss=4.87, falarm=0.79, confusion=1.59
#Eval for threshold 0.55 DER=8.61, miss=6.97, falarm=0.46, confusion=1.18
#Eval for threshold 0.6 DER=10.29, miss=9.17, falarm=0.30, confusion=0.83
#Eval for threshold 0.7 DER=14.21, miss=13.69, falarm=0.12, confusion=0.40
#Eval for threshold 0.8 DER=19.51, miss=19.29, falarm=0.05, confusion=0.17

#grep -r Eval logs/run_ts_vad2_streaming_hltsz_stage2_3_threshold0.9.log
#Eval for threshold 0.9 DER=25.37, miss=22.60, falarm=2.05, confusion=0.72
#Eval for threshold 0.9 DER=25.27, miss=22.51, falarm=2.26, confusion=0.50
#Eval for threshold 0.9 DER=43.72, miss=42.35, falarm=1.14, confusion=0.23
#Eval for threshold 0.9 DER=18.50, miss=17.10, falarm=0.69, confusion=0.70
#Eval for threshold 0.9 DER=17.74, miss=16.26, falarm=1.04, confusion=0.43
#Eval for threshold 0.9 DER=29.27, miss=29.15, falarm=0.03, confusion=0.09
#Eval for threshold 0.9 DER=18.60, miss=14.07, falarm=2.79, confusion=1.74
#Eval for threshold 0.9 DER=22.45, miss=19.14, falarm=2.65, confusion=0.66
#Eval for threshold 0.9 DER=42.12, miss=40.77, falarm=1.13, confusion=0.22
#Eval for threshold 0.9 DER=11.45, miss=8.92, falarm=0.78, confusion=1.75
#Eval for threshold 0.9 DER=15.14, miss=13.47, falarm=1.11, confusion=0.56
#Eval for threshold 0.9 DER=27.26, miss=27.11, falarm=0.05, confusion=0.11

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
   echo "compute CDER for magicdata-ramc"
   threshold="0.2 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.7 0.8 0.9"
   predict_rttm_dir=/data/maduo/exp/speaker_diarization/ts_vad2_streaming/magicdata-ramc_ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr1e4_single_backend_transformer_multi_backend_transformer_rs_len4_streaming/magicdata-ramc_collar0.0_decoding_chunk_size25_num_decoding_left_chunks-1_simulate_streamingfalse_
   oracle_rttm_dir=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
   infer_sets="dev test cssd_testset"
   for name in $infer_sets;do
    for thres in $threshold;do
     echo "currently, compute $name set in $thres threshold mode"
     python3 cder/score.py -s $predict_rttm_dir/$name/res_rttm_${thres}  -r $oracle_rttm_dir/$name/rttm_debug_nog0
    done
   done
fi
# grep -r Avg logs/run_ts_vad2_streaming_hltsz_stage3_threshold0.9.log
# grep -r 'Avg CDER' logs/run_ts_vad2_streaming_hltsz_stage3.log
# dev of magicdata-ramc
#Avg CDER : 1.025
#Avg CDER : 0.843
#Avg CDER : 0.709
#Avg CDER : 0.530
#Avg CDER : 0.394
#Avg CDER : 0.258
#Avg CDER : 0.170
#Avg CDER : 0.131
#Avg CDER : 0.114
#Avg CDER : 0.107
#Avg CDER : 0.146

# test of magicdata-ramc
#Avg CDER : 0.647
#Avg CDER : 0.550
#Avg CDER : 0.508
#Avg CDER : 0.415
#Avg CDER : 0.333
#Avg CDER : 0.233
#Avg CDER : 0.178
#Avg CDER : 0.154
#Avg CDER : 0.121
#Avg CDER : Error!
#Avg CDER : Error!

# csssd_testset of magicdata-ramc
#Avg CDER : 0.364
#Avg CDER : 0.305
#Avg CDER : 0.265
#Avg CDER : 0.215
#Avg CDER : 0.170
#Avg CDER : 0.136
#Avg CDER : 0.118
#Avg CDER : 0.107
#Avg CDER : 0.105
#Avg CDER : 0.150
#Avg CDER : Error!





if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is cam++ 200k speaker model
    #  oracle target speaker embedding is from cam++ pretrain model
    # checkpoint is from https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/files
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/data/maduo/datasets/musan
    rir_path=/data/maduo/datasets/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
    dataset_name="magicdata-ramc" # dataset name

    # for loading speaker embedding file
    spk_path=/data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2_streaming/magicdata-ramc_ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr1e4_single_backend_transformer_multi_backend_transformer_rs_len8_streaming
    data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
    rs_len=8
    segment_shift=2
    num_transformer_layer=2
    CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15115 \
   ts_vad2_streaming/train_accelerate_ddp.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --grad-clip false\
    --lr 1e-4\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --exp-dir $exp_dir\
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --num-transformer-layer $num_transformer_layer
fi

if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 1 ];then
 #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_single_backend_transformer_multi_backend_transformer_ts_len10_streaming
 exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2_streaming/magicdata-ramc_ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr1e4_single_backend_transformer_multi_backend_transformer_rs_len8_streaming
 model_file=$exp_dir/best-valid-der.pt
 #model_file=$exp_dir/epoch-1.pt
 rs_len=8
 segment_shift=1
 decoding_chunk_size=25
 num_decoding_left_chunks=-1
 simulate_streaming=false
 batch_size=1
 if $simulate_streaming;then
   fn_name="self.forward_chunk_by_chunk_temp"
 else
   fn_name=""
 fi

 #single_backend_type="mamba2"
 #multi_backend_type="transformer"
 #d_state=256
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 #infer_sets="Eval Test"
 #infer_sets="Test"
 infer_sets="dev test cssd_testset"
 rttm_dir=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0
 #collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 dataset_name="magicdata-ramc" # dataset name
 # for loading speaker embedding file
 spk_path=/data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 #data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
 data_dir="/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format"
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_collar${c}_decoding_chunk_size${decoding_chunk_size}_num_decoding_left_chunks${num_decoding_left_chunks}_simulate_streaming${simulate_streaming}_${fn_name}
  python3 ts_vad2_streaming/infer.py \
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
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --num-transformer-layer $num_transformer_layer\
    --decoding-chunk-size $decoding_chunk_size\
    --num-decoding-left-chunks $num_decoding_left_chunks\
    --simulate-streaming $simulate_streaming\
    --batch-size $batch_size

 done
done
fi
# grep -r Eval logs/run_ts_vad2_streaming_hltsz_stage10-12.log
# dev of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=16.17, miss=0.25, falarm=12.26, confusion=3.66
#Eval for threshold 0.3 DER=14.00, miss=0.50, falarm=9.39, confusion=4.11
#Eval for threshold 0.35 DER=13.30, miss=0.72, falarm=8.32, confusion=4.26
#Eval for threshold 0.4 DER=12.79, miss=1.04, falarm=7.47, confusion=4.28
#Eval for threshold 0.45 DER=12.49, miss=1.51, falarm=6.77, confusion=4.21
#Eval for threshold 0.5 DER=12.34, miss=2.05, falarm=6.21, confusion=4.08
#Eval for threshold 0.55 DER=12.37, miss=2.73, falarm=5.75, confusion=3.89
#Eval for threshold 0.6 DER=12.49, miss=3.42, falarm=5.35, confusion=3.73
#Eval for threshold 0.7 DER=13.06, miss=5.13, falarm=4.57, confusion=3.36
#Eval for threshold 0.8 DER=14.47, miss=7.84, falarm=3.77, confusion=2.85

# test of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=20.64, miss=0.27, falarm=18.45, confusion=1.93
#Eval for threshold 0.3 DER=17.45, miss=0.60, falarm=14.33, confusion=2.52
#Eval for threshold 0.35 DER=16.22, miss=0.91, falarm=12.52, confusion=2.78
#Eval for threshold 0.4 DER=15.06, miss=1.42, falarm=10.58, confusion=3.06
#Eval for threshold 0.45 DER=13.98, miss=2.18, falarm=8.33, confusion=3.47
#Eval for threshold 0.5 DER=13.62, miss=3.50, falarm=6.70, confusion=3.41
#Eval for threshold 0.55 DER=14.02, miss=5.21, falarm=5.87, confusion=2.94
#Eval for threshold 0.6 DER=14.67, miss=6.87, falarm=5.35, confusion=2.44
#Eval for threshold 0.7 DER=16.09, miss=9.88, falarm=4.51, confusion=1.70
#Eval for threshold 0.8 DER=18.12, miss=13.28, falarm=3.65, confusion=1.19

# cssd_testset of masgicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=33.86, miss=2.52, falarm=29.56, confusion=1.78
#Eval for threshold 0.3 DER=26.10, miss=3.98, falarm=19.25, confusion=2.87
#Eval for threshold 0.35 DER=23.45, miss=5.08, falarm=15.01, confusion=3.37
#Eval for threshold 0.4 DER=21.80, miss=6.47, falarm=11.63, confusion=3.71
#Eval for threshold 0.45 DER=21.13, miss=8.31, falarm=9.14, confusion=3.68
#Eval for threshold 0.5 DER=21.36, miss=10.69, falarm=7.38, confusion=3.29
#Eval for threshold 0.55 DER=22.21, miss=13.36, falarm=6.15, confusion=2.71
#Eval for threshold 0.6 DER=23.46, miss=16.14, falarm=5.19, confusion=2.14
#Eval for threshold 0.7 DER=27.04, miss=22.13, falarm=3.66, confusion=1.25
#Eval for threshold 0.8 DER=32.42, miss=29.40, falarm=2.38, confusion=0.65

# dev of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=7.07, miss=0.04, falarm=3.91, confusion=3.11
#Eval for threshold 0.3 DER=6.02, miss=0.07, falarm=2.55, confusion=3.39
#Eval for threshold 0.35 DER=5.70, miss=0.13, falarm=2.08, confusion=3.49
#Eval for threshold 0.4 DER=5.49, miss=0.23, falarm=1.74, confusion=3.52
#Eval for threshold 0.45 DER=5.41, miss=0.42, falarm=1.48, confusion=3.51
#Eval for threshold 0.5 DER=5.41, miss=0.64, falarm=1.31, confusion=3.46 as report
#Eval for threshold 0.55 DER=5.53, miss=0.95, falarm=1.22, confusion=3.36
#Eval for threshold 0.6 DER=5.70, miss=1.26, falarm=1.14, confusion=3.30
#Eval for threshold 0.7 DER=6.27, miss=2.14, falarm=1.02, confusion=3.12
#Eval for threshold 0.8 DER=7.50, miss=3.79, falarm=0.91, confusion=2.80

# test of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=11.41, miss=0.04, falarm=10.14, confusion=1.23
#Eval for threshold 0.3 DER=9.48, miss=0.12, falarm=7.74, confusion=1.62
#Eval for threshold 0.35 DER=8.77, miss=0.23, falarm=6.72, confusion=1.82
#Eval for threshold 0.4 DER=8.02, miss=0.47, falarm=5.46, confusion=2.09
#Eval for threshold 0.45 DER=7.20, miss=0.86, falarm=3.78, confusion=2.55
#Eval for threshold 0.5 DER=6.95, miss=1.74, falarm=2.58, confusion=2.62 as report
#Eval for threshold 0.55 DER=7.41, miss=3.03, falarm=2.11, confusion=2.28
#Eval for threshold 0.6 DER=8.12, miss=4.32, falarm=1.91, confusion=1.89
#Eval for threshold 0.7 DER=9.47, miss=6.50, falarm=1.65, confusion=1.32
#Eval for threshold 0.8 DER=11.27, miss=8.89, falarm=1.41, confusion=0.97

# cssd_testset of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=13.99, miss=0.62, falarm=12.92, confusion=0.45
#Eval for threshold 0.3 DER=9.23, miss=1.08, falarm=7.20, confusion=0.96
#Eval for threshold 0.35 DER=7.60, miss=1.45, falarm=4.84, confusion=1.31
#Eval for threshold 0.4 DER=6.60, miss=1.95, falarm=3.01, confusion=1.65
#Eval for threshold 0.45 DER=6.21, miss=2.74, falarm=1.66, confusion=1.82 as report
#Eval for threshold 0.5 DER=6.53, miss=4.00, falarm=0.85, confusion=1.68
#Eval for threshold 0.55 DER=7.43, miss=5.61, falarm=0.47, confusion=1.35
#Eval for threshold 0.6 DER=8.70, miss=7.40, falarm=0.30, confusion=1.00
#Eval for threshold 0.7 DER=12.30, miss=11.62, falarm=0.16, confusion=0.52
#Eval for threshold 0.8 DER=17.57, miss=17.22, falarm=0.09, confusion=0.26

# grep -r Eval  logs/run_ts_vad2_streaming_hltsz_stage11_12_threshold0.9.log
#Eval for threshold 0.9 DER=18.59, miss=14.08, falarm=2.78, confusion=1.73
#Eval for threshold 0.9 DER=22.45, miss=19.14, falarm=2.65, confusion=0.66
#Eval for threshold 0.9 DER=42.11, miss=40.76, falarm=1.13, confusion=0.22
#Eval for threshold 0.9 DER=11.46, miss=8.93, falarm=0.78, confusion=1.75
#Eval for threshold 0.9 DER=15.15, miss=13.47, falarm=1.11, confusion=0.56
#Eval for threshold 0.9 DER=27.27, miss=27.11, falarm=0.05, confusion=0.11


if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ];then
   echo "compute CDER for magicdata-ramc"
   threshold="0.2 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.7 0.8 0.9"
   predict_rttm_dir=/data/maduo/exp/speaker_diarization/ts_vad2_streaming/magicdata-ramc_ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr1e4_single_backend_transformer_multi_backend_transformer_rs_len8_streaming/magicdata-ramc_collar0.0_decoding_chunk_size25_num_decoding_left_chunks-1_simulate_streamingfalse_
   oracle_rttm_dir=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
   infer_sets="dev test cssd_testset"
   for name in $infer_sets;do
    for thres in $threshold;do
     echo "currently, compute $name set in $thres threshold mode"
     python3 cder/score.py -s $predict_rttm_dir/$name/res_rttm_${thres}  -r $oracle_rttm_dir/$name/rttm_debug_nog0
    done
   done
fi

# grep -r Avg logs/run_ts_vad2_streaming_hltsz_stage10-12.log
# dev of magicdata-ramc, 
#Avg CDER : 0.565
#Avg CDER : 0.409
#Avg CDER : 0.327
#Avg CDER : 0.269
#Avg CDER : 0.207
#Avg CDER : 0.148
#Avg CDER : 0.128
#Avg CDER : 0.120
#Avg CDER : 0.108
#Avg CDER : 0.102
#Avg CDER : 0.105
# test of magicdata-ramc
#Avg CDER : 0.498
#Avg CDER : 0.424
#Avg CDER : 0.385
#Avg CDER : 0.355
#Avg CDER : 0.308
#Avg CDER : 0.222
#Avg CDER : 0.149
#Avg CDER : 0.111
#Avg CDER : 0.113
#Avg CDER : Error!
#Avg CDER : Error!

# cssd_testset of magicdata-ramc
#Avg CDER : 0.370
#Avg CDER : 0.295
#Avg CDER : 0.249
#Avg CDER : 0.207
#Avg CDER : 0.169
#Avg CDER : 0.139
#Avg CDER : 0.119
#Avg CDER : 0.105
#Avg CDER : 0.094
#Avg CDER : 0.123
#Avg CDER : 0.176


if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is cam++ 200k speaker model
    #  oracle target speaker embedding is from cam++ pretrain model
    # checkpoint is from https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/files
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/data/maduo/datasets/musan
    rir_path=/data/maduo/datasets/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
    dataset_name="magicdata-ramc" # dataset name

    # for loading speaker embedding file
    spk_path=/data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2_streaming/magicdata-ramc_ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr1e4_single_backend_transformer_multi_backend_transformer_rs_len6_streaming
    data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
    rs_len=6
    segment_shift=2
    num_transformer_layer=2
    CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15115 \
   ts_vad2_streaming/train_accelerate_ddp.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --grad-clip false\
    --lr 1e-4\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --exp-dir $exp_dir\
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --num-transformer-layer $num_transformer_layer
fi

if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ];then
 #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_single_backend_transformer_multi_backend_transformer_ts_len10_streaming
 exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2_streaming/magicdata-ramc_ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr1e4_single_backend_transformer_multi_backend_transformer_rs_len6_streaming
 model_file=$exp_dir/best-valid-der.pt
 #model_file=$exp_dir/epoch-1.pt
 rs_len=6
 segment_shift=1
 decoding_chunk_size=25
 num_decoding_left_chunks=-1
 simulate_streaming=false
 batch_size=1
 if $simulate_streaming;then
   fn_name="self.forward_chunk_by_chunk_temp"
 else
   fn_name=""
 fi

 #single_backend_type="mamba2"
 #multi_backend_type="transformer"
 #d_state=256
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 #infer_sets="Eval Test"
 #infer_sets="Test"
 infer_sets="dev test cssd_testset"
 rttm_dir=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0
 #collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 dataset_name="magicdata-ramc" # dataset name
 # for loading speaker embedding file
 spk_path=/data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 #data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
 data_dir="/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format"
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_collar${c}_decoding_chunk_size${decoding_chunk_size}_num_decoding_left_chunks${num_decoding_left_chunks}_simulate_streaming${simulate_streaming}_${fn_name}
  python3 ts_vad2_streaming/infer.py \
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
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --num-transformer-layer $num_transformer_layer\
    --decoding-chunk-size $decoding_chunk_size\
    --num-decoding-left-chunks $num_decoding_left_chunks\
    --simulate-streaming $simulate_streaming\
    --batch-size $batch_size

 done
done
fi

if [ ${stage} -le 15 ] && [ ${stop_stage} -ge 15 ];then
   echo "compute CDER for magicdata-ramc"
   threshold="0.2 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.7 0.8 0.9"
   predict_rttm_dir=/data/maduo/exp/speaker_diarization/ts_vad2_streaming/magicdata-ramc_ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr1e4_single_backend_transformer_multi_backend_transformer_rs_len6_streaming/magicdata-ramc_collar0.0_decoding_chunk_size25_num_decoding_left_chunks-1_simulate_streamingfalse_
   oracle_rttm_dir=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
   infer_sets="dev test cssd_testset"
   for name in $infer_sets;do
    for thres in $threshold;do
     echo "currently, compute $name set in $thres threshold mode"
     python3 cder/score.py -s $predict_rttm_dir/$name/res_rttm_${thres}  -r $oracle_rttm_dir/$name/rttm_debug_nog0
    done
   done
fi



if [ ${stage} -le 16 ] && [ ${stop_stage} -ge 16 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is cam++ 200k speaker model
    #  oracle target speaker embedding is from cam++ pretrain model
    # checkpoint is from https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/files
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/data/maduo/datasets/musan
    rir_path=/data/maduo/datasets/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
    dataset_name="magicdata-ramc" # dataset name

    # for loading speaker embedding file
    spk_path=/data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2_streaming/magicdata-ramc_ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr5e5_single_backend_transformer_multi_backend_transformer_rs_len16_shift0.8_streaming
    data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
    rs_len=16
    segment_shift=0.8
    num_transformer_layer=2
    CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15115 \
   ts_vad2_streaming/train_accelerate_ddp.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --grad-clip false\
    --lr 5e-5\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --exp-dir $exp_dir\
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --num-transformer-layer $num_transformer_layer
fi

if [ ${stage} -le 17 ] && [ ${stop_stage} -ge 17 ];then
 #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_single_backend_transformer_multi_backend_transformer_ts_len10_streaming
 exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2_streaming/magicdata-ramc_ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr5e5_single_backend_transformer_multi_backend_transformer_rs_len16_shift0.8_streaming
 model_file=$exp_dir/best-valid-der.pt
 #model_file=$exp_dir/epoch-1.pt
 rs_len=16
 segment_shift=0.8
 decoding_chunk_size=25
 num_decoding_left_chunks=-1
 simulate_streaming=false
 batch_size=1
 if $simulate_streaming;then
   fn_name="self.forward_chunk_by_chunk_temp"
 else
   fn_name=""
 fi

 #single_backend_type="mamba2"
 #multi_backend_type="transformer"
 #d_state=256
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 #infer_sets="Eval Test"
 #infer_sets="Test"
 infer_sets="dev test cssd_testset"
 rttm_dir=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0
 #collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 dataset_name="magicdata-ramc" # dataset name
 # for loading speaker embedding file
 spk_path=/data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 #data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
 data_dir="/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format"
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_collar${c}_decoding_chunk_size${decoding_chunk_size}_num_decoding_left_chunks${num_decoding_left_chunks}_simulate_streaming${simulate_streaming}_${fn_name}
  python3 ts_vad2_streaming/infer.py \
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
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --num-transformer-layer $num_transformer_layer\
    --decoding-chunk-size $decoding_chunk_size\
    --num-decoding-left-chunks $num_decoding_left_chunks\
    --simulate-streaming $simulate_streaming\
    --batch-size $batch_size

 done
done
fi

#grep -r Eval logs/run_ts_vad2_streaming_hltsz_stage16-18.log
# dev of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=17.08, miss=0.27, falarm=12.96, confusion=3.85
#Eval for threshold 0.3 DER=14.59, miss=0.51, falarm=9.73, confusion=4.35
#Eval for threshold 0.35 DER=13.86, miss=0.71, falarm=8.67, confusion=4.48
#Eval for threshold 0.4 DER=13.28, miss=1.00, falarm=7.75, confusion=4.53
#Eval for threshold 0.45 DER=12.92, miss=1.42, falarm=7.01, confusion=4.49
#Eval for threshold 0.5 DER=12.79, miss=1.97, falarm=6.45, confusion=4.37
#Eval for threshold 0.55 DER=12.81, miss=2.66, falarm=6.00, confusion=4.15
#Eval for threshold 0.6 DER=12.94, miss=3.39, falarm=5.60, confusion=3.95
#Eval for threshold 0.7 DER=13.57, miss=5.25, falarm=4.81, confusion=3.51
#Eval for threshold 0.8 DER=15.15, miss=8.14, falarm=4.03, confusion=2.98
#Eval for threshold 0.9 DER=19.60, miss=15.06, falarm=3.03, confusion=1.51

# test of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=21.69, miss=0.32, falarm=19.47, confusion=1.91
#Eval for threshold 0.3 DER=18.03, miss=0.68, falarm=14.74, confusion=2.61
#Eval for threshold 0.35 DER=16.36, miss=1.03, falarm=12.24, confusion=3.08
#Eval for threshold 0.4 DER=14.85, miss=1.77, falarm=9.53, confusion=3.54
#Eval for threshold 0.45 DER=14.34, miss=3.23, falarm=7.65, confusion=3.46
#Eval for threshold 0.5 DER=14.56, miss=4.82, falarm=6.64, confusion=3.10
#Eval for threshold 0.55 DER=14.90, miss=6.25, falarm=5.94, confusion=2.71
#Eval for threshold 0.6 DER=15.35, miss=7.63, falarm=5.40, confusion=2.31
#Eval for threshold 0.7 DER=16.79, miss=10.55, falarm=4.60, confusion=1.64
#Eval for threshold 0.8 DER=18.95, miss=14.10, falarm=3.76, confusion=1.08
#Eval for threshold 0.9 DER=23.32, miss=19.95, falarm=2.80, confusion=0.58

# cssd_testset of magicdata-ramc. collar=0.0
#Eval for threshold 0.2 DER=35.60, miss=2.66, falarm=30.78, confusion=2.16
#Eval for threshold 0.3 DER=27.90, miss=3.88, falarm=20.71, confusion=3.32
#Eval for threshold 0.35 DER=25.17, miss=4.75, falarm=16.49, confusion=3.93
#Eval for threshold 0.4 DER=23.26, miss=6.03, falarm=12.91, confusion=4.33
#Eval for threshold 0.45 DER=22.36, miss=7.90, falarm=10.18, confusion=4.29 as report
#Eval for threshold 0.5 DER=22.56, miss=10.47, falarm=8.24, confusion=3.85
#Eval for threshold 0.55 DER=23.49, miss=13.42, falarm=6.90, confusion=3.17
#Eval for threshold 0.6 DER=24.94, miss=16.56, falarm=5.92, confusion=2.46
#Eval for threshold 0.7 DER=28.84, miss=22.97, falarm=4.43, confusion=1.43
#Eval for threshold 0.8 DER=34.07, miss=30.26, falarm=3.06, confusion=0.74
#Eval for threshold 0.9 DER=43.46, miss=41.54, falarm=1.62, confusion=0.30

# dev of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=7.86, miss=0.04, falarm=4.63, confusion=3.19
#Eval for threshold 0.3 DER=6.41, miss=0.07, falarm=2.83, confusion=3.50
#Eval for threshold 0.35 DER=6.02, miss=0.11, falarm=2.32, confusion=3.60
#Eval for threshold 0.4 DER=5.74, miss=0.19, falarm=1.90, confusion=3.65
#Eval for threshold 0.45 DER=5.60, miss=0.35, falarm=1.60, confusion=3.66
#Eval for threshold 0.5 DER=5.60, miss=0.59, falarm=1.39, confusion=3.61 as report
#Eval for threshold 0.55 DER=5.70, miss=0.92, falarm=1.27, confusion=3.52
#Eval for threshold 0.6 DER=5.87, miss=1.25, falarm=1.20, confusion=3.42
#Eval for threshold 0.7 DER=6.51, miss=2.24, falarm=1.06, confusion=3.20
#Eval for threshold 0.8 DER=7.98, miss=4.16, falarm=0.95, confusion=2.87
#Eval for threshold 0.9 DER=12.32, miss=10.06, falarm=0.79, confusion=1.46

# test of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=12.44, miss=0.04, falarm=11.22, confusion=1.18
#Eval for threshold 0.3 DER=10.08, miss=0.12, falarm=8.32, confusion=1.64
#Eval for threshold 0.35 DER=8.99, miss=0.25, falarm=6.70, confusion=2.03
#Eval for threshold 0.4 DER=7.82, miss=0.65, falarm=4.65, confusion=2.51
#Eval for threshold 0.45 DER=7.49, miss=1.72, falarm=3.22, confusion=2.54 as report
#Eval for threshold 0.5 DER=7.78, miss=2.93, falarm=2.55, confusion=2.30
#Eval for threshold 0.55 DER=8.15, miss=3.96, falarm=2.16, confusion=2.03
#Eval for threshold 0.6 DER=8.59, miss=4.94, falarm=1.90, confusion=1.74
#Eval for threshold 0.7 DER=9.91, miss=7.02, falarm=1.62, confusion=1.27
#Eval for threshold 0.8 DER=11.94, miss=9.69, falarm=1.39, confusion=0.86
#Eval for threshold 0.9 DER=15.97, miss=14.37, falarm=1.14, confusion=0.46

# cssd_testset of magicdata-ramc,collar=0.25
#Eval for threshold 0.2 DER=16.29, miss=0.66, falarm=15.19, confusion=0.43
#Eval for threshold 0.3 DER=10.99, miss=1.04, falarm=8.99, confusion=0.96
#Eval for threshold 0.35 DER=9.01, miss=1.30, falarm=6.30, confusion=1.41
#Eval for threshold 0.4 DER=7.62, miss=1.75, falarm=4.03, confusion=1.85
#Eval for threshold 0.45 DER=6.95, miss=2.61, falarm=2.32, confusion=2.03 as report
#Eval for threshold 0.5 DER=7.17, miss=4.05, falarm=1.18, confusion=1.94
#Eval for threshold 0.55 DER=8.13, miss=5.93, falarm=0.63, confusion=1.58
#Eval for threshold 0.6 DER=9.66, miss=8.13, falarm=0.40, confusion=1.13
#Eval for threshold 0.7 DER=13.67, miss=12.93, falarm=0.22, confusion=0.52
#Eval for threshold 0.8 DER=19.11, miss=18.77, falarm=0.12, confusion=0.22
#Eval for threshold 0.9 DER=29.11, miss=28.97, falarm=0.05, confusion=0.09

if [ ${stage} -le 18 ] && [ ${stop_stage} -ge 18 ];then
   echo "compute CDER for magicdata-ramc"
   threshold="0.2 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.7 0.8 0.9"
   predict_rttm_dir=/data/maduo/exp/speaker_diarization/ts_vad2_streaming/magicdata-ramc_ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr5e5_single_backend_transformer_multi_backend_transformer_rs_len16_shift0.8_streaming/magicdata-ramc_collar0.0_decoding_chunk_size25_num_decoding_left_chunks-1_simulate_streamingfalse_
   oracle_rttm_dir=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
   infer_sets="dev test cssd_testset"
   for name in $infer_sets;do
    for thres in $threshold;do
     echo "currently, compute $name set in $thres threshold mode"
     python3 cder/score.py -s $predict_rttm_dir/$name/res_rttm_${thres}  -r $oracle_rttm_dir/$name/rttm_debug_nog0
    done
   done
fi

#grep -r Avg logs/run_ts_vad2_streaming_hltsz_stage16-18.log
# dev of magicdata-ramc, 
#Avg CDER : 0.701
#Avg CDER : 0.396
#Avg CDER : 0.311
#Avg CDER : 0.242
#Avg CDER : 0.186
#Avg CDER : 0.152
#Avg CDER : 0.123
#Avg CDER : 0.114
#Avg CDER : 0.105
#Avg CDER : 0.108
#Avg CDER : 0.109

# test of magicdata-ramc
#Avg CDER : 0.537
#Avg CDER : 0.458
#Avg CDER : 0.415
#Avg CDER : 0.377
#Avg CDER : 0.284
#Avg CDER : 0.219
#Avg CDER : 0.186
#Avg CDER : 0.159
#Avg CDER : 0.117
#Avg CDER : Error!
#Avg CDER : Error!
# cssd_testset of magicdata-ramc
#Avg CDER : 0.358
#Avg CDER : 0.283
#Avg CDER : 0.252
#Avg CDER : 0.215
#Avg CDER : 0.177
#Avg CDER : 0.141
#Avg CDER : 0.117
#Avg CDER : 0.102
#Avg CDER : 0.088
#Avg CDER : 0.157
#Avg CDER : Error!



if [ ${stage} -le 19 ] && [ ${stop_stage} -ge 19 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is cam++ 200k speaker model
    #  oracle target speaker embedding is from cam++ pretrain model
    # checkpoint is from https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/files
    # how to look for port ?                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/data/maduo/datasets/musan
    rir_path=/data/maduo/datasets/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
    dataset_name="magicdata-ramc" # dataset name

    # for loading speaker embedding file
    spk_path=/data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2_streaming/magicdata-ramc_ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr1e5_single_backend_mamba2_multi_backend_transformer_rs_len16_shift0.8_streaming
    data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
    rs_len=16
    segment_shift=0.8
    single_backend_type="mamba2"
    num_transformer_layer=2
    CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15115 \
   ts_vad2_streaming/train_accelerate_ddp.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --grad-clip false\
    --lr 1e-5\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --exp-dir $exp_dir\
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --num-transformer-layer $num_transformer_layer\
    --single-backend-type $single_backend_type
fi

if [ ${stage} -le 20 ] && [ ${stop_stage} -ge 20 ];then
 #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_single_backend_transformer_multi_backend_transformer_ts_len10_streaming
 exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2_streaming/magicdata-ramc_ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr1e5_single_backend_mamba2_multi_backend_transformer_rs_len16_shift0.8_streaming
 model_file=$exp_dir/best-valid-der.pt
 #model_file=$exp_dir/epoch-1.pt
 rs_len=16
 segment_shift=0.8
 decoding_chunk_size=25
 num_decoding_left_chunks=-1
 simulate_streaming=false
 batch_size=1
 if $simulate_streaming;then
   fn_name="self.forward_chunk_by_chunk_temp"
 else
   fn_name=""
 fi

 single_backend_type="mamba2"
 #multi_backend_type="transformer"
 #d_state=256
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 #infer_sets="Eval Test"
 #infer_sets="Test"
 infer_sets="dev test cssd_testset"
 rttm_dir=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0
 #collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 dataset_name="magicdata-ramc" # dataset name
 # for loading speaker embedding file
 spk_path=/data/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 #data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
 data_dir="/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format"
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_collar${c}_decoding_chunk_size${decoding_chunk_size}_num_decoding_left_chunks${num_decoding_left_chunks}_simulate_streaming${simulate_streaming}_${fn_name}
  python3 ts_vad2_streaming/infer.py \
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
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --num-transformer-layer $num_transformer_layer\
     --single-backend-type $single_backend_type\
    --decoding-chunk-size $decoding_chunk_size\
    --num-decoding-left-chunks $num_decoding_left_chunks\
    --simulate-streaming $simulate_streaming\
    --batch-size $batch_size

 done
done
fi

if [ ${stage} -le 21 ] && [ ${stop_stage} -ge 21 ];then
   echo "compute CDER for magicdata-ramc"
   threshold="0.2 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.7 0.8 0.9"
   predict_rttm_dir=/data/maduo/exp/speaker_diarization/ts_vad2_streaming/magicdata-ramc_ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr1e5_single_backend_mamba2_multi_backend_transformer_rs_len16_shift0.8_streaming/magicdata-ramc_collar0.0_decoding_chunk_size25_num_decoding_left_chunks-1_simulate_streamingfalse_
   oracle_rttm_dir=/data/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
   infer_sets="dev test cssd_testset"
   for name in $infer_sets;do
    for thres in $threshold;do
     echo "currently, compute $name set in $thres threshold mode"
     python3 cder/score.py -s $predict_rttm_dir/$name/res_rttm_${thres}  -r $oracle_rttm_dir/$name/rttm_debug_nog0
    done
   done
fi
