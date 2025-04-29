#!/usr/bin/env bash
stage=0
stop_stage=1000

. utils/parse_options.sh
. path_for_speaker_diarization_hltsz_4090.sh
if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is cam++ 200k speaker model
    #  oracle target speaker embedding is from cam++ pretrain model
    # checkpoint is from https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/files
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/data2/shared_datasets/speechdata/14_musan
    rir_path=/data1/home/maduo/datasets/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/data1/home/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
    dataset_name="magicdata-ramc" # dataset name

    # for loading speaker embedding file
    spk_path=/data1/home/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/data1/home/maduo/exp/speaker_diarization/ts_vad2_streaming/magicdata-ramc_ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr1e4_single_backend_transformer_multi_backend_transformer_rs_len6_streaming
    data_dir="/data1/home/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format" # oracle target audio , mix audio and labels path
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
 exp_dir=/data1/home/maduo/exp/speaker_diarization/ts_vad2_streaming/magicdata-ramc_ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr1e4_single_backend_transformer_multi_backend_transformer_rs_len6_streaming
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
 rttm_dir=/data1/home/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0
 #collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/data1/home/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 dataset_name="magicdata-ramc" # dataset name
 # for loading speaker embedding file
 spk_path=/data1/home/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 #data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
 data_dir="/data1/home/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format"
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
# grep -r Eval logs/run_ts_vad2_streaming_hltsz_4090_stage13-15.log
# dev of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=15.77, miss=0.26, falarm=11.96, confusion=3.56
#Eval for threshold 0.3 DER=13.68, miss=0.53, falarm=9.19, confusion=3.96
#Eval for threshold 0.35 DER=12.99, miss=0.74, falarm=8.18, confusion=4.07
#Eval for threshold 0.4 DER=12.55, miss=1.07, falarm=7.36, confusion=4.12
#Eval for threshold 0.45 DER=12.24, miss=1.52, falarm=6.65, confusion=4.07
#Eval for threshold 0.5 DER=12.14, miss=2.07, falarm=6.10, confusion=3.96
#Eval for threshold 0.55 DER=12.15, miss=2.74, falarm=5.63, confusion=3.79
#Eval for threshold 0.6 DER=12.27, miss=3.48, falarm=5.23, confusion=3.57
#Eval for threshold 0.7 DER=12.91, miss=5.33, falarm=4.41, confusion=3.17
#Eval for threshold 0.8 DER=14.46, miss=8.41, falarm=3.58, confusion=2.47
#Eval for threshold 0.9 DER=18.99, miss=15.13, falarm=2.55, confusion=1.32

# test of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=20.18, miss=0.30, falarm=18.00, confusion=1.88
#Eval for threshold 0.3 DER=16.85, miss=0.67, falarm=13.69, confusion=2.49
#Eval for threshold 0.35 DER=15.49, miss=0.98, falarm=11.63, confusion=2.88
#Eval for threshold 0.4 DER=14.30, miss=1.49, falarm=9.53, confusion=3.28
#Eval for threshold 0.45 DER=13.47, miss=2.49, falarm=7.51, confusion=3.47
#Eval for threshold 0.5 DER=13.55, miss=4.15, falarm=6.35, confusion=3.05
#Eval for threshold 0.55 DER=14.09, miss=5.75, falarm=5.73, confusion=2.60
#Eval for threshold 0.6 DER=14.66, miss=7.17, falarm=5.28, confusion=2.21
#Eval for threshold 0.7 DER=15.95, miss=9.92, falarm=4.41, confusion=1.62
#Eval for threshold 0.8 DER=18.18, miss=13.57, falarm=3.51, confusion=1.09
#Eval for threshold 0.9 DER=23.07, miss=20.03, falarm=2.48, confusion=0.56

# cssd_testset of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=33.34, miss=2.67, falarm=28.96, confusion=1.70
#Eval for threshold 0.3 DER=26.25, miss=3.94, falarm=19.69, confusion=2.62
#Eval for threshold 0.35 DER=23.72, miss=4.90, falarm=15.63, confusion=3.20
#Eval for threshold 0.4 DER=21.83, miss=6.16, falarm=11.91, confusion=3.75
#Eval for threshold 0.45 DER=21.00, miss=8.06, falarm=9.11, confusion=3.83
#Eval for threshold 0.5 DER=21.42, miss=10.71, falarm=7.37, confusion=3.34
#Eval for threshold 0.55 DER=22.50, miss=13.67, falarm=6.19, confusion=2.64
#Eval for threshold 0.6 DER=23.91, miss=16.62, falarm=5.30, confusion=2.00
#Eval for threshold 0.7 DER=27.19, miss=22.27, falarm=3.82, confusion=1.10
#Eval for threshold 0.8 DER=31.68, miss=28.62, falarm=2.50, confusion=0.56
#Eval for threshold 0.9 DER=40.96, miss=39.54, falarm=1.21, confusion=0.21

# dev of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=6.88, miss=0.04, falarm=3.74, confusion=3.09
#Eval for threshold 0.3 DER=5.88, miss=0.08, falarm=2.45, confusion=3.36
#Eval for threshold 0.35 DER=5.58, miss=0.14, falarm=2.01, confusion=3.44
#Eval for threshold 0.4 DER=5.43, miss=0.24, falarm=1.72, confusion=3.47
#Eval for threshold 0.45 DER=5.37, miss=0.42, falarm=1.49, confusion=3.45
#Eval for threshold 0.5 DER=5.40, miss=0.65, falarm=1.33, confusion=3.42  as report
#Eval for threshold 0.55 DER=5.49, miss=0.94, falarm=1.23, confusion=3.33
#Eval for threshold 0.6 DER=5.65, miss=1.29, falarm=1.15, confusion=3.21
#Eval for threshold 0.7 DER=6.24, miss=2.26, falarm=1.01, confusion=2.96
#Eval for threshold 0.8 DER=7.60, miss=4.28, falarm=0.90, confusion=2.41
#Eval for threshold 0.9 DER=11.89, miss=9.81, falarm=0.77, confusion=1.32

# test of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=11.19, miss=0.04, falarm=9.90, confusion=1.24
#Eval for threshold 0.3 DER=9.10, miss=0.14, falarm=7.32, confusion=1.65
#Eval for threshold 0.35 DER=8.22, miss=0.26, falarm=5.97, confusion=1.99
#Eval for threshold 0.4 DER=7.42, miss=0.49, falarm=4.58, confusion=2.35
#Eval for threshold 0.45 DER=6.82, miss=1.13, falarm=3.08, confusion=2.61 as report
#Eval for threshold 0.5 DER=7.02, miss=2.41, falarm=2.33, confusion=2.28
#Eval for threshold 0.55 DER=7.62, miss=3.63, falarm=2.06, confusion=1.93
#Eval for threshold 0.6 DER=8.15, miss=4.60, falarm=1.90, confusion=1.65
#Eval for threshold 0.7 DER=9.32, miss=6.45, falarm=1.63, confusion=1.25
#Eval for threshold 0.8 DER=11.30, miss=9.04, falarm=1.36, confusion=0.89
#Eval for threshold 0.9 DER=15.71, miss=14.16, falarm=1.06, confusion=0.48
#
# cssd_testset of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=13.81, miss=0.68, falarm=12.75, confusion=0.37
#Eval for threshold 0.3 DER=9.37, miss=1.09, falarm=7.48, confusion=0.79
#Eval for threshold 0.35 DER=7.87, miss=1.42, falarm=5.26, confusion=1.19
#Eval for threshold 0.4 DER=6.59, miss=1.84, falarm=3.06, confusion=1.69
#Eval for threshold 0.45 DER=6.03, miss=2.65, falarm=1.40, confusion=1.98 as report
#Eval for threshold 0.5 DER=6.59, miss=4.20, falarm=0.65, confusion=1.74
#Eval for threshold 0.55 DER=7.78, miss=6.14, falarm=0.34, confusion=1.31
#Eval for threshold 0.6 DER=9.28, miss=8.14, falarm=0.22, confusion=0.92
#Eval for threshold 0.7 DER=12.56, miss=12.02, falarm=0.12, confusion=0.42
#Eval for threshold 0.8 DER=16.83, miss=16.56, falarm=0.06, confusion=0.21
#Eval for threshold 0.9 DER=25.93, miss=25.79, falarm=0.03, confusion=0.11

if [ ${stage} -le 15 ] && [ ${stop_stage} -ge 15 ];then
   echo "compute CDER for magicdata-ramc"
   threshold="0.2 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.7 0.8 0.9"
   predict_rttm_dir=/data1/home/maduo/exp/speaker_diarization/ts_vad2_streaming/magicdata-ramc_ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr1e4_single_backend_transformer_multi_backend_transformer_rs_len6_streaming/magicdata-ramc_collar0.0_decoding_chunk_size25_num_decoding_left_chunks-1_simulate_streamingfalse_
   oracle_rttm_dir=/data1/home/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format 
   infer_sets="dev test cssd_testset"
   for name in $infer_sets;do
    for thres in $threshold;do
     echo "currently, compute $name set in $thres threshold mode"
     python3 cder/score.py -s $predict_rttm_dir/$name/res_rttm_${thres}  -r $oracle_rttm_dir/$name/rttm_debug_nog0
    done
   done
fi

#grep -r Avg logs/run_ts_vad2_streaming_hltsz_4090_stage13-15.log
# dev of magicdata-ramc
#Avg CDER : 0.599
#Avg CDER : 0.406
#Avg CDER : 0.331
#Avg CDER : 0.266
#Avg CDER : 0.217
#Avg CDER : 0.172
#Avg CDER : 0.146
#Avg CDER : 0.128
#Avg CDER : 0.103
#Avg CDER : 0.102
#Avg CDER : 0.107
#
# test of magicdata-ramc
#Avg CDER : 0.482
#Avg CDER : 0.403
#Avg CDER : 0.366
#Avg CDER : 0.324
#Avg CDER : 0.255
#Avg CDER : 0.177
#Avg CDER : 0.132
#Avg CDER : 0.132
#Avg CDER : 0.111
#Avg CDER : Error!
#Avg CDER : Error!
#
#cssd_testset of magicdata-ramc
#Avg CDER : 0.346
#Avg CDER : 0.277
#Avg CDER : 0.244
#Avg CDER : 0.205
#Avg CDER : 0.165
#Avg CDER : 0.131
#Avg CDER : 0.109
#Avg CDER : 0.100
#Avg CDER : 0.089
#Avg CDER : 0.120
#Avg CDER : Error!
