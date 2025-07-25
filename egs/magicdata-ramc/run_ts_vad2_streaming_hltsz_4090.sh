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
#
#
#
if [ ${stage} -le 16 ] && [ ${stop_stage} -ge 16 ];then
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
    exp_dir=/data1/home/maduo/exp/speaker_diarization/ts_vad2_streaming/magicdata-ramc_ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr1e4_single_backend_transformer_multi_backend_transformer_rs_len6_shift0.8_streaming
    data_dir="/data1/home/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format" # oracle target audio , mix audio and labels path
    rs_len=6
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

if [ ${stage} -le 17 ] && [ ${stop_stage} -ge 17 ];then
 #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_single_backend_transformer_multi_backend_transformer_ts_len10_streaming
 exp_dir=/data1/home/maduo/exp/speaker_diarization/ts_vad2_streaming/magicdata-ramc_ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr1e4_single_backend_transformer_multi_backend_transformer_rs_len6_shift0.8_streaming
 model_file=$exp_dir/best-valid-der.pt
 #model_file=$exp_dir/epoch-1.pt
 rs_len=6
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

# grep -r Eval logs/run_ts_vad2_streaming_hltsz_4090_stage16-18.log
# dev of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=16.93, miss=0.28, falarm=13.13, confusion=3.52
#Eval for threshold 0.3 DER=14.25, miss=0.55, falarm=9.59, confusion=4.10
#Eval for threshold 0.35 DER=13.44, miss=0.76, falarm=8.40, confusion=4.27
#Eval for threshold 0.4 DER=12.89, miss=1.11, falarm=7.44, confusion=4.33
#Eval for threshold 0.45 DER=12.57, miss=1.61, falarm=6.68, confusion=4.28
#Eval for threshold 0.5 DER=12.42, miss=2.23, falarm=6.06, confusion=4.13
#Eval for threshold 0.55 DER=12.51, miss=3.00, falarm=5.60, confusion=3.92
#Eval for threshold 0.6 DER=12.70, miss=3.80, falarm=5.21, confusion=3.70
#Eval for threshold 0.7 DER=13.59, miss=6.05, falarm=4.42, confusion=3.13
#Eval for threshold 0.8 DER=15.95, miss=10.15, falarm=3.60, confusion=2.21
#Eval for threshold 0.9 DER=21.12, miss=17.45, falarm=2.57, confusion=1.10

# test of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=20.58, miss=0.29, falarm=18.40, confusion=1.89
#Eval for threshold 0.3 DER=17.12, miss=0.65, falarm=13.94, confusion=2.54
#Eval for threshold 0.35 DER=15.61, miss=0.96, falarm=11.70, confusion=2.94
#Eval for threshold 0.4 DER=14.28, miss=1.54, falarm=9.37, confusion=3.38
#Eval for threshold 0.45 DER=13.50, miss=2.63, falarm=7.39, confusion=3.48
#Eval for threshold 0.5 DER=13.61, miss=4.13, falarm=6.33, confusion=3.15
#Eval for threshold 0.55 DER=14.06, miss=5.63, falarm=5.71, confusion=2.72
#Eval for threshold 0.6 DER=14.55, miss=6.99, falarm=5.25, confusion=2.31
#Eval for threshold 0.7 DER=15.88, miss=9.78, falarm=4.41, confusion=1.70
#Eval for threshold 0.8 DER=17.98, miss=13.25, falarm=3.55, confusion=1.18
#Eval for threshold 0.9 DER=22.57, miss=19.38, falarm=2.54, confusion=0.66

# cssd_testset of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=34.97, miss=2.51, falarm=30.62, confusion=1.84
#Eval for threshold 0.3 DER=27.51, miss=3.69, falarm=21.01, confusion=2.82
#Eval for threshold 0.35 DER=24.76, miss=4.48, falarm=16.90, confusion=3.37
#Eval for threshold 0.4 DER=22.73, miss=5.56, falarm=13.36, confusion=3.81
#Eval for threshold 0.45 DER=21.66, miss=7.12, falarm=10.55, confusion=3.99 as report
#Eval for threshold 0.5 DER=21.46, miss=9.25, falarm=8.49, confusion=3.71 
#Eval for threshold 0.55 DER=22.10, miss=11.84, falarm=7.08, confusion=3.18
#Eval for threshold 0.6 DER=23.24, miss=14.60, falarm=6.04, confusion=2.61
#Eval for threshold 0.7 DER=26.63, miss=20.72, falarm=4.34, confusion=1.56
#Eval for threshold 0.8 DER=31.62, miss=27.98, falarm=2.84, confusion=0.80
#Eval for threshold 0.9 DER=41.21, miss=39.52, falarm=1.40, confusion=0.29

# dev of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=7.88, miss=0.04, falarm=4.85, confusion=2.99
#Eval for threshold 0.3 DER=6.31, miss=0.08, falarm=2.84, confusion=3.40
#Eval for threshold 0.35 DER=5.89, miss=0.13, falarm=2.26, confusion=3.51
#Eval for threshold 0.4 DER=5.63, miss=0.25, falarm=1.83, confusion=3.55
#Eval for threshold 0.45 DER=5.52, miss=0.43, falarm=1.53, confusion=3.55
#Eval for threshold 0.5 DER=5.51, miss=0.69, falarm=1.33, confusion=3.49 as report
#Eval for threshold 0.55 DER=5.65, miss=1.05, falarm=1.20, confusion=3.39
#Eval for threshold 0.6 DER=5.86, miss=1.46, falarm=1.13, confusion=3.28
#Eval for threshold 0.7 DER=6.75, miss=2.86, falarm=1.01, confusion=2.88
#Eval for threshold 0.8 DER=9.02, miss=6.05, falarm=0.88, confusion=2.09
#Eval for threshold 0.9 DER=14.07, miss=12.29, falarm=0.73, confusion=1.05

# test of magicdata-ramc,collar=0.25
#Eval for threshold 0.2 DER=11.42, miss=0.03, falarm=10.17, confusion=1.22
#Eval for threshold 0.3 DER=9.24, miss=0.11, falarm=7.47, confusion=1.67
#Eval for threshold 0.35 DER=8.30, miss=0.22, falarm=6.10, confusion=1.98
#Eval for threshold 0.4 DER=7.37, miss=0.49, falarm=4.48, confusion=2.40
#Eval for threshold 0.45 DER=6.77, miss=1.20, falarm=2.97, confusion=2.60
#Eval for threshold 0.5 DER=6.99, miss=2.32, falarm=2.30, confusion=2.37 as report
#Eval for threshold 0.55 DER=7.48, miss=3.40, falarm=2.02, confusion=2.06
#Eval for threshold 0.6 DER=7.96, miss=4.36, falarm=1.86, confusion=1.75
#Eval for threshold 0.7 DER=9.17, miss=6.22, falarm=1.63, confusion=1.32
#Eval for threshold 0.8 DER=11.01, miss=8.65, falarm=1.39, confusion=0.97
#Eval for threshold 0.9 DER=15.05, miss=13.38, falarm=1.10, confusion=0.57


# cssd_testset of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=14.37, miss=0.63, falarm=13.32, confusion=0.42
#Eval for threshold 0.3 DER=9.71, miss=1.01, falarm=7.81, confusion=0.90
#Eval for threshold 0.35 DER=8.08, miss=1.26, falarm=5.58, confusion=1.24
#Eval for threshold 0.4 DER=6.84, miss=1.61, falarm=3.62, confusion=1.61 
#Eval for threshold 0.45 DER=6.24, miss=2.26, falarm=2.13, confusion=1.85 as report
#Eval for threshold 0.5 DER=6.31, miss=3.33, falarm=1.16, confusion=1.83
#Eval for threshold 0.55 DER=7.00, miss=4.81, falarm=0.65, confusion=1.54
#Eval for threshold 0.6 DER=8.16, miss=6.55, falarm=0.40, confusion=1.22
#Eval for threshold 0.7 DER=11.61, miss=10.75, falarm=0.20, confusion=0.66
#Eval for threshold 0.8 DER=16.70, miss=16.31, falarm=0.10, confusion=0.28
#Eval for threshold 0.9 DER=26.57, miss=26.41, falarm=0.04, confusion=0.12

if [ ${stage} -le 18 ] && [ ${stop_stage} -ge 18 ];then
   echo "compute CDER for magicdata-ramc"
   threshold="0.2 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.7 0.8 0.9"
   predict_rttm_dir=/data1/home/maduo/exp/speaker_diarization/ts_vad2_streaming/magicdata-ramc_ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr1e4_single_backend_transformer_multi_backend_transformer_rs_len6_shift0.8_streaming/magicdata-ramc_collar0.0_decoding_chunk_size25_num_decoding_left_chunks-1_simulate_streamingfalse_
   oracle_rttm_dir=/data1/home/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
   infer_sets="dev test cssd_testset"
   for name in $infer_sets;do
    for thres in $threshold;do
     echo "currently, compute $name set in $thres threshold mode"
     python3 cder/score.py -s $predict_rttm_dir/$name/res_rttm_${thres}  -r $oracle_rttm_dir/$name/rttm_debug_nog0
    done
   done
fi

#grep -r Avg logs/run_ts_vad2_streaming_hltsz_4090_stage16-18.log
# dev of magicdata-ramc
#Avg CDER : 0.799
#Avg CDER : 0.489
#Avg CDER : 0.393
#Avg CDER : 0.313
#Avg CDER : 0.249
#Avg CDER : 0.197
#Avg CDER : 0.168
#Avg CDER : 0.145
#Avg CDER : 0.113
#Avg CDER : 0.117
#Avg CDER : 0.106
#
# test of magicdata-ramc
#Avg CDER : 0.520
#Avg CDER : 0.452
#Avg CDER : 0.417
#Avg CDER : 0.375
#Avg CDER : 0.290
#Avg CDER : 0.185
#Avg CDER : 0.147
#Avg CDER : 0.130
#Avg CDER : 0.110
#Avg CDER : Error!
#Avg CDER : Error!
#
# cssd_testset of magicdata-ram
#Avg CDER : 0.364
#Avg CDER : 0.292
#Avg CDER : 0.258
#Avg CDER : 0.221
#Avg CDER : 0.180
#Avg CDER : 0.147
#Avg CDER : 0.123
#Avg CDER : 0.111
#Avg CDER : 0.096
#Avg CDER : 0.104
#Avg CDER : 0.152
#
#
#
#
if [ ${stage} -le 19 ] && [ ${stop_stage} -ge 19 ];then
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
    exp_dir=/data1/home/maduo/exp/speaker_diarization/ts_vad2_streaming/magicdata-ramc_ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr5e5_single_backend_mamba2_multi_backend_transformer_rs_len6_shift0.8_streaming
    data_dir="/data1/home/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format" # oracle target audio , mix audio and labels path
    rs_len=6
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
    --lr 5e-5\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --single-backend-type $single_backend_type\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --exp-dir $exp_dir\
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --num-transformer-layer $num_transformer_layer
fi

if [ ${stage} -le 20 ] && [ ${stop_stage} -ge 20 ];then
 exp_dir=/data1/home/maduo/exp/speaker_diarization/ts_vad2_streaming/magicdata-ramc_ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr5e5_single_backend_mamba2_multi_backend_transformer_rs_len6_shift0.8_streaming
 model_file=$exp_dir/best-valid-der.pt
 #model_file=$exp_dir/epoch-1.pt
 rs_len=6
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
    --single-backend-type $single_backend_type\
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
#grep -r Eval logs/run_ts_vad2_streaming_hltsz_4090_stage19-21.log
# dev of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=16.68, miss=0.21, falarm=14.02, confusion=2.45
#Eval for threshold 0.3 DER=13.53, miss=0.38, falarm=9.75, confusion=3.40
#Eval for threshold 0.35 DER=12.72, miss=0.56, falarm=8.47, confusion=3.69
#Eval for threshold 0.4 DER=12.19, miss=0.80, falarm=7.56, confusion=3.84
#Eval for threshold 0.45 DER=11.87, miss=1.11, falarm=6.87, confusion=3.89
#Eval for threshold 0.5 DER=11.65, miss=1.54, falarm=6.30, confusion=3.80 as report
#Eval for threshold 0.55 DER=11.59, miss=2.08, falarm=5.84, confusion=3.67
#Eval for threshold 0.6 DER=11.62, miss=2.71, falarm=5.43, confusion=3.47
#Eval for threshold 0.7 DER=12.17, miss=4.60, falarm=4.61, confusion=2.95
#Eval for threshold 0.8 DER=14.08, miss=8.27, falarm=3.72, confusion=2.10
#Eval for threshold 0.9 DER=18.30, miss=14.47, falarm=2.58, confusion=1.25

# test of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=18.58, miss=0.24, falarm=16.62, confusion=1.72
#Eval for threshold 0.3 DER=15.88, miss=0.51, falarm=13.16, confusion=2.21
#Eval for threshold 0.35 DER=14.85, miss=0.75, falarm=11.63, confusion=2.47
#Eval for threshold 0.4 DER=13.77, miss=1.09, falarm=9.80, confusion=2.88
#Eval for threshold 0.45 DER=12.68, miss=1.76, falarm=7.54, confusion=3.39 as report
#Eval for threshold 0.5 DER=12.84, miss=3.50, falarm=6.41, confusion=2.94
#Eval for threshold 0.55 DER=13.28, miss=4.96, falarm=5.86, confusion=2.45
#Eval for threshold 0.6 DER=13.70, miss=6.21, falarm=5.41, confusion=2.08
#Eval for threshold 0.7 DER=14.75, miss=8.61, falarm=4.54, confusion=1.60
#Eval for threshold 0.8 DER=16.59, miss=11.83, falarm=3.60, confusion=1.16
#Eval for threshold 0.9 DER=21.00, miss=17.78, falarm=2.51, confusion=0.71

# cssd_testset of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=26.95, miss=2.85, falarm=22.04, confusion=2.06
#Eval for threshold 0.3 DER=22.34, miss=4.00, falarm=15.38, confusion=2.96
#Eval for threshold 0.35 DER=20.90, miss=4.77, falarm=12.79, confusion=3.34
#Eval for threshold 0.4 DER=20.06, miss=5.87, falarm=10.66, confusion=3.53
#Eval for threshold 0.45 DER=19.93, miss=7.39, falarm=9.21, confusion=3.32 as report
#Eval for threshold 0.5 DER=20.29, miss=9.28, falarm=8.16, confusion=2.85
#Eval for threshold 0.55 DER=20.96, miss=11.28, falarm=7.32, confusion=2.35
#Eval for threshold 0.6 DER=21.78, miss=13.38, falarm=6.51, confusion=1.89
#Eval for threshold 0.7 DER=24.00, miss=17.97, falarm=4.87, confusion=1.17
#Eval for threshold 0.8 DER=27.70, miss=23.79, falarm=3.27, confusion=0.65
#Eval for threshold 0.9 DER=36.05, miss=34.15, falarm=1.66, confusion=0.25

# dev of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=8.09, miss=0.04, falarm=5.97, confusion=2.07
#Eval for threshold 0.3 DER=5.92, miss=0.07, falarm=2.90, confusion=2.95
#Eval for threshold 0.35 DER=5.47, miss=0.11, falarm=2.16, confusion=3.20
#Eval for threshold 0.4 DER=5.27, miss=0.20, falarm=1.74, confusion=3.33
#Eval for threshold 0.45 DER=5.17, miss=0.31, falarm=1.47, confusion=3.38
#Eval for threshold 0.5 DER=5.15, miss=0.48, falarm=1.32, confusion=3.35 as report
#Eval for threshold 0.55 DER=5.20, miss=0.70, falarm=1.22, confusion=3.28
#Eval for threshold 0.6 DER=5.29, miss=0.99, falarm=1.14, confusion=3.16
#Eval for threshold 0.7 DER=5.86, miss=2.05, falarm=1.02, confusion=2.79
#Eval for threshold 0.8 DER=7.76, miss=4.84, falarm=0.91, confusion=2.02
#Eval for threshold 0.9 DER=11.67, miss=9.70, falarm=0.74, confusion=1.23

# test of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=9.80, miss=0.04, falarm=8.50, confusion=1.27
#Eval for threshold 0.3 DER=8.31, miss=0.11, falarm=6.64, confusion=1.57
#Eval for threshold 0.35 DER=7.77, miss=0.19, falarm=5.83, confusion=1.74
#Eval for threshold 0.4 DER=7.09, miss=0.33, falarm=4.68, confusion=2.09
#Eval for threshold 0.45 DER=6.27, miss=0.67, falarm=2.94, confusion=2.66 as report
#Eval for threshold 0.5 DER=6.58, miss=2.08, falarm=2.20, confusion=2.29
#Eval for threshold 0.55 DER=7.12, miss=3.21, falarm=2.02, confusion=1.89
#Eval for threshold 0.6 DER=7.57, miss=4.06, falarm=1.90, confusion=1.62
#Eval for threshold 0.7 DER=8.54, miss=5.58, falarm=1.67, confusion=1.29
#Eval for threshold 0.8 DER=10.07, miss=7.66, falarm=1.40, confusion=1.01
#Eval for threshold 0.9 DER=13.73, miss=11.97, falarm=1.09, confusion=0.67

# cssd_testset of magicdata-ramc ,collar=0.25
#Eval for threshold 0.2 DER=8.39, miss=0.76, falarm=7.04, confusion=0.60
#Eval for threshold 0.3 DER=5.86, miss=1.17, falarm=3.58, confusion=1.11
#Eval for threshold 0.35 DER=5.08, miss=1.43, falarm=2.26, confusion=1.39
#Eval for threshold 0.4 DER=4.70, miss=1.86, falarm=1.26, confusion=1.58 
#Eval for threshold 0.45 DER=4.84, miss=2.64, falarm=0.67, confusion=1.53 as report
#Eval for threshold 0.5 DER=5.42, miss=3.69, falarm=0.46, confusion=1.27
#Eval for threshold 0.55 DER=6.19, miss=4.85, falarm=0.36, confusion=0.98
#Eval for threshold 0.6 DER=7.11, miss=6.09, falarm=0.28, confusion=0.74
#Eval for threshold 0.7 DER=9.40, miss=8.84, falarm=0.16, confusion=0.40
#Eval for threshold 0.8 DER=12.79, miss=12.50, falarm=0.08, confusion=0.21
#Eval for threshold 0.9 DER=20.44, miss=20.30, falarm=0.03, confusion=0.11


if [ ${stage} -le 21 ] && [ ${stop_stage} -ge 21 ];then
   echo "compute CDER for magicdata-ramc"
   threshold="0.2 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.7 0.8 0.9"
   predict_rttm_dir=/data1/home/maduo/exp/speaker_diarization/ts_vad2_streaming/magicdata-ramc_ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr5e5_single_backend_mamba2_multi_backend_transformer_rs_len6_shift0.8_streaming/magicdata-ramc_collar0.0_decoding_chunk_size25_num_decoding_left_chunks-1_simulate_streamingfalse_
   oracle_rttm_dir=/data1/home/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
   infer_sets="dev test cssd_testset"
   for name in $infer_sets;do
    for thres in $threshold;do
     echo "currently, compute $name set in $thres threshold mode"
     python3 cder/score.py -s $predict_rttm_dir/$name/res_rttm_${thres}  -r $oracle_rttm_dir/$name/rttm_debug_nog0
    done
   done
fi

#grep -r Avg logs/run_ts_vad2_streaming_hltsz_4090_stage19-21.log
# dev of magicdata-ramc
#Avg CDER : 0.689
#Avg CDER : 0.513
#Avg CDER : 0.373
#Avg CDER : 0.278
#Avg CDER : 0.197
#Avg CDER : 0.155
#Avg CDER : 0.130
#Avg CDER : 0.117
#Avg CDER : 0.103
#Avg CDER : 0.100
#Avg CDER : 0.113
#
#test of magicdata-ramc
#Avg CDER : 0.409
#Avg CDER : 0.331
#Avg CDER : 0.304
#Avg CDER : 0.284
#Avg CDER : 0.240
#Avg CDER : 0.148
#Avg CDER : 0.128
#Avg CDER : 0.126
#Avg CDER : 0.105
#Avg CDER : Error!
#Avg CDER : Error!
#
#cssd_testset of magicdata-ramc
#Avg CDER : 0.258
#Avg CDER : 0.199
#Avg CDER : 0.172
#Avg CDER : 0.147
#Avg CDER : 0.123
#Avg CDER : 0.109
#Avg CDER : 0.103
#Avg CDER : 0.097
#Avg CDER : 0.092
#Avg CDER : 0.085
#Avg CDER : 0.151
#
#
#
#
#
if [ ${stage} -le 22 ] && [ ${stop_stage} -ge 22 ];then
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
    exp_dir=/data1/home/maduo/exp/speaker_diarization/ts_vad2_streaming/magicdata-ramc_ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr5e5_single_backend_mamba2_multi_backend_transformer_rs_len6_shift2_streaming
    data_dir="/data1/home/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format" # oracle target audio , mix audio and labels path
    rs_len=6
    segment_shift=2
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
    --lr 5e-5\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --single-backend-type $single_backend_type\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --exp-dir $exp_dir\
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --num-transformer-layer $num_transformer_layer
fi

if [ ${stage} -le 23 ] && [ ${stop_stage} -ge 23 ];then
 exp_dir=/data1/home/maduo/exp/speaker_diarization/ts_vad2_streaming/magicdata-ramc_ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr5e5_single_backend_mamba2_multi_backend_transformer_rs_len6_shift2_streaming
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
    --single-backend-type $single_backend_type\
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

#grep -r Eval logs/run_ts_vad2_streaming_hltsz_4090_stage22-24.log
#
# dev of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=16.52, miss=0.22, falarm=13.79, confusion=2.51
#Eval for threshold 0.3 DER=13.59, miss=0.41, falarm=9.91, confusion=3.27
#Eval for threshold 0.35 DER=12.76, miss=0.57, falarm=8.64, confusion=3.56
#Eval for threshold 0.4 DER=12.16, miss=0.80, falarm=7.61, confusion=3.76
#Eval for threshold 0.45 DER=11.80, miss=1.12, falarm=6.85, confusion=3.84
#Eval for threshold 0.5 DER=11.57, miss=1.53, falarm=6.27, confusion=3.77
#Eval for threshold 0.55 DER=11.53, miss=2.07, falarm=5.82, confusion=3.64
#Eval for threshold 0.6 DER=11.55, miss=2.69, falarm=5.39, confusion=3.47
#Eval for threshold 0.7 DER=12.06, miss=4.52, falarm=4.58, confusion=2.97
#Eval for threshold 0.8 DER=13.74, miss=7.82, falarm=3.69, confusion=2.23
#Eval for threshold 0.9 DER=17.60, miss=13.40, falarm=2.64, confusion=1.56

# test of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=17.90, miss=0.25, falarm=15.89, confusion=1.76
#Eval for threshold 0.3 DER=15.56, miss=0.54, falarm=12.85, confusion=2.17
#Eval for threshold 0.35 DER=14.72, miss=0.76, falarm=11.61, confusion=2.36
#Eval for threshold 0.4 DER=13.87, miss=1.11, falarm=10.17, confusion=2.59
#Eval for threshold 0.45 DER=12.66, miss=1.63, falarm=7.99, confusion=3.05
#Eval for threshold 0.5 DER=12.49, miss=3.14, falarm=6.45, confusion=2.90
#Eval for threshold 0.55 DER=13.13, miss=4.95, falarm=5.86, confusion=2.32
#Eval for threshold 0.6 DER=13.48, miss=6.06, falarm=5.39, confusion=2.02
#Eval for threshold 0.7 DER=14.34, miss=8.15, falarm=4.55, confusion=1.64
#Eval for threshold 0.8 DER=15.99, miss=11.13, falarm=3.61, confusion=1.25
#Eval for threshold 0.9 DER=20.12, miss=16.83, falarm=2.51, confusion=0.79

# cssd_testset of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=26.64, miss=2.85, falarm=21.74, confusion=2.04
#Eval for threshold 0.3 DER=22.15, miss=3.92, falarm=15.38, confusion=2.85
#Eval for threshold 0.35 DER=20.73, miss=4.58, falarm=12.93, confusion=3.21
#Eval for threshold 0.4 DER=19.77, miss=5.49, falarm=10.83, confusion=3.45
#Eval for threshold 0.45 DER=19.42, miss=6.81, falarm=9.27, confusion=3.34
#Eval for threshold 0.5 DER=19.67, miss=8.58, falarm=8.21, confusion=2.88
#Eval for threshold 0.55 DER=20.18, miss=10.46, falarm=7.32, confusion=2.39
#Eval for threshold 0.6 DER=20.82, miss=12.41, falarm=6.48, confusion=1.93
#Eval for threshold 0.7 DER=22.88, miss=16.83, falarm=4.82, confusion=1.23
#Eval for threshold 0.8 DER=26.74, miss=22.87, falarm=3.21, confusion=0.66
#Eval for threshold 0.9 DER=35.19, miss=33.27, falarm=1.66, confusion=0.26

# dev of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=7.97, miss=0.05, falarm=5.81, confusion=2.11
#Eval for threshold 0.3 DER=6.04, miss=0.07, falarm=3.17, confusion=2.80
#Eval for threshold 0.35 DER=5.56, miss=0.12, falarm=2.41, confusion=3.04
#Eval for threshold 0.4 DER=5.30, miss=0.19, falarm=1.86, confusion=3.25
#Eval for threshold 0.45 DER=5.20, miss=0.32, falarm=1.55, confusion=3.33
#Eval for threshold 0.5 DER=5.16, miss=0.47, falarm=1.37, confusion=3.32 as report 
#Eval for threshold 0.55 DER=5.19, miss=0.68, falarm=1.25, confusion=3.26
#Eval for threshold 0.6 DER=5.31, miss=0.98, falarm=1.16, confusion=3.17
#Eval for threshold 0.7 DER=5.86, miss=2.05, falarm=1.03, confusion=2.78
#Eval for threshold 0.8 DER=7.46, miss=4.41, falarm=0.91, confusion=2.14
#Eval for threshold 0.9 DER=11.02, miss=8.69, falarm=0.76, confusion=1.56

# test of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=9.22, miss=0.04, falarm=7.91, confusion=1.28
#Eval for threshold 0.3 DER=7.98, miss=0.11, falarm=6.35, confusion=1.52
#Eval for threshold 0.35 DER=7.60, miss=0.18, falarm=5.78, confusion=1.64
#Eval for threshold 0.4 DER=7.17, miss=0.34, falarm=5.02, confusion=1.81
#Eval for threshold 0.45 DER=6.28, miss=0.57, falarm=3.44, confusion=2.28
#Eval for threshold 0.5 DER=6.22, miss=1.71, falarm=2.28, confusion=2.24 as report
#Eval for threshold 0.55 DER=7.00, miss=3.22, falarm=2.07, confusion=1.72
#Eval for threshold 0.6 DER=7.40, miss=3.96, falarm=1.94, confusion=1.51
#Eval for threshold 0.7 DER=8.13, miss=5.12, falarm=1.71, confusion=1.31
#Eval for threshold 0.8 DER=9.43, miss=6.91, falarm=1.44, confusion=1.09
#Eval for threshold 0.9 DER=12.86, miss=11.02, falarm=1.10, confusion=0.74

# cssd_testset of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=8.15, miss=0.78, falarm=6.71, confusion=0.66
#Eval for threshold 0.3 DER=5.64, miss=1.12, falarm=3.45, confusion=1.07
#Eval for threshold 0.35 DER=4.91, miss=1.34, falarm=2.28, confusion=1.29
#Eval for threshold 0.4 DER=4.50, miss=1.67, falarm=1.30, confusion=1.53
#Eval for threshold 0.45 DER=4.49, miss=2.24, falarm=0.70, confusion=1.55 as report
#Eval for threshold 0.5 DER=4.95, miss=3.19, falarm=0.48, confusion=1.27
#Eval for threshold 0.55 DER=5.56, miss=4.17, falarm=0.37, confusion=1.02
#Eval for threshold 0.6 DER=6.32, miss=5.23, falarm=0.28, confusion=0.81
#Eval for threshold 0.7 DER=8.43, miss=7.79, falarm=0.15, confusion=0.49
#Eval for threshold 0.8 DER=11.98, miss=11.64, falarm=0.07, confusion=0.27
#Eval for threshold 0.9 DER=19.98, miss=19.82, falarm=0.03, confusion=0.13


if [ ${stage} -le 24 ] && [ ${stop_stage} -ge 24 ];then
   echo "compute CDER for magicdata-ramc"
   threshold="0.2 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.7 0.8 0.9"
   predict_rttm_dir=/data1/home/maduo/exp/speaker_diarization/ts_vad2_streaming/magicdata-ramc_ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr5e5_single_backend_mamba2_multi_backend_transformer_rs_len6_shift2_streaming/magicdata-ramc_collar0.0_decoding_chunk_size25_num_decoding_left_chunks-1_simulate_streamingfalse_
   oracle_rttm_dir=/data1/home/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
   infer_sets="dev test cssd_testset"
   for name in $infer_sets;do
    for thres in $threshold;do
     echo "currently, compute $name set in $thres threshold mode"
     python3 cder/score.py -s $predict_rttm_dir/$name/res_rttm_${thres}  -r $oracle_rttm_dir/$name/rttm_debug_nog0
    done
   done
fi
# grep -r Avg logs/run_ts_vad2_streaming_hltsz_4090_stage22-24.log
# dev of magicdata-ramc
#Avg CDER : 0.606
#Avg CDER : 0.488
#Avg CDER : 0.387
#Avg CDER : 0.306
#Avg CDER : 0.227
#Avg CDER : 0.184
#Avg CDER : 0.140
#Avg CDER : 0.115
#Avg CDER : 0.103
#Avg CDER : 0.098
#Avg CDER : 0.096
#
# test of magicdata-ramc
#Avg CDER : 0.384
#Avg CDER : 0.315
#Avg CDER : 0.293
#Avg CDER : 0.271
#Avg CDER : 0.240
#Avg CDER : 0.153
#Avg CDER : 0.119
#Avg CDER : 0.119
#Avg CDER : 0.107
#Avg CDER : Error!
#Avg CDER : Error!
#
#cssd_testset of magicdata-ramc
#Avg CDER : 0.267
#Avg CDER : 0.201
#Avg CDER : 0.178
#Avg CDER : 0.153
#Avg CDER : 0.126
#Avg CDER : 0.111
#Avg CDER : 0.105
#Avg CDER : 0.102
#Avg CDER : 0.095
#Avg CDER : 0.092
#Avg CDER : 0.105
#
#
#
#
if [ ${stage} -le 25 ] && [ ${stop_stage} -ge 25 ];then
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
    exp_dir=/data1/home/maduo/exp/speaker_diarization/ts_vad2_streaming/magicdata-ramc_ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr5e5_single_backend_mamba2_multi_backend_transformer_rs_len6_shift0.4_streaming
    data_dir="/data1/home/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format" # oracle target audio , mix audio and labels path
    rs_len=6
    segment_shift=0.4
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
    --lr 5e-5\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --single-backend-type $single_backend_type\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --exp-dir $exp_dir\
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --num-transformer-layer $num_transformer_layer
fi

if [ ${stage} -le 26 ] && [ ${stop_stage} -ge 26 ];then
 exp_dir=/data1/home/maduo/exp/speaker_diarization/ts_vad2_streaming/magicdata-ramc_ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr5e5_single_backend_mamba2_multi_backend_transformer_rs_len6_shift0.4_streaming
 model_file=$exp_dir/best-valid-der.pt
 #model_file=$exp_dir/epoch-1.pt
 rs_len=6
 segment_shift=0.4
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
    --single-backend-type $single_backend_type\
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

if [ ${stage} -le 27 ] && [ ${stop_stage} -ge 27 ];then
   echo "compute CDER for magicdata-ramc"
   threshold="0.2 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.7 0.8 0.9"
   predict_rttm_dir=/data1/home/maduo/exp/speaker_diarization/ts_vad2_streaming/magicdata-ramc_ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr5e5_single_backend_mamba2_multi_backend_transformer_rs_len6_shift0.4_streaming/magicdata-ramc_collar0.0_decoding_chunk_size25_num_decoding_left_chunks-1_simulate_streamingfalse_
   oracle_rttm_dir=/data1/home/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
   infer_sets="dev test cssd_testset"
   for name in $infer_sets;do
    for thres in $threshold;do
     echo "currently, compute $name set in $thres threshold mode"
     python3 cder/score.py -s $predict_rttm_dir/$name/res_rttm_${thres}  -r $oracle_rttm_dir/$name/rttm_debug_nog0
    done
   done
fi




## compared with stage 19-21, stage30-32 will unidirectional mamba2, Avoid leaking information on the left
if [ ${stage} -le 30 ] && [ ${stop_stage} -ge 30 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is cam++ 200k speaker model
    #  oracle target speaker embedding is from cam++ pretrain model
    # checkpoint is from https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/files
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/share/workspace/shared_datasets/speechdata/14_musan
    rir_path=/share/workspace/shared_datasets/speechdata/21_RIRS_NOISES/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/share/workspace/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
    dataset_name="magicdata-ramc" # dataset name

    # for loading speaker embedding file
    spk_path=/share/workspace/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/share/workspace/maduo/exp/speaker_diarization/ts_vad2_streaming/magicdata-ramc_ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr5e5_single_backend_mamba2_unidirectional_multi_backend_transformer_rs_len6_shift0.8_streaming
    data_dir="/share/workspace/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format" # oracle target audio , mix audio and labels path
    rs_len=6
    segment_shift=0.8
    single_backend_type="mamba2_unidirectional"
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
    --single-backend-type $single_backend_type\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --exp-dir $exp_dir\
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --num-transformer-layer $num_transformer_layer
fi

if [ ${stage} -le 31 ] && [ ${stop_stage} -ge 31 ];then
 exp_dir=/share/workspace/maduo/exp/speaker_diarization/ts_vad2_streaming/magicdata-ramc_ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr5e5_single_backend_mamba2_unidirectional_multi_backend_transformer_rs_len6_shift0.8_streaming
 
 model_file=$exp_dir/best-valid-der.pt
 #model_file=$exp_dir/epoch-1.pt
 rs_len=6
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

 single_backend_type="mamba2_unidirectional"
 #multi_backend_type="transformer"
 #d_state=256
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 #infer_sets="Eval Test"
 #infer_sets="Test"
 infer_sets="dev test cssd_testset"
 rttm_dir=/share/workspace/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0
 #collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/share/workspace/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 dataset_name="magicdata-ramc" # dataset name
 # for loading speaker embedding file
 spk_path=/share/workspace/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 #data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
 data_dir="/share/workspace/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format"
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
    --single-backend-type $single_backend_type\
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
# grep -r Eval logs/run_ts_vad2_streaming_hltsz_4090_stage30-32.log
# dev of magicdata-ramc. collar=0.0
#Eval for threshold 0.2 DER=16.57, miss=0.22, falarm=13.80, confusion=2.55
#Eval for threshold 0.3 DER=13.62, miss=0.42, falarm=9.74, confusion=3.46
#Eval for threshold 0.35 DER=12.70, miss=0.59, falarm=8.41, confusion=3.71
#Eval for threshold 0.4 DER=12.05, miss=0.84, falarm=7.39, confusion=3.82
#Eval for threshold 0.45 DER=11.67, miss=1.20, falarm=6.63, confusion=3.84
#Eval for threshold 0.5 DER=11.48, miss=1.68, falarm=6.02, confusion=3.78
#Eval for threshold 0.55 DER=11.46, miss=2.25, falarm=5.56, confusion=3.65
#Eval for threshold 0.6 DER=11.58, miss=2.99, falarm=5.15, confusion=3.45
#Eval for threshold 0.7 DER=12.23, miss=4.94, falarm=4.33, confusion=2.95
#Eval for threshold 0.8 DER=14.08, miss=8.33, falarm=3.59, confusion=2.16
#Eval for threshold 0.9 DER=18.74, miss=15.04, falarm=2.60, confusion=1.10

# test of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=18.90, miss=0.26, falarm=17.01, confusion=1.63
#Eval for threshold 0.3 DER=15.96, miss=0.53, falarm=13.29, confusion=2.14
#Eval for threshold 0.35 DER=14.95, miss=0.78, falarm=11.81, confusion=2.36
#Eval for threshold 0.4 DER=14.10, miss=1.15, falarm=10.38, confusion=2.57
#Eval for threshold 0.45 DER=13.12, miss=1.70, falarm=8.36, confusion=3.06
#Eval for threshold 0.5 DER=12.50, miss=2.94, falarm=6.26, confusion=3.29
#Eval for threshold 0.55 DER=13.27, miss=5.25, falarm=5.63, confusion=2.39
#Eval for threshold 0.6 DER=13.68, miss=6.52, falarm=5.17, confusion=2.00
#Eval for threshold 0.7 DER=14.63, miss=8.70, falarm=4.36, confusion=1.56
#Eval for threshold 0.8 DER=16.39, miss=11.71, falarm=3.54, confusion=1.14
#Eval for threshold 0.9 DER=20.45, miss=17.16, falarm=2.56, confusion=0.72

# cssd_testset of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=27.20, miss=2.77, falarm=22.76, confusion=1.67
#Eval for threshold 0.3 DER=22.25, miss=3.86, falarm=15.91, confusion=2.48
#Eval for threshold 0.35 DER=20.64, miss=4.55, falarm=13.25, confusion=2.84
#Eval for threshold 0.4 DER=19.49, miss=5.39, falarm=11.03, confusion=3.07
#Eval for threshold 0.45 DER=18.95, miss=6.56, falarm=9.31, confusion=3.08
#Eval for threshold 0.5 DER=19.03, miss=8.20, falarm=8.09, confusion=2.75
#Eval for threshold 0.55 DER=19.52, miss=10.07, falarm=7.19, confusion=2.26
#Eval for threshold 0.6 DER=20.24, miss=12.08, falarm=6.36, confusion=1.80
#Eval for threshold 0.7 DER=22.50, miss=16.62, falarm=4.79, confusion=1.09
#Eval for threshold 0.8 DER=26.53, miss=22.68, falarm=3.26, confusion=0.58
#Eval for threshold 0.9 DER=35.11, miss=33.15, falarm=1.72, confusion=0.24

# dev of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=7.95, miss=0.04, falarm=5.66, confusion=2.24
#Eval for threshold 0.3 DER=6.08, miss=0.07, falarm=2.98, confusion=3.03
#Eval for threshold 0.35 DER=5.62, miss=0.11, falarm=2.25, confusion=3.26
#Eval for threshold 0.4 DER=5.34, miss=0.18, falarm=1.79, confusion=3.37
#Eval for threshold 0.45 DER=5.24, miss=0.33, falarm=1.51, confusion=3.40
#Eval for threshold 0.5 DER=5.22, miss=0.51, falarm=1.32, confusion=3.39
#Eval for threshold 0.55 DER=5.30, miss=0.75, falarm=1.23, confusion=3.32
#Eval for threshold 0.6 DER=5.48, miss=1.13, falarm=1.15, confusion=3.20
#Eval for threshold 0.7 DER=6.07, miss=2.23, falarm=1.00, confusion=2.85
#Eval for threshold 0.8 DER=7.79, miss=4.77, falarm=0.90, confusion=2.12
#Eval for threshold 0.9 DER=12.09, miss=10.24, falarm=0.76, confusion=1.09

# test of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=10.16, miss=0.03, falarm=8.92, confusion=1.20
#Eval for threshold 0.3 DER=8.44, miss=0.09, falarm=6.81, confusion=1.54
#Eval for threshold 0.35 DER=7.95, miss=0.18, falarm=6.09, confusion=1.69
#Eval for threshold 0.4 DER=7.55, miss=0.33, falarm=5.39, confusion=1.83
#Eval for threshold 0.45 DER=6.96, miss=0.59, falarm=4.07, confusion=2.30
#Eval for threshold 0.5 DER=6.40, miss=1.42, falarm=2.30, confusion=2.67
#Eval for threshold 0.55 DER=7.28, miss=3.40, falarm=2.05, confusion=1.83
#Eval for threshold 0.6 DER=7.70, miss=4.25, falarm=1.91, confusion=1.54
#Eval for threshold 0.7 DER=8.47, miss=5.53, falarm=1.68, confusion=1.27
#Eval for threshold 0.8 DER=9.93, miss=7.52, falarm=1.42, confusion=0.98
#Eval for threshold 0.9 DER=13.26, miss=11.49, falarm=1.11, confusion=0.67

# cssd_testset of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=8.44, miss=0.75, falarm=7.16, confusion=0.54
#Eval for threshold 0.3 DER=5.69, miss=1.09, falarm=3.64, confusion=0.96
#Eval for threshold 0.35 DER=4.92, miss=1.32, falarm=2.43, confusion=1.17
#Eval for threshold 0.4 DER=4.44, miss=1.60, falarm=1.48, confusion=1.36
#Eval for threshold 0.45 DER=4.32, miss=2.04, falarm=0.82, confusion=1.45
#Eval for threshold 0.5 DER=4.61, miss=2.82, falarm=0.48, confusion=1.30
#Eval for threshold 0.55 DER=5.23, miss=3.80, falarm=0.37, confusion=1.06
#Eval for threshold 0.6 DER=6.00, miss=4.89, falarm=0.29, confusion=0.82
#Eval for threshold 0.7 DER=8.21, miss=7.59, falarm=0.16, confusion=0.47
#Eval for threshold 0.8 DER=12.01, miss=11.69, falarm=0.08, confusion=0.23
#Eval for threshold 0.9 DER=20.01, miss=19.86, falarm=0.03, confusion=0.12

if [ ${stage} -le 32 ] && [ ${stop_stage} -ge 32 ];then
   echo "compute CDER for magicdata-ramc"
   threshold="0.2 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.7 0.8 0.9"
   predict_rttm_dir=/share/workspace/maduo/exp/speaker_diarization/ts_vad2_streaming/magicdata-ramc_ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr5e5_single_backend_mamba2_unidirectional_multi_backend_transformer_rs_len6_shift0.8_streaming/magicdata-ramc_collar0.0_decoding_chunk_size25_num_decoding_left_chunks-1_simulate_streamingfalse_
   oracle_rttm_dir=/share/workspace/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
   infer_sets="dev test cssd_testset"
   for name in $infer_sets;do
    for thres in $threshold;do
     echo "currently, compute $name set in $thres threshold mode"
     python3 cder/score.py -s $predict_rttm_dir/$name/res_rttm_${thres}  -r $oracle_rttm_dir/$name/rttm_debug_nog0
    done
   done
fi
#grep -r Avg logs/run_ts_vad2_streaming_hltsz_4090_stage32.log
# dev
#Avg CDER : 0.830
#Avg CDER : 0.519
#Avg CDER : 0.405
#Avg CDER : 0.318
#Avg CDER : 0.251
#Avg CDER : 0.189
#Avg CDER : 0.150
#Avg CDER : 0.132
#Avg CDER : 0.107
#Avg CDER : 0.098
#Avg CDER : 0.095
# test 
#Avg CDER : 0.411
#Avg CDER : 0.334
#Avg CDER : 0.301
#Avg CDER : 0.277
#Avg CDER : 0.241
#Avg CDER : 0.170
#Avg CDER : 0.124
#Avg CDER : 0.119
#Avg CDER : Error!
#Avg CDER : Error!
#Avg CDER : Error!
# cssd_testset
#Avg CDER : 0.291
#Avg CDER : 0.219
#Avg CDER : 0.188
#Avg CDER : 0.160
#Avg CDER : 0.134
#Avg CDER : 0.115
#Avg CDER : 0.106
#Avg CDER : 0.101
#Avg CDER : 0.095
#Avg CDER : 0.090
#Avg CDER : 0.094


if [ ${stage} -le 33 ] && [ ${stop_stage} -ge 33 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is cam++ 200k speaker model
    #  oracle target speaker embedding is from cam++ pretrain model
    # checkpoint is from https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/files
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/share/workspace/shared_datasets/speechdata/14_musan
    rir_path=/share/workspace/shared_datasets/speechdata/21_RIRS_NOISES/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/share/workspace/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
    dataset_name="magicdata-ramc" # dataset name

    # for loading speaker embedding file
    spk_path=/share/workspace/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/share/workspace/maduo/exp/speaker_diarization/ts_vad2_streaming/magicdata-ramc_ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr5e5_single_backend_mamba2_unidirectional_multi_backend_transformer_rs_len8_shift0.8_streaming
    data_dir="/share/workspace/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format" # oracle target audio , mix audio and labels path
    rs_len=8
    segment_shift=0.8
    single_backend_type="mamba2_unidirectional"
    num_transformer_layer=2
    CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15215 \
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
    --single-backend-type $single_backend_type\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --exp-dir $exp_dir\
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --num-transformer-layer $num_transformer_layer
fi

if [ ${stage} -le 34 ] && [ ${stop_stage} -ge 34 ];then
 exp_dir=/share/workspace/maduo/exp/speaker_diarization/ts_vad2_streaming/magicdata-ramc_ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr5e5_single_backend_mamba2_unidirectional_multi_backend_transformer_rs_len8_shift0.8_streaming

 model_file=$exp_dir/best-valid-der.pt
 #model_file=$exp_dir/epoch-1.pt
 rs_len=8
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

 single_backend_type="mamba2_unidirectional"
 #multi_backend_type="transformer"
 #d_state=256
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 #infer_sets="Eval Test"
 #infer_sets="Test"
 infer_sets="dev test cssd_testset"
 rttm_dir=/share/workspace/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0
 #collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/share/workspace/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 dataset_name="magicdata-ramc" # dataset name
 # for loading speaker embedding file
 spk_path=/share/workspace/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 #data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
 data_dir="/share/workspace/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format"
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
    --single-backend-type $single_backend_type\
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

# grep -r Eval logs/run_ts_vad2_streaming_hltsz_4090_stage33-35.log
# dev of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=16.30, miss=0.22, falarm=13.36, confusion=2.72
#Eval for threshold 0.3 DER=13.41, miss=0.43, falarm=9.42, confusion=3.56
#Eval for threshold 0.35 DER=12.59, miss=0.62, falarm=8.21, confusion=3.76
#Eval for threshold 0.4 DER=12.02, miss=0.90, falarm=7.30, confusion=3.82
#Eval for threshold 0.45 DER=11.60, miss=1.22, falarm=6.55, confusion=3.83
#Eval for threshold 0.5 DER=11.38, miss=1.63, falarm=5.96, confusion=3.79
#Eval for threshold 0.55 DER=11.33, miss=2.19, falarm=5.48, confusion=3.65
#Eval for threshold 0.6 DER=11.43, miss=2.83, falarm=5.10, confusion=3.50
#Eval for threshold 0.7 DER=12.01, miss=4.52, falarm=4.33, confusion=3.16
#Eval for threshold 0.8 DER=13.39, miss=7.29, falarm=3.57, confusion=2.53
#Eval for threshold 0.9 DER=18.12, miss=14.29, falarm=2.62, confusion=1.20

# test of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=18.91, miss=0.28, falarm=16.95, confusion=1.67
#Eval for threshold 0.3 DER=16.06, miss=0.61, falarm=13.22, confusion=2.23
#Eval for threshold 0.35 DER=15.08, miss=0.90, falarm=11.72, confusion=2.45
#Eval for threshold 0.4 DER=14.10, miss=1.34, falarm=10.05, confusion=2.70
#Eval for threshold 0.45 DER=12.79, miss=2.03, falarm=7.52, confusion=3.25
#Eval for threshold 0.5 DER=12.92, miss=3.90, falarm=6.09, confusion=2.93
#Eval for threshold 0.55 DER=13.50, miss=5.60, falarm=5.51, confusion=2.39
#Eval for threshold 0.6 DER=13.90, miss=6.75, falarm=5.07, confusion=2.08
#Eval for threshold 0.7 DER=14.95, miss=9.05, falarm=4.30, confusion=1.59
#Eval for threshold 0.8 DER=16.76, miss=12.10, falarm=3.51, confusion=1.15
#Eval for threshold 0.9 DER=20.80, miss=17.54, falarm=2.55, confusion=0.72

# cssd_testset of magicdata-ramc, collar=0.0
#Eval for threshold 0.2 DER=27.82, miss=2.73, falarm=23.23, confusion=1.87
#Eval for threshold 0.3 DER=22.37, miss=3.98, falarm=15.57, confusion=2.83
#Eval for threshold 0.35 DER=20.74, miss=4.78, falarm=12.77, confusion=3.19
#Eval for threshold 0.4 DER=19.71, miss=5.81, falarm=10.49, confusion=3.41
#Eval for threshold 0.45 DER=19.26, miss=7.21, falarm=8.73, confusion=3.32
#Eval for threshold 0.5 DER=19.51, miss=9.05, falarm=7.55, confusion=2.90
#Eval for threshold 0.55 DER=20.08, miss=11.08, falarm=6.62, confusion=2.37
#Eval for threshold 0.6 DER=20.95, miss=13.26, falarm=5.79, confusion=1.90
#Eval for threshold 0.7 DER=23.64, miss=18.27, falarm=4.26, confusion=1.10
#Eval for threshold 0.8 DER=28.10, miss=24.65, falarm=2.87, confusion=0.58
#Eval for threshold 0.9 DER=37.37, miss=35.61, falarm=1.54, confusion=0.22

# dev of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=7.83, miss=0.04, falarm=5.42, confusion=2.38
#Eval for threshold 0.3 DER=6.03, miss=0.07, falarm=2.81, confusion=3.15
#Eval for threshold 0.35 DER=5.63, miss=0.12, falarm=2.19, confusion=3.32
#Eval for threshold 0.4 DER=5.39, miss=0.21, falarm=1.80, confusion=3.37
#Eval for threshold 0.45 DER=5.26, miss=0.33, falarm=1.54, confusion=3.39
#Eval for threshold 0.5 DER=5.22, miss=0.48, falarm=1.36, confusion=3.38
#Eval for threshold 0.55 DER=5.27, miss=0.73, falarm=1.23, confusion=3.31
#Eval for threshold 0.6 DER=5.40, miss=1.01, falarm=1.14, confusion=3.25
#Eval for threshold 0.7 DER=5.92, miss=1.86, falarm=1.00, confusion=3.05
#Eval for threshold 0.8 DER=7.11, miss=3.68, falarm=0.90, confusion=2.54
#Eval for threshold 0.9 DER=11.45, miss=9.49, falarm=0.76, confusion=1.20

# test of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=10.29, miss=0.04, falarm=9.05, confusion=1.19
#Eval for threshold 0.3 DER=8.69, miss=0.12, falarm=7.00, confusion=1.56
#Eval for threshold 0.35 DER=8.18, miss=0.23, falarm=6.23, confusion=1.72
#Eval for threshold 0.4 DER=7.64, miss=0.43, falarm=5.28, confusion=1.93
#Eval for threshold 0.45 DER=6.61, miss=0.77, falarm=3.32, confusion=2.52
#Eval for threshold 0.5 DER=6.84, miss=2.26, falarm=2.25, confusion=2.32
#Eval for threshold 0.55 DER=7.51, miss=3.65, falarm=2.01, confusion=1.85
#Eval for threshold 0.6 DER=7.88, miss=4.39, falarm=1.86, confusion=1.64
#Eval for threshold 0.7 DER=8.74, miss=5.79, falarm=1.63, confusion=1.32
#Eval for threshold 0.8 DER=10.21, miss=7.82, falarm=1.39, confusion=1.01
#Eval for threshold 0.9 DER=13.55, miss=11.78, falarm=1.09, confusion=0.68

# cssd_testset of magicdata-ramc, collar=0.25
#Eval for threshold 0.2 DER=9.14, miss=0.73, falarm=7.83, confusion=0.57
#Eval for threshold 0.3 DER=6.06, miss=1.13, falarm=3.85, confusion=1.08
#Eval for threshold 0.35 DER=5.27, miss=1.38, falarm=2.56, confusion=1.33
#Eval for threshold 0.4 DER=4.83, miss=1.72, falarm=1.56, confusion=1.55
#Eval for threshold 0.45 DER=4.72, miss=2.28, falarm=0.83, confusion=1.61
#Eval for threshold 0.5 DER=5.10, miss=3.19, falarm=0.47, confusion=1.44
#Eval for threshold 0.55 DER=5.77, miss=4.27, falarm=0.34, confusion=1.16
#Eval for threshold 0.6 DER=6.65, miss=5.48, falarm=0.26, confusion=0.91
#Eval for threshold 0.7 DER=9.19, miss=8.53, falarm=0.14, confusion=0.51
#Eval for threshold 0.8 DER=13.35, miss=13.05, falarm=0.06, confusion=0.24
#Eval for threshold 0.9 DER=22.31, miss=22.17, falarm=0.03, confusion=0.11

if [ ${stage} -le 35 ] && [ ${stop_stage} -ge 35 ];then
   echo "compute CDER for magicdata-ramc"
   threshold="0.2 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.7 0.8 0.9"
   predict_rttm_dir=/share/workspace/maduo/exp/speaker_diarization/ts_vad2_streaming/magicdata-ramc_ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr5e5_single_backend_mamba2_unidirectional_multi_backend_transformer_rs_len8_shift0.8_streaming/magicdata-ramc_collar0.0_decoding_chunk_size25_num_decoding_left_chunks-1_simulate_streamingfalse_
   oracle_rttm_dir=/share/workspace/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
   infer_sets="dev test cssd_testset"
   for name in $infer_sets;do
    for thres in $threshold;do
     echo "currently, compute $name set in $thres threshold mode"
     python3 cder/score.py -s $predict_rttm_dir/$name/res_rttm_${thres}  -r $oracle_rttm_dir/$name/rttm_debug_nog0
    done
   done
fi

#grep -r Avg logs/run_ts_vad2_streaming_hltsz_4090_stage35.log
# dev 
#Avg CDER : 1.080
#Avg CDER : 0.700
#Avg CDER : 0.542
#Avg CDER : 0.414
#Avg CDER : 0.317
#Avg CDER : 0.251
#Avg CDER : 0.195
#Avg CDER : 0.161
#Avg CDER : 0.118
#Avg CDER : 0.105
#Avg CDER : 0.103

# test
#Avg CDER : 0.445
#Avg CDER : 0.363
#Avg CDER : 0.332
#Avg CDER : 0.310
#Avg CDER : 0.267
#Avg CDER : 0.175
#Avg CDER : 0.129
#Avg CDER : 0.124
#Avg CDER : Error!
#Avg CDER : Error!
#Avg CDER : Error!

# cssd_testset
#Avg CDER : 0.311
#Avg CDER : 0.231
#Avg CDER : 0.199
#Avg CDER : 0.167
#Avg CDER : 0.138
#Avg CDER : 0.118
#Avg CDER : 0.110
#Avg CDER : 0.103
#Avg CDER : 0.096
#Avg CDER : 0.092
#Avg CDER : 0.084

if [ ${stage} -le 36 ] && [ ${stop_stage} -ge 36 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is cam++ 200k speaker model
    #  oracle target speaker embedding is from cam++ pretrain model
    # checkpoint is from https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/files
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/share/workspace/shared_datasets/speechdata/14_musan
    rir_path=/share/workspace/shared_datasets/speechdata/21_RIRS_NOISES/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/share/workspace/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
    dataset_name="magicdata-ramc" # dataset name

    # for loading speaker embedding file
    spk_path=/share/workspace/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/share/workspace/maduo/exp/speaker_diarization/ts_vad2_streaming/magicdata-ramc_ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr1e5_single_backend_mamba2_unidirectional_multi_backend_transformer_rs_len16_shift0.8_streaming_epoch20
    data_dir="/share/workspace/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format" # oracle target audio , mix audio and labels path
    rs_len=16
    segment_shift=0.8
    single_backend_type="mamba2_unidirectional"
    num_transformer_layer=2
    CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15315 \
   ts_vad2_streaming/train_accelerate_ddp.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --start-batch 70500\
    --keep-last-k 10\
    --keep-last-epoch 10\
    --grad-clip false\
    --lr 1e-5\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --single-backend-type $single_backend_type\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --exp-dir $exp_dir\
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --num-transformer-layer $num_transformer_layer
fi
# note: lr=1e-4, epoch=14, loss  and der  are explode 
# you can see the below log
# grep -r Eval logs/run_ts_vad2_streaming_hltsz_4090_stage36-38_1.log
# grep -r Eval logs/run_ts_vad2_streaming_hltsz_4090_stage36-38.log
# lr=5e-5, 2025-07-19 06:39:09,552 (train_accelerate_ddp:679) INFO: [Train] - Epoch 17, batch_idx_train: 92762, num_updates: 83500, {'loss': nan, 'DER': 1.0, 'ACC': np.float64(0.7868553533912078), 'MI': 1.0, 'FA': 0.0, 'CF': 0.0}, batch size: 64, grad_norm: None, grad_scale: , lr: 3.4583333333333334e-06,



if [ ${stage} -le 37 ] && [ ${stop_stage} -ge 37 ];then
 exp_dir=/share/workspace/maduo/exp/speaker_diarization/ts_vad2_streaming/magicdata-ramc_ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr1e5_single_backend_mamba2_unidirectional_multi_backend_transformer_rs_len16_shift0.8_streaming_epoch20

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

 single_backend_type="mamba2_unidirectional"
 #multi_backend_type="transformer"
 #d_state=256
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 #infer_sets="Eval Test"
 #infer_sets="Test"
 infer_sets="dev test cssd_testset"
 rttm_dir=/share/workspace/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0
 #collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/share/workspace/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 dataset_name="magicdata-ramc" # dataset name
 # for loading speaker embedding file
 spk_path=/share/workspace/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 #data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
 data_dir="/share/workspace/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format"
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
    --single-backend-type $single_backend_type\
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
# grep -r Eval logs/run_ts_vad2_streaming_hltsz_4090_stage36-38_5.log
# dev , collar=0.0
# Eval for threshold 0.2 DER=17.35, miss=0.28, falarm=14.01, confusion=3.05
#Eval for threshold 0.3 DER=14.27, miss=0.57, falarm=9.84, confusion=3.86
#Eval for threshold 0.35 DER=13.39, miss=0.81, falarm=8.47, confusion=4.11
#Eval for threshold 0.4 DER=12.80, miss=1.15, falarm=7.42, confusion=4.22
#Eval for threshold 0.45 DER=12.50, miss=1.61, falarm=6.67, confusion=4.22
#Eval for threshold 0.5 DER=12.36, miss=2.20, falarm=6.06, confusion=4.09
#Eval for threshold 0.55 DER=12.38, miss=2.92, falarm=5.58, confusion=3.89
#Eval for threshold 0.6 DER=12.60, miss=3.81, falarm=5.17, confusion=3.62
#Eval for threshold 0.7 DER=13.64, miss=6.23, falarm=4.40, confusion=3.01
#Eval for threshold 0.8 DER=16.11, miss=10.32, falarm=3.62, confusion=2.17
#Eval for threshold 0.9 DER=21.74, miss=18.11, falarm=2.64, confusion=0.99

# test, collar=0.0
#Eval for threshold 0.2 DER=20.81, miss=0.32, falarm=18.65, confusion=1.84
#Eval for threshold 0.3 DER=17.19, miss=0.71, falarm=13.99, confusion=2.49
#Eval for threshold 0.35 DER=15.78, miss=1.05, falarm=11.93, confusion=2.80
#Eval for threshold 0.4 DER=14.29, miss=1.56, falarm=9.45, confusion=3.28
#Eval for threshold 0.45 DER=13.29, miss=2.55, falarm=7.15, confusion=3.60
#Eval for threshold 0.5 DER=13.54, miss=4.30, falarm=6.07, confusion=3.17
#Eval for threshold 0.55 DER=14.18, miss=6.09, falarm=5.49, confusion=2.60
#Eval for threshold 0.6 DER=14.88, miss=7.65, falarm=5.06, confusion=2.17
#Eval for threshold 0.7 DER=16.41, miss=10.53, falarm=4.26, confusion=1.63
#Eval for threshold 0.8 DER=18.77, miss=14.16, falarm=3.49, confusion=1.13
#Eval for threshold 0.9 DER=23.81, miss=20.70, falarm=2.54, confusion=0.57
# cssd_testset, collar=0.0
#Eval for threshold 0.2 DER=32.59, miss=2.50, falarm=28.09, confusion=2.00
#Eval for threshold 0.3 DER=25.38, miss=3.55, falarm=18.73, confusion=3.09
#Eval for threshold 0.35 DER=22.90, miss=4.24, falarm=14.99, confusion=3.67
#Eval for threshold 0.4 DER=21.19, miss=5.23, falarm=11.82, confusion=4.14
#Eval for threshold 0.45 DER=20.45, miss=6.76, falarm=9.47, confusion=4.21
#Eval for threshold 0.5 DER=20.79, miss=9.11, falarm=8.08, confusion=3.60
#Eval for threshold 0.55 DER=21.75, miss=11.82, falarm=7.15, confusion=2.78
#Eval for threshold 0.6 DER=23.10, miss=14.68, falarm=6.30, confusion=2.12
#Eval for threshold 0.7 DER=26.71, miss=20.80, falarm=4.70, confusion=1.21
#Eval for threshold 0.8 DER=32.12, miss=28.33, falarm=3.19, confusion=0.60
#Eval for threshold 0.9 DER=43.14, miss=41.36, falarm=1.58, confusion=0.20
# dev, collar=0.25
#Eval for threshold 0.2 DER=8.42, miss=0.06, falarm=5.89, confusion=2.48
#Eval for threshold 0.3 DER=6.52, miss=0.12, falarm=3.23, confusion=3.17
#Eval for threshold 0.35 DER=6.04, miss=0.18, falarm=2.49, confusion=3.36
#Eval for threshold 0.4 DER=5.77, miss=0.30, falarm=1.97, confusion=3.50
#Eval for threshold 0.45 DER=5.66, miss=0.50, falarm=1.63, confusion=3.53
#Eval for threshold 0.5 DER=5.65, miss=0.77, falarm=1.41, confusion=3.47 as report
#Eval for threshold 0.55 DER=5.77, miss=1.14, falarm=1.28, confusion=3.35
#Eval for threshold 0.6 DER=6.01, miss=1.65, falarm=1.18, confusion=3.18
#Eval for threshold 0.7 DER=7.03, miss=3.24, falarm=1.05, confusion=2.74
#Eval for threshold 0.8 DER=9.37, miss=6.41, falarm=0.92, confusion=2.03
#Eval for threshold 0.9 DER=14.84, miss=13.17, falarm=0.75, confusion=0.92

# test, collar=0.25
#Eval for threshold 0.2 DER=11.67, miss=0.05, falarm=10.43, confusion=1.19
#Eval for threshold 0.3 DER=9.47, miss=0.15, falarm=7.71, confusion=1.62
#Eval for threshold 0.35 DER=8.56, miss=0.29, falarm=6.42, confusion=1.85
#Eval for threshold 0.4 DER=7.47, miss=0.52, falarm=4.66, confusion=2.29
#Eval for threshold 0.45 DER=6.71, miss=1.12, falarm=2.92, confusion=2.67 as report
#Eval for threshold 0.5 DER=7.04, miss=2.44, falarm=2.25, confusion=2.35
#Eval for threshold 0.55 DER=7.70, miss=3.80, falarm=2.00, confusion=1.90
#Eval for threshold 0.6 DER=8.39, miss=4.94, falarm=1.87, confusion=1.59
#Eval for threshold 0.7 DER=9.78, miss=6.90, falarm=1.62, confusion=1.26
#Eval for threshold 0.8 DER=11.82, miss=9.51, falarm=1.39, confusion=0.92

# cssd_testset, collar=0.25
#Eval for threshold 0.9 DER=16.30, miss=14.71, falarm=1.11, confusion=0.48
#Eval for threshold 0.2 DER=12.77, miss=0.66, falarm=11.63, confusion=0.48
#Eval for threshold 0.3 DER=8.15, miss=0.99, falarm=6.15, confusion=1.00
#Eval for threshold 0.35 DER=6.66, miss=1.20, falarm=4.10, confusion=1.36
#Eval for threshold 0.4 DER=5.67, miss=1.53, falarm=2.39, confusion=1.76
#Eval for threshold 0.45 DER=5.30, miss=2.16, falarm=1.16, confusion=1.97 as report
#Eval for threshold 0.5 DER=5.80, miss=3.48, falarm=0.63, confusion=1.69
#Eval for threshold 0.55 DER=6.89, miss=5.22, falarm=0.50, confusion=1.18
#Eval for threshold 0.6 DER=8.32, miss=7.10, falarm=0.41, confusion=0.81
#Eval for threshold 0.7 DER=12.04, miss=11.41, falarm=0.26, confusion=0.38
#Eval for threshold 0.8 DER=17.58, miss=17.29, falarm=0.13, confusion=0.16
#Eval for threshold 0.9 DER=29.09, miss=28.96, falarm=0.05, confusion=0.08

if [ ${stage} -le 38 ] && [ ${stop_stage} -ge 38 ];then
   echo "compute CDER for magicdata-ramc"
   threshold="0.2 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.7 0.8 0.9"
   predict_rttm_dir=/share/workspace/maduo/exp/speaker_diarization/ts_vad2_streaming/magicdata-ramc_ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr1e5_single_backend_mamba2_unidirectional_multi_backend_transformer_rs_len16_shift0.8_streaming_epoch20/magicdata-ramc_collar0.0_decoding_chunk_size25_num_decoding_left_chunks-1_simulate_streamingfalse_
   oracle_rttm_dir=/share/workspace/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
   infer_sets="dev test cssd_testset"
   for name in $infer_sets;do
    for thres in $threshold;do
     echo "currently, compute $name set in $thres threshold mode"
     python3 cder/score.py -s $predict_rttm_dir/$name/res_rttm_${thres}  -r $oracle_rttm_dir/$name/rttm_debug_nog0
    done
   done
fi
# grep -r Avg logs/run_ts_vad2_streaming_hltsz_4090_stage36-38_5.log
#Avg CDER : 0.871
#Avg CDER : 0.556
#Avg CDER : 0.448
#Avg CDER : 0.339
#Avg CDER : 0.255
#Avg CDER : 0.178
#Avg CDER : 0.138
#Avg CDER : 0.116
#Avg CDER : 0.098
#Avg CDER : 0.098
#Avg CDER : 0.102
#Avg CDER : 0.541
#Avg CDER : 0.461
#Avg CDER : 0.425
#Avg CDER : 0.393
#Avg CDER : 0.309
#Avg CDER : 0.205
#Avg CDER : 0.145
#Avg CDER : 0.136
#Avg CDER : Error!
#Avg CDER : Error!
#Avg CDER : Error!
#Avg CDER : 0.360
#Avg CDER : 0.279
#Avg CDER : 0.238
#Avg CDER : 0.200
#Avg CDER : 0.155
#Avg CDER : 0.122
#Avg CDER : 0.109
#Avg CDER : 0.102
#Avg CDER : 0.095
#Avg CDER : 0.100
#Avg CDER : 0.199


if [ ${stage} -le 40 ] && [ ${stop_stage} -ge 40 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is cam++ 200k speaker model
    #  oracle target speaker embedding is from cam++ pretrain model
    # checkpoint is from https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/files
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/share/workspace/shared_datasets/speechdata/14_musan
    rir_path=/share/workspace/shared_datasets/speechdata/21_RIRS_NOISES/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/share/workspace/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
    dataset_name="magicdata-ramc" # dataset name

    # for loading speaker embedding file
    spk_path=/share/workspace/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/share/workspace/maduo/exp/speaker_diarization/ts_vad2_streaming/magicdata-ramc_ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr3e5_single_backend_mamba2_unidirectional_multi_backend_transformer_rs_len16_shift0.8_streaming_epoch20
    data_dir="/share/workspace/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format" # oracle target audio , mix audio and labels path
    rs_len=16
    segment_shift=0.8
    single_backend_type="mamba2_unidirectional"
    num_transformer_layer=2
    CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15315 \
   ts_vad2_streaming/train_accelerate_ddp.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --keep-last-k 3\
    --keep-last-epoch 3\
    --grad-clip false\
    --lr 3e-5\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --single-backend-type $single_backend_type\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --exp-dir $exp_dir\
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --num-transformer-layer $num_transformer_layer
fi
# note: lr=1e-4, epoch=14, loss  and der  are explode
# you can see the below log
# grep -r Eval logs/run_ts_vad2_streaming_hltsz_4090_stage36-38_1.log
# grep -r Eval logs/run_ts_vad2_streaming_hltsz_4090_stage36-38.log
# lr=5e-5, 2025-07-19 06:39:09,552 (train_accelerate_ddp:679) INFO: [Train] - Epoch 17, batch_idx_train: 92762, num_updates: 83500, {'loss': nan, 'DER': 1.0, 'ACC': np.float64(0.7868553533912078), 'MI': 1.0, 'FA': 0.0, 'CF': 0.0}, batch size: 64, grad_norm: None, grad_scale: , lr: 3.4583333333333334e-06,
# lr=1e-5, working

if [ ${stage} -le 41 ] && [ ${stop_stage} -ge 41 ];then
 exp_dir=/share/workspace/maduo/exp/speaker_diarization/ts_vad2_streaming/magicdata-ramc_ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr3e5_single_backend_mamba2_unidirectional_multi_backend_transformer_rs_len16_shift0.8_streaming_epoch20

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

 single_backend_type="mamba2_unidirectional"
 #multi_backend_type="transformer"
 #d_state=256
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 #infer_sets="Eval Test"
 #infer_sets="Test"
 infer_sets="dev test cssd_testset"
 rttm_dir=/share/workspace/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0
 #collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/share/workspace/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 dataset_name="magicdata-ramc" # dataset name
 # for loading speaker embedding file
 spk_path=/share/workspace/maduo/model_hub/ts_vad/spk_embed/magicdata-ramc/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 #data_dir="/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc" # oracle target audio , mix audio and labels path
 data_dir="/share/workspace/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format"
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
    --single-backend-type $single_backend_type\
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


if [ ${stage} -le 42 ] && [ ${stop_stage} -ge 42 ];then
   echo "compute CDER for magicdata-ramc"
   threshold="0.2 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.7 0.8 0.9"
   predict_rttm_dir=/share/workspace/maduo/exp/speaker_diarization/ts_vad2_streaming/magicdata-ramc_ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr3e5_single_backend_mamba2_unidirectional_multi_backend_transformer_rs_len16_shift0.8_streaming_epoch20/magicdata-ramc_collar0.0_decoding_chunk_size25_num_decoding_left_chunks-1_simulate_streamingfalse_
   oracle_rttm_dir=/share/workspace/maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format
   infer_sets="dev test cssd_testset"
   for name in $infer_sets;do
    for thres in $threshold;do
     echo "currently, compute $name set in $thres threshold mode"
     python3 cder/score.py -s $predict_rttm_dir/$name/res_rttm_${thres}  -r $oracle_rttm_dir/$name/rttm_debug_nog0
    done
   done
fi
