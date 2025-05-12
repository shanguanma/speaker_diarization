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

