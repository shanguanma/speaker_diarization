#!/usr/bin/env bash

stage=0
stop_stage=1000
. utils/parse_options.sh
. path_for_speaker_diarization_hltsz.sh

#if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
#    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
#    # # speech encoder is cam++ 200k speaker model
#    #  oracle target speaker embedding is from cam++ pretrain model
#    # checkpoint is from https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/files
#    # how to look for port ?
#    # netstat -tuln
#    export NCCL_DEBUG=INFO
#    export PYTHONFAULTHANDLER=1
#    musan_path=/data/maduo/datasets/musan
#    rir_path=/data/maduo/datasets/RIRS_NOISES
#    # for loading pretrain model weigt
#    speech_encoder_type="CAM++"
#    speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
#    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
#    dataset_name="alimeeting" # dataset name
#
#    # for loading speaker embedding file
#    spk_path=/data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
#    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
#
#    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
#    exp_dir=/data/maduo/exp/speaker_diarization/alimeeting/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_single_backend_transformer_multi_backend_transformer_ts_len10_streaming
#    data_dir="/data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
#    rs_len=10
#    segment_shift=2
#    #single_backend_type="transformer"
#    #multi_backend_type="transformer"
#    #d_state=64
#    num_transformer_layer=2
#    CUDA_VISIABLE_DEVICES=0,1 \
#  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15615 \
#   ts_vad2_streaming/train_accelerate_ddp.py \
#    --world-size 2 \
#    --num-epochs 20\
#    --start-epoch 1\
#    --keep-last-k 1\
#    --keep-last-epoch 1\
#    --freeze-updates 4000\
#    --grad-clip true\
#    --lr 2e-4\
#    --musan-path $musan_path \
#    --rir-path $rir_path \
#    --speech-encoder-type $speech_encoder_type\
#    --speech-encoder-path $speech_encoder_path\
#    --spk-path $spk_path\
#    --speaker-embedding-name-dir $speaker_embedding_name_dir\
#    --exp-dir $exp_dir\
#    --data-dir $data_dir\
#    --dataset-name $dataset_name\
#    --rs-len $rs_len\
#    --segment-shift $segment_shift\
#    --num-transformer-layer $num_transformer_layer\
#fi
#
#
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then                                               
 #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_single_backend_transformer_multi_backend_transformer_ts_len10_streaming
 #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr1e4_single_backend_transformer_multi_backend_transformer_ts_len4_streaming
 exp_dir=/data/maduo/exp/speaker_diarization/ts_vad2_streaming/ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr1e4_single_backend_transformer_multi_backend_transformer_ts_len4_streaming
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
 infer_sets="Eval Test"
 #infer_sets="Test"
 #infer_sets="Eval"
 rttm_dir=/data/maduo/model_hub/ts_vad/
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0
 #collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 dataset_name="alimeeting" # dataset name
 # for loading speaker embedding file
 spk_path=/data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding/ # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_collar${c}_rs_len_${rs_len}_decoding_chunk_size${decoding_chunk_size}_num_decoding_left_chunks${num_decoding_left_chunks}_simulate_streaming${simulate_streaming}_${fn_name}
  python3 ts_vad2_streaming/infer.py \
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

# grep -r Eval logs/run_ts_vad2_streaming_hltsz_stage0_againe.log
#2025-04-17 23:56:27,391 (infer:254) INFO: currently, it will infer Eval set.
# Eval of alimeeting, collar=0.0
#Eval for threshold 0.20: DER 32.47%, MS 2.80%, FA 28.35%, SC 1.32%
#Eval for threshold 0.30: DER 24.37%, MS 4.43%, FA 18.21%, SC 1.74%
#Eval for threshold 0.35: DER 21.56%, MS 5.35%, FA 14.35%, SC 1.87%
#Eval for threshold 0.40: DER 19.71%, MS 6.42%, FA 11.37%, SC 1.92%
#Eval for threshold 0.45: DER 18.39%, MS 7.63%, FA 8.80%, SC 1.95%
#Eval for threshold 0.50: DER 17.73%, MS 8.96%, FA 6.90%, SC 1.87%
#Eval for threshold 0.55: DER 17.51%, MS 10.43%, FA 5.28%, SC 1.80%
#Eval for threshold 0.60: DER 17.73%, MS 12.11%, FA 4.01%, SC 1.61%
#Eval for threshold 0.70: DER 19.57%, MS 16.08%, FA 2.33%, SC 1.16%
#Eval for threshold 0.80: DER 23.24%, MS 21.17%, FA 1.33%, SC 0.74%


# Test of alimeeting, collar=0.0
#Eval for threshold 0.20: DER 31.96%, MS 2.41%, FA 28.10%, SC 1.44%
#Eval for threshold 0.30: DER 23.45%, MS 4.08%, FA 17.45%, SC 1.91%
#Eval for threshold 0.35: DER 20.92%, MS 5.15%, FA 13.69%, SC 2.08%
#Eval for threshold 0.40: DER 19.20%, MS 6.34%, FA 10.67%, SC 2.19%
#Eval for threshold 0.45: DER 18.15%, MS 7.69%, FA 8.26%, SC 2.20%
#Eval for threshold 0.50: DER 17.67%, MS 9.18%, FA 6.34%, SC 2.15%
#Eval for threshold 0.55: DER 17.73%, MS 10.86%, FA 4.85%, SC 2.02%
#Eval for threshold 0.60: DER 18.18%, MS 12.66%, FA 3.64%, SC 1.88%
#Eval for threshold 0.70: DER 20.34%, MS 16.88%, FA 1.99%, SC 1.46%
#Eval for threshold 0.80: DER 24.31%, MS 22.34%, FA 1.04%, SC 0.93%
#2025-04-18 00:36:06,861 (infer:254) INFO: currently, it will infer Eval set.

# Eval of alimeeting, collar=0.25
#Eval for threshold 0.20: DER 19.79%, MS 1.02%, FA 18.29%, SC 0.48%
#Eval for threshold 0.30: DER 13.38%, MS 1.79%, FA 10.93%, SC 0.67%
#Eval for threshold 0.35: DER 11.26%, MS 2.25%, FA 8.28%, SC 0.73%
#Eval for threshold 0.40: DER 9.87%, MS 2.79%, FA 6.33%, SC 0.75%
#Eval for threshold 0.45: DER 8.83%, MS 3.38%, FA 4.68%, SC 0.78%
#Eval for threshold 0.50: DER 8.35%, MS 4.10%, FA 3.46%, SC 0.79%
#Eval for threshold 0.55: DER 8.24%, MS 4.97%, FA 2.47%, SC 0.80%
#Eval for threshold 0.60: DER 8.40%, MS 5.97%, FA 1.72%, SC 0.71%
#Eval for threshold 0.70: DER 9.85%, MS 8.53%, FA 0.81%, SC 0.52%
#Eval for threshold 0.80: DER 12.72%, MS 12.03%, FA 0.38%, SC 0.31%

# Test of alimeeting, collar=0.25
#Eval for threshold 0.20: DER 20.44%, MS 1.06%, FA 18.77%, SC 0.61%
#Eval for threshold 0.30: DER 13.33%, MS 1.96%, FA 10.53%, SC 0.84%
#Eval for threshold 0.35: DER 11.28%, MS 2.56%, FA 7.79%, SC 0.93%
#Eval for threshold 0.40: DER 9.98%, MS 3.22%, FA 5.77%, SC 0.99%
#Eval for threshold 0.45: DER 9.20%, MS 4.03%, FA 4.17%, SC 1.00%
#Eval for threshold 0.50: DER 8.83%, MS 4.86%, FA 2.99%, SC 0.99%
#Eval for threshold 0.55: DER 8.88%, MS 5.83%, FA 2.11%, SC 0.94%
#Eval for threshold 0.60: DER 9.18%, MS 6.92%, FA 1.37%, SC 0.89%
#Eval for threshold 0.70: DER 10.79%, MS 9.60%, FA 0.49%, SC 0.70%
#Eval for threshold 0.80: DER 13.86%, MS 13.28%, FA 0.16%, SC 0.42%
