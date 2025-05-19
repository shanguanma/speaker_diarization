#!/usr/bin/env bash

stage=0
stop_stage=1000

. utils/parse_options.sh
. path_for_dia_cuda11.8_py3111_aistation.sh



if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is cam++ 200k speaker model
    #  oracle target speaker embedding is from cam++ pretrain model
    # checkpoint is from https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/files
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/maduo/datasets/musan
    rir_path=/maduo/datasets/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
    dataset_name="alimeeting" # dataset name

    # for loading speaker embedding file
    spk_path=/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/maduo/exp/speaker_diarization/ts_vad2_streaming/ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_speech_encoder_cam++_zh_en_speaker_embedding_epoch20_front_fix_seed_lr1e4_single_backend_transformer_multi_backend_transformer_rs_len8_shift0.8_streaming
    mkdir -p $exp_dir
    data_dir="/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    rs_len=8
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
    --num-transformer-layer $num_transformer_layer\

fi




if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
 exp_dir=/maduo/exp/speaker_diarization/ts_vad2_streaming/ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_speech_encoder_cam++_zh_en_speaker_embedding_epoch20_front_fix_seed_lr1e4_single_backend_transformer_multi_backend_transformer_rs_len8_shift0.8_streaming
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
 rttm_dir=/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0
 #collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 dataset_name="alimeeting" # dataset name
 # for loading speaker embedding file
 spk_path=/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 data_dir="/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
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
#grep -r Eval  logs/run_ts_vad2_streaming_aistation_stage1-2.log
# Eval of alimeeting, collar=0.0
#Eval for threshold 0.20: DER 29.12%, MS 2.50%, FA 25.09%, SC 1.53%
#Eval for threshold 0.30: DER 21.85%, MS 4.06%, FA 15.95%, SC 1.85%
#Eval for threshold 0.35: DER 19.54%, MS 5.00%, FA 12.65%, SC 1.89%
#Eval for threshold 0.40: DER 17.98%, MS 5.95%, FA 10.10%, SC 1.93%
#Eval for threshold 0.45: DER 16.94%, MS 7.04%, FA 7.98%, SC 1.91%
#Eval for threshold 0.50: DER 16.40%, MS 8.29%, FA 6.25%, SC 1.86%
#Eval for threshold 0.55: DER 16.27%, MS 9.70%, FA 4.83%, SC 1.74% as report
#Eval for threshold 0.60: DER 16.66%, MS 11.32%, FA 3.74%, SC 1.60%
#Eval for threshold 0.70: DER 18.41%, MS 15.02%, FA 2.19%, SC 1.20%
#Eval for threshold 0.80: DER 21.98%, MS 20.02%, FA 1.27%, SC 0.69%

# Test of alimeeting, collar=0.0
#Eval for threshold 0.20: DER 28.09%, MS 2.27%, FA 24.48%, SC 1.34%
#Eval for threshold 0.30: DER 21.14%, MS 3.81%, FA 15.62%, SC 1.71%
#Eval for threshold 0.35: DER 19.08%, MS 4.77%, FA 12.49%, SC 1.81%
#Eval for threshold 0.40: DER 17.70%, MS 5.85%, FA 9.97%, SC 1.89%
#Eval for threshold 0.45: DER 16.87%, MS 7.04%, FA 7.91%, SC 1.93%
#Eval for threshold 0.50: DER 16.58%, MS 8.42%, FA 6.24%, SC 1.92%
#Eval for threshold 0.55: DER 16.61%, MS 9.90%, FA 4.84%, SC 1.87% as report
#Eval for threshold 0.60: DER 17.03%, MS 11.60%, FA 3.71%, SC 1.72%
#Eval for threshold 0.70: DER 19.04%, MS 15.61%, FA 2.09%, SC 1.35%
#Eval for threshold 0.80: DER 22.96%, MS 21.06%, FA 1.02%, SC 0.88%
#2025-05-15 00:19:51,534 (infer:254) INFO: currently, it will infer Eval set.

# Eval of alimeeting, collar=0.25
#Eval for threshold 0.20: DER 16.59%, MS 0.87%, FA 15.17%, SC 0.55%
#Eval for threshold 0.30: DER 11.12%, MS 1.54%, FA 8.90%, SC 0.68%
#Eval for threshold 0.35: DER 9.51%, MS 1.97%, FA 6.84%, SC 0.70%
#Eval for threshold 0.40: DER 8.38%, MS 2.44%, FA 5.20%, SC 0.74%
#Eval for threshold 0.45: DER 7.70%, MS 3.03%, FA 3.90%, SC 0.77%
#Eval for threshold 0.50: DER 7.32%, MS 3.69%, FA 2.88%, SC 0.76%
#Eval for threshold 0.55: DER 7.24%, MS 4.46%, FA 2.03%, SC 0.75% as report
#Eval for threshold 0.60: DER 7.63%, MS 5.39%, FA 1.53%, SC 0.71%
#Eval for threshold 0.70: DER 9.03%, MS 7.67%, FA 0.78%, SC 0.59%
#Eval for threshold 0.80: DER 11.73%, MS 11.01%, FA 0.40%, SC 0.32%

# Test of alimeeting, collar=0.25
#Eval for threshold 0.20: DER 16.73%, MS 0.97%, FA 15.24%, SC 0.52%
#Eval for threshold 0.30: DER 11.34%, MS 1.73%, FA 8.91%, SC 0.71%
#Eval for threshold 0.35: DER 9.85%, MS 2.22%, FA 6.86%, SC 0.77%
#Eval for threshold 0.40: DER 8.86%, MS 2.78%, FA 5.24%, SC 0.83%
#Eval for threshold 0.45: DER 8.27%, MS 3.43%, FA 3.95%, SC 0.89%
#Eval for threshold 0.50: DER 8.08%, MS 4.23%, FA 2.95%, SC 0.90%
#Eval for threshold 0.55: DER 8.06%, MS 5.07%, FA 2.08%, SC 0.91% as report
#Eval for threshold 0.60: DER 8.33%, MS 6.07%, FA 1.41%, SC 0.86%
#Eval for threshold 0.70: DER 9.79%, MS 8.53%, FA 0.60%, SC 0.67%
#Eval for threshold 0.80: DER 12.74%, MS 12.16%, FA 0.18%, SC 0.40%






if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is cam++ 200k speaker model
    #  oracle target speaker embedding is from cam++ pretrain model
    # checkpoint is from https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/files
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/maduo/datasets/musan
    rir_path=/maduo/datasets/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
    dataset_name="alimeeting" # dataset name

    # for loading speaker embedding file
    spk_path=/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    #speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/maduo/exp/speaker_diarization/ts_vad2_streaming/ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_speech_encoder_cam++_zh_200k_speaker_embedding_epoch20_front_fix_seed_lr1e4_single_backend_transformer_multi_backend_transformer_rs_len8_shift0.8_streaming
    mkdir -p $exp_dir
    data_dir="/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    rs_len=8
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
    --num-transformer-layer $num_transformer_layer\

fi




if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
 exp_dir=/maduo/exp/speaker_diarization/ts_vad2_streaming/ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_speech_encoder_cam++_zh_200k_speaker_embedding_epoch20_front_fix_seed_lr1e4_single_backend_transformer_multi_backend_transformer_rs_len8_shift0.8_streaming
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
 rttm_dir=/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0
 #collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 dataset_name="alimeeting" # dataset name
 # for loading speaker embedding file
 spk_path=/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 #speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
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
#grep -r Eval logs/run_ts_vad2_streaming_aistation_stage3-4.log
#Eval for threshold 0.20: DER 30.27%, MS 2.50%, FA 26.31%, SC 1.46%
#Eval for threshold 0.30: DER 22.81%, MS 4.14%, FA 16.86%, SC 1.81%
#Eval for threshold 0.35: DER 20.42%, MS 5.03%, FA 13.46%, SC 1.92%
#Eval for threshold 0.40: DER 18.78%, MS 6.00%, FA 10.85%, SC 1.94%
#Eval for threshold 0.45: DER 17.50%, MS 7.10%, FA 8.50%, SC 1.91%
#Eval for threshold 0.50: DER 16.90%, MS 8.35%, FA 6.67%, SC 1.88%
#Eval for threshold 0.55: DER 16.80%, MS 9.80%, FA 5.21%, SC 1.79% as report
#Eval for threshold 0.60: DER 17.12%, MS 11.38%, FA 4.08%, SC 1.66%
#Eval for threshold 0.70: DER 18.81%, MS 15.25%, FA 2.37%, SC 1.19%
#Eval for threshold 0.80: DER 22.37%, MS 20.25%, FA 1.43%, SC 0.69%
#Eval for threshold 0.20: DER 29.96%, MS 2.31%, FA 26.14%, SC 1.51%
#Eval for threshold 0.30: DER 22.41%, MS 3.94%, FA 16.57%, SC 1.90%
#Eval for threshold 0.35: DER 20.14%, MS 4.95%, FA 13.18%, SC 2.00%
#Eval for threshold 0.40: DER 18.61%, MS 6.08%, FA 10.43%, SC 2.10%
#Eval for threshold 0.45: DER 17.62%, MS 7.35%, FA 8.15%, SC 2.12%
#Eval for threshold 0.50: DER 17.10%, MS 8.74%, FA 6.34%, SC 2.02%
#Eval for threshold 0.55: DER 17.04%, MS 10.26%, FA 4.91%, SC 1.87% as report
#Eval for threshold 0.60: DER 17.44%, MS 11.98%, FA 3.75%, SC 1.70%
#Eval for threshold 0.70: DER 19.56%, MS 16.16%, FA 2.09%, SC 1.31%
#Eval for threshold 0.80: DER 23.57%, MS 21.68%, FA 1.08%, SC 0.81%
#2025-05-17 17:04:42,688 (infer:254) INFO: currently, it will infer Eval set.
#Eval for threshold 0.20: DER 18.01%, MS 0.87%, FA 16.62%, SC 0.53%
#Eval for threshold 0.30: DER 12.23%, MS 1.61%, FA 9.97%, SC 0.65%
#Eval for threshold 0.35: DER 10.49%, MS 2.02%, FA 7.75%, SC 0.72%
#Eval for threshold 0.40: DER 9.25%, MS 2.48%, FA 6.03%, SC 0.74%
#Eval for threshold 0.45: DER 8.25%, MS 3.03%, FA 4.44%, SC 0.77%
#Eval for threshold 0.50: DER 7.83%, MS 3.73%, FA 3.29%, SC 0.81%
#Eval for threshold 0.55: DER 7.68%, MS 4.54%, FA 2.35%, SC 0.79% as report
#Eval for threshold 0.60: DER 7.95%, MS 5.51%, FA 1.70%, SC 0.75%
#Eval for threshold 0.70: DER 9.24%, MS 7.92%, FA 0.77%, SC 0.56%
#Eval for threshold 0.80: DER 12.13%, MS 11.42%, FA 0.41%, SC 0.30%
#Eval for threshold 0.20: DER 19.12%, MS 0.99%, FA 17.54%, SC 0.59%
#Eval for threshold 0.30: DER 12.94%, MS 1.79%, FA 10.34%, SC 0.81%
#Eval for threshold 0.35: DER 11.13%, MS 2.32%, FA 7.93%, SC 0.88%
#Eval for threshold 0.40: DER 9.87%, MS 2.97%, FA 5.95%, SC 0.95%
#Eval for threshold 0.45: DER 9.04%, MS 3.68%, FA 4.40%, SC 0.97%
#Eval for threshold 0.50: DER 8.56%, MS 4.48%, FA 3.16%, SC 0.93%
#Eval for threshold 0.55: DER 8.50%, MS 5.37%, FA 2.27%, SC 0.86% as report
#Eval for threshold 0.60: DER 8.71%, MS 6.38%, FA 1.55%, SC 0.78%
#Eval for threshold 0.70: DER 10.24%, MS 9.02%, FA 0.59%, SC 0.63%
#Eval for threshold 0.80: DER 13.36%, MS 12.83%, FA 0.17%, SC 0.35%



if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is cam++ 200k speaker model
    #  oracle target speaker embedding is from cam++ pretrain model
    # checkpoint is from https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/files
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/maduo/datasets/musan
    rir_path=/maduo/datasets/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
    dataset_name="alimeeting" # dataset name

    # for loading speaker embedding file
    spk_path=/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/maduo/exp/speaker_diarization/ts_vad2_streaming/ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_speech_encoder_cam++_zh_en_speaker_embedding_epoch20_front_fix_seed_lr5e5_single_backend_transformer_multi_backend_transformer_rs_len16_shift0.8_streaming
    mkdir -p $exp_dir
    data_dir="/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    rs_len=16
    segment_shift=0.8
    num_transformer_layer=2
    CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15215 \
   ts_vad2_streaming/train_accelerate_ddp.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 17\
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
    --num-transformer-layer $num_transformer_layer\

fi




if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ];then
 exp_dir=/maduo/exp/speaker_diarization/ts_vad2_streaming/ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_speech_encoder_cam++_zh_en_speaker_embedding_epoch20_front_fix_seed_lr5e5_single_backend_transformer_multi_backend_transformer_rs_len16_shift0.8_streaming
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
 infer_sets="Eval Test"
 #infer_sets="Test"
 #infer_sets="Eval"
 rttm_dir=/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0
 #collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 dataset_name="alimeeting" # dataset name
 # for loading speaker embedding file
 spk_path=/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 data_dir="/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
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
