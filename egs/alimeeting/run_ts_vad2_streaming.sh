#!/usr/bin/env bash
stage=0
stop_stage=1000

. utils/parse_options.sh
#. path_for_speaker_diarization.sh
. path_for_dia_pt2.4.sh


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is cam++ 200k speaker model
    #  oracle target speaker embedding is from cam++ pretrain model
    # checkpoint is from https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/files
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
    dataset_name="alimeeting" # dataset name

    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_single_backend_transformer_multi_backend_transformer_ts_len10_streaming
    data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    rs_len=10
    segment_shift=2
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
    --lr 2e-4\
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
# loss is nan at epoch 4. give up it.



if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
 #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_single_backend_transformer_multi_backend_transformer_ts_len10_streaming
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr1e4_single_backend_transformer_multi_backend_transformer_ts_len10_streaming
 model_file=$exp_dir/best-valid-der.pt
 #model_file=$exp_dir/epoch-1.pt
 rs_len=10
 segment_shift=1
 decoding_chunk_size=25
 num_decoding_left_chunks=-1
 simulate_streaming=true
 batch_size=1
 #fn_name="self.forward_chunk_by_chunk_temp"
 if $simulate_streaming;then
   fn_name="self.forward_chunk_by_chunk_temp_version3"
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
 infer_sets="Eval"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 #collar="0.0 0.25"
 #collar=0.0
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 dataset_name="alimeeting" # dataset name
 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
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
# The inference  is same as train (dynamic chunk)
# decoding_chunk_size=0
# num_decoding_left_chunks=-1
# simulate_streaming=false
# batch_size=1
#Model DER:  0.18620008195862398
#Model ACC:  0.9345291036522448
#2025-01-17 19:53:41,484 (infer:84) INFO: frame_len: 0.04!!
#100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 25/25 [00:13<00:00,  1.81it/s]
#Eval for threshold 0.20: DER 18.45%, MS 0.93%, FA 16.99%, SC 0.53%
#
#Eval for threshold 0.30: DER 12.65%, MS 1.62%, FA 10.32%, SC 0.71%
#
#Eval for threshold 0.35: DER 10.80%, MS 2.05%, FA 7.97%, SC 0.78%
#
#Eval for threshold 0.40: DER 9.43%, MS 2.53%, FA 6.06%, SC 0.84%
#
#Eval for threshold 0.45: DER 8.51%, MS 3.10%, FA 4.55%, SC 0.86%
#
#Eval for threshold 0.50: DER 8.09%, MS 3.81%, FA 3.36%, SC 0.92%
#
#Eval for threshold 0.55: DER 7.96%, MS 4.62%, FA 2.45%, SC 0.89% as report
#
#Eval for threshold 0.60: DER 8.14%, MS 5.62%, FA 1.69%, SC 0.82%
#
#Eval for threshold 0.70: DER 9.43%, MS 8.08%, FA 0.79%, SC 0.56%
#
#Eval for threshold 0.80: DER 12.34%, MS 11.63%, FA 0.41%, SC 0.30%

#The inference is same full offline mode (chunk_size=max_len=rs_len*100)
# # decoding_chunk_size=-1
# num_decoding_left_chunks=-1
# simulate_streaming=false
# batch_size=1
# Model DER:  0.19075626929984874
#Model ACC:  0.9338257874253976
#2025-01-17 20:08:11,606 (infer:84) INFO: frame_len: 0.04!!
#100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 25/25 [00:13<00:00,  1.80it/s]
#Eval for threshold 0.20: DER 18.91%, MS 0.91%, FA 17.44%, SC 0.56%
#
#Eval for threshold 0.30: DER 12.97%, MS 1.59%, FA 10.67%, SC 0.72%
#
#Eval for threshold 0.35: DER 11.12%, MS 2.01%, FA 8.33%, SC 0.78%
#
#Eval for threshold 0.40: DER 9.79%, MS 2.52%, FA 6.38%, SC 0.90%
#
#Eval for threshold 0.45: DER 8.81%, MS 3.10%, FA 4.79%, SC 0.92%
#
#Eval for threshold 0.50: DER 8.30%, MS 3.79%, FA 3.53%, SC 0.98%
#
#Eval for threshold 0.55: DER 8.07%, MS 4.62%, FA 2.53%, SC 0.92%
#
#Eval for threshold 0.60: DER 8.25%, MS 5.58%, FA 1.81%, SC 0.86%
#
#Eval for threshold 0.70: DER 9.49%, MS 8.04%, FA 0.83%, SC 0.62%
#
#Eval for threshold 0.80: DER 12.49%, MS 11.74%, FA 0.44%, SC 0.32%

# The inference is online mode (use kv cache tech).
# fn_name="self.forward_chunk_by_chunk"
# using fn self.forward_chunk_by_chunk()
# decoding_chunk_size=25 # chunk size 1s audio
# num_decoding_left_chunks=-1
# simulate_streaming=true
# batch_size=1
# cat logs/run_ts_vad2_streaming_stage2-rs_len10_A100_lr_1e-4.infer.log
## Model DER:  0.4611780595441592
#Model ACC:  0.842007855374139
#100%|██████████| 25/25 [00:17<00:00,  1.40it/s]
#Eval for threshold 0.20: DER 181.51%, MS 0.00%, FA 181.51%, SC 0.00%
#
#Eval for threshold 0.30: DER 181.51%, MS 0.00%, FA 181.51%, SC 0.00%
#
#Eval for threshold 0.35: DER 178.06%, MS 0.00%, FA 178.06%, SC 0.00%
#
#Eval for threshold 0.40: DER 143.60%, MS 0.03%, FA 143.39%, SC 0.18%
#
#Eval for threshold 0.45: DER 59.52%, MS 1.21%, FA 55.68%, SC 2.64%
#
#Eval for threshold 0.50: DER 32.09%, MS 19.51%, FA 5.03%, SC 7.55%
#
#Eval for threshold 0.55: DER 60.94%, MS 60.41%, FA 0.23%, SC 0.29%
#
#Eval for threshold 0.60: DER 91.02%, MS 91.02%, FA 0.01%, SC 0.00%
#
#Eval for threshold 0.70: DER 100.00%, MS 100.00%, FA 0.00%, SC 0.00%
#
#Eval for threshold 0.80: DER 100.00%, MS 100.00%, FA 0.00%, SC 0.00%

# The inference is online mode (use max_len to build mask, and final chunk_mask = mask* chunk_mask).
# decoding_chunk_size=25 # chunk size 1s audio
# num_decoding_left_chunks=-1
# simulate_streaming=false
# batch_size=1
# tail -n 50 logs/run_ts_vad2_streaming_stage2-rs_len10_A100_lr_1e-4.infer_1.log
# Model DER:  0.1873443236008868
#Model ACC:  0.9352578412493591
#100%|██████████| 25/25 [00:18<00:00,  1.37it/s]
#Eval for threshold 0.20: DER 18.15%, MS 0.89%, FA 16.68%, SC 0.59%
#
#Eval for threshold 0.30: DER 12.43%, MS 1.59%, FA 10.12%, SC 0.72%
#
#Eval for threshold 0.35: DER 10.64%, MS 1.98%, FA 7.86%, SC 0.80%
#
#Eval for threshold 0.40: DER 9.34%, MS 2.46%, FA 6.03%, SC 0.85%
#
#Eval for threshold 0.45: DER 8.52%, MS 3.02%, FA 4.59%, SC 0.91%
#
#Eval for threshold 0.50: DER 8.03%, MS 3.71%, FA 3.38%, SC 0.93%
#
#Eval for threshold 0.55: DER 7.88%, MS 4.50%, FA 2.47%, SC 0.90% as report
#
#Eval for threshold 0.60: DER 8.03%, MS 5.42%, FA 1.76%, SC 0.85%
#
#Eval for threshold 0.70: DER 9.31%, MS 7.82%, FA 0.87%, SC 0.62%
#
#Eval for threshold 0.80: DER 11.95%, MS 11.13%, FA 0.43%, SC 0.39%

# The inference is online mode (use kv cache tech).
# fn_name="self.forward_chunk_by_chunk_temp" .Consider the right context of the embed.
# using fn self.forward_chunk_by_chunk_temp() # version2
# decoding_chunk_size=25 # chunk size 1s audio
# num_decoding_left_chunks=-1
# simulate_streaming=true
# batch_size=1
# Model DER:  0.4543531817639755
#Model ACC:  0.843802638203343
#100%|██████████| 25/25 [00:18<00:00,  1.34it/s]
#Eval for threshold 0.20: DER 181.51%, MS 0.00%, FA 181.51%, SC 0.00%
#
#Eval for threshold 0.30: DER 181.51%, MS 0.00%, FA 181.51%, SC 0.00%
#
#Eval for threshold 0.35: DER 177.89%, MS 0.00%, FA 177.89%, SC 0.00%
#
#Eval for threshold 0.40: DER 142.37%, MS 0.02%, FA 142.14%, SC 0.21%
#
#Eval for threshold 0.45: DER 59.64%, MS 1.11%, FA 55.88%, SC 2.66%
#
#Eval for threshold 0.50: DER 31.51%, MS 18.64%, FA 5.18%, SC 7.69%
#
#Eval for threshold 0.55: DER 60.04%, MS 59.47%, FA 0.24%, SC 0.32%
#
#Eval for threshold 0.60: DER 91.42%, MS 91.41%, FA 0.01%, SC 0.00%
#
#Eval for threshold 0.70: DER 100.00%, MS 100.00%, FA 0.00%, SC 0.00%
#
#Eval for threshold 0.80: DER 100.00%, MS 100.00%, FA 0.00%, SC 0.00%

# The inference is online mode (use kv cache tech).
# fn_name="self.forward_chunk_by_chunk_temp" .Consider the right context of the embed.
# using fn self.forward_chunk_by_chunk_temp() # version1
# decoding_chunk_size=25 # chunk size 1s audio
# num_decoding_left_chunks=-1
# simulate_streaming=true
# batch_size=1

#tail -n 50 logs/run_ts_vad2_streaming_stage2-rs_len10_A100_lr_1e-4.infer_3.log
#
#Model DER:  0.44988028131918334
#Model ACC:  0.8456859424630564
#100%|██████████| 25/25 [00:19<00:00,  1.31it/s]
#Eval for threshold 0.20: DER 181.51%, MS 0.00%, FA 181.51%, SC 0.00%
#
#Eval for threshold 0.30: DER 181.51%, MS 0.00%, FA 181.51%, SC 0.00%
#
#Eval for threshold 0.35: DER 177.88%, MS 0.00%, FA 177.88%, SC 0.00%
#
#Eval for threshold 0.40: DER 141.52%, MS 0.02%, FA 141.29%, SC 0.21%
#
#Eval for threshold 0.45: DER 58.58%, MS 1.12%, FA 54.83%, SC 2.63%
#
#Eval for threshold 0.50: DER 31.13%, MS 18.55%, FA 5.16%, SC 7.41%
#
#Eval for threshold 0.55: DER 59.88%, MS 59.29%, FA 0.25%, SC 0.34%
#
#Eval for threshold 0.60: DER 91.49%, MS 91.48%, FA 0.01%, SC 0.00%
#
#Eval for threshold 0.70: DER 100.00%, MS 100.00%, FA 0.00%, SC 0.00%
#
#Eval for threshold 0.80: DER 100.00%, MS 100.00%, FA 0.00%, SC 0.00%

# The inference is online mode (use kv cache tech).
# fn_name="self.forward_chunk_by_chunk_temp" .Consider the right context of the embed.
# using fn self.forward_chunk_by_chunk_temp() # version3
# decoding_chunk_size=25 # chunk size 1s audio
# num_decoding_left_chunks=-1
# simulate_streaming=true
# batch_size=1


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is cam++ 200k speaker model
    #  oracle target speaker embedding is from cam++ pretrain model
    # checkpoint is from https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/files
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
    dataset_name="alimeeting" # dataset name

    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr1e4_single_backend_transformer_multi_backend_transformer_ts_len10_streaming
    data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    rs_len=10
    segment_shift=2
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
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is cam++ 200k speaker model
    #  oracle target speaker embedding is from cam++ pretrain model
    # checkpoint is from https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/files
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="CAM++"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
    dataset_name="alimeeting" # dataset name

    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr1e4_single_backend_transformer_multi_backend_transformer_ts_len4_streaming
    data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
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
    --num-transformer-layer $num_transformer_layer\

fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ];then
 #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_single_backend_transformer_multi_backend_transformer_ts_len10_streaming
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr1e4_single_backend_transformer_multi_backend_transformer_ts_len4_streaming
 model_file=$exp_dir/best-valid-der.pt
 #model_file=$exp_dir/epoch-1.pt
 rs_len=10
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
 infer_sets="Eval"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 #collar="0.0 0.25"
 #collar=0.0
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 dataset_name="alimeeting" # dataset name
 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
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
# cat logs/run_ts_vad2_streaming_stage5-rs_len4_A100_lr_1e-4.infer.log
#Model DER:  0.19776027949841646
#Model ACC:  0.9318008731562509
#100%|██████████| 25/25 [00:18<00:00,  1.32it/s]
#Eval for threshold 0.20: DER 22.91%, MS 0.89%, FA 21.63%, SC 0.39%
#
#Eval for threshold 0.30: DER 15.82%, MS 1.52%, FA 13.73%, SC 0.57%
#
#Eval for threshold 0.35: DER 13.20%, MS 1.94%, FA 10.62%, SC 0.64%
#
#Eval for threshold 0.40: DER 11.21%, MS 2.37%, FA 8.10%, SC 0.74%
#
#Eval for threshold 0.45: DER 9.85%, MS 2.91%, FA 6.14%, SC 0.79%
#
#Eval for threshold 0.50: DER 8.99%, MS 3.50%, FA 4.63%, SC 0.86%
#
#Eval for threshold 0.55: DER 8.49%, MS 4.32%, FA 3.35%, SC 0.82%
#
#Eval for threshold 0.60: DER 8.51%, MS 5.26%, FA 2.39%, SC 0.86%
#
#Eval for threshold 0.70: DER 9.63%, MS 7.77%, FA 1.08%, SC 0.77%
#
#Eval for threshold 0.80: DER 12.27%, MS 11.34%, FA 0.44%, SC 0.49%



if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ];then
 #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_single_backend_transformer_multi_backend_transformer_ts_len10_streaming
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr1e4_single_backend_transformer_multi_backend_transformer_ts_len4_streaming
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
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0
 #collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 dataset_name="alimeeting" # dataset name
 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
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
#cat logs/run_ts_vad2_streaming_stage6-rs_len4_A100_lr_1e-4_rs_len4.infer.log
#Model DER:  0.17412961720723932
#Model ACC:  0.9320734013605518
#100%|██████████| 25/25 [00:15<00:00,  1.57it/s]
#Eval for threshold 0.20: DER 19.73%, MS 1.02%, FA 18.24%, SC 0.47%
#
#Eval for threshold 0.30: DER 13.36%, MS 1.79%, FA 10.91%, SC 0.67%
#
#Eval for threshold 0.35: DER 11.24%, MS 2.25%, FA 8.25%, SC 0.74%
#
#Eval for threshold 0.40: DER 9.81%, MS 2.77%, FA 6.29%, SC 0.75%
#
#Eval for threshold 0.45: DER 8.84%, MS 3.37%, FA 4.68%, SC 0.79%
#
#Eval for threshold 0.50: DER 8.33%, MS 4.09%, FA 3.43%, SC 0.81%
#
#Eval for threshold 0.55: DER 8.26%, MS 4.98%, FA 2.47%, SC 0.81% as report
#
#Eval for threshold 0.60: DER 8.43%, MS 5.99%, FA 1.72%, SC 0.72%
#
#Eval for threshold 0.70: DER 9.85%, MS 8.51%, FA 0.80%, SC 0.53%
#
#Eval for threshold 0.80: DER 12.71%, MS 12.02%, FA 0.38%, SC 0.32%


if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ];then
 #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_single_backend_transformer_multi_backend_transformer_ts_len10_streaming
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr1e4_single_backend_transformer_multi_backend_transformer_ts_len4_streaming
 model_file=$exp_dir/best-valid-der.pt
 #model_file=$exp_dir/epoch-1.pt
 rs_len=4
 segment_shift=1
 decoding_chunk_size=25
 num_decoding_left_chunks=3
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
 infer_sets="Eval"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 #collar="0.0 0.25"
 #collar=0.0
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 dataset_name="alimeeting" # dataset name
 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
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
# cat logs/run_ts_vad2_streaming_stage7-rs_len4_A100_lr_1e-4_rs_len4_left_chunk_3_chunksize25.infer.log
#Model DER:  0.1740630639562891
#Model ACC:  0.9320638247143643
#100%|██████████| 25/25 [00:17<00:00,  1.43it/s]
#Eval for threshold 0.20: DER 19.75%, MS 1.02%, FA 18.26%, SC 0.48%
#
#Eval for threshold 0.30: DER 13.35%, MS 1.77%, FA 10.89%, SC 0.68%
#
#Eval for threshold 0.35: DER 11.23%, MS 2.24%, FA 8.25%, SC 0.74%
#
#Eval for threshold 0.40: DER 9.80%, MS 2.76%, FA 6.29%, SC 0.75%
#
#Eval for threshold 0.45: DER 8.81%, MS 3.37%, FA 4.64%, SC 0.80%
#
#Eval for threshold 0.50: DER 8.32%, MS 4.09%, FA 3.42%, SC 0.81%
#
#Eval for threshold 0.55: DER 8.25%, MS 4.95%, FA 2.48%, SC 0.81%
#
#Eval for threshold 0.60: DER 8.40%, MS 5.97%, FA 1.70%, SC 0.72%
#
#Eval for threshold 0.70: DER 9.86%, MS 8.55%, FA 0.79%, SC 0.52%
#
#Eval for threshold 0.80: DER 12.76%, MS 12.06%, FA 0.38%, SC 0.32%

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ];then
 #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_single_backend_transformer_multi_backend_transformer_ts_len10_streaming
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr1e4_single_backend_transformer_multi_backend_transformer_ts_len4_streaming
 model_file=$exp_dir/best-valid-der.pt
 #model_file=$exp_dir/epoch-1.pt
 rs_len=4
 segment_shift=1
 decoding_chunk_size=16
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
 infer_sets="Eval"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 #collar="0.0 0.25"
 #collar=0.0
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 dataset_name="alimeeting" # dataset name
 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
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

