#!/usr/bin/env bash

stage=0
stop_stage=1000
. utils/parse_options.sh
#. path_for_speaker_diarization_hltsz.sh
. path_for_speaker_diarization_hltsz_uv_env.sh

#compared with stage122-123 of run_ts_vad2.sh, stage124-125 will use mamba2 to replace transformer
if [ ${stage} -le 124 ] && [ ${stop_stage} -ge 124 ];then
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
    dataset_name="alimeeting" # dataset name

    # for loading speaker embedding file
    spk_path=/data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/data/maduo/exp/speaker_diarization/alimeeting/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr1e4_single_backend_mamba2_multi_backend_transformer_ts_len6_d_state_128
    data_dir="/data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    rs_len=6
    segment_shift=2
    single_backend_type="mamba2"
    multi_backend_type="transformer"
    d_state=128
    num_transformer_layer=2
    CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15915 \
   ts_vad2/train_accelerate_ddp2_debug2.py \
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
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --select-encoder-layer-nums 6\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --exp-dir $exp_dir\
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state
fi

if [ ${stage} -le 125 ] && [ ${stop_stage} -ge 125 ];then
 exp_dir=/data/maduo/exp/speaker_diarization/alimeeting/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr1e4_single_backend_mamba2_multi_backend_transformer_ts_len6_d_state_128
 model_file=$exp_dir/best-valid-der.pt
 rs_len=6
 segment_shift=1
 single_backend_type="mamba2"
 multi_backend_type="transformer"
 d_state=128
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/data/maduo/model_hub/ts_vad/
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 dataset_name="alimeeting" # dataset name
 # for loading speaker embedding file
 spk_path=/data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_collar${c}
  python3 ts_vad2/infer2.py \
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
    --d-state $d_state
 done
done
fi
#grep -r Eval logs/run_ts_vad2_hltsz_stage125.log
# collar=0.0
# dev of alimeeting
#Eval for threshold 0.20: DER 20.61%, MS 2.24%, FA 17.40%, SC 0.96%
#Eval for threshold 0.30: DER 16.21%, MS 3.50%, FA 11.47%, SC 1.23%
#Eval for threshold 0.35: DER 14.95%, MS 4.17%, FA 9.50%, SC 1.28%
#Eval for threshold 0.40: DER 14.04%, MS 4.86%, FA 7.89%, SC 1.28%
#Eval for threshold 0.45: DER 13.52%, MS 5.64%, FA 6.60%, SC 1.27%
#Eval for threshold 0.50: DER 13.28%, MS 6.52%, FA 5.53%, SC 1.22%
#Eval for threshold 0.55: DER 13.21%, MS 7.47%, FA 4.56%, SC 1.18%
#Eval for threshold 0.60: DER 13.31%, MS 8.48%, FA 3.77%, SC 1.07%
#Eval for threshold 0.70: DER 14.26%, MS 10.91%, FA 2.48%, SC 0.87%
#Eval for threshold 0.80: DER 16.60%, MS 14.50%, FA 1.53%, SC 0.57%
## test of alimeeting
#Eval for threshold 0.20: DER 20.59%, MS 2.24%, FA 17.31%, SC 1.04%
#Eval for threshold 0.30: DER 16.28%, MS 3.61%, FA 11.44%, SC 1.23%
#Eval for threshold 0.35: DER 15.07%, MS 4.32%, FA 9.45%, SC 1.29%
#Eval for threshold 0.40: DER 14.26%, MS 5.13%, FA 7.79%, SC 1.35%
#Eval for threshold 0.45: DER 13.81%, MS 5.95%, FA 6.43%, SC 1.42%
#Eval for threshold 0.50: DER 13.65%, MS 6.92%, FA 5.28%, SC 1.45%
#Eval for threshold 0.55: DER 13.70%, MS 8.02%, FA 4.29%, SC 1.39%
#Eval for threshold 0.60: DER 14.00%, MS 9.22%, FA 3.47%, SC 1.30%
#Eval for threshold 0.70: DER 15.42%, MS 12.20%, FA 2.19%, SC 1.03%
#Eval for threshold 0.80: DER 18.29%, MS 16.30%, FA 1.26%, SC 0.73%
#
## collar=0.25
## dev of alimeeting
#Eval for threshold 0.20: DER 9.30%, MS 0.79%, FA 8.27%, SC 0.24%
#Eval for threshold 0.30: DER 6.63%, MS 1.33%, FA 4.92%, SC 0.39%
#Eval for threshold 0.35: DER 5.93%, MS 1.60%, FA 3.91%, SC 0.41%
#Eval for threshold 0.40: DER 5.42%, MS 1.93%, FA 3.07%, SC 0.42%
#Eval for threshold 0.45: DER 5.14%, MS 2.27%, FA 2.46%, SC 0.42%
#Eval for threshold 0.50: DER 5.08%, MS 2.72%, FA 1.98%, SC 0.39%
#Eval for threshold 0.55: DER 5.11%, MS 3.17%, FA 1.56%, SC 0.38%
#Eval for threshold 0.60: DER 5.30%, MS 3.71%, FA 1.25%, SC 0.35%
#Eval for threshold 0.70: DER 6.04%, MS 5.02%, FA 0.74%, SC 0.28%
#Eval for threshold 0.80: DER 7.66%, MS 7.02%, FA 0.48%, SC 0.16%
## test of alimeeting
#Eval for threshold 0.20: DER 10.33%, MS 0.95%, FA 8.98%, SC 0.39%
#Eval for threshold 0.30: DER 7.32%, MS 1.63%, FA 5.22%, SC 0.47%
#Eval for threshold 0.35: DER 6.52%, MS 1.99%, FA 4.04%, SC 0.49%
#Eval for threshold 0.40: DER 6.00%, MS 2.40%, FA 3.08%, SC 0.53%
#Eval for threshold 0.45: DER 5.77%, MS 2.85%, FA 2.32%, SC 0.61%
#Eval for threshold 0.50: DER 5.73%, MS 3.33%, FA 1.76%, SC 0.64%
#Eval for threshold 0.55: DER 5.83%, MS 3.95%, FA 1.26%, SC 0.62%
#Eval for threshold 0.60: DER 6.14%, MS 4.65%, FA 0.92%, SC 0.57%
#Eval for threshold 0.70: DER 7.28%, MS 6.46%, FA 0.44%, SC 0.38%
#Eval for threshold 0.80: DER 9.49%, MS 9.05%, FA 0.20%, SC 0.24%


#compared with stage124-125 of run_ts_vad2_hltsz.sh, stage126-127 d_state will reduce from 128 to 64
if [ ${stage} -le 126 ] && [ ${stop_stage} -ge 126 ];then
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
    dataset_name="alimeeting" # dataset name

    # for loading speaker embedding file
    spk_path=/data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/data/maduo/exp/speaker_diarization/alimeeting/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr1e4_single_backend_mamba2_multi_backend_transformer_ts_len6_d_state_64
    data_dir="/data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    rs_len=6
    segment_shift=2
    single_backend_type="mamba2"
    multi_backend_type="transformer"
    d_state=64
    num_transformer_layer=2
    CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15815 \
   ts_vad2/train_accelerate_ddp2_debug2.py \
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
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --select-encoder-layer-nums 6\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --exp-dir $exp_dir\
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state
fi

if [ ${stage} -le 127 ] && [ ${stop_stage} -ge 127 ];then
 exp_dir=/data/maduo/exp/speaker_diarization/alimeeting/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr1e4_single_backend_mamba2_multi_backend_transformer_ts_len6_d_state_64
 model_file=$exp_dir/best-valid-der.pt
 rs_len=6
 segment_shift=1
 single_backend_type="mamba2"
 multi_backend_type="transformer"
 d_state=64
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/data/maduo/model_hub/ts_vad/
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 dataset_name="alimeeting" # dataset name
 # for loading speaker embedding file
 spk_path=/data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_collar${c}
  python3 ts_vad2/infer2.py \
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
    --d-state $d_state
 done
done
fi
#grep -r Eval  logs/run_ts_vad2_hltsz_stage127.log
# collar=0.0
# dev of alimeeting
#Eval for threshold 0.20: DER 20.35%, MS 2.43%, FA 16.90%, SC 1.02%
#Eval for threshold 0.30: DER 16.03%, MS 3.78%, FA 11.06%, SC 1.20%
#Eval for threshold 0.35: DER 14.81%, MS 4.44%, FA 9.11%, SC 1.26%
#Eval for threshold 0.40: DER 14.02%, MS 5.19%, FA 7.53%, SC 1.29%
#Eval for threshold 0.45: DER 13.49%, MS 5.99%, FA 6.23%, SC 1.26%
#Eval for threshold 0.50: DER 13.32%, MS 6.92%, FA 5.18%, SC 1.21%
#Eval for threshold 0.55: DER 13.37%, MS 7.93%, FA 4.31%, SC 1.13%
#Eval for threshold 0.60: DER 13.53%, MS 8.91%, FA 3.60%, SC 1.02%
#Eval for threshold 0.70: DER 14.57%, MS 11.41%, FA 2.39%, SC 0.77%
#Eval for threshold 0.80: DER 17.07%, MS 15.07%, FA 1.48%, SC 0.52%

# test of alimeeting
#Eval for threshold 0.20: DER 19.43%, MS 2.45%, FA 15.89%, SC 1.09%
#Eval for threshold 0.30: DER 15.55%, MS 3.88%, FA 10.36%, SC 1.30%
#Eval for threshold 0.35: DER 14.50%, MS 4.66%, FA 8.48%, SC 1.35%
#Eval for threshold 0.40: DER 13.88%, MS 5.50%, FA 6.98%, SC 1.39%
#Eval for threshold 0.45: DER 13.48%, MS 6.33%, FA 5.74%, SC 1.40%
#Eval for threshold 0.50: DER 13.39%, MS 7.32%, FA 4.71%, SC 1.37%
#Eval for threshold 0.55: DER 13.56%, MS 8.39%, FA 3.85%, SC 1.32%
#Eval for threshold 0.60: DER 13.95%, MS 9.61%, FA 3.13%, SC 1.21%
#Eval for threshold 0.70: DER 15.58%, MS 12.62%, FA 2.00%, SC 0.96%
#Eval for threshold 0.80: DER 18.52%, MS 16.65%, FA 1.18%, SC 0.68%

# collar=0.25
# dev of alimeeting
#Eval for threshold 0.20: DER 9.24%, MS 0.88%, FA 8.07%, SC 0.29%
#Eval for threshold 0.30: DER 6.55%, MS 1.45%, FA 4.75%, SC 0.35%
#Eval for threshold 0.35: DER 5.79%, MS 1.70%, FA 3.73%, SC 0.36%
#Eval for threshold 0.40: DER 5.29%, MS 2.05%, FA 2.88%, SC 0.37%
#Eval for threshold 0.45: DER 5.08%, MS 2.47%, FA 2.24%, SC 0.37%
#Eval for threshold 0.50: DER 5.07%, MS 2.93%, FA 1.77%, SC 0.37%
#Eval for threshold 0.55: DER 5.16%, MS 3.43%, FA 1.38%, SC 0.34%
#Eval for threshold 0.60: DER 5.34%, MS 3.94%, FA 1.10%, SC 0.30%
#Eval for threshold 0.70: DER 6.27%, MS 5.37%, FA 0.70%, SC 0.21%
#Eval for threshold 0.80: DER 8.10%, MS 7.54%, FA 0.44%, SC 0.12%
# test of alimeeting
#Eval for threshold 0.20: DER 9.22%, MS 1.09%, FA 7.70%, SC 0.44%
#Eval for threshold 0.30: DER 6.65%, MS 1.79%, FA 4.31%, SC 0.55%
#Eval for threshold 0.35: DER 5.97%, MS 2.15%, FA 3.23%, SC 0.59%
#Eval for threshold 0.40: DER 5.65%, MS 2.60%, FA 2.44%, SC 0.61%
#Eval for threshold 0.45: DER 5.49%, MS 3.03%, FA 1.84%, SC 0.62%
#Eval for threshold 0.50: DER 5.55%, MS 3.57%, FA 1.38%, SC 0.61%
#Eval for threshold 0.55: DER 5.76%, MS 4.17%, FA 1.01%, SC 0.58%
#Eval for threshold 0.60: DER 6.16%, MS 4.90%, FA 0.74%, SC 0.52%
#Eval for threshold 0.70: DER 7.43%, MS 6.72%, FA 0.35%, SC 0.37%
#Eval for threshold 0.80: DER 9.68%, MS 9.28%, FA 0.16%, SC 0.23%

##compared with stage122-123 of run_ts_vad2.sh, stage128-129 rs_len will increase from 4 to 6
if [ ${stage} -le 128 ] && [ ${stop_stage} -ge 128 ];then
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
    dataset_name="alimeeting" # dataset name

    # for loading speaker embedding file
    spk_path=/data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/data/maduo/exp/speaker_diarization/alimeeting/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_single_backend_transformer_multi_backend_transformer_ts_len6
    data_dir="/data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    rs_len=6
    segment_shift=2
    single_backend_type="transformer"
    multi_backend_type="transformer"
    d_state=64
    num_transformer_layer=2
    CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15815 \
   ts_vad2/train_accelerate_ddp2_debug2.py \
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
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --select-encoder-layer-nums 6\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --exp-dir $exp_dir\
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state
fi

if [ ${stage} -le 129 ] && [ ${stop_stage} -ge 129 ];then
 exp_dir=/data/maduo/exp/speaker_diarization/alimeeting/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_single_backend_transformer_multi_backend_transformer_ts_len6
 model_file=$exp_dir/best-valid-der.pt
 rs_len=6
 segment_shift=1
 single_backend_type="transformer"
 multi_backend_type="transformer"
 d_state=64
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/data/maduo/model_hub/ts_vad/
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 dataset_name="alimeeting" # dataset name
 # for loading speaker embedding file
 spk_path=/data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_collar${c}
  python3 ts_vad2/infer2.py \
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
    --d-state $d_state
 done
done
fi
#grep -r 'Eval' logs/run_ts_vad2_hltsz_stage128-129.log
# collar=0.0
# Eval of alimeeting
#Eval for threshold 0.20: DER 17.16%, MS 2.66%, FA 13.40%, SC 1.10%
#Eval for threshold 0.30: DER 14.37%, MS 3.82%, FA 9.33%, SC 1.21%
#Eval for threshold 0.35: DER 13.58%, MS 4.47%, FA 7.86%, SC 1.25%
#Eval for threshold 0.40: DER 13.07%, MS 5.10%, FA 6.71%, SC 1.26%
#Eval for threshold 0.45: DER 12.75%, MS 5.80%, FA 5.74%, SC 1.21%
#Eval for threshold 0.50: DER 12.57%, MS 6.53%, FA 4.87%, SC 1.16%
#Eval for threshold 0.55: DER 12.53%, MS 7.32%, FA 4.13%, SC 1.08%
#Eval for threshold 0.60: DER 12.75%, MS 8.23%, FA 3.50%, SC 1.02%
#Eval for threshold 0.70: DER 13.74%, MS 10.48%, FA 2.47%, SC 0.80%
#Eval for threshold 0.80: DER 15.70%, MS 13.53%, FA 1.58%, SC 0.60%

# Test of alimeeting
#Eval for threshold 0.20: DER 16.92%, MS 2.61%, FA 13.21%, SC 1.09%
#Eval for threshold 0.30: DER 14.18%, MS 3.80%, FA 9.09%, SC 1.28%
#Eval for threshold 0.35: DER 13.43%, MS 4.43%, FA 7.67%, SC 1.33%
#Eval for threshold 0.40: DER 12.93%, MS 5.10%, FA 6.47%, SC 1.36%
#Eval for threshold 0.45: DER 12.67%, MS 5.84%, FA 5.46%, SC 1.37%
#Eval for threshold 0.50: DER 12.65%, MS 6.68%, FA 4.61%, SC 1.36%
#Eval for threshold 0.55: DER 12.78%, MS 7.58%, FA 3.90%, SC 1.30%
#Eval for threshold 0.60: DER 13.07%, MS 8.60%, FA 3.27%, SC 1.21%
#Eval for threshold 0.70: DER 14.25%, MS 10.99%, FA 2.24%, SC 1.01%
#Eval for threshold 0.80: DER 16.50%, MS 14.33%, FA 1.42%, SC 0.75%
# collar=0.25
# Eval of alimeeting
#Eval for threshold 0.20: DER 7.36%, MS 0.92%, FA 6.08%, SC 0.36%
#Eval for threshold 0.30: DER 5.72%, MS 1.39%, FA 3.89%, SC 0.44%
#Eval for threshold 0.35: DER 5.28%, MS 1.64%, FA 3.17%, SC 0.46%
#Eval for threshold 0.40: DER 4.99%, MS 1.93%, FA 2.63%, SC 0.43%
#Eval for threshold 0.45: DER 4.82%, MS 2.24%, FA 2.16%, SC 0.42%
#Eval for threshold 0.50: DER 4.77%, MS 2.57%, FA 1.79%, SC 0.40%
#Eval for threshold 0.55: DER 4.80%, MS 2.99%, FA 1.45%, SC 0.35%
#Eval for threshold 0.60: DER 4.95%, MS 3.45%, FA 1.17%, SC 0.33%
#Eval for threshold 0.70: DER 5.64%, MS 4.64%, FA 0.75%, SC 0.25%
#Eval for threshold 0.80: DER 7.05%, MS 6.38%, FA 0.51%, SC 0.17%
# Test of alimeeting
#Eval for threshold 0.20: DER 7.47%, MS 1.10%, FA 5.96%, SC 0.40%
#Eval for threshold 0.30: DER 5.73%, MS 1.60%, FA 3.63%, SC 0.49%
#Eval for threshold 0.35: DER 5.27%, MS 1.88%, FA 2.87%, SC 0.52%
#Eval for threshold 0.40: DER 4.98%, MS 2.20%, FA 2.24%, SC 0.55%
#Eval for threshold 0.45: DER 4.87%, MS 2.56%, FA 1.76%, SC 0.55%
#Eval for threshold 0.50: DER 4.92%, MS 3.00%, FA 1.36%, SC 0.56%
#Eval for threshold 0.55: DER 5.06%, MS 3.47%, FA 1.07%, SC 0.52%
#Eval for threshold 0.60: DER 5.33%, MS 4.04%, FA 0.83%, SC 0.46%
#Eval for threshold 0.70: DER 6.26%, MS 5.44%, FA 0.45%, SC 0.37%
#Eval for threshold 0.80: DER 7.97%, MS 7.48%, FA 0.24%, SC 0.25%


#compared with stage126-127 of run_ts_vad2_hltsz.sh, stage130-131 rs_len will reduce from 6 to 4
if [ ${stage} -le 130 ] && [ ${stop_stage} -ge 130 ];then
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
    dataset_name="alimeeting" # dataset name

    # for loading speaker embedding file
    spk_path=/data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/data/maduo/exp/speaker_diarization/alimeeting/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr1e4_single_backend_mamba2_multi_backend_transformer_ts_len4_d_state_64
    data_dir="/data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    rs_len=4
    segment_shift=2
    single_backend_type="mamba2"
    multi_backend_type="transformer"
    d_state=64
    num_transformer_layer=2
    CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15715 \
   ts_vad2/train_accelerate_ddp2_debug2.py \
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
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --select-encoder-layer-nums 6\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --exp-dir $exp_dir\
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state
fi

if [ ${stage} -le 131 ] && [ ${stop_stage} -ge 131 ];then
 exp_dir=/data/maduo/exp/speaker_diarization/alimeeting/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr1e4_single_backend_mamba2_multi_backend_transformer_ts_len4_d_state_64
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 single_backend_type="mamba2"
 multi_backend_type="transformer"
 d_state=64
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/data/maduo/model_hub/ts_vad/
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 dataset_name="alimeeting" # dataset name
 # for loading speaker embedding file
 spk_path=/data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_collar${c}
  python3 ts_vad2/infer2.py \
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
    --d-state $d_state
 done
done
fi
# grep -r Eval logs/run_ts_vad2_hltsz_stage130-131_rs_len4_d_state64.log
# collar=0.0
# Eval of alimeeting
#Eval for threshold 0.20: DER 23.35%, MS 2.33%, FA 20.03%, SC 0.99%
#Eval for threshold 0.30: DER 17.47%, MS 3.79%, FA 12.37%, SC 1.31%
#Eval for threshold 0.35: DER 15.95%, MS 4.60%, FA 9.95%, SC 1.40%
#Eval for threshold 0.40: DER 14.95%, MS 5.41%, FA 8.09%, SC 1.44%
#Eval for threshold 0.45: DER 14.36%, MS 6.38%, FA 6.54%, SC 1.44%
#Eval for threshold 0.50: DER 14.09%, MS 7.42%, FA 5.27%, SC 1.39%
#Eval for threshold 0.55: DER 14.11%, MS 8.51%, FA 4.31%, SC 1.29%
#Eval for threshold 0.60: DER 14.44%, MS 9.83%, FA 3.47%, SC 1.14%
#Eval for threshold 0.70: DER 15.92%, MS 12.85%, FA 2.24%, SC 0.83%
#Eval for threshold 0.80: DER 19.04%, MS 17.13%, FA 1.38%, SC 0.52%

# Test of alimeeting
#Eval for threshold 0.20: DER 22.24%, MS 2.24%, FA 18.89%, SC 1.11%
#Eval for threshold 0.30: DER 16.96%, MS 3.75%, FA 11.83%, SC 1.38%
#Eval for threshold 0.35: DER 15.54%, MS 4.59%, FA 9.49%, SC 1.47%
#Eval for threshold 0.40: DER 14.66%, MS 5.54%, FA 7.63%, SC 1.49%
#Eval for threshold 0.45: DER 14.19%, MS 6.56%, FA 6.13%, SC 1.49%
#Eval for threshold 0.50: DER 14.12%, MS 7.73%, FA 4.94%, SC 1.46%
#Eval for threshold 0.55: DER 14.32%, MS 8.99%, FA 3.95%, SC 1.38%
#Eval for threshold 0.60: DER 14.79%, MS 10.36%, FA 3.17%, SC 1.26%
#Eval for threshold 0.70: DER 16.76%, MS 13.81%, FA 1.95%, SC 0.99%
#Eval for threshold 0.80: DER 20.42%, MS 18.65%, FA 1.09%, SC 0.67%

# collar=0.25
# Eval of alimeeting
#Eval for threshold 0.20: DER 11.86%, MS 0.87%, FA 10.71%, SC 0.28%
#Eval for threshold 0.30: DER 7.83%, MS 1.48%, FA 5.93%, SC 0.42%
#Eval for threshold 0.35: DER 6.78%, MS 1.85%, FA 4.48%, SC 0.45%
#Eval for threshold 0.40: DER 6.12%, MS 2.23%, FA 3.41%, SC 0.47%
#Eval for threshold 0.45: DER 5.79%, MS 2.71%, FA 2.58%, SC 0.50%
#Eval for threshold 0.50: DER 5.60%, MS 3.23%, FA 1.89%, SC 0.48%
#Eval for threshold 0.55: DER 5.72%, MS 3.82%, FA 1.46%, SC 0.44%
#Eval for threshold 0.60: DER 6.05%, MS 4.57%, FA 1.09%, SC 0.39%
#Eval for threshold 0.70: DER 7.29%, MS 6.44%, FA 0.60%, SC 0.25%
#Eval for threshold 0.80: DER 9.83%, MS 9.29%, FA 0.40%, SC 0.14%

# Test of alimeeting
#Eval for threshold 0.20: DER 11.85%, MS 1.00%, FA 10.44%, SC 0.42%
#Eval for threshold 0.30: DER 7.86%, MS 1.79%, FA 5.53%, SC 0.54%
#Eval for threshold 0.35: DER 6.87%, MS 2.23%, FA 4.06%, SC 0.58%
#Eval for threshold 0.40: DER 6.31%, MS 2.75%, FA 2.97%, SC 0.60%
#Eval for threshold 0.45: DER 6.05%, MS 3.28%, FA 2.15%, SC 0.61%
#Eval for threshold 0.50: DER 6.09%, MS 3.94%, FA 1.55%, SC 0.59%
#Eval for threshold 0.55: DER 6.32%, MS 4.69%, FA 1.07%, SC 0.56%
#Eval for threshold 0.60: DER 6.76%, MS 5.50%, FA 0.77%, SC 0.49%
#Eval for threshold 0.70: DER 8.38%, MS 7.66%, FA 0.36%, SC 0.35%
#Eval for threshold 0.80: DER 11.29%, MS 10.92%, FA 0.15%, SC 0.22%


##compared with stage122-123 of run_ts_vad2.sh, stage132-133 rs_len will increase from 4 to 8
if [ ${stage} -le 132 ] && [ ${stop_stage} -ge 132 ];then
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
    dataset_name="alimeeting" # dataset name

    # for loading speaker embedding file
    spk_path=/data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/data/maduo/exp/speaker_diarization/alimeeting/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_single_backend_transformer_multi_backend_transformer_ts_len8
    data_dir="/data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    rs_len=8
    segment_shift=2
    single_backend_type="transformer"
    multi_backend_type="transformer"
    d_state=64
    num_transformer_layer=2
    CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15815 \
   ts_vad2/train_accelerate_ddp2_debug2.py \
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
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --select-encoder-layer-nums 6\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --exp-dir $exp_dir\
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state
fi

if [ ${stage} -le 133 ] && [ ${stop_stage} -ge 133 ];then
 exp_dir=/data/maduo/exp/speaker_diarization/alimeeting/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_single_backend_transformer_multi_backend_transformer_ts_len8
 model_file=$exp_dir/best-valid-der.pt
 rs_len=8
 segment_shift=1
 single_backend_type="transformer"
 multi_backend_type="transformer"
 d_state=64
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/data/maduo/model_hub/ts_vad/
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 dataset_name="alimeeting" # dataset name
 # for loading speaker embedding file
 spk_path=/data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_collar${c}
  python3 ts_vad2/infer2.py \
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
    --d-state $d_state
 done
done
fi
#grep -r Eval logs/run_ts_vad2_hltsz_stage132-133_rs_len8_transformer.log
# collar=0.0
# Eval of alimeeting
#Eval for threshold 0.20: DER 17.96%, MS 2.36%, FA 14.66%, SC 0.95%
#Eval for threshold 0.30: DER 14.55%, MS 3.46%, FA 10.04%, SC 1.05%
#Eval for threshold 0.35: DER 13.53%, MS 4.04%, FA 8.40%, SC 1.09%
#Eval for threshold 0.40: DER 12.88%, MS 4.70%, FA 7.07%, SC 1.11%
#Eval for threshold 0.45: DER 12.42%, MS 5.33%, FA 6.00%, SC 1.09%
#Eval for threshold 0.50: DER 12.33%, MS 6.14%, FA 5.17%, SC 1.03%
#Eval for threshold 0.55: DER 12.27%, MS 6.90%, FA 4.43%, SC 0.95%
#Eval for threshold 0.60: DER 12.42%, MS 7.77%, FA 3.77%, SC 0.88%
#Eval for threshold 0.70: DER 13.28%, MS 9.96%, FA 2.61%, SC 0.71%
#Eval for threshold 0.80: DER 15.11%, MS 12.92%, FA 1.66%, SC 0.53%

# Test of alimeeting
#Eval for threshold 0.20: DER 17.31%, MS 2.43%, FA 13.85%, SC 1.03%
#Eval for threshold 0.30: DER 14.34%, MS 3.61%, FA 9.54%, SC 1.19%
#Eval for threshold 0.35: DER 13.51%, MS 4.26%, FA 8.02%, SC 1.23%
#Eval for threshold 0.40: DER 12.99%, MS 4.93%, FA 6.80%, SC 1.27%
#Eval for threshold 0.45: DER 12.64%, MS 5.63%, FA 5.75%, SC 1.26%
#Eval for threshold 0.50: DER 12.48%, MS 6.39%, FA 4.83%, SC 1.26%
#Eval for threshold 0.55: DER 12.53%, MS 7.25%, FA 4.06%, SC 1.21%
#Eval for threshold 0.60: DER 12.78%, MS 8.21%, FA 3.43%, SC 1.15%
#Eval for threshold 0.70: DER 13.77%, MS 10.48%, FA 2.34%, SC 0.96%
#Eval for threshold 0.80: DER 15.85%, MS 13.69%, FA 1.44%, SC 0.71%

# collar=0.25
# Eval of alimeeting
#Eval for threshold 0.20: DER 7.72%, MS 0.81%, FA 6.64%, SC 0.27%
#Eval for threshold 0.30: DER 5.66%, MS 1.24%, FA 4.11%, SC 0.31%
#Eval for threshold 0.35: DER 5.05%, MS 1.44%, FA 3.28%, SC 0.33%
#Eval for threshold 0.40: DER 4.69%, MS 1.74%, FA 2.62%, SC 0.33%
#Eval for threshold 0.45: DER 4.47%, MS 2.01%, FA 2.15%, SC 0.32%
#Eval for threshold 0.50: DER 4.48%, MS 2.36%, FA 1.82%, SC 0.29%
#Eval for threshold 0.55: DER 4.49%, MS 2.71%, FA 1.53%, SC 0.25%
#Eval for threshold 0.60: DER 4.60%, MS 3.12%, FA 1.25%, SC 0.23%
#Eval for threshold 0.70: DER 5.21%, MS 4.20%, FA 0.83%, SC 0.18%
#Eval for threshold 0.80: DER 6.56%, MS 5.88%, FA 0.55%, SC 0.13%

# Test of alimeeting
#Eval for threshold 0.20: DER 7.56%, MS 0.95%, FA 6.25%, SC 0.36%
#Eval for threshold 0.30: DER 5.67%, MS 1.45%, FA 3.79%, SC 0.42%
#Eval for threshold 0.35: DER 5.15%, MS 1.75%, FA 2.97%, SC 0.44%
#Eval for threshold 0.40: DER 4.88%, MS 2.06%, FA 2.36%, SC 0.46%
#Eval for threshold 0.45: DER 4.74%, MS 2.41%, FA 1.86%, SC 0.47%
#Eval for threshold 0.50: DER 4.70%, MS 2.78%, FA 1.44%, SC 0.47%
#Eval for threshold 0.55: DER 4.78%, MS 3.20%, FA 1.13%, SC 0.45%
#Eval for threshold 0.60: DER 5.00%, MS 3.69%, FA 0.89%, SC 0.42%
#Eval for threshold 0.70: DER 5.79%, MS 4.94%, FA 0.53%, SC 0.32%
#Eval for threshold 0.80: DER 7.21%, MS 6.73%, FA 0.27%, SC 0.21%


##compared with stage122-123 of run_ts_vad2.sh, stage134-135 rs_len will increase from 4 to 10
if [ ${stage} -le 134 ] && [ ${stop_stage} -ge 134 ];then
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
    dataset_name="alimeeting" # dataset name

    # for loading speaker embedding file
    spk_path=/data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/data/maduo/exp/speaker_diarization/alimeeting/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_single_backend_transformer_multi_backend_transformer_ts_len10
    data_dir="/data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    rs_len=10
    segment_shift=2
    single_backend_type="transformer"
    multi_backend_type="transformer"
    d_state=64
    num_transformer_layer=2
    CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15715 \
   ts_vad2/train_accelerate_ddp2_debug2.py \
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
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --select-encoder-layer-nums 6\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --exp-dir $exp_dir\
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state
fi

if [ ${stage} -le 135 ] && [ ${stop_stage} -ge 135 ];then
 exp_dir=/data/maduo/exp/speaker_diarization/alimeeting/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_single_backend_transformer_multi_backend_transformer_ts_len10
 model_file=$exp_dir/best-valid-der.pt
 rs_len=10
 segment_shift=1
 single_backend_type="transformer"
 multi_backend_type="transformer"
 d_state=64
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/data/maduo/model_hub/ts_vad/
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 dataset_name="alimeeting" # dataset name
 # for loading speaker embedding file
 spk_path=/data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_collar${c}
  python3 ts_vad2/infer2.py \
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
    --d-state $d_state
 done
done
fi
#grep -r Eval  logs/run_ts_vad2_hltsz_stage134-135_rs_len10_transformer.log
# collar=0.0
# Eval of alimeeting
#Eval for threshold 0.20: DER 17.76%, MS 2.41%, FA 14.37%, SC 0.99%
#Eval for threshold 0.30: DER 14.57%, MS 3.50%, FA 9.92%, SC 1.15%
#Eval for threshold 0.35: DER 13.69%, MS 4.11%, FA 8.43%, SC 1.16%
#Eval for threshold 0.40: DER 13.05%, MS 4.74%, FA 7.17%, SC 1.14%
#Eval for threshold 0.45: DER 12.60%, MS 5.34%, FA 6.12%, SC 1.15%
#Eval for threshold 0.50: DER 12.34%, MS 6.07%, FA 5.18%, SC 1.09%
#Eval for threshold 0.55: DER 12.34%, MS 6.88%, FA 4.42%, SC 1.04%
#Eval for threshold 0.60: DER 12.50%, MS 7.76%, FA 3.74%, SC 1.00%
#Eval for threshold 0.70: DER 13.35%, MS 9.89%, FA 2.61%, SC 0.84%
#Eval for threshold 0.80: DER 15.15%, MS 12.76%, FA 1.77%, SC 0.63%

# Test of alimeeting
#Eval for threshold 0.20: DER 17.97%, MS 2.46%, FA 14.41%, SC 1.10%
#Eval for threshold 0.30: DER 14.92%, MS 3.61%, FA 10.01%, SC 1.30%
#Eval for threshold 0.35: DER 14.05%, MS 4.24%, FA 8.44%, SC 1.37%
#Eval for threshold 0.40: DER 13.45%, MS 4.91%, FA 7.13%, SC 1.42%
#Eval for threshold 0.45: DER 13.08%, MS 5.67%, FA 5.97%, SC 1.43%
#Eval for threshold 0.50: DER 12.91%, MS 6.49%, FA 5.00%, SC 1.43%
#Eval for threshold 0.55: DER 12.96%, MS 7.35%, FA 4.22%, SC 1.39%
#Eval for threshold 0.60: DER 13.18%, MS 8.36%, FA 3.50%, SC 1.32%
#Eval for threshold 0.70: DER 14.28%, MS 10.79%, FA 2.38%, SC 1.11%
#Eval for threshold 0.80: DER 16.38%, MS 14.07%, FA 1.49%, SC 0.81%
# 
# collar=0.25
# Eval of alimeeting
#Eval for threshold 0.20: DER 7.46%, MS 0.80%, FA 6.37%, SC 0.30%
#Eval for threshold 0.30: DER 5.62%, MS 1.25%, FA 4.02%, SC 0.36%
#Eval for threshold 0.35: DER 5.10%, MS 1.49%, FA 3.24%, SC 0.37%
#Eval for threshold 0.40: DER 4.79%, MS 1.77%, FA 2.67%, SC 0.35%
#Eval for threshold 0.45: DER 4.57%, MS 2.04%, FA 2.16%, SC 0.37%
#Eval for threshold 0.50: DER 4.43%, MS 2.34%, FA 1.75%, SC 0.34%
#Eval for threshold 0.55: DER 4.48%, MS 2.73%, FA 1.41%, SC 0.34%
#Eval for threshold 0.60: DER 4.65%, MS 3.14%, FA 1.19%, SC 0.31%
#Eval for threshold 0.70: DER 5.29%, MS 4.22%, FA 0.81%, SC 0.26%
#Eval for threshold 0.80: DER 6.55%, MS 5.84%, FA 0.55%, SC 0.17%

# Test of alimeeting
#Eval for threshold 0.20: DER 8.30%, MS 1.02%, FA 6.84%, SC 0.45%
#Eval for threshold 0.30: DER 6.31%, MS 1.55%, FA 4.21%, SC 0.55%
#Eval for threshold 0.35: DER 5.76%, MS 1.82%, FA 3.34%, SC 0.60%
#Eval for threshold 0.40: DER 5.44%, MS 2.16%, FA 2.64%, SC 0.65%
#Eval for threshold 0.45: DER 5.24%, MS 2.52%, FA 2.06%, SC 0.67%
#Eval for threshold 0.50: DER 5.20%, MS 2.95%, FA 1.59%, SC 0.66%
#Eval for threshold 0.55: DER 5.30%, MS 3.43%, FA 1.23%, SC 0.64%
#Eval for threshold 0.60: DER 5.51%, MS 3.99%, FA 0.92%, SC 0.60%
#Eval for threshold 0.70: DER 6.34%, MS 5.37%, FA 0.51%, SC 0.46%
#Eval for threshold 0.80: DER 7.85%, MS 7.29%, FA 0.26%, SC 0.30%



##compared with stage122-123 of run_ts_vad2.sh, stage136-137 rs_len will increase from 4 to 12
if [ ${stage} -le 136 ] && [ ${stop_stage} -ge 136 ];then
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
    dataset_name="alimeeting" # dataset name

    # for loading speaker embedding file
    spk_path=/data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/data/maduo/exp/speaker_diarization/alimeeting/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_single_backend_transformer_multi_backend_transformer_ts_len12
    data_dir="/data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    rs_len=12
    segment_shift=2
    single_backend_type="transformer"
    multi_backend_type="transformer"
    d_state=64
    num_transformer_layer=2
    CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15715 \
   ts_vad2/train_accelerate_ddp2_debug2.py \
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
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --select-encoder-layer-nums 6\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --exp-dir $exp_dir\
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state
fi

if [ ${stage} -le 137 ] && [ ${stop_stage} -ge 137 ];then
 exp_dir=/data/maduo/exp/speaker_diarization/alimeeting/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_single_backend_transformer_multi_backend_transformer_ts_len12
 model_file=$exp_dir/best-valid-der.pt
 rs_len=12
 segment_shift=1
 single_backend_type="transformer"
 multi_backend_type="transformer"
 d_state=64
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/data/maduo/model_hub/ts_vad/
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 dataset_name="alimeeting" # dataset name
 # for loading speaker embedding file
 spk_path=/data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_collar${c}
  python3 ts_vad2/infer2.py \
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
    --d-state $d_state
 done
done
fi

##compared with stage122-123 of run_ts_vad2.sh, stage138-139 rs_len will increase from 4 to 15
if [ ${stage} -le 138 ] && [ ${stop_stage} -ge 138 ];then
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
    dataset_name="alimeeting" # dataset name

    # for loading speaker embedding file
    spk_path=/data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/data/maduo/exp/speaker_diarization/alimeeting/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_single_backend_transformer_multi_backend_transformer_ts_len15
    data_dir="/data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    rs_len=15
    segment_shift=2
    single_backend_type="transformer"
    multi_backend_type="transformer"
    d_state=64
    num_transformer_layer=2
    CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15615 \
   ts_vad2/train_accelerate_ddp2_debug2.py \
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
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --select-encoder-layer-nums 6\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --exp-dir $exp_dir\
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state
fi

if [ ${stage} -le 139 ] && [ ${stop_stage} -ge 139 ];then
 exp_dir=/data/maduo/exp/speaker_diarization/alimeeting/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_single_backend_transformer_multi_backend_transformer_ts_len15
 model_file=$exp_dir/best-valid-der.pt
 rs_len=15
 segment_shift=1
 single_backend_type="transformer"
 multi_backend_type="transformer"
 d_state=64
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/data/maduo/model_hub/ts_vad/
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 dataset_name="alimeeting" # dataset name
 # for loading speaker embedding file
 spk_path=/data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_collar${c}
  python3 ts_vad2/infer2.py \
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
    --d-state $d_state
 done
done
fi



# compared stage104-105 of run_ts_vad2.sh stage140-141 use cam++_200k to replese cam++ zh_en
if [ ${stage} -le 140 ] && [ ${stop_stage} -ge 140 ];then
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
    speech_encoder_type="w2v-bert2"
    speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
    speech_encoder_config="/data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    dataset_name="alimeeting" # dataset name

    # for loading speaker embedding file
    spk_path=/data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/data/maduo/exp/speaker_diarization/alimeeting/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch40_front_fix_seed_lr1e4_single_backend_transformer_multi_backend_transformer_rs_len4_shift2
    data_dir="/data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    rs_len=4
    segment_shift=2
    single_backend_type="transformer"
    multi_backend_type="transformer"
    d_state=64
    num_transformer_layer=2
    CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15615 \
   ts_vad2/train_accelerate_ddp2_debug2.py \
    --world-size 2 \
    --num-epochs 40\
    --start-epoch 1\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-updates 4000\
    --grad-clip true\
    --lr 1e-4\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --speech-encoder-config $speech_encoder_config\
    --select-encoder-layer-nums 6\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --exp-dir $exp_dir\
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state
fi


if [ ${stage} -le 141 ] && [ ${stop_stage} -ge 141 ];then
 
 exp_dir=/data/maduo/exp/speaker_diarization/alimeeting/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch40_front_fix_seed_lr1e4_single_backend_transformer_multi_backend_transformer_rs_len4_shift2 
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 single_backend_type="transformer"
 multi_backend_type="transformer"
 d_state=64
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/data/maduo/model_hub/ts_vad/
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 dataset_name="alimeeting" # dataset name
 speech_encoder_type="w2v-bert2"
 speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 speech_encoder_config="/data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 spk_path=/data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_collar${c}
  python3 ts_vad2/infer2.py \
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
    --d-state $d_state
 done
done
fi



# compared stage138-139 of run_ts_vad2.sh stage144-145 use cam++_200k to replese cam++ zh_en
if [ ${stage} -le 144 ] && [ ${stop_stage} -ge 144 ];then                                                                                                                                                                                                                                                                                                                                                                                                                                                                    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
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
    speech_encoder_type="w2v-bert2"
    speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
    speech_encoder_config="/data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    dataset_name="alimeeting" # dataset name

    # for loading speaker embedding file
    spk_path=/data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/data/maduo/exp/speaker_diarization/alimeeting/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch40_front_fix_seed_lr1e4_single_backend_mamba2_multi_backend_transformer_rs_len10_shift2_d_state256
    data_dir="/data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    rs_len=10
    segment_shift=2
    single_backend_type="mamba2"
    multi_backend_type="transformer"
    d_state=256
    num_transformer_layer=2
    CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15715 \
   ts_vad2/train_accelerate_ddp2_debug2.py \
    --world-size 2 \
    --num-epochs 40\
    --start-epoch 1\
    --keep-last-k 10\
    --keep-last-epoch 10\
    --freeze-updates 4000\
    --grad-clip true\
    --lr 1e-4\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --speech-encoder-config $speech_encoder_config\
    --select-encoder-layer-nums 6\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --exp-dir $exp_dir\
    --data-dir $data_dir\
    --dataset-name $dataset_name\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --single-backend-type $single_backend_type\
    --multi-backend-type $multi_backend_type\
    --num-transformer-layer $num_transformer_layer\
    --d-state $d_state
fi


if [ ${stage} -le 145 ] && [ ${stop_stage} -ge 145 ];then
 exp_dir=/data/maduo/exp/speaker_diarization/alimeeting/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch40_front_fix_seed_lr1e4_single_backend_mamba2_multi_backend_transformer_rs_len10_shift2_d_state256
 model_file=$exp_dir/best-valid-der.pt
 rs_len=10
 segment_shift=1
 single_backend_type="mamba2"
 multi_backend_type="transformer"
 d_state=256
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/data/maduo/model_hub/ts_vad/
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 dataset_name="alimeeting" # dataset name
 speech_encoder_type="w2v-bert2"
 speech_encoder_path="/data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 speech_encoder_config="/data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 spk_path=/data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
 for c in $collar;do
  for name in $infer_sets;do
    results_path=$exp_dir/${dataset_name}_collar${c}
  python3 ts_vad2/infer2.py \
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
    --d-state $d_state
 done
done
fi
