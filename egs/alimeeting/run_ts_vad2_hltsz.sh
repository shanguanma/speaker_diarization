#!/usr/bin/env bash

stage=0
stop_stage=1000
. utils/parse_options.sh
. path_for_speaker_diarization_hltsz.sh


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


