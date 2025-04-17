#!/usr/bin/env bash

stage=0
stop_stage=1000

. utils/parse_options.sh
. path_for_dia_cuda11.8_py3111_aistation.sh

if [ ${stage} -le 148 ] && [ ${stop_stage} -ge 148 ];then
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
    speech_encoder_type="CAM++_ots_vad"
    speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt"

    #speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
    dataset_name="alimeeting" # dataset name

    # for loading speaker embedding file
    spk_path=/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_en_zh_advanced_feature_dir_epoch20_front_fix_seed_lr2e4_single_backend_conformer_6layer__multi_backend_lstm_ts_len6_ots_vad_style_v1
    mkdir -p $exp_dir
    data_dir="/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    rs_len=6
    segment_shift=2
    single_backend_type="conformer_ots_vad"
    multi_backend_type="lstm_ots_vad"
    d_state=256
    ots_vad_style="v1"
    num_transformer_layer=2
    CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15115 \
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
    --d-state $d_state\
    --ots-vad-style $ots_vad_style
fi


if [ ${stage} -le 149 ] && [ ${stop_stage} -ge 149 ];then
 exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_en_zh_advanced_feature_dir_epoch20_front_fix_seed_lr2e4_single_backend_conformer_6layer__multi_backend_lstm_ts_len6_ots_vad_style_v1
 model_file=$exp_dir/best-valid-der.pt
 rs_len=6
 segment_shift=1
 single_backend_type="conformer_ots_vad"
 multi_backend_type="lstm_ots_vad"
 ots_vad_style="v1"
 d_state=256
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++_ots_vad"
 speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt"
 dataset_name="alimeeting" # dataset name
 # for loading speaker embedding file
 spk_path=/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 data_dir="/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
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
    --d-state $d_state\
    --ots-vad-style $ots_vad_style
 done
done
fi

#grep -r Eval logs/run_ts_vad2_aistation_stage148-149.log 
# Eval of alimeeting, collar=0.0
#Eval for threshold 0.20: DER 26.25%, MS 3.17%, FA 21.18%, SC 1.89%
#Eval for threshold 0.30: DER 20.34%, MS 5.01%, FA 12.76%, SC 2.57%
#Eval for threshold 0.35: DER 18.65%, MS 5.97%, FA 9.83%, SC 2.84%
#Eval for threshold 0.40: DER 17.67%, MS 7.10%, FA 7.48%, SC 3.09%
#Eval for threshold 0.45: DER 17.30%, MS 8.38%, FA 5.68%, SC 3.24%
#Eval for threshold 0.50: DER 17.41%, MS 9.85%, FA 4.30%, SC 3.26%
#Eval for threshold 0.55: DER 18.05%, MS 11.65%, FA 3.31%, SC 3.09%
#Eval for threshold 0.60: DER 18.97%, MS 13.60%, FA 2.60%, SC 2.77%
#Eval for threshold 0.70: DER 21.94%, MS 18.38%, FA 1.57%, SC 1.98%
#Eval for threshold 0.80: DER 26.75%, MS 24.60%, FA 0.96%, SC 1.19%
# Test of alimeeting, collar=0.0
#Eval for threshold 0.20: DER 26.74%, MS 2.90%, FA 21.50%, SC 2.34%
#Eval for threshold 0.30: DER 21.35%, MS 4.73%, FA 13.35%, SC 3.26%
#Eval for threshold 0.35: DER 19.90%, MS 5.79%, FA 10.46%, SC 3.64%
#Eval for threshold 0.40: DER 18.97%, MS 6.95%, FA 8.05%, SC 3.97%
#Eval for threshold 0.45: DER 18.56%, MS 8.34%, FA 6.02%, SC 4.20%
#Eval for threshold 0.50: DER 18.61%, MS 9.97%, FA 4.37%, SC 4.27%
#Eval for threshold 0.55: DER 19.27%, MS 12.01%, FA 3.25%, SC 4.01%
#Eval for threshold 0.60: DER 20.37%, MS 14.36%, FA 2.42%, SC 3.58%
#Eval for threshold 0.70: DER 23.83%, MS 19.99%, FA 1.30%, SC 2.55%
#Eval for threshold 0.80: DER 29.50%, MS 27.39%, FA 0.64%, SC 1.46%

# Eval of alimeeting, collar=0.25
#Eval for threshold 0.20: DER 16.13%, MS 1.14%, FA 14.02%, SC 0.98%
#Eval for threshold 0.30: DER 11.31%, MS 1.93%, FA 7.82%, SC 1.56%
#Eval for threshold 0.35: DER 9.91%, MS 2.40%, FA 5.68%, SC 1.82%
#Eval for threshold 0.40: DER 9.11%, MS 2.93%, FA 4.02%, SC 2.15%
#Eval for threshold 0.45: DER 8.80%, MS 3.65%, FA 2.77%, SC 2.37%
#Eval for threshold 0.50: DER 8.86%, MS 4.56%, FA 1.87%, SC 2.43%
#Eval for threshold 0.55: DER 9.43%, MS 5.81%, FA 1.23%, SC 2.39%
#Eval for threshold 0.60: DER 10.31%, MS 7.32%, FA 0.88%, SC 2.12%
#Eval for threshold 0.70: DER 13.05%, MS 11.13%, FA 0.49%, SC 1.43%
#Eval for threshold 0.80: DER 17.27%, MS 16.07%, FA 0.34%, SC 0.86%

# Test of alimeeting, collar=0.25
#Eval for threshold 0.20: DER 17.92%, MS 1.32%, FA 15.08%, SC 1.51%
#Eval for threshold 0.30: DER 13.30%, MS 2.25%, FA 8.64%, SC 2.40%
#Eval for threshold 0.35: DER 12.00%, MS 2.80%, FA 6.35%, SC 2.86%
#Eval for threshold 0.40: DER 11.21%, MS 3.45%, FA 4.54%, SC 3.22%
#Eval for threshold 0.45: DER 10.78%, MS 4.25%, FA 2.96%, SC 3.57%
#Eval for threshold 0.50: DER 10.86%, MS 5.36%, FA 1.77%, SC 3.73%
#Eval for threshold 0.55: DER 11.46%, MS 6.86%, FA 1.09%, SC 3.52%
#Eval for threshold 0.60: DER 12.47%, MS 8.72%, FA 0.67%, SC 3.08%
#Eval for threshold 0.70: DER 15.58%, MS 13.30%, FA 0.23%, SC 2.05%
#Eval for threshold 0.80: DER 20.72%, MS 19.57%, FA 0.07%, SC 1.08%


if [ ${stage} -le 155 ] && [ ${stop_stage} -ge 155 ];then
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
    exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_epoch20_front_fix_seed_lr2e4_single_backend_transformer_multi_backend_transformer_rs_len8_shift0.8
    mkdir -p $exp_dir
    data_dir="/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    rs_len=8
    segment_shift=0.8
    single_backend_type="transformer"
    multi_backend_type="transformer"
    num_transformer_layer=2
    CUDA_VISIABLE_DEVICES=0,1 \
    NCCL_ASYNC_ERROR_HANDLING=1\
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15115 \
   ts_vad2/train_accelerate_ddp2_debug2.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 18\
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
    --num-transformer-layer $num_transformer_layer
fi

if [ ${stage} -le 156 ] && [ ${stop_stage} -ge 156 ];then
 exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_epoch20_front_fix_seed_lr2e4_single_backend_transformer_multi_backend_transformer_rs_len8_shift0.8
 model_file=$exp_dir/best-valid-der.pt
 rs_len=8
 segment_shift=0.8
 single_backend_type="transformer"
 multi_backend_type="transformer"
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0
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
    --num-transformer-layer $num_transformer_layer
 done
done
fi


if [ ${stage} -le 160 ] && [ ${stop_stage} -ge 160 ];then
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
    exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_epoch20_front_fix_seed_lr2e4_single_backend_transformer_multi_backend_transformer_rs_len8_shift2_enhance_ratio0.3_offline_se_aug
    #exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_epoch20_front_fix_seed_lr2e4_single_backend_transformer_multi_backend_transformer_rs_len8_shift2_enhance_ratio0.3
    mkdir -p $exp_dir
    data_dir="/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    enhanced_audio_dir="/maduo/datasets/zipenhancer_alimeeting/"
    rs_len=8
    segment_shift=2
    enhance_ratio=0.3
    single_backend_type="transformer"
    multi_backend_type="transformer"
    num_transformer_layer=2
    CUDA_VISIABLE_DEVICES=0,1 \
   TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15215 \
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
    --enhance-ratio $enhance_ratio\
    --enhanced-audio-dir $enhanced_audio_dir
fi

if [ ${stage} -le 161 ] && [ ${stop_stage} -ge 161 ];then
 exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_epoch20_front_fix_seed_lr2e4_single_backend_transformer_multi_backend_transformer_rs_len8_shift2_enhance_ratio0.3_offline_se_aug
 model_file=$exp_dir/best-valid-der.pt
 rs_len=8
 segment_shift=1
 single_backend_type="transformer"
 multi_backend_type="transformer"
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0
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
    --num-transformer-layer $num_transformer_layer
 done
done
fi


# continue to train from epoch3(using online extract speech enhancement audio) via offline extract speech enhancement audio
if [ ${stage} -le 162 ] && [ ${stop_stage} -ge 162 ];then
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
    #exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_epoch20_front_fix_seed_lr2e4_single_backend_transformer_multi_backend_transformer_rs_len8_shift2_enhance_ratio0.3_offline_se_aug
    exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_epoch20_front_fix_seed_lr2e4_single_backend_transformer_multi_backend_transformer_rs_len8_shift2_enhance_ratio0.3
    mkdir -p $exp_dir
    data_dir="/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    enhanced_audio_dir="/maduo/datasets/zipenhancer_alimeeting/"
    rs_len=8
    segment_shift=2
    enhance_ratio=0.3
    single_backend_type="transformer"
    multi_backend_type="transformer"
    num_transformer_layer=2
    CUDA_VISIABLE_DEVICES=0,1 \
   TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15315 \
    ts_vad2/train_accelerate_ddp2_debug2.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 3\
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
    --enhance-ratio $enhance_ratio\
    --enhanced-audio-dir $enhanced_audio_dir
fi

if [ ${stage} -le 163 ] && [ ${stop_stage} -ge 163 ];then
 exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_epoch20_front_fix_seed_lr2e4_single_backend_transformer_multi_backend_transformer_rs_len8_shift2_enhance_ratio0.3
 model_file=$exp_dir/best-valid-der.pt
 rs_len=8
 segment_shift=1
 single_backend_type="transformer"
 multi_backend_type="transformer"
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0
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
    --num-transformer-layer $num_transformer_layer
 done
done
fi
