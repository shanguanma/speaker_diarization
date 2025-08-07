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
#Eval for threshold 0.45: DER 17.30%, MS 8.38%, FA 5.68%, SC 3.24% as report
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
#Eval for threshold 0.45: DER 18.56%, MS 8.34%, FA 6.02%, SC 4.20% as report
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
#Eval for threshold 0.45: DER 8.80%, MS 3.65%, FA 2.77%, SC 2.37% as report
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
#Eval for threshold 0.45: DER 10.78%, MS 4.25%, FA 2.96%, SC 3.57% as report
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
# Eval of alimeeting, collar=0.0
# Eval for threshold 0.20: DER 14.70%, MS 3.10%, FA 10.44%, SC 1.16%
#Eval for threshold 0.30: DER 12.88%, MS 4.25%, FA 7.47%, SC 1.16%
#Eval for threshold 0.35: DER 12.47%, MS 4.81%, FA 6.50%, SC 1.16%
#Eval for threshold 0.40: DER 12.15%, MS 5.35%, FA 5.63%, SC 1.17% 
#Eval for threshold 0.45: DER 12.06%, MS 5.99%, FA 4.91%, SC 1.15% as report
#Eval for threshold 0.50: DER 12.03%, MS 6.64%, FA 4.27%, SC 1.12%
#Eval for threshold 0.55: DER 12.13%, MS 7.37%, FA 3.71%, SC 1.05%
#Eval for threshold 0.60: DER 12.35%, MS 8.15%, FA 3.21%, SC 0.99%
#Eval for threshold 0.70: DER 13.10%, MS 9.91%, FA 2.37%, SC 0.82%
#Eval for threshold 0.80: DER 14.75%, MS 12.45%, FA 1.67%, SC 0.63%

# Test of alimeeting, collar=0.0
#Eval for threshold 0.20: DER 14.18%, MS 3.36%, FA 9.72%, SC 1.09%
#Eval for threshold 0.30: DER 12.77%, MS 4.56%, FA 6.99%, SC 1.22%
#Eval for threshold 0.35: DER 12.46%, MS 5.15%, FA 6.05%, SC 1.26%
#Eval for threshold 0.40: DER 12.32%, MS 5.77%, FA 5.26%, SC 1.29%
#Eval for threshold 0.45: DER 12.28%, MS 6.43%, FA 4.56%, SC 1.28% as report
#Eval for threshold 0.50: DER 12.38%, MS 7.13%, FA 3.99%, SC 1.27%
#Eval for threshold 0.55: DER 12.56%, MS 7.88%, FA 3.47%, SC 1.21%
#Eval for threshold 0.60: DER 12.85%, MS 8.69%, FA 3.03%, SC 1.13%
#Eval for threshold 0.70: DER 13.84%, MS 10.63%, FA 2.25%, SC 0.97%
#Eval for threshold 0.80: DER 15.58%, MS 13.25%, FA 1.56%, SC 0.76%

# Eval of alimeeting, collar=0.25
#Eval for threshold 0.20: DER 5.59%, MS 1.07%, FA 4.15%, SC 0.37%
#Eval for threshold 0.30: DER 4.68%, MS 1.55%, FA 2.76%, SC 0.37%
#Eval for threshold 0.35: DER 4.49%, MS 1.80%, FA 2.32%, SC 0.38%
#Eval for threshold 0.40: DER 4.36%, MS 2.04%, FA 1.94%, SC 0.39%
#Eval for threshold 0.45: DER 4.35%, MS 2.32%, FA 1.62%, SC 0.41%  as report
#Eval for threshold 0.50: DER 4.41%, MS 2.65%, FA 1.36%, SC 0.40%
#Eval for threshold 0.55: DER 4.50%, MS 2.98%, FA 1.16%, SC 0.36%
#Eval for threshold 0.60: DER 4.65%, MS 3.32%, FA 0.99%, SC 0.35%
#Eval for threshold 0.70: DER 5.23%, MS 4.27%, FA 0.69%, SC 0.27%
#Eval for threshold 0.80: DER 6.44%, MS 5.76%, FA 0.49%, SC 0.18%

# Test of alimeeting, collar=0.25
#Eval for threshold 0.20: DER 5.53%, MS 1.45%, FA 3.71%, SC 0.37%
#Eval for threshold 0.30: DER 4.80%, MS 1.99%, FA 2.35%, SC 0.46%
#Eval for threshold 0.35: DER 4.67%, MS 2.27%, FA 1.90%, SC 0.50%
#Eval for threshold 0.40: DER 4.63%, MS 2.55%, FA 1.54%, SC 0.54%
#Eval for threshold 0.45: DER 4.65%, MS 2.87%, FA 1.22%, SC 0.56% as report
#Eval for threshold 0.50: DER 4.78%, MS 3.26%, FA 0.97%, SC 0.55%
#Eval for threshold 0.55: DER 4.93%, MS 3.64%, FA 0.77%, SC 0.52%
#Eval for threshold 0.60: DER 5.19%, MS 4.07%, FA 0.64%, SC 0.47%
#Eval for threshold 0.70: DER 5.95%, MS 5.17%, FA 0.42%, SC 0.36%
#Eval for threshold 0.80: DER 7.25%, MS 6.72%, FA 0.27%, SC 0.25%



# lr=2e-4, Epoch 18, batch_idx_train: 80200, num_updates: 61500, {'loss': nan, 'DER': 0.1880349422041825,  
if [ ${stage} -le 157 ] && [ ${stop_stage} -ge 157 ];then
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
    #exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_epoch20_front_fix_seed_lr2e4_single_backend_transformer_multi_backend_transformer_rs_len16_shift0.8
    exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_epoch20_front_fix_seed_lr1e4_single_backend_transformer_multi_backend_transformer_rs_len16_shift0.8
    mkdir -p $exp_dir
    data_dir="/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    rs_len=16
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
    --start-epoch 12\
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
    --num-transformer-layer $num_transformer_layer
fi

if [ ${stage} -le 158 ] && [ ${stop_stage} -ge 158 ];then
 #exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_epoch20_front_fix_seed_lr2e4_single_backend_transformer_multi_backend_transformer_rs_len16_shift0.8
 exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_epoch20_front_fix_seed_lr1e4_single_backend_transformer_multi_backend_transformer_rs_len16_shift0.8
 model_file=$exp_dir/best-valid-der.pt
 rs_len=16
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
#  grep -r Eval logs/run_ts_vad2_aistation_stage157-158_7.log
#2025-04-26 17:10:59,970 (infer2:255) INFO: currently, it will infer Eval set.
# Eval of alimeeting, collar=0.0
#Eval for threshold 0.20: DER 14.92%, MS 3.58%, FA 10.11%, SC 1.23%
#Eval for threshold 0.30: DER 13.40%, MS 4.88%, FA 7.23%, SC 1.29%
#Eval for threshold 0.35: DER 12.99%, MS 5.52%, FA 6.20%, SC 1.27%
#Eval for threshold 0.40: DER 12.82%, MS 6.21%, FA 5.35%, SC 1.26%
#Eval for threshold 0.45: DER 12.80%, MS 6.91%, FA 4.65%, SC 1.23%
#Eval for threshold 0.50: DER 12.87%, MS 7.67%, FA 4.02%, SC 1.18%
#Eval for threshold 0.55: DER 13.09%, MS 8.47%, FA 3.49%, SC 1.12%
#Eval for threshold 0.60: DER 13.38%, MS 9.37%, FA 2.98%, SC 1.03%
#Eval for threshold 0.70: DER 14.58%, MS 11.52%, FA 2.18%, SC 0.89%
#Eval for threshold 0.80: DER 16.63%, MS 14.47%, FA 1.50%, SC 0.65%

# Test of alimeeting, collar=0.0
#Eval for threshold 0.20: DER 14.22%, MS 3.96%, FA 9.01%, SC 1.25%
#Eval for threshold 0.30: DER 13.07%, MS 5.42%, FA 6.29%, SC 1.36%
#Eval for threshold 0.35: DER 12.88%, MS 6.14%, FA 5.38%, SC 1.37%
#Eval for threshold 0.40: DER 12.88%, MS 6.91%, FA 4.63%, SC 1.35%
#Eval for threshold 0.45: DER 13.00%, MS 7.73%, FA 3.94%, SC 1.34%
#Eval for threshold 0.50: DER 13.31%, MS 8.61%, FA 3.41%, SC 1.29%
#Eval for threshold 0.55: DER 13.71%, MS 9.58%, FA 2.94%, SC 1.19%
#Eval for threshold 0.60: DER 14.24%, MS 10.59%, FA 2.53%, SC 1.12%
#Eval for threshold 0.70: DER 15.74%, MS 13.00%, FA 1.80%, SC 0.93%
#Eval for threshold 0.80: DER 18.25%, MS 16.33%, FA 1.21%, SC 0.71%
#2025-04-26 19:04:47,969 (infer2:255) INFO: currently, it will infer Eval set.

# Eval of alimeeting, collar=0.25
#Eval for threshold 0.20: DER 5.86%, MS 1.35%, FA 4.07%, SC 0.44%
#Eval for threshold 0.30: DER 5.10%, MS 1.94%, FA 2.68%, SC 0.48%
#Eval for threshold 0.35: DER 4.91%, MS 2.22%, FA 2.22%, SC 0.47%
#Eval for threshold 0.40: DER 4.86%, MS 2.55%, FA 1.84%, SC 0.47% as report
#Eval for threshold 0.45: DER 4.92%, MS 2.90%, FA 1.55%, SC 0.47%
#Eval for threshold 0.50: DER 5.05%, MS 3.28%, FA 1.32%, SC 0.45%
#Eval for threshold 0.55: DER 5.24%, MS 3.70%, FA 1.11%, SC 0.43%
#Eval for threshold 0.60: DER 5.46%, MS 4.16%, FA 0.92%, SC 0.38%
#Eval for threshold 0.70: DER 6.33%, MS 5.33%, FA 0.68%, SC 0.32%
#Eval for threshold 0.80: DER 7.81%, MS 7.09%, FA 0.50%, SC 0.22%

# Test of alimeeting,collar=0.25
#Eval for threshold 0.20: DER 5.64%, MS 1.79%, FA 3.35%, SC 0.49%
#Eval for threshold 0.30: DER 5.04%, MS 2.47%, FA 2.01%, SC 0.56%
#Eval for threshold 0.35: DER 4.98%, MS 2.81%, FA 1.59%, SC 0.58% as report
#Eval for threshold 0.40: DER 5.08%, MS 3.22%, FA 1.28%, SC 0.58%
#Eval for threshold 0.45: DER 5.22%, MS 3.64%, FA 1.01%, SC 0.57%
#Eval for threshold 0.50: DER 5.50%, MS 4.13%, FA 0.83%, SC 0.55%
#Eval for threshold 0.55: DER 5.83%, MS 4.65%, FA 0.67%, SC 0.51%
#Eval for threshold 0.60: DER 6.23%, MS 5.21%, FA 0.55%, SC 0.47%
#Eval for threshold 0.70: DER 7.35%, MS 6.64%, FA 0.35%, SC 0.37%
#Eval for threshold 0.80: DER 9.21%, MS 8.77%, FA 0.19%, SC 0.25%


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

#2025-04-18 03:58:26,732 (infer2:255) INFO: currently, it will infer Eval set.
# Eval of alimeeting, collar=0.0
#Eval for threshold 0.20: DER 17.46%, MS 2.60%, FA 13.84%, SC 1.02%
#Eval for threshold 0.30: DER 14.54%, MS 3.75%, FA 9.62%, SC 1.17%
#Eval for threshold 0.35: DER 13.71%, MS 4.31%, FA 8.22%, SC 1.17%
#Eval for threshold 0.40: DER 13.18%, MS 4.97%, FA 7.06%, SC 1.14%
#Eval for threshold 0.45: DER 12.82%, MS 5.64%, FA 6.05%, SC 1.13%
#Eval for threshold 0.50: DER 12.61%, MS 6.35%, FA 5.16%, SC 1.10%
#Eval for threshold 0.55: DER 12.65%, MS 7.17%, FA 4.45%, SC 1.04%
#Eval for threshold 0.60: DER 12.79%, MS 8.05%, FA 3.77%, SC 0.97%
#Eval for threshold 0.70: DER 13.67%, MS 10.28%, FA 2.62%, SC 0.77%
#Eval for threshold 0.80: DER 15.47%, MS 13.12%, FA 1.79%, SC 0.57%

# Test of alimeeting, collar=0.0
#Eval for threshold 0.20: DER 16.47%, MS 2.55%, FA 12.79%, SC 1.12%
#Eval for threshold 0.30: DER 13.86%, MS 3.74%, FA 8.86%, SC 1.26%
#Eval for threshold 0.35: DER 13.23%, MS 4.39%, FA 7.52%, SC 1.32%
#Eval for threshold 0.40: DER 12.82%, MS 5.08%, FA 6.39%, SC 1.35%
#Eval for threshold 0.45: DER 12.58%, MS 5.79%, FA 5.43%, SC 1.35%
#Eval for threshold 0.50: DER 12.52%, MS 6.60%, FA 4.59%, SC 1.34%
#Eval for threshold 0.55: DER 12.66%, MS 7.49%, FA 3.89%, SC 1.27%
#Eval for threshold 0.60: DER 12.94%, MS 8.49%, FA 3.28%, SC 1.18%
#Eval for threshold 0.70: DER 14.07%, MS 10.81%, FA 2.29%, SC 0.97%
#Eval for threshold 0.80: DER 16.25%, MS 14.06%, FA 1.44%, SC 0.75%
#2025-04-18 05:05:31,185 (infer2:255) INFO: currently, it will infer Eval set.

# Eval of alimeeting, collar=0.25
#Eval for threshold 0.20: DER 7.37%, MS 0.90%, FA 6.14%, SC 0.33%
#Eval for threshold 0.30: DER 5.64%, MS 1.36%, FA 3.91%, SC 0.37%
#Eval for threshold 0.35: DER 5.21%, MS 1.59%, FA 3.23%, SC 0.39%
#Eval for threshold 0.40: DER 4.98%, MS 1.87%, FA 2.74%, SC 0.37%
#Eval for threshold 0.45: DER 4.80%, MS 2.17%, FA 2.25%, SC 0.37%
#Eval for threshold 0.50: DER 4.70%, MS 2.50%, FA 1.83%, SC 0.37% as report 
#Eval for threshold 0.55: DER 4.76%, MS 2.90%, FA 1.52%, SC 0.34%
#Eval for threshold 0.60: DER 4.87%, MS 3.32%, FA 1.24%, SC 0.31%
#Eval for threshold 0.70: DER 5.52%, MS 4.52%, FA 0.78%, SC 0.22%
#Eval for threshold 0.80: DER 6.81%, MS 6.11%, FA 0.56%, SC 0.14%

# Test of alimeeting, collar=0.25
#Eval for threshold 0.20: DER 6.95%, MS 1.11%, FA 5.40%, SC 0.43%
#Eval for threshold 0.30: DER 5.37%, MS 1.61%, FA 3.27%, SC 0.49%
#Eval for threshold 0.35: DER 5.04%, MS 1.90%, FA 2.62%, SC 0.53%
#Eval for threshold 0.40: DER 4.85%, MS 2.23%, FA 2.07%, SC 0.55%
#Eval for threshold 0.45: DER 4.78%, MS 2.58%, FA 1.64%, SC 0.56%
#Eval for threshold 0.50: DER 4.81%, MS 2.97%, FA 1.29%, SC 0.56% as report
#Eval for threshold 0.55: DER 4.97%, MS 3.44%, FA 1.00%, SC 0.53%
#Eval for threshold 0.60: DER 5.21%, MS 3.98%, FA 0.77%, SC 0.47%
#Eval for threshold 0.70: DER 6.10%, MS 5.28%, FA 0.46%, SC 0.36%
#Eval for threshold 0.80: DER 7.70%, MS 7.18%, FA 0.25%, SC 0.27%

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




if [ ${stage} -le 164 ] && [ ${stop_stage} -ge 164 ];then
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
    exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_epoch20_front_fix_seed_lr2e4_single_backend_transformer_multi_backend_transformer_rs_len8_shift0.8_enhance_ratio0.3_offline_se_aug
    #exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_epoch20_front_fix_seed_lr2e4_single_backend_transformer_multi_backend_transformer_rs_len8_shift2_enhance_ratio0.3
    mkdir -p $exp_dir
    data_dir="/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    enhanced_audio_dir="/maduo/datasets/zipenhancer_alimeeting/"
    rs_len=8
    segment_shift=0.8
    enhance_ratio=0.3
    single_backend_type="transformer"
    multi_backend_type="transformer"
    num_transformer_layer=2
    CUDA_VISIABLE_DEVICES=0,1 \
   TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15415 \
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

if [ ${stage} -le 165 ] && [ ${stop_stage} -ge 165 ];then
 exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_epoch20_front_fix_seed_lr2e4_single_backend_transformer_multi_backend_transformer_rs_len8_shift0.8_enhance_ratio0.3_offline_se_aug
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
# grep -r Eval logs/run_ts_vad2_aistation_stage164-165.log
#2025-04-19 13:17:55,581 (infer2:255) INFO: currently, it will infer Eval set.
# Eval of alimeeting, collar=0.0
#Eval for threshold 0.20: DER 18.34%, MS 2.70%, FA 14.52%, SC 1.12%
#Eval for threshold 0.30: DER 15.22%, MS 3.93%, FA 10.04%, SC 1.25%
#Eval for threshold 0.35: DER 14.37%, MS 4.57%, FA 8.47%, SC 1.34%
#Eval for threshold 0.40: DER 13.79%, MS 5.23%, FA 7.22%, SC 1.34%
#Eval for threshold 0.45: DER 13.48%, MS 5.96%, FA 6.18%, SC 1.33%
#Eval for threshold 0.50: DER 13.31%, MS 6.75%, FA 5.29%, SC 1.27% as report
#Eval for threshold 0.55: DER 13.43%, MS 7.67%, FA 4.56%, SC 1.20%
#Eval for threshold 0.60: DER 13.64%, MS 8.60%, FA 3.92%, SC 1.11%
#Eval for threshold 0.70: DER 14.50%, MS 10.82%, FA 2.79%, SC 0.89%
#Eval for threshold 0.80: DER 16.37%, MS 13.89%, FA 1.87%, SC 0.62%

# Test of alimeeting, collar=0.0
#Eval for threshold 0.20: DER 16.69%, MS 2.52%, FA 13.09%, SC 1.08%
#Eval for threshold 0.30: DER 14.07%, MS 3.77%, FA 9.06%, SC 1.25%
#Eval for threshold 0.35: DER 13.40%, MS 4.44%, FA 7.66%, SC 1.30%
#Eval for threshold 0.40: DER 12.98%, MS 5.14%, FA 6.50%, SC 1.34%
#Eval for threshold 0.45: DER 12.73%, MS 5.90%, FA 5.49%, SC 1.34%
#Eval for threshold 0.50: DER 12.73%, MS 6.75%, FA 4.65%, SC 1.33% as report
#Eval for threshold 0.55: DER 12.88%, MS 7.68%, FA 3.93%, SC 1.27%
#Eval for threshold 0.60: DER 13.18%, MS 8.66%, FA 3.31%, SC 1.21%
#Eval for threshold 0.70: DER 14.37%, MS 11.07%, FA 2.29%, SC 1.01%
#Eval for threshold 0.80: DER 16.46%, MS 14.29%, FA 1.42%, SC 0.75%

# Eval of alimeeting, collar=0.25
#2025-04-19 14:36:55,628 (infer2:255) INFO: currently, it will infer Eval set.
#Eval for threshold 0.20: DER 8.22%, MS 0.91%, FA 6.95%, SC 0.35%
#Eval for threshold 0.30: DER 6.22%, MS 1.38%, FA 4.41%, SC 0.43%
#Eval for threshold 0.35: DER 5.72%, MS 1.66%, FA 3.59%, SC 0.47%
#Eval for threshold 0.40: DER 5.39%, MS 1.94%, FA 2.98%, SC 0.47%
#Eval for threshold 0.45: DER 5.23%, MS 2.30%, FA 2.46%, SC 0.47%
#Eval for threshold 0.50: DER 5.17%, MS 2.71%, FA 2.03%, SC 0.44% as report
#Eval for threshold 0.55: DER 5.32%, MS 3.20%, FA 1.70%, SC 0.41%
#Eval for threshold 0.60: DER 5.47%, MS 3.68%, FA 1.41%, SC 0.39%
#Eval for threshold 0.70: DER 6.10%, MS 4.88%, FA 0.95%, SC 0.27%
#Eval for threshold 0.80: DER 7.49%, MS 6.72%, FA 0.60%, SC 0.17%

# Test of alimeeting, collar=0.25
#Eval for threshold 0.20: DER 7.14%, MS 1.07%, FA 5.70%, SC 0.38%
#Eval for threshold 0.30: DER 5.52%, MS 1.61%, FA 3.47%, SC 0.45%
#Eval for threshold 0.35: DER 5.15%, MS 1.93%, FA 2.76%, SC 0.46%
#Eval for threshold 0.40: DER 4.95%, MS 2.25%, FA 2.21%, SC 0.49%
#Eval for threshold 0.45: DER 4.85%, MS 2.61%, FA 1.74%, SC 0.49% 
#Eval for threshold 0.50: DER 4.91%, MS 3.05%, FA 1.35%, SC 0.50% as report
#Eval for threshold 0.55: DER 5.10%, MS 3.56%, FA 1.06%, SC 0.48%
#Eval for threshold 0.60: DER 5.36%, MS 4.09%, FA 0.81%, SC 0.46%
#Eval for threshold 0.70: DER 6.28%, MS 5.43%, FA 0.48%, SC 0.37%
#Eval for threshold 0.80: DER 7.80%, MS 7.28%, FA 0.26%, SC 0.26%



# when lr=1e-4,  Epoch 3, batch_idx_train: 9320, num_updates: 8500  loss is nan.
if [ ${stage} -le 166 ] && [ ${stop_stage} -ge 166 ];then
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
    # cam++_zh_200k
    #speech_encoder_type="CAM++"
    #speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    # w2v-bert2
    speech_encoder_type="w2v-bert2"
    speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
    speech_encoder_config="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
    
    dataset_name="alimeeting" # dataset name

    # for loading speaker embedding file
    spk_path=/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    #exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_wav-bert2.0_epoch20_front_fix_seed_lr1e4_single_backend_transformer_multi_backend_transformer_rs_len8_shift0.8 # 
    exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_wav-bert2.0_epoch20_front_fix_seed_lr5e5_single_backend_transformer_multi_backend_transformer_rs_len8_shift0.8
    mkdir -p $exp_dir
    data_dir="/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    rs_len=8
    segment_shift=0.8
    single_backend_type="transformer"
    multi_backend_type="transformer"
    num_transformer_layer=2
    CUDA_VISIABLE_DEVICES=0,1 \
    NCCL_ASYNC_ERROR_HANDLING=1\
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 16115 \
   ts_vad2/train_accelerate_ddp2_debug2.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-updates 4000\
    --grad-clip true\
    --lr 5e-5\
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
    --num-transformer-layer $num_transformer_layer
fi

if [ ${stage} -le 167 ] && [ ${stop_stage} -ge 167 ];then
 #exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_wav-bert2.0_epoch20_front_fix_seed_lr1e4_single_backend_transformer_multi_backend_transformer_rs_len8_shift0.8
 exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_wav-bert2.0_epoch20_front_fix_seed_lr5e5_single_backend_transformer_multi_backend_transformer_rs_len8_shift0.8
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
 #speech_encoder_type="CAM++"
 #speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"

 # w2v-bert2
 speech_encoder_type="w2v-bert2"
 speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 speech_encoder_config="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
 
 
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
    --speech-encoder-config $speech_encoder_config\
    --select-encoder-layer-nums 6\
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
#grep -r Eval  logs/run_ts_vad2_aistation_stage167_2.log
#2025-05-28 17:30:09,036 (infer2:255) INFO: currently, it will infer Eval set.
# Eval of alimeeting, collar=0.0
#Eval for threshold 0.20: DER 15.90%, MS 2.63%, FA 12.34%, SC 0.93%
#Eval for threshold 0.30: DER 13.44%, MS 4.03%, FA 8.30%, SC 1.10%
#Eval for threshold 0.35: DER 12.78%, MS 4.74%, FA 6.94%, SC 1.11%
#Eval for threshold 0.40: DER 12.35%, MS 5.46%, FA 5.80%, SC 1.09%
#Eval for threshold 0.45: DER 12.18%, MS 6.27%, FA 4.87%, SC 1.04% as report
#Eval for threshold 0.50: DER 12.19%, MS 7.11%, FA 4.10%, SC 0.98%
#Eval for threshold 0.55: DER 12.37%, MS 7.99%, FA 3.49%, SC 0.89%
#Eval for threshold 0.60: DER 12.74%, MS 8.98%, FA 2.97%, SC 0.80%
#Eval for threshold 0.70: DER 13.99%, MS 11.23%, FA 2.11%, SC 0.64%
#Eval for threshold 0.80: DER 16.35%, MS 14.56%, FA 1.35%, SC 0.43%

# Test of alimeeting, collar=0.0
#Eval for threshold 0.20: DER 15.58%, MS 2.60%, FA 12.05%, SC 0.92%
#Eval for threshold 0.30: DER 13.20%, MS 4.02%, FA 8.04%, SC 1.13%
#Eval for threshold 0.35: DER 12.63%, MS 4.81%, FA 6.63%, SC 1.19%
#Eval for threshold 0.40: DER 12.34%, MS 5.61%, FA 5.51%, SC 1.21%
#Eval for threshold 0.45: DER 12.31%, MS 6.53%, FA 4.58%, SC 1.20% as report
#Eval for threshold 0.50: DER 12.52%, MS 7.53%, FA 3.83%, SC 1.16%
#Eval for threshold 0.55: DER 12.93%, MS 8.59%, FA 3.22%, SC 1.11%
#Eval for threshold 0.60: DER 13.47%, MS 9.78%, FA 2.65%, SC 1.04%
#Eval for threshold 0.70: DER 15.17%, MS 12.55%, FA 1.76%, SC 0.86%
#Eval for threshold 0.80: DER 17.99%, MS 16.26%, FA 1.10%, SC 0.63%
#2025-05-28 21:41:04,614 (infer2:255) INFO: currently, it will infer Eval set.

# Eval of alimeeting, collar=0.25
#Eval for threshold 0.20: DER 6.22%, MS 0.94%, FA 5.05%, SC 0.22%
#Eval for threshold 0.30: DER 4.83%, MS 1.43%, FA 3.11%, SC 0.28%
#Eval for threshold 0.35: DER 4.44%, MS 1.70%, FA 2.45%, SC 0.29%
#Eval for threshold 0.40: DER 4.21%, MS 1.99%, FA 1.94%, SC 0.28%
#Eval for threshold 0.45: DER 4.09%, MS 2.35%, FA 1.49%, SC 0.25% as report
#Eval for threshold 0.50: DER 4.13%, MS 2.74%, FA 1.18%, SC 0.20%
#Eval for threshold 0.55: DER 4.31%, MS 3.17%, FA 0.97%, SC 0.18%
#Eval for threshold 0.60: DER 4.58%, MS 3.65%, FA 0.78%, SC 0.15%
#Eval for threshold 0.70: DER 5.53%, MS 4.85%, FA 0.56%, SC 0.12%
#Eval for threshold 0.80: DER 7.39%, MS 6.92%, FA 0.41%, SC 0.06%

# Test of alimeeting, collar=0.25
#Eval for threshold 0.20: DER 6.56%, MS 1.07%, FA 5.14%, SC 0.35%
#Eval for threshold 0.30: DER 5.04%, MS 1.67%, FA 2.93%, SC 0.44%
#Eval for threshold 0.35: DER 4.72%, MS 2.03%, FA 2.21%, SC 0.48%
#Eval for threshold 0.40: DER 4.54%, MS 2.40%, FA 1.66%, SC 0.48%
#Eval for threshold 0.45: DER 4.53%, MS 2.83%, FA 1.21%, SC 0.50% as report
#Eval for threshold 0.50: DER 4.73%, MS 3.35%, FA 0.90%, SC 0.48%
#Eval for threshold 0.55: DER 5.05%, MS 3.92%, FA 0.68%, SC 0.45%
#Eval for threshold 0.60: DER 5.52%, MS 4.60%, FA 0.52%, SC 0.41%
#Eval for threshold 0.70: DER 6.80%, MS 6.21%, FA 0.29%, SC 0.31%
#Eval for threshold 0.80: DER 8.84%, MS 8.46%, FA 0.17%, SC 0.21%

# lr=5e-5
#end of epoch 4, batch_idx: 3909 batch_idx_train: 15639, {'loss': nan, 'DER': 1.0,
# will reduce lr and try again
if [ ${stage} -le 168 ] && [ ${stop_stage} -ge 168 ];then
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
    # cam++_zh_200k
    #speech_encoder_type="CAM++"
    #speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    # w2v-bert2
    speech_encoder_type="w2v-bert2"
    speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
    speech_encoder_config="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    dataset_name="alimeeting" # dataset name

    # for loading speaker embedding file
    spk_path=/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    #exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_wav-bert2.0_epoch20_front_fix_seed_lr1e4_single_backend_transformer_multi_backend_transformer_rs_len8_shift0.8 #
    exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_real_cam++_zh_200k_feature_dir_wav-bert2.0_epoch20_front_fix_seed_lr2e5_single_backend_transformer_multi_backend_transformer_rs_len8_shift0.8
    mkdir -p $exp_dir
    data_dir="/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    rs_len=8
    segment_shift=0.8
    single_backend_type="transformer"
    multi_backend_type="transformer"
    num_transformer_layer=2
    CUDA_VISIABLE_DEVICES=0,1 \
    NCCL_ASYNC_ERROR_HANDLING=1\
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 16215 \
   ts_vad2/train_accelerate_ddp2_debug2.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-updates 4000\
    --grad-clip true\
    --lr 2e-5\
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
    --num-transformer-layer $num_transformer_layer
fi

if [ ${stage} -le 169 ] && [ ${stop_stage} -ge 169 ];then
 #exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_wav-bert2.0_epoch20_front_fix_seed_lr1e4_single_backend_transformer_multi_backend_transformer_rs_len8_shift0.8
 exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_real_cam++_zh_200k_feature_dir_wav-bert2.0_epoch20_front_fix_seed_lr2e5_single_backend_transformer_multi_backend_transformer_rs_len8_shift0.8
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
 #speech_encoder_type="CAM++"
 #speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"

 # w2v-bert2
 speech_encoder_type="w2v-bert2"
 speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 speech_encoder_config="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"


 dataset_name="alimeeting" # dataset name
 # for loading speaker embedding file
 spk_path=/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
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
    --speech-encoder-config $speech_encoder_config\
    --select-encoder-layer-nums 6\
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
#grep -r Eval  logs/run_ts_vad2_aistation_stage168-169_lr2e-5.log
#Eval for threshold 0.20: DER 16.75%, MS 2.60%, FA 13.05%, SC 1.10%
#Eval for threshold 0.30: DER 13.76%, MS 4.00%, FA 8.54%, SC 1.21%
#Eval for threshold 0.35: DER 12.99%, MS 4.76%, FA 7.01%, SC 1.22%
#Eval for threshold 0.40: DER 12.58%, MS 5.52%, FA 5.86%, SC 1.20%
#Eval for threshold 0.45: DER 12.36%, MS 6.33%, FA 4.86%, SC 1.16% as report
#Eval for threshold 0.50: DER 12.40%, MS 7.26%, FA 4.04%, SC 1.10%
#Eval for threshold 0.55: DER 12.62%, MS 8.25%, FA 3.35%, SC 1.02%
#Eval for threshold 0.60: DER 13.02%, MS 9.31%, FA 2.81%, SC 0.90%
#Eval for threshold 0.70: DER 14.53%, MS 11.93%, FA 1.93%, SC 0.68%
#Eval for threshold 0.80: DER 17.23%, MS 15.58%, FA 1.20%, SC 0.45%
#Eval for threshold 0.20: DER 18.68%, MS 2.60%, FA 15.08%, SC 1.00%
#Eval for threshold 0.30: DER 14.95%, MS 4.14%, FA 9.58%, SC 1.23%
#Eval for threshold 0.35: DER 14.07%, MS 4.97%, FA 7.81%, SC 1.29%
#Eval for threshold 0.40: DER 13.58%, MS 5.89%, FA 6.35%, SC 1.34%
#Eval for threshold 0.45: DER 13.41%, MS 6.90%, FA 5.12%, SC 1.39% as report
#Eval for threshold 0.50: DER 13.52%, MS 8.03%, FA 4.11%, SC 1.38%
#Eval for threshold 0.55: DER 13.92%, MS 9.30%, FA 3.29%, SC 1.33%
#Eval for threshold 0.60: DER 14.60%, MS 10.71%, FA 2.66%, SC 1.23%
#Eval for threshold 0.70: DER 16.86%, MS 14.17%, FA 1.73%, SC 0.96%
#Eval for threshold 0.80: DER 20.69%, MS 19.05%, FA 1.00%, SC 0.63%
#2025-06-01 15:48:08,084 (infer2:255) INFO: currently, it will infer Eval set.
#Eval for threshold 0.20: DER 6.63%, MS 0.95%, FA 5.37%, SC 0.31%
#Eval for threshold 0.30: DER 4.83%, MS 1.46%, FA 3.04%, SC 0.33%
#Eval for threshold 0.35: DER 4.44%, MS 1.76%, FA 2.33%, SC 0.35%
#Eval for threshold 0.40: DER 4.28%, MS 2.10%, FA 1.81%, SC 0.36%
#Eval for threshold 0.45: DER 4.25%, MS 2.46%, FA 1.43%, SC 0.36% as report
#Eval for threshold 0.50: DER 4.35%, MS 2.88%, FA 1.12%, SC 0.35%
#Eval for threshold 0.55: DER 4.61%, MS 3.38%, FA 0.90%, SC 0.33%
#Eval for threshold 0.60: DER 4.99%, MS 3.96%, FA 0.75%, SC 0.28%
#Eval for threshold 0.70: DER 6.09%, MS 5.37%, FA 0.54%, SC 0.18%
#Eval for threshold 0.80: DER 8.05%, MS 7.55%, FA 0.39%, SC 0.11%
#Eval for threshold 0.20: DER 9.89%, MS 1.12%, FA 8.41%, SC 0.36%
#Eval for threshold 0.30: DER 6.99%, MS 1.80%, FA 4.72%, SC 0.47%
#Eval for threshold 0.35: DER 6.32%, MS 2.18%, FA 3.60%, SC 0.54%
#Eval for threshold 0.40: DER 5.91%, MS 2.61%, FA 2.66%, SC 0.64% as report
#Eval for threshold 0.45: DER 5.77%, MS 3.15%, FA 1.90%, SC 0.73%
#Eval for threshold 0.50: DER 5.85%, MS 3.75%, FA 1.32%, SC 0.78%
#Eval for threshold 0.55: DER 6.15%, MS 4.50%, FA 0.90%, SC 0.76%
#Eval for threshold 0.60: DER 6.72%, MS 5.41%, FA 0.62%, SC 0.69%
#Eval for threshold 0.70: DER 8.59%, MS 7.78%, FA 0.33%, SC 0.48%
#Eval for threshold 0.80: DER 11.73%, MS 11.33%, FA 0.17%, SC 0.24%


if [ ${stage} -le 170 ] && [ ${stop_stage} -ge 170 ];then
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
    exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_epoch20_front_fix_seed_lr2e4_single_backend_transformer_multi_backend_transformer_rs_len8_shift0.4
    mkdir -p $exp_dir
    data_dir="/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    rs_len=8
    segment_shift=0.4
    single_backend_type="transformer"
    multi_backend_type="transformer"
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
    --num-transformer-layer $num_transformer_layer
fi

if [ ${stage} -le 171 ] && [ ${stop_stage} -ge 171 ];then
 exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_epoch20_front_fix_seed_lr2e4_single_backend_transformer_multi_backend_transformer_rs_len8_shift0.4
 model_file=$exp_dir/best-valid-der.pt
 rs_len=8
 segment_shift=0.4
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
#grep -r Eval  logs/run_ts_vad2_aistation_stage170-171_3.log
# Eval of alimeeting, collar=0.0
##Eval for threshold 0.20: DER 14.96%, MS 3.67%, FA 9.98%, SC 1.31%
#Eval for threshold 0.30: DER 13.42%, MS 4.75%, FA 7.31%, SC 1.37%
#Eval for threshold 0.35: DER 12.98%, MS 5.30%, FA 6.29%, SC 1.39%
#Eval for threshold 0.40: DER 12.71%, MS 5.84%, FA 5.50%, SC 1.37%
#Eval for threshold 0.45: DER 12.58%, MS 6.40%, FA 4.83%, SC 1.34%
#Eval for threshold 0.50: DER 12.54%, MS 6.97%, FA 4.24%, SC 1.33%
#Eval for threshold 0.55: DER 12.65%, MS 7.62%, FA 3.74%, SC 1.28%
#Eval for threshold 0.60: DER 12.87%, MS 8.35%, FA 3.28%, SC 1.24%
#Eval for threshold 0.70: DER 13.59%, MS 10.10%, FA 2.44%, SC 1.06%
#Eval for threshold 0.80: DER 15.06%, MS 12.59%, FA 1.68%, SC 0.80%

# Test of alimeeting, collar=0.0
#Eval for threshold 0.20: DER 15.13%, MS 3.59%, FA 10.21%, SC 1.33%
#Eval for threshold 0.30: DER 13.64%, MS 4.71%, FA 7.48%, SC 1.45%
#Eval for threshold 0.35: DER 13.25%, MS 5.29%, FA 6.50%, SC 1.47%
#Eval for threshold 0.40: DER 13.01%, MS 5.87%, FA 5.66%, SC 1.49%
#Eval for threshold 0.45: DER 12.90%, MS 6.49%, FA 4.91%, SC 1.49%
#Eval for threshold 0.50: DER 12.92%, MS 7.16%, FA 4.26%, SC 1.49%
#Eval for threshold 0.55: DER 13.05%, MS 7.92%, FA 3.69%, SC 1.44%
#Eval for threshold 0.60: DER 13.30%, MS 8.74%, FA 3.20%, SC 1.36%
#Eval for threshold 0.70: DER 14.23%, MS 10.69%, FA 2.33%, SC 1.21%
#Eval for threshold 0.80: DER 15.96%, MS 13.41%, FA 1.58%, SC 0.96%
#2025-05-01 04:14:11,085 (infer2:255) INFO: currently, it will infer Eval set.
# Eval of alimeeting, collar=0.25
#Eval for threshold 0.20: DER 6.01%, MS 1.33%, FA 4.23%, SC 0.45%
#Eval for threshold 0.30: DER 5.23%, MS 1.77%, FA 2.96%, SC 0.51%
#Eval for threshold 0.35: DER 4.95%, MS 2.00%, FA 2.41%, SC 0.54%
#Eval for threshold 0.40: DER 4.81%, MS 2.21%, FA 2.06%, SC 0.54%
#Eval for threshold 0.45: DER 4.78%, MS 2.47%, FA 1.77%, SC 0.54%
#Eval for threshold 0.50: DER 4.77%, MS 2.70%, FA 1.51%, SC 0.56%
#Eval for threshold 0.55: DER 4.88%, MS 3.03%, FA 1.30%, SC 0.55%
#Eval for threshold 0.60: DER 5.05%, MS 3.40%, FA 1.12%, SC 0.52%
#Eval for threshold 0.70: DER 5.61%, MS 4.39%, FA 0.78%, SC 0.43%
#Eval for threshold 0.80: DER 6.61%, MS 5.78%, FA 0.54%, SC 0.29%

# Test of alimeeting, collar=0.25
#Eval for threshold 0.20: DER 6.59%, MS 1.55%, FA 4.49%, SC 0.55%
#Eval for threshold 0.30: DER 5.67%, MS 2.03%, FA 3.00%, SC 0.63%
#Eval for threshold 0.35: DER 5.42%, MS 2.31%, FA 2.45%, SC 0.66%
#Eval for threshold 0.40: DER 5.28%, MS 2.60%, FA 1.99%, SC 0.69%
#Eval for threshold 0.45: DER 5.24%, MS 2.91%, FA 1.61%, SC 0.71%
#Eval for threshold 0.50: DER 5.30%, MS 3.27%, FA 1.31%, SC 0.73%
#Eval for threshold 0.55: DER 5.43%, MS 3.66%, FA 1.06%, SC 0.71%
#Eval for threshold 0.60: DER 5.66%, MS 4.12%, FA 0.87%, SC 0.66%
#Eval for threshold 0.70: DER 6.43%, MS 5.31%, FA 0.56%, SC 0.56%
#Eval for threshold 0.80: DER 7.73%, MS 6.97%, FA 0.33%, SC 0.43%

# compared with stage155-156, stage172-173 will use remove silence audio(using oracle rttm) to train tsvad
if [ ${stage} -le 172 ] && [ ${stop_stage} -ge 172 ];then
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
    spk_path=/maduo/model_hub/ts_vad/spk_embed/alimeeting_wo_sil/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/maduo/exp/speaker_diarization/ts_vad2/alimeeting_wo_sil/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_epoch20_front_fix_seed_lr2e4_single_backend_transformer_multi_backend_transformer_rs_len8_shift0.8
    mkdir -p $exp_dir
    data_dir="/maduo/datasets/alimeeting_wo_sil" # oracle target audio , mix audio and labels path
    rs_len=8
    segment_shift=0.8
    single_backend_type="transformer"
    multi_backend_type="transformer"
    num_transformer_layer=2
    CUDA_VISIABLE_DEVICES=0,1 \
    NCCL_ASYNC_ERROR_HANDLING=1\
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15215 \
   ts_vad2/train_accelerate_ddp2_debug2.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 10\
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

if [ ${stage} -le 173 ] && [ ${stop_stage} -ge 173 ];then
 exp_dir=/maduo/exp/speaker_diarization/ts_vad2/alimeeting_wo_sil/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_epoch20_front_fix_seed_lr2e4_single_backend_transformer_multi_backend_transformer_rs_len8_shift0.8
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
 # alimeeting_wo_sil groundtruth rttm
 # cp -r  /maduo/datasets/alimeeting_wo_sil/Eval_Ali/Eval_Ali_far/alimeeting_eval.rttm /maduo/datasets/alimeeting_wo_sil/
 # cp -r  /maduo/datasets/alimeeting_wo_sil/Test_Ali/Test_Ali_far/alimeeting_test.rttm /maduo/datasets/alimeeting_wo_sil/ 
 rttm_dir=/maduo/datasets/alimeeting_wo_sil/
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 dataset_name="alimeeting" # dataset name
 # for loading speaker embedding file
 spk_path=/maduo/model_hub/ts_vad/spk_embed/alimeeting_wo_sil/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 data_dir="/maduo/datasets/alimeeting_wo_sil" # oracle target audio , mix audio and labels path
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

#grep -r Eval logs/run_ts_vad2_aistation_stage173.log
# Eval of alimeeting, collar=0.0
#2025-05-08 09:55:46,412 (infer2:255) INFO: currently, it will infer Eval set.
#Eval for threshold 0.20: DER 20.73%, MS 0.07%, FA 17.78%, SC 2.87%
#Eval for threshold 0.30: DER 12.00%, MS 0.25%, FA 7.02%, SC 4.73%
#Eval for threshold 0.35: DER 10.04%, MS 0.92%, FA 4.01%, SC 5.10%
#Eval for threshold 0.40: DER 9.24%, MS 2.24%, FA 2.13%, SC 4.86% as report
#Eval for threshold 0.45: DER 9.39%, MS 3.94%, FA 1.16%, SC 4.28%
#Eval for threshold 0.50: DER 10.23%, MS 5.74%, FA 0.70%, SC 3.79%
#Eval for threshold 0.55: DER 11.34%, MS 7.65%, FA 0.56%, SC 3.12%
#Eval for threshold 0.60: DER 12.79%, MS 9.61%, FA 0.50%, SC 2.67%
#Eval for threshold 0.70: DER 16.47%, MS 14.29%, FA 0.40%, SC 1.78%
#Eval for threshold 0.80: DER 21.81%, MS 20.43%, FA 0.30%, SC 1.08%

# Test of alimeeting, collar=0.0
#Eval for threshold 0.20: DER 32.02%, MS 0.05%, FA 28.22%, SC 3.75%
#Eval for threshold 0.30: DER 18.46%, MS 0.32%, FA 11.62%, SC 6.53%
#Eval for threshold 0.35: DER 14.98%, MS 1.48%, FA 6.23%, SC 7.27%
#Eval for threshold 0.40: DER 13.67%, MS 3.63%, FA 3.05%, SC 7.00% as report
#Eval for threshold 0.45: DER 13.81%, MS 6.30%, FA 1.35%, SC 6.17%
#Eval for threshold 0.50: DER 14.96%, MS 9.10%, FA 0.61%, SC 5.24%
#Eval for threshold 0.55: DER 16.78%, MS 12.24%, FA 0.47%, SC 4.07%
#Eval for threshold 0.60: DER 18.91%, MS 15.39%, FA 0.41%, SC 3.11%
#Eval for threshold 0.70: DER 23.56%, MS 21.48%, FA 0.33%, SC 1.75%
#Eval for threshold 0.80: DER 28.78%, MS 27.55%, FA 0.25%, SC 0.98%
#2025-05-08 10:43:08,076 (infer2:255) INFO: currently, it will infer Eval set.

# Eval of alimeeting, collar=0.25
#Eval for threshold 0.20: DER 15.22%, MS 0.00%, FA 13.37%, SC 1.85%
#Eval for threshold 0.30: DER 8.12%, MS 0.06%, FA 4.88%, SC 3.17%
#Eval for threshold 0.35: DER 6.53%, MS 0.44%, FA 2.53%, SC 3.56%
#Eval for threshold 0.40: DER 5.94%, MS 1.36%, FA 1.11%, SC 3.47% as report
#Eval for threshold 0.45: DER 6.11%, MS 2.65%, FA 0.37%, SC 3.10%
#Eval for threshold 0.50: DER 6.88%, MS 4.13%, FA 0.06%, SC 2.70%
#Eval for threshold 0.55: DER 7.91%, MS 5.72%, FA 0.00%, SC 2.19%
#Eval for threshold 0.60: DER 9.20%, MS 7.34%, FA 0.00%, SC 1.86%
#Eval for threshold 0.70: DER 12.77%, MS 11.65%, FA 0.00%, SC 1.12%
#Eval for threshold 0.80: DER 18.05%, MS 17.46%, FA 0.00%, SC 0.58%

# Test of alimeeting, collar=0.25
#Eval for threshold 0.20: DER 27.67%, MS 0.00%, FA 24.80%, SC 2.87%
#Eval for threshold 0.30: DER 15.24%, MS 0.12%, FA 9.85%, SC 5.27%
#Eval for threshold 0.35: DER 12.03%, MS 0.99%, FA 5.07%, SC 5.97%
#Eval for threshold 0.40: DER 10.82%, MS 2.81%, FA 2.17%, SC 5.84% as report
#Eval for threshold 0.45: DER 10.97%, MS 5.16%, FA 0.67%, SC 5.14%
#Eval for threshold 0.50: DER 12.13%, MS 7.74%, FA 0.07%, SC 4.32%
#Eval for threshold 0.55: DER 13.88%, MS 10.64%, FA 0.00%, SC 3.24%
#Eval for threshold 0.60: DER 16.06%, MS 13.67%, FA 0.00%, SC 2.39%
#Eval for threshold 0.70: DER 20.76%, MS 19.61%, FA 0.00%, SC 1.15%
#Eval for threshold 0.80: DER 26.06%, MS 25.54%, FA 0.00%, SC 0.52%


# compared with stage155-156,stage174-175 will cam++_zh_200k speaker embedding and cam++_zh_200k ckpt init speech encoder
if [ ${stage} -le 174 ] && [ ${stop_stage} -ge 174 ];then
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
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_real_cam++_zh_200k_feature_dir_cam++_zh_200k_ckpt_init_speech_encoder_epoch20_front_fix_seed_lr2e4_single_backend_transformer_multi_backend_transformer_rs_len8_shift0.8
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
    --num-transformer-layer $num_transformer_layer
fi

if [ ${stage} -le 175 ] && [ ${stop_stage} -ge 175 ];then
 exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_real_cam++_zh_200k_feature_dir_cam++_zh_200k_ckpt_init_speech_encoder_epoch20_front_fix_seed_lr2e4_single_backend_transformer_multi_backend_transformer_rs_len8_shift0.8
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
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
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


#grep -r Eval logs/run_ts_vad2_aistation_stage175.log
#2025-05-12 10:47:08,807 (infer2:255) INFO: currently, it will infer Eval set.
# Eval of alimeeting, collar=0.0
#Eval for threshold 0.20: DER 18.17%, MS 2.34%, FA 14.91%, SC 0.92%
#Eval for threshold 0.30: DER 14.62%, MS 3.50%, FA 10.05%, SC 1.07%
#Eval for threshold 0.35: DER 13.58%, MS 4.14%, FA 8.29%, SC 1.14%
#Eval for threshold 0.40: DER 12.91%, MS 4.76%, FA 7.00%, SC 1.15%
#Eval for threshold 0.45: DER 12.49%, MS 5.46%, FA 5.91%, SC 1.11%
#Eval for threshold 0.50: DER 12.34%, MS 6.26%, FA 5.00%, SC 1.07% as report
#Eval for threshold 0.55: DER 12.37%, MS 7.11%, FA 4.22%, SC 1.04%
#Eval for threshold 0.60: DER 12.57%, MS 8.08%, FA 3.53%, SC 0.97%
#Eval for threshold 0.70: DER 13.50%, MS 10.27%, FA 2.45%, SC 0.78%
#Eval for threshold 0.80: DER 15.34%, MS 13.24%, FA 1.55%, SC 0.55%

# Test of alimeeting, collar=0.0
#Eval for threshold 0.20: DER 17.45%, MS 2.35%, FA 14.12%, SC 0.98%
#Eval for threshold 0.30: DER 14.38%, MS 3.48%, FA 9.71%, SC 1.18%
#Eval for threshold 0.35: DER 13.51%, MS 4.14%, FA 8.15%, SC 1.22%
#Eval for threshold 0.40: DER 12.97%, MS 4.84%, FA 6.87%, SC 1.26%
#Eval for threshold 0.45: DER 12.63%, MS 5.59%, FA 5.78%, SC 1.27% 
#Eval for threshold 0.50: DER 12.50%, MS 6.37%, FA 4.88%, SC 1.25% as report
#Eval for threshold 0.55: DER 12.54%, MS 7.25%, FA 4.07%, SC 1.22%
#Eval for threshold 0.60: DER 12.81%, MS 8.25%, FA 3.41%, SC 1.15%
#Eval for threshold 0.70: DER 13.86%, MS 10.61%, FA 2.30%, SC 0.95%
#Eval for threshold 0.80: DER 15.99%, MS 13.87%, FA 1.42%, SC 0.69%
#2025-05-12 11:25:24,761 (infer2:255) INFO: currently, it will infer Eval set.

# Eval of alimeeting, collar=0.25
#Eval for threshold 0.20: DER 7.91%, MS 0.79%, FA 6.86%, SC 0.25%
#Eval for threshold 0.30: DER 5.70%, MS 1.22%, FA 4.17%, SC 0.30%
#Eval for threshold 0.35: DER 5.11%, MS 1.48%, FA 3.30%, SC 0.33%
#Eval for threshold 0.40: DER 4.83%, MS 1.81%, FA 2.69%, SC 0.33%
#Eval for threshold 0.45: DER 4.62%, MS 2.12%, FA 2.18%, SC 0.31%
#Eval for threshold 0.50: DER 4.57%, MS 2.51%, FA 1.76%, SC 0.31% as report
#Eval for threshold 0.55: DER 4.61%, MS 2.88%, FA 1.42%, SC 0.31%
#Eval for threshold 0.60: DER 4.81%, MS 3.36%, FA 1.15%, SC 0.30%
#Eval for threshold 0.70: DER 5.50%, MS 4.48%, FA 0.79%, SC 0.23%
#Eval for threshold 0.80: DER 6.76%, MS 6.09%, FA 0.51%, SC 0.16%

# Test of alimeeting, collar=0.25
#Eval for threshold 0.20: DER 7.72%, MS 0.95%, FA 6.43%, SC 0.34%
#Eval for threshold 0.30: DER 5.79%, MS 1.45%, FA 3.92%, SC 0.42%
#Eval for threshold 0.35: DER 5.30%, MS 1.75%, FA 3.12%, SC 0.44%
#Eval for threshold 0.40: DER 5.01%, MS 2.08%, FA 2.47%, SC 0.46%
#Eval for threshold 0.45: DER 4.84%, MS 2.42%, FA 1.94%, SC 0.48%
#Eval for threshold 0.50: DER 4.79%, MS 2.81%, FA 1.52%, SC 0.46% as report
#Eval for threshold 0.55: DER 4.88%, MS 3.25%, FA 1.17%, SC 0.46%
#Eval for threshold 0.60: DER 5.09%, MS 3.77%, FA 0.90%, SC 0.42%
#Eval for threshold 0.70: DER 5.91%, MS 5.09%, FA 0.51%, SC 0.32%
#Eval for threshold 0.80: DER 7.36%, MS 6.89%, FA 0.26%, SC 0.21%

# compared with stage172-173, stage176-177 will increase dropout rate 
if [ ${stage} -le 176 ] && [ ${stop_stage} -ge 176 ];then
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
    spk_path=/maduo/model_hub/ts_vad/spk_embed/alimeeting_wo_sil/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    data_dir="/maduo/datasets/alimeeting_wo_sil" # oracle target audio , mix audio and labels path
    rs_len=8
    segment_shift=0.8
    lr=2e-4
    lr_type="PolynomialDecayLR" # default PolynomialDecayLR`,  `CosineAnnealingLR`, `ReduceLROnPlateau` 
    dropout=0.3
    single_backend_type="transformer"
    multi_backend_type="transformer"
    num_transformer_layer=2
    exp_dir=/maduo/exp/speaker_diarization/ts_vad2/alimeeting_wo_sil/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_epoch20_front_fix_seed_lr${lr}_single_backend_${single_backend_type}_multi_backend_${multi_backend_type}_rs_len${rs_len}_shift${segment_shift}_lr_type_${lr_type}_dropout${dropout}
    mkdir -p $exp_dir
    CUDA_VISIABLE_DEVICES=0,1 \
    NCCL_ASYNC_ERROR_HANDLING=1\
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15215 \
   ts_vad2/train_accelerate_ddp2_debug2.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-updates 4000\
    --grad-clip true\
    --lr $lr\
    --lr-type $lr_type\
    --dropout $dropout\
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

if [ ${stage} -le 177 ] && [ ${stop_stage} -ge 177 ];then
 rs_len=8
 segment_shift=0.8
 single_backend_type="transformer"
 multi_backend_type="transformer"
 lr=2e-4
 lr_type="PolynomialDecayLR" # default PolynomialDecayLR`,  `CosineAnnealingLR`, `ReduceLROnPlateau`
 dropout=0.3
 exp_dir=/maduo/exp/speaker_diarization/ts_vad2/alimeeting_wo_sil/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_epoch20_front_fix_seed_lr${lr}_single_backend_${single_backend_type}_multi_backend_${multi_backend_type}_rs_len${rs_len}_shift${segment_shift}_lr_type_${lr_type}_dropout${dropout}
 model_file=$exp_dir/best-valid-der.pt
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 # alimeeting_wo_sil groundtruth rttm
 # cp -r  /maduo/datasets/alimeeting_wo_sil/Eval_Ali/Eval_Ali_far/alimeeting_eval.rttm /maduo/datasets/alimeeting_wo_sil/
 # cp -r  /maduo/datasets/alimeeting_wo_sil/Test_Ali/Test_Ali_far/alimeeting_test.rttm /maduo/datasets/alimeeting_wo_sil/
 rttm_dir=/maduo/datasets/alimeeting_wo_sil/
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 dataset_name="alimeeting" # dataset name
 # for loading speaker embedding file
 spk_path=/maduo/model_hub/ts_vad/spk_embed/alimeeting_wo_sil/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 data_dir="/maduo/datasets/alimeeting_wo_sil" # oracle target audio , mix audio and labels path
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
# grep -r Eval logs/run_ts_vad2_aistation_stage177.log
#2025-05-12 12:23:52,641 (infer2:255) INFO: currently, it will infer Eval set.
# Eval of alimeeting, collar=0.0
#Eval for threshold 0.20: DER 40.22%, MS 0.00%, FA 37.43%, SC 2.78%
#Eval for threshold 0.30: DER 19.07%, MS 1.06%, FA 11.62%, SC 6.40%
#Eval for threshold 0.35: DER 15.72%, MS 3.90%, FA 5.74%, SC 6.08%
#Eval for threshold 0.40: DER 15.16%, MS 7.23%, FA 2.59%, SC 5.34%
#Eval for threshold 0.45: DER 16.41%, MS 11.17%, FA 1.07%, SC 4.17%
#Eval for threshold 0.50: DER 18.83%, MS 14.88%, FA 0.49%, SC 3.47%
#Eval for threshold 0.55: DER 21.71%, MS 18.92%, FA 0.32%, SC 2.47%
#Eval for threshold 0.60: DER 25.04%, MS 22.93%, FA 0.27%, SC 1.84%
#Eval for threshold 0.70: DER 32.45%, MS 31.23%, FA 0.19%, SC 1.03%
#Eval for threshold 0.80: DER 41.76%, MS 41.00%, FA 0.12%, SC 0.65%

# Test of alimeeting, collar=0.0
#Eval for threshold 0.20: DER 46.86%, MS 0.01%, FA 44.06%, SC 2.79%
#Eval for threshold 0.30: DER 22.75%, MS 2.11%, FA 13.32%, SC 7.32%
#Eval for threshold 0.35: DER 19.76%, MS 6.04%, FA 6.87%, SC 6.85%
#Eval for threshold 0.40: DER 19.38%, MS 10.14%, FA 3.28%, SC 5.96%
#Eval for threshold 0.45: DER 20.84%, MS 14.42%, FA 1.38%, SC 5.04%
#Eval for threshold 0.50: DER 23.10%, MS 18.48%, FA 0.47%, SC 4.15%
#Eval for threshold 0.55: DER 26.02%, MS 22.53%, FA 0.28%, SC 3.21%
#Eval for threshold 0.60: DER 29.11%, MS 26.54%, FA 0.23%, SC 2.34%
#Eval for threshold 0.70: DER 35.48%, MS 34.07%, FA 0.16%, SC 1.25%
#Eval for threshold 0.80: DER 41.69%, MS 40.99%, FA 0.09%, SC 0.61%
#2025-05-12 13:11:06,176 (infer2:255) INFO: currently, it will infer Eval set.

# Eval of alimeeting, collar=0.25
#Eval for threshold 0.20: DER 32.76%, MS 0.00%, FA 30.74%, SC 2.02%
#Eval for threshold 0.30: DER 14.71%, MS 0.74%, FA 9.16%, SC 4.81%
#Eval for threshold 0.35: DER 11.89%, MS 2.82%, FA 4.38%, SC 4.69%
#Eval for threshold 0.40: DER 11.41%, MS 5.60%, FA 1.75%, SC 4.06%
#Eval for threshold 0.45: DER 12.58%, MS 8.95%, FA 0.54%, SC 3.09%
#Eval for threshold 0.50: DER 14.93%, MS 12.33%, FA 0.07%, SC 2.53%
#Eval for threshold 0.55: DER 17.87%, MS 16.21%, FA 0.00%, SC 1.65%
#Eval for threshold 0.60: DER 21.18%, MS 20.04%, FA 0.00%, SC 1.14%
#Eval for threshold 0.70: DER 28.89%, MS 28.36%, FA 0.00%, SC 0.53%
#Eval for threshold 0.80: DER 38.99%, MS 38.72%, FA 0.00%, SC 0.27%

# Test of alimeeting, collar=0.25
#Eval for threshold 0.20: DER 40.99%, MS 0.00%, FA 38.85%, SC 2.14%
#Eval for threshold 0.30: DER 19.35%, MS 1.67%, FA 11.73%, SC 5.95%
#Eval for threshold 0.35: DER 16.64%, MS 4.99%, FA 6.03%, SC 5.62%
#Eval for threshold 0.40: DER 16.22%, MS 8.61%, FA 2.65%, SC 4.96%
#Eval for threshold 0.45: DER 17.69%, MS 12.59%, FA 0.88%, SC 4.23%
#Eval for threshold 0.50: DER 20.07%, MS 16.53%, FA 0.11%, SC 3.43%
#Eval for threshold 0.55: DER 23.14%, MS 20.55%, FA 0.00%, SC 2.59%
#Eval for threshold 0.60: DER 26.45%, MS 24.68%, FA 0.00%, SC 1.77%
#Eval for threshold 0.70: DER 33.08%, MS 32.28%, FA 0.00%, SC 0.80%
#Eval for threshold 0.80: DER 39.49%, MS 39.22%, FA 0.00%, SC 0.27%

# compared with stage172-173, stage178-179 will use ReduceLROnPlateau lr_scheduler 
if [ ${stage} -le 178 ] && [ ${stop_stage} -ge 178 ];then
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
    spk_path=/maduo/model_hub/ts_vad/spk_embed/alimeeting_wo_sil/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    data_dir="/maduo/datasets/alimeeting_wo_sil" # oracle target audio , mix audio and labels path
    rs_len=8
    segment_shift=0.8
    lr=2e-4
    lr_type="ReduceLROnPlateau" # default PolynomialDecayLR`,  `CosineAnnealingLR`, `ReduceLROnPlateau`
    dropout=0.1
    single_backend_type="transformer"
    multi_backend_type="transformer"
    num_transformer_layer=2
    exp_dir=/maduo/exp/speaker_diarization/ts_vad2/alimeeting_wo_sil/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_epoch20_front_fix_seed_lr${lr}_single_backend_${single_backend_type}_multi_backend_${multi_backend_type}_rs_len${rs_len}_shift${segment_shift}_lr_type_${lr_type}_dropout${dropout}
    mkdir -p $exp_dir
    CUDA_VISIABLE_DEVICES=0,1 \
    NCCL_ASYNC_ERROR_HANDLING=1\
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15215 \
   ts_vad2/train_accelerate_ddp2_debug3.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-updates 4000\
    --grad-clip true\
    --lr $lr\
    --lr-type $lr_type\
    --dropout $dropout\
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

if [ ${stage} -le 179 ] && [ ${stop_stage} -ge 179 ];then
 rs_len=8
 segment_shift=0.8
 single_backend_type="transformer"
 multi_backend_type="transformer"
 lr=2e-4
 lr_type="ReduceLROnPlateau" # default PolynomialDecayLR`,  `CosineAnnealingLR`, `ReduceLROnPlateau`
 dropout=0.1
 exp_dir=/maduo/exp/speaker_diarization/ts_vad2/alimeeting_wo_sil/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_epoch20_front_fix_seed_lr${lr}_single_backend_${single_backend_type}_multi_backend_${multi_backend_type}_rs_len${rs_len}_shift${segment_shift}_lr_type_${lr_type}_dropout${dropout}
 model_file=$exp_dir/best-valid-der.pt
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 # alimeeting_wo_sil groundtruth rttm
 # cp -r  /maduo/datasets/alimeeting_wo_sil/Eval_Ali/Eval_Ali_far/alimeeting_eval.rttm /maduo/datasets/alimeeting_wo_sil/
 # cp -r  /maduo/datasets/alimeeting_wo_sil/Test_Ali/Test_Ali_far/alimeeting_test.rttm /maduo/datasets/alimeeting_wo_sil/
 rttm_dir=/maduo/datasets/alimeeting_wo_sil/
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
 dataset_name="alimeeting" # dataset name
 # for loading speaker embedding file
 spk_path=/maduo/model_hub/ts_vad/spk_embed/alimeeting_wo_sil/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 data_dir="/maduo/datasets/alimeeting_wo_sil" # oracle target audio , mix audio and labels path
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

# compared with stage155-156. stage180-181 will use more small step(0.4s) to cut audio to get train sample
if [ ${stage} -le 180 ] && [ ${stop_stage} -ge 180 ];then
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
    exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_epoch20_front_fix_seed_lr2e4_single_backend_transformer_multi_backend_transformer_rs_len8_shift0.4_again
    mkdir -p $exp_dir
    data_dir="/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    rs_len=8
    segment_shift=0.4
    single_backend_type="transformer"
    multi_backend_type="transformer"
    num_transformer_layer=2
    CUDA_VISIABLE_DEVICES=0,1 \
    NCCL_ASYNC_ERROR_HANDLING=1\
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15115 \
   ts_vad2/train_accelerate_ddp2_debug2.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 17\
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

if [ ${stage} -le 181 ] && [ ${stop_stage} -ge 181 ];then
 exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_epoch20_front_fix_seed_lr2e4_single_backend_transformer_multi_backend_transformer_rs_len8_shift0.4_again
 model_file=$exp_dir/best-valid-der.pt
 rs_len=8
 segment_shift=0.4
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
# grep -r Eval logs/run_ts_vad2_aistation_stage180-181_5.log

# Eval of alimeeting, collar=0.0
#Eval for threshold 0.20: DER 14.94%, MS 3.08%, FA 10.93%, SC 0.93%
#Eval for threshold 0.30: DER 13.06%, MS 4.18%, FA 7.92%, SC 0.97%
#Eval for threshold 0.35: DER 12.54%, MS 4.69%, FA 6.84%, SC 1.01%
#Eval for threshold 0.40: DER 12.15%, MS 5.25%, FA 5.88%, SC 1.02%
#Eval for threshold 0.45: DER 11.91%, MS 5.80%, FA 5.10%, SC 1.01% as report
#Eval for threshold 0.50: DER 11.84%, MS 6.41%, FA 4.45%, SC 0.98%
#Eval for threshold 0.55: DER 11.88%, MS 7.04%, FA 3.88%, SC 0.96%
#Eval for threshold 0.60: DER 12.03%, MS 7.78%, FA 3.34%, SC 0.91%
#Eval for threshold 0.70: DER 12.74%, MS 9.51%, FA 2.46%, SC 0.77%
#Eval for threshold 0.80: DER 14.23%, MS 11.98%, FA 1.68%, SC 0.57%

# Test of alimeeting, collar=0.0
#Eval for threshold 0.20: DER 14.77%, MS 3.23%, FA 10.41%, SC 1.12%
#Eval for threshold 0.30: DER 13.14%, MS 4.34%, FA 7.57%, SC 1.23%
#Eval for threshold 0.35: DER 12.68%, MS 4.90%, FA 6.53%, SC 1.25%
#Eval for threshold 0.40: DER 12.40%, MS 5.47%, FA 5.67%, SC 1.26%
#Eval for threshold 0.45: DER 12.29%, MS 6.08%, FA 4.95%, SC 1.26% as report
#Eval for threshold 0.50: DER 12.29%, MS 6.72%, FA 4.33%, SC 1.24%
#Eval for threshold 0.55: DER 12.40%, MS 7.43%, FA 3.76%, SC 1.20%
#Eval for threshold 0.60: DER 12.66%, MS 8.25%, FA 3.28%, SC 1.14%
#Eval for threshold 0.70: DER 13.61%, MS 10.19%, FA 2.43%, SC 0.99%
#Eval for threshold 0.80: DER 15.25%, MS 12.79%, FA 1.67%, SC 0.79%
#2025-05-28 08:31:57,600 (infer2:255) INFO: currently, it will infer Eval set.

# Eval of alimeeting, collar=0.25
#Eval for threshold 0.20: DER 5.68%, MS 1.06%, FA 4.36%, SC 0.26%
#Eval for threshold 0.30: DER 4.64%, MS 1.49%, FA 2.89%, SC 0.25%
#Eval for threshold 0.35: DER 4.40%, MS 1.71%, FA 2.42%, SC 0.27%
#Eval for threshold 0.40: DER 4.22%, MS 1.94%, FA 2.00%, SC 0.28%
#Eval for threshold 0.45: DER 4.13%, MS 2.18%, FA 1.67%, SC 0.28% as report
#Eval for threshold 0.50: DER 4.15%, MS 2.45%, FA 1.41%, SC 0.30%
#Eval for threshold 0.55: DER 4.23%, MS 2.74%, FA 1.19%, SC 0.30%
#Eval for threshold 0.60: DER 4.38%, MS 3.09%, FA 0.99%, SC 0.30%
#Eval for threshold 0.70: DER 4.93%, MS 3.98%, FA 0.70%, SC 0.25%
#Eval for threshold 0.80: DER 5.92%, MS 5.26%, FA 0.50%, SC 0.16%

# Test of alimeeting, collar=0.25
#Eval for threshold 0.20: DER 5.78%, MS 1.37%, FA 4.01%, SC 0.39%
#Eval for threshold 0.30: DER 4.91%, MS 1.89%, FA 2.57%, SC 0.45%
#Eval for threshold 0.35: DER 4.68%, MS 2.14%, FA 2.09%, SC 0.46%
#Eval for threshold 0.40: DER 4.54%, MS 2.38%, FA 1.69%, SC 0.46%
#Eval for threshold 0.45: DER 4.52%, MS 2.67%, FA 1.37%, SC 0.47% as report
#Eval for threshold 0.50: DER 4.58%, MS 2.98%, FA 1.14%, SC 0.46%
#Eval for threshold 0.55: DER 4.70%, MS 3.35%, FA 0.92%, SC 0.44%
#Eval for threshold 0.60: DER 4.95%, MS 3.79%, FA 0.75%, SC 0.41%
#Eval for threshold 0.70: DER 5.71%, MS 4.86%, FA 0.49%, SC 0.36%
#Eval for threshold 0.80: DER 6.94%, MS 6.37%, FA 0.30%, SC 0.27%



#  Epoch 15, batch_idx_train: 120285, num_updates: 108000, {'loss': nan, 'DER': 1.0, 'ACC': 0.680859375, 'MI': 1.0, 'FA': 0.0, 'CF': 0.0}, when lr=5e-5 
if [ ${stage} -le 182 ] && [ ${stop_stage} -ge 182 ];then
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
    # cam++_zh_200k
    #speech_encoder_type="CAM++"
    #speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    # w2v-bert2
    speech_encoder_type="w2v-bert2"
    speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
    speech_encoder_config="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    dataset_name="alimeeting" # dataset name

    # for loading speaker embedding file
    spk_path=/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    #exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_wav-bert2.0_epoch20_front_fix_seed_lr1e4_single_backend_transformer_multi_backend_transformer_rs_len8_shift0.8 #
    #exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_wav-bert2.0_epoch20_front_fix_seed_lr5e5_single_backend_transformer_multi_backend_transformer_rs_len8_shift0.4
    exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_wav-bert2.0_epoch20_front_fix_seed_lr2e5_single_backend_transformer_multi_backend_transformer_rs_len8_shift0.4
    mkdir -p $exp_dir
    data_dir="/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    rs_len=8
    segment_shift=0.4
    single_backend_type="transformer"
    multi_backend_type="transformer"
    num_transformer_layer=2
    CUDA_VISIABLE_DEVICES=0,1 \
    TORCH_NCCL_ASYNC_ERROR_HANDLING=1\
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 16115 \
   ts_vad2/train_accelerate_ddp2_debug2.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 13\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-updates 4000\
    --grad-clip true\
    --lr 2e-5\
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
    --num-transformer-layer $num_transformer_layer
fi

if [ ${stage} -le 183 ] && [ ${stop_stage} -ge 183 ];then
 #exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_wav-bert2.0_epoch20_front_fix_seed_lr1e4_single_backend_transformer_multi_backend_transformer_rs_len8_shift0.8
 exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_wav-bert2.0_epoch20_front_fix_seed_lr2e5_single_backend_transformer_multi_backend_transformer_rs_len8_shift0.4
 model_file=$exp_dir/best-valid-der.pt
 rs_len=8
 segment_shift=0.4
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
 #speech_encoder_type="CAM++"
 #speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"

 # w2v-bert2
 speech_encoder_type="w2v-bert2"
 speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 speech_encoder_config="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"


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
    --speech-encoder-config $speech_encoder_config\
    --select-encoder-layer-nums 6\
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
#grep -r Eval  logs/run_ts_vad2_aistation_stage182-183_1r2e-5_from_epoch13.log
#Eval for threshold 0.20: DER 15.24%, MS 2.61%, FA 11.83%, SC 0.80%
#Eval for threshold 0.30: DER 12.92%, MS 3.81%, FA 8.15%, SC 0.95%
#Eval for threshold 0.35: DER 12.31%, MS 4.43%, FA 6.90%, SC 0.98%
#Eval for threshold 0.40: DER 11.95%, MS 5.10%, FA 5.88%, SC 0.98%
#Eval for threshold 0.45: DER 11.76%, MS 5.79%, FA 5.00%, SC 0.97% as report
#Eval for threshold 0.50: DER 11.72%, MS 6.52%, FA 4.27%, SC 0.94%
#Eval for threshold 0.55: DER 11.89%, MS 7.37%, FA 3.66%, SC 0.87%
#Eval for threshold 0.60: DER 12.16%, MS 8.23%, FA 3.14%, SC 0.79%
#Eval for threshold 0.70: DER 13.18%, MS 10.37%, FA 2.19%, SC 0.62%
#Eval for threshold 0.80: DER 15.33%, MS 13.45%, FA 1.43%, SC 0.45%
#Eval for threshold 0.20: DER 14.98%, MS 2.69%, FA 11.46%, SC 0.83%
#Eval for threshold 0.30: DER 12.84%, MS 3.95%, FA 7.91%, SC 0.98%
#Eval for threshold 0.35: DER 12.29%, MS 4.57%, FA 6.69%, SC 1.03%
#Eval for threshold 0.40: DER 12.04%, MS 5.32%, FA 5.64%, SC 1.07%
#Eval for threshold 0.45: DER 11.96%, MS 6.10%, FA 4.78%, SC 1.08% as report
#Eval for threshold 0.50: DER 12.04%, MS 6.93%, FA 4.06%, SC 1.06%
#Eval for threshold 0.55: DER 12.29%, MS 7.86%, FA 3.42%, SC 1.00%
#Eval for threshold 0.60: DER 12.70%, MS 8.85%, FA 2.90%, SC 0.96%
#Eval for threshold 0.70: DER 13.96%, MS 11.16%, FA 2.00%, SC 0.80%
#Eval for threshold 0.80: DER 16.37%, MS 14.50%, FA 1.26%, SC 0.61%
#2025-06-08 13:09:49,597 (infer2:255) INFO: currently, it will infer Eval set.
#Eval for threshold 0.20: DER 5.93%, MS 0.89%, FA 4.86%, SC 0.18%
#Eval for threshold 0.30: DER 4.56%, MS 1.31%, FA 3.02%, SC 0.23%
#Eval for threshold 0.35: DER 4.26%, MS 1.55%, FA 2.46%, SC 0.25%
#Eval for threshold 0.40: DER 4.08%, MS 1.82%, FA 2.02%, SC 0.24%
#Eval for threshold 0.45: DER 3.95%, MS 2.09%, FA 1.63%, SC 0.23% as report
#Eval for threshold 0.50: DER 3.97%, MS 2.43%, FA 1.32%, SC 0.23%
#Eval for threshold 0.55: DER 4.09%, MS 2.80%, FA 1.09%, SC 0.20%
#Eval for threshold 0.60: DER 4.27%, MS 3.19%, FA 0.90%, SC 0.17%
#Eval for threshold 0.70: DER 5.06%, MS 4.31%, FA 0.64%, SC 0.12%
#Eval for threshold 0.80: DER 6.57%, MS 6.03%, FA 0.45%, SC 0.09%
#Eval for threshold 0.20: DER 6.31%, MS 1.13%, FA 4.92%, SC 0.27%
#Eval for threshold 0.30: DER 4.95%, MS 1.64%, FA 2.97%, SC 0.34%
#Eval for threshold 0.35: DER 4.61%, MS 1.88%, FA 2.36%, SC 0.37%
#Eval for threshold 0.40: DER 4.46%, MS 2.22%, FA 1.85%, SC 0.39%
#Eval for threshold 0.45: DER 4.40%, MS 2.58%, FA 1.43%, SC 0.39% as report
#Eval for threshold 0.50: DER 4.49%, MS 2.98%, FA 1.11%, SC 0.40%
#Eval for threshold 0.55: DER 4.68%, MS 3.45%, FA 0.85%, SC 0.37%
#Eval for threshold 0.60: DER 4.99%, MS 3.98%, FA 0.66%, SC 0.35%
#Eval for threshold 0.70: DER 5.95%, MS 5.29%, FA 0.38%, SC 0.28%
#Eval for threshold 0.80: DER 7.63%, MS 7.22%, FA 0.22%, SC 0.20%




# compared stage138-139 of run_ts_vad2.sh stage190-191 will use w2v-bert2 to replase cam++
if [ ${stage} -le 190 ] && [ ${stop_stage} -ge 190 ];then
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
    dataset_name="alimeeting" # dataset name
    speech_encoder_type="w2v-bert2"
    speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
    speech_encoder_config="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
    # for loading speaker embedding file
    spk_path=/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_epoch40_front_fix_seed_lr1e5_w2v-bert2_speech_encoder_single_backend_mamba2_multi_backend_transformer_rs_len10_shift2_d_state256
    mkdir -p $exp_dir
    data_dir="/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    rs_len=10
    segment_shift=2
    single_backend_type="mamba2"
    multi_backend_type="transformer"
    d_state=256
    num_transformer_layer=2
    CUDA_VISIABLE_DEVICES=0,1 \
    NCCL_ASYNC_ERROR_HANDLING=1\
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15115 \
   ts_vad2/train_accelerate_ddp2_debug2.py \
    --world-size 2 \
    --num-epochs 40\
    --start-epoch 8\
    --start-batch 34500\
    --keep-last-k 2\
    --keep-last-epoch 2\
    --freeze-updates 4000\
    --grad-clip true\
    --lr 1e-5\
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
# note: lr=1e-4,  end of epoch 4, batch_idx: 1564,  batch_idx_train: 6259, {'loss': nan, 'DER': 1.0, 'ACC': 0.68146875, 'MI': 1.0, 'FA': 0.0, 'CF': 0.0}, batch size: 64, grad_norm: nan, grad_scale: , lr: 7.649444444444445e-05,
# lr=5e-5, 2025-07-21 16:54:31,252 (train_accelerate_ddp2_debug2:731) INFO: [Train] - Epoch 6, batch_idx_train: 9325, num_updates: 9000, {'loss': nan, 'DER': 1.0, 'ACC': 0.684375, 'MI': 1.0, 'FA': 0.0, 'CF': 0.0}, batch size: 64, grad_norm: nan, grad_scale: , lr: 3.0419444444444444e-05, 
# lr=1e-5, epoch27 train loss is  nan


if [ ${stage} -le 191 ] && [ ${stop_stage} -ge 191 ];then
 exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_epoch40_front_fix_seed_lr1e5_w2v-bert2_speech_encoder_single_backend_mamba2_multi_backend_transformer_rs_len10_shift2_d_state256
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
 rttm_dir=/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 dataset_name="alimeeting" # dataset name
 speech_encoder_type="w2v-bert2"
 speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 speech_encoder_config="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
 # for loading speaker embedding file
 spk_path=/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
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


# compared stage138-139 of run_ts_vad2.sh stage190-191 will use w2v-bert2 to replase cam++
if [ ${stage} -le 192 ] && [ ${stop_stage} -ge 192 ];then
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
    dataset_name="alimeeting" # dataset name
    speech_encoder_type="w2v-bert2"
    speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
    speech_encoder_config="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
    # for loading speaker embedding file
    spk_path=/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_epoch40_front_fix_seed_lr5e6_w2v-bert2_speech_encoder_single_backend_mamba2_multi_backend_transformer_rs_len10_shift2_d_state256
    mkdir -p $exp_dir
    data_dir="/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    rs_len=10
    segment_shift=2
    single_backend_type="mamba2"
    multi_backend_type="transformer"
    d_state=256
    num_transformer_layer=2
    CUDA_VISIABLE_DEVICES=0,1 \
    NCCL_ASYNC_ERROR_HANDLING=1\
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15215 \
   ts_vad2/train_accelerate_ddp2_debug2.py \
    --world-size 2 \
    --num-epochs 40\
    --start-epoch 1\
    --start-batch 36000\
    --keep-last-k 2\
    --keep-last-epoch 2\
    --freeze-updates 4000\
    --grad-clip true\
    --lr 5e-6\
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
# note: lr=1e-4,  end of epoch 4, batch_idx: 1564,  batch_idx_train: 6259, {'loss': nan, 'DER': 1.0, 'ACC': 0.68146875, 'MI': 1.0, 'FA': 0.0, 'CF': 0.0}, batch size: 64, grad_norm: nan, grad_scale: , lr: 7.649444444444445e-05,
# lr=5e-5, 2025-07-21 16:54:31,252 (train_accelerate_ddp2_debug2:731) INFO: [Train] - Epoch 6, batch_idx_train: 9325, num_updates: 9000, {'loss': nan, 'DER': 1.0, 'ACC': 0.684375, 'MI': 1.0, 'FA': 0.0, 'CF': 0.0}, batch size: 64, grad_norm: nan, grad_scale: , lr: 3.0419444444444444e-05,



if [ ${stage} -le 193 ] && [ ${stop_stage} -ge 193 ];then
 exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_epoch40_front_fix_seed_lr5e6_w2v-bert2_speech_encoder_single_backend_mamba2_multi_backend_transformer_rs_len10_shift2_d_state256
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
 rttm_dir=/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 dataset_name="alimeeting" # dataset name
 speech_encoder_type="w2v-bert2"
 speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 speech_encoder_config="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
 # for loading speaker embedding file
 spk_path=/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
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
#grep -r Eval logs/run_ts_vad2_aistation_stage192-193_lr5e6_6.log
# 
#Eval for threshold 0.20: DER 20.11%, MS 2.36%, FA 16.32%, SC 1.43%
#Eval for threshold 0.30: DER 15.78%, MS 4.12%, FA 10.11%, SC 1.55%
#Eval for threshold 0.35: DER 14.65%, MS 5.13%, FA 7.93%, SC 1.59%
#Eval for threshold 0.40: DER 14.09%, MS 6.19%, FA 6.33%, SC 1.57%
#Eval for threshold 0.45: DER 13.96%, MS 7.44%, FA 5.01%, SC 1.51%
#Eval for threshold 0.50: DER 14.23%, MS 8.83%, FA 4.04%, SC 1.36%
#Eval for threshold 0.55: DER 14.69%, MS 10.29%, FA 3.20%, SC 1.20%
#Eval for threshold 0.60: DER 15.53%, MS 11.92%, FA 2.57%, SC 1.03%
#Eval for threshold 0.70: DER 18.21%, MS 15.83%, FA 1.68%, SC 0.70%
#Eval for threshold 0.80: DER 23.10%, MS 21.67%, FA 1.05%, SC 0.38%
#Eval for threshold 0.20: DER 20.65%, MS 2.49%, FA 16.91%, SC 1.25%
#Eval for threshold 0.30: DER 16.36%, MS 4.35%, FA 10.47%, SC 1.54%
#Eval for threshold 0.35: DER 15.23%, MS 5.45%, FA 8.19%, SC 1.59%
#Eval for threshold 0.40: DER 14.65%, MS 6.67%, FA 6.36%, SC 1.62%
#Eval for threshold 0.45: DER 14.61%, MS 8.10%, FA 4.93%, SC 1.58%
#Eval for threshold 0.50: DER 14.98%, MS 9.67%, FA 3.81%, SC 1.50%
#Eval for threshold 0.55: DER 15.79%, MS 11.48%, FA 2.93%, SC 1.38%
#Eval for threshold 0.60: DER 16.96%, MS 13.52%, FA 2.26%, SC 1.18%
#Eval for threshold 0.70: DER 20.41%, MS 18.35%, FA 1.27%, SC 0.80%
#Eval for threshold 0.80: DER 26.15%, MS 25.03%, FA 0.68%, SC 0.44%
#2025-08-06 07:58:00,544 (infer2:255) INFO: currently, it will infer Eval set.
#Eval for threshold 0.20: DER 9.47%, MS 0.77%, FA 8.22%, SC 0.48%
#Eval for threshold 0.30: DER 6.47%, MS 1.51%, FA 4.44%, SC 0.53%
#Eval for threshold 0.35: DER 5.72%, MS 1.94%, FA 3.21%, SC 0.57%
#Eval for threshold 0.40: DER 5.44%, MS 2.47%, FA 2.38%, SC 0.59%
#Eval for threshold 0.45: DER 5.41%, MS 3.10%, FA 1.71%, SC 0.59%
#Eval for threshold 0.50: DER 5.69%, MS 3.86%, FA 1.28%, SC 0.56%
#Eval for threshold 0.55: DER 6.12%, MS 4.67%, FA 0.93%, SC 0.52%
#Eval for threshold 0.60: DER 6.77%, MS 5.65%, FA 0.69%, SC 0.44%
#Eval for threshold 0.70: DER 8.86%, MS 8.12%, FA 0.46%, SC 0.28%
#Eval for threshold 0.80: DER 12.67%, MS 12.19%, FA 0.35%, SC 0.13%
#Eval for threshold 0.20: DER 11.62%, MS 1.04%, FA 10.16%, SC 0.42%
#Eval for threshold 0.30: DER 8.24%, MS 1.90%, FA 5.74%, SC 0.61%
#Eval for threshold 0.35: DER 7.34%, MS 2.46%, FA 4.17%, SC 0.71%
#Eval for threshold 0.40: DER 6.82%, MS 3.06%, FA 2.96%, SC 0.80%
#Eval for threshold 0.45: DER 6.76%, MS 3.90%, FA 2.05%, SC 0.81%
#Eval for threshold 0.50: DER 7.03%, MS 4.86%, FA 1.40%, SC 0.77%
#Eval for threshold 0.55: DER 7.68%, MS 6.06%, FA 0.93%, SC 0.69%
#Eval for threshold 0.60: DER 8.65%, MS 7.49%, FA 0.63%, SC 0.54%
#Eval for threshold 0.70: DER 11.43%, MS 10.92%, FA 0.21%, SC 0.31%
#Eval for threshold 0.80: DER 16.17%, MS 15.96%, FA 0.10%, SC 0.11%


# compared stage138-139 of run_ts_vad2.sh stage190-191 will use w2v-bert2 to replase cam++
if [ ${stage} -le 194 ] && [ ${stop_stage} -ge 194 ];then
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
    dataset_name="alimeeting" # dataset name
    speech_encoder_type="w2v-bert2"
    speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
    speech_encoder_config="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
    # for loading speaker embedding file
    spk_path=/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_epoch40_front_fix_seed_lr1e6_w2v-bert2_speech_encoder_single_backend_mamba2_multi_backend_transformer_rs_len10_shift2_d_state256
    mkdir -p $exp_dir
    data_dir="/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    rs_len=10
    segment_shift=2
    single_backend_type="mamba2"
    multi_backend_type="transformer"
    d_state=256
    num_transformer_layer=2
    CUDA_VISIABLE_DEVICES=0,1 \
    NCCL_ASYNC_ERROR_HANDLING=1\
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15315 \
   ts_vad2/train_accelerate_ddp2_debug2.py \
    --world-size 2 \
    --num-epochs 40\
    --start-epoch 24\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-updates 4000\
    --grad-clip true\
    --lr 1e-6\
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
# note: lr=1e-4,  end of epoch 4, batch_idx: 1564,  batch_idx_train: 6259, {'loss': nan, 'DER': 1.0, 'ACC': 0.68146875, 'MI': 1.0, 'FA': 0.0, 'CF': 0.0}, batch size: 64, grad_norm: nan, grad_scale: , lr: 7.649444444444445e-05,
# lr=5e-5, 2025-07-21 16:54:31,252 (train_accelerate_ddp2_debug2:731) INFO: [Train] - Epoch 6, batch_idx_train: 9325, num_updates: 9000, {'loss': nan, 'DER': 1.0, 'ACC': 0.684375, 'MI': 1.0, 'FA': 0.0, 'CF': 0.0}, batch size: 64, grad_norm: nan, grad_scale: , lr: 3.0419444444444444e-05,

if [ ${stage} -le 195 ] && [ ${stop_stage} -ge 195 ];then
 exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_epoch40_front_fix_seed_lr1e6_w2v-bert2_speech_encoder_single_backend_mamba2_multi_backend_transformer_rs_len10_shift2_d_state256
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
 rttm_dir=/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 dataset_name="alimeeting" # dataset name
 speech_encoder_type="w2v-bert2"
 speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 speech_encoder_config="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
 # for loading speaker embedding file
 spk_path=/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
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
#grep -r Eval logs/run_ts_vad2_aistation_stage194-195_lr1e6_h20-3e.log
#Eval for threshold 0.20: DER 36.18%, MS 2.56%, FA 30.66%, SC 2.96%
#Eval for threshold 0.30: DER 30.55%, MS 4.18%, FA 20.96%, SC 5.42%
#Eval for threshold 0.35: DER 29.01%, MS 5.26%, FA 17.74%, SC 6.01%
#Eval for threshold 0.40: DER 27.85%, MS 6.67%, FA 14.81%, SC 6.37%
#Eval for threshold 0.45: DER 26.99%, MS 8.31%, FA 12.09%, SC 6.60%
#Eval for threshold 0.50: DER 26.14%, MS 10.12%, FA 9.47%, SC 6.56%
#Eval for threshold 0.55: DER 25.78%, MS 12.21%, FA 7.21%, SC 6.36%
#Eval for threshold 0.60: DER 25.93%, MS 14.73%, FA 5.20%, SC 6.00%
#Eval for threshold 0.70: DER 29.26%, MS 22.53%, FA 2.43%, SC 4.30%
#Eval for threshold 0.80: DER 36.63%, MS 33.26%, FA 1.20%, SC 2.17%
#Eval for threshold 0.20: DER 37.52%, MS 2.39%, FA 33.16%, SC 1.97%
#Eval for threshold 0.30: DER 31.33%, MS 3.95%, FA 24.61%, SC 2.77%
#Eval for threshold 0.35: DER 28.64%, MS 5.00%, FA 20.44%, SC 3.21%
#Eval for threshold 0.40: DER 26.17%, MS 6.19%, FA 16.20%, SC 3.78%
#Eval for threshold 0.45: DER 24.52%, MS 7.66%, FA 12.60%, SC 4.25%
#Eval for threshold 0.50: DER 23.37%, MS 9.46%, FA 9.41%, SC 4.50%
#Eval for threshold 0.55: DER 23.08%, MS 11.71%, FA 6.92%, SC 4.45%
#Eval for threshold 0.60: DER 23.80%, MS 14.66%, FA 4.87%, SC 4.28%
#Eval for threshold 0.70: DER 28.90%, MS 23.97%, FA 2.04%, SC 2.89%
#Eval for threshold 0.80: DER 39.92%, MS 37.98%, FA 0.68%, SC 1.26%
#2025-08-06 15:24:51,286 (infer2:255) INFO: currently, it will infer Eval set.
#Eval for threshold 0.20: DER 27.12%, MS 1.02%, FA 23.92%, SC 2.18%
#Eval for threshold 0.30: DER 22.31%, MS 1.90%, FA 15.28%, SC 5.12%
#Eval for threshold 0.35: DER 21.26%, MS 2.74%, FA 12.87%, SC 5.65%
#Eval for threshold 0.40: DER 20.46%, MS 3.84%, FA 10.61%, SC 6.01%
#Eval for threshold 0.45: DER 19.80%, MS 5.12%, FA 8.43%, SC 6.25%
#Eval for threshold 0.50: DER 18.94%, MS 6.54%, FA 6.27%, SC 6.13%
#Eval for threshold 0.55: DER 18.46%, MS 8.19%, FA 4.36%, SC 5.91%
#Eval for threshold 0.60: DER 18.58%, MS 10.26%, FA 2.75%, SC 5.58%
#Eval for threshold 0.70: DER 21.75%, MS 17.07%, FA 0.90%, SC 3.78%
#Eval for threshold 0.80: DER 28.21%, MS 26.03%, FA 0.36%, SC 1.82%
#Eval for threshold 0.20: DER 30.38%, MS 1.03%, FA 28.19%, SC 1.17%
#Eval for threshold 0.30: DER 24.86%, MS 1.88%, FA 21.10%, SC 1.87%
#Eval for threshold 0.35: DER 22.10%, MS 2.59%, FA 17.21%, SC 2.30%
#Eval for threshold 0.40: DER 19.30%, MS 3.36%, FA 12.99%, SC 2.94%
#Eval for threshold 0.45: DER 17.46%, MS 4.39%, FA 9.50%, SC 3.57%
#Eval for threshold 0.50: DER 16.05%, MS 5.73%, FA 6.43%, SC 3.89%
#Eval for threshold 0.55: DER 15.68%, MS 7.53%, FA 4.30%, SC 3.85%
#Eval for threshold 0.60: DER 16.38%, MS 10.09%, FA 2.64%, SC 3.65%
#Eval for threshold 0.70: DER 21.68%, MS 18.77%, FA 0.77%, SC 2.14%
#Eval for threshold 0.80: DER 32.54%, MS 31.72%, FA 0.09%, SC 0.73%


if [ ${stage} -le 196 ] && [ ${stop_stage} -ge 196 ];then
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
    dataset_name="alimeeting" # dataset name
    speech_encoder_type="w2v-bert2"
    speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
    speech_encoder_config="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
    # for loading speaker embedding file
    spk_path=/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_epoch40_front_fix_seed_lr1e4_w2v-bert2_speech_encoder_single_backend_mamba2_multi_backend_transformer_rs_len4_shift2_d_state256
    mkdir -p $exp_dir
    data_dir="/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    rs_len=4
    segment_shift=2
    single_backend_type="mamba2"
    multi_backend_type="transformer"
    d_state=256
    num_transformer_layer=2
    CUDA_VISIABLE_DEVICES=0,1 \
    NCCL_ASYNC_ERROR_HANDLING=1\
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15415 \
   ts_vad2/train_accelerate_ddp2_debug2.py \
    --world-size 2 \
    --num-epochs 40\
    --start-epoch 14\
    --keep-last-k 20\
    --keep-last-epoch 20\
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

if [ ${stage} -le 197 ] && [ ${stop_stage} -ge 197 ];then
 exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_epoch40_front_fix_seed_lr1e4_w2v-bert2_speech_encoder_single_backend_mamba2_multi_backend_transformer_rs_len4_shift2_d_state256
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
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
 rttm_dir=/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 dataset_name="alimeeting" # dataset name
 speech_encoder_type="w2v-bert2"
 speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 speech_encoder_config="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
 # for loading speaker embedding file
 spk_path=/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
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

# grep -r Eval logs/run_ts_vad2_aistation_stage197_lr1e4_infer_a800.log
# Eval, collar=0.0
#2025-08-07 09:44:25,755 (infer2:255) INFO: currently, it will infer Eval set.
#Eval for threshold 0.20: DER 17.62%, MS 2.03%, FA 14.90%, SC 0.70%
#Eval for threshold 0.30: DER 13.79%, MS 3.19%, FA 9.75%, SC 0.84%
#Eval for threshold 0.35: DER 12.73%, MS 3.82%, FA 8.05%, SC 0.87%
#Eval for threshold 0.40: DER 12.13%, MS 4.49%, FA 6.76%, SC 0.88%
#Eval for threshold 0.45: DER 11.77%, MS 5.24%, FA 5.64%, SC 0.88%
#Eval for threshold 0.50: DER 11.65%, MS 6.01%, FA 4.79%, SC 0.86%
#Eval for threshold 0.55: DER 11.63%, MS 6.82%, FA 4.00%, SC 0.81%
#Eval for threshold 0.60: DER 11.82%, MS 7.75%, FA 3.32%, SC 0.76%
#Eval for threshold 0.70: DER 12.91%, MS 10.04%, FA 2.27%, SC 0.60%
#Eval for threshold 0.80: DER 15.24%, MS 13.32%, FA 1.50%, SC 0.41%

# Test , collar=0.0
#Eval for threshold 0.20: DER 17.73%, MS 2.17%, FA 14.80%, SC 0.77%
#Eval for threshold 0.30: DER 13.80%, MS 3.44%, FA 9.41%, SC 0.95%
#Eval for threshold 0.35: DER 12.79%, MS 4.14%, FA 7.63%, SC 1.03%
#Eval for threshold 0.40: DER 12.22%, MS 4.91%, FA 6.23%, SC 1.08%
#Eval for threshold 0.45: DER 11.92%, MS 5.75%, FA 5.11%, SC 1.06%
#Eval for threshold 0.50: DER 11.93%, MS 6.63%, FA 4.24%, SC 1.06%
#Eval for threshold 0.55: DER 12.17%, MS 7.65%, FA 3.49%, SC 1.03%
#Eval for threshold 0.60: DER 12.64%, MS 8.83%, FA 2.87%, SC 0.94%
#Eval for threshold 0.70: DER 14.22%, MS 11.56%, FA 1.90%, SC 0.76%
#Eval for threshold 0.80: DER 17.17%, MS 15.46%, FA 1.16%, SC 0.55%

# Eval, collar=0.25
#2025-08-07 09:51:57,922 (infer2:255) INFO: currently, it will infer Eval set.
#Eval for threshold 0.20: DER 7.35%, MS 0.77%, FA 6.43%, SC 0.16%
#Eval for threshold 0.30: DER 5.09%, MS 1.21%, FA 3.68%, SC 0.20%
#Eval for threshold 0.35: DER 4.54%, MS 1.47%, FA 2.88%, SC 0.19%
#Eval for threshold 0.40: DER 4.25%, MS 1.79%, FA 2.28%, SC 0.19%
#Eval for threshold 0.45: DER 4.08%, MS 2.06%, FA 1.81%, SC 0.20%
#Eval for threshold 0.50: DER 4.09%, MS 2.40%, FA 1.48%, SC 0.21%
#Eval for threshold 0.55: DER 4.19%, MS 2.78%, FA 1.20%, SC 0.21%
#Eval for threshold 0.60: DER 4.31%, MS 3.18%, FA 0.93%, SC 0.19%
#Eval for threshold 0.70: DER 5.09%, MS 4.32%, FA 0.64%, SC 0.13%
#Eval for threshold 0.80: DER 6.66%, MS 6.11%, FA 0.48%, SC 0.07%

# Test, collar=0.25
#Eval for threshold 0.20: DER 8.35%, MS 0.97%, FA 7.10%, SC 0.28%
#Eval for threshold 0.30: DER 5.66%, MS 1.55%, FA 3.74%, SC 0.37%
#Eval for threshold 0.35: DER 5.05%, MS 1.85%, FA 2.78%, SC 0.42%
#Eval for threshold 0.40: DER 4.71%, MS 2.22%, FA 2.04%, SC 0.45%
#Eval for threshold 0.45: DER 4.53%, MS 2.60%, FA 1.49%, SC 0.45%
#Eval for threshold 0.50: DER 4.61%, MS 3.07%, FA 1.10%, SC 0.45%
#Eval for threshold 0.55: DER 4.84%, MS 3.62%, FA 0.80%, SC 0.42%
#Eval for threshold 0.60: DER 5.22%, MS 4.26%, FA 0.59%, SC 0.38%
#Eval for threshold 0.70: DER 6.42%, MS 5.82%, FA 0.33%, SC 0.27%
#Eval for threshold 0.80: DER 8.57%, MS 8.24%, FA 0.17%, SC 0.16%


if [ ${stage} -le 198 ] && [ ${stop_stage} -ge 198 ];then
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
    dataset_name="alimeeting" # dataset name
    speech_encoder_type="w2v-bert2"
    speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
    speech_encoder_config="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
    # for loading speaker embedding file
    spk_path=/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_epoch40_front_fix_seed_lr5e5_w2v-bert2_speech_encoder_single_backend_mamba2_multi_backend_transformer_rs_len4_shift2_d_state256
    mkdir -p $exp_dir
    data_dir="/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    rs_len=4
    segment_shift=2
    single_backend_type="mamba2"
    multi_backend_type="transformer"
    d_state=256
    num_transformer_layer=2
    CUDA_VISIABLE_DEVICES=0,1 \
    NCCL_ASYNC_ERROR_HANDLING=1\
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15515 \
   ts_vad2/train_accelerate_ddp2_debug2.py \
    --world-size 2 \
    --num-epochs 40\
    --start-epoch 14\
    --keep-last-k 20\
    --keep-last-epoch 20\
    --freeze-updates 4000\
    --grad-clip true\
    --lr 5e-5\
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

if [ ${stage} -le 199 ] && [ ${stop_stage} -ge 199 ];then
 exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_epoch40_front_fix_seed_lr5e5_w2v-bert2_speech_encoder_single_backend_mamba2_multi_backend_transformer_rs_len4_shift2_d_state256
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
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
 rttm_dir=/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 dataset_name="alimeeting" # dataset name
 speech_encoder_type="w2v-bert2"
 speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 speech_encoder_config="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
 # for loading speaker embedding file
 spk_path=/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
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
#grep -r Eval logs/run_ts_vad2_aistation_stage199_lr5e5_infer_a800.log
# Eval , collar=0.0
#2025-08-07 10:01:50,620 (infer2:255) INFO: currently, it will infer Eval set.
#Eval for threshold 0.20: DER 15.88%, MS 2.44%, FA 12.65%, SC 0.79%
#Eval for threshold 0.30: DER 12.91%, MS 3.64%, FA 8.36%, SC 0.91%
#Eval for threshold 0.35: DER 12.17%, MS 4.27%, FA 6.98%, SC 0.92%
#Eval for threshold 0.40: DER 11.76%, MS 4.92%, FA 5.92%, SC 0.93%
#Eval for threshold 0.45: DER 11.56%, MS 5.56%, FA 5.06%, SC 0.93%
#Eval for threshold 0.50: DER 11.46%, MS 6.25%, FA 4.28%, SC 0.93%
#Eval for threshold 0.55: DER 11.52%, MS 7.02%, FA 3.62%, SC 0.88%
#Eval for threshold 0.60: DER 11.80%, MS 7.91%, FA 3.11%, SC 0.78%
#Eval for threshold 0.70: DER 12.89%, MS 10.15%, FA 2.13%, SC 0.60%
#Eval for threshold 0.80: DER 15.38%, MS 13.52%, FA 1.41%, SC 0.44%

# Test, collar=0.0
#Eval for threshold 0.20: DER 16.72%, MS 2.49%, FA 13.28%, SC 0.94%
#Eval for threshold 0.30: DER 13.55%, MS 3.78%, FA 8.65%, SC 1.13%
#Eval for threshold 0.35: DER 12.81%, MS 4.49%, FA 7.13%, SC 1.19%
#Eval for threshold 0.40: DER 12.34%, MS 5.21%, FA 5.90%, SC 1.23%
#Eval for threshold 0.45: DER 12.15%, MS 6.02%, FA 4.90%, SC 1.23%
#Eval for threshold 0.50: DER 12.19%, MS 6.91%, FA 4.07%, SC 1.21%
#Eval for threshold 0.55: DER 12.40%, MS 7.90%, FA 3.36%, SC 1.15%
#Eval for threshold 0.60: DER 12.87%, MS 9.02%, FA 2.79%, SC 1.06%
#Eval for threshold 0.70: DER 14.44%, MS 11.71%, FA 1.89%, SC 0.84%
#Eval for threshold 0.80: DER 17.47%, MS 15.70%, FA 1.18%, SC 0.60%
#2025-08-07 10:09:21,930 (infer2:255) INFO: currently, it will infer Eval set.

# Eval, collar=0.25
#Eval for threshold 0.20: DER 6.14%, MS 0.90%, FA 5.05%, SC 0.20%
#Eval for threshold 0.30: DER 4.55%, MS 1.37%, FA 2.93%, SC 0.25%
#Eval for threshold 0.35: DER 4.17%, MS 1.59%, FA 2.32%, SC 0.25%
#Eval for threshold 0.40: DER 4.02%, MS 1.85%, FA 1.89%, SC 0.27%
#Eval for threshold 0.45: DER 3.96%, MS 2.10%, FA 1.58%, SC 0.29% as report
#Eval for threshold 0.50: DER 3.94%, MS 2.40%, FA 1.26%, SC 0.29%
#Eval for threshold 0.55: DER 4.04%, MS 2.73%, FA 1.04%, SC 0.27%
#Eval for threshold 0.60: DER 4.24%, MS 3.14%, FA 0.88%, SC 0.22%
#Eval for threshold 0.70: DER 5.02%, MS 4.26%, FA 0.61%, SC 0.16%
#Eval for threshold 0.80: DER 6.76%, MS 6.21%, FA 0.45%, SC 0.10%

# Test, collar=0.25
#Eval for threshold 0.20: DER 7.68%, MS 1.11%, FA 6.18%, SC 0.38%
#Eval for threshold 0.30: DER 5.58%, MS 1.64%, FA 3.44%, SC 0.50%
#Eval for threshold 0.35: DER 5.16%, MS 1.97%, FA 2.63%, SC 0.56%
#Eval for threshold 0.40: DER 4.85%, MS 2.29%, FA 1.98%, SC 0.58% 
#Eval for threshold 0.45: DER 4.76%, MS 2.65%, FA 1.48%, SC 0.62% as report
#Eval for threshold 0.50: DER 4.81%, MS 3.13%, FA 1.06%, SC 0.63%
#Eval for threshold 0.55: DER 5.04%, MS 3.67%, FA 0.76%, SC 0.60%
#Eval for threshold 0.60: DER 5.42%, MS 4.32%, FA 0.57%, SC 0.53%
#Eval for threshold 0.70: DER 6.61%, MS 5.91%, FA 0.33%, SC 0.37%
#Eval for threshold 0.80: DER 8.84%, MS 8.44%, FA 0.19%, SC 0.21%


if [ ${stage} -le 200 ] && [ ${stop_stage} -ge 200 ];then
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
    dataset_name="alimeeting" # dataset name
    speech_encoder_type="w2v-bert2"
    speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
    speech_encoder_config="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
    # for loading speaker embedding file
    spk_path=/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_epoch40_front_fix_seed_lr1e5_w2v-bert2_speech_encoder_single_backend_mamba2_multi_backend_transformer_rs_len4_shift2_d_state256
    mkdir -p $exp_dir
    data_dir="/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    rs_len=4
    segment_shift=2
    single_backend_type="mamba2"
    multi_backend_type="transformer"
    d_state=256
    num_transformer_layer=2
    CUDA_VISIABLE_DEVICES=0,1 \
    NCCL_ASYNC_ERROR_HANDLING=1\
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15615 \
   ts_vad2/train_accelerate_ddp2_debug2.py \
    --world-size 2 \
    --num-epochs 40\
    --start-epoch 14\
    --keep-last-k 20\
    --keep-last-epoch 20\
    --freeze-updates 4000\
    --grad-clip true\
    --lr 1e-5\
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

if [ ${stage} -le 201 ] && [ ${stop_stage} -ge 201 ];then
 exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_epoch40_front_fix_seed_lr1e5_w2v-bert2_speech_encoder_single_backend_mamba2_multi_backend_transformer_rs_len4_shift2_d_state256
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
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
 rttm_dir=/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 dataset_name="alimeeting" # dataset name
 speech_encoder_type="w2v-bert2"
 speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 speech_encoder_config="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
 # for loading speaker embedding file
 spk_path=/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
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

# grep -r Eval logs/run_ts_vad2_aistation_stage201_lr1e5_infer_a800.log
# Eval , collar=0.0
#2025-08-07 10:32:19,628 (infer2:255) INFO: currently, it will infer Eval set.
#Eval for threshold 0.20: DER 21.82%, MS 1.87%, FA 18.75%, SC 1.21%
#Eval for threshold 0.30: DER 16.44%, MS 3.19%, FA 11.90%, SC 1.35%
#Eval for threshold 0.35: DER 14.96%, MS 3.95%, FA 9.59%, SC 1.43%
#Eval for threshold 0.40: DER 14.02%, MS 4.86%, FA 7.71%, SC 1.45%
#Eval for threshold 0.45: DER 13.48%, MS 5.89%, FA 6.11%, SC 1.48%
#Eval for threshold 0.50: DER 13.34%, MS 7.12%, FA 4.87%, SC 1.35%
#Eval for threshold 0.55: DER 13.53%, MS 8.37%, FA 3.96%, SC 1.20%
#Eval for threshold 0.60: DER 14.06%, MS 9.78%, FA 3.18%, SC 1.10%
#Eval for threshold 0.70: DER 16.02%, MS 13.19%, FA 2.01%, SC 0.82%
#Eval for threshold 0.80: DER 19.61%, MS 17.91%, FA 1.24%, SC 0.46%

# Test, collar=0.0
#Eval for threshold 0.20: DER 22.72%, MS 2.03%, FA 19.56%, SC 1.14%
#Eval for threshold 0.30: DER 17.64%, MS 3.50%, FA 12.64%, SC 1.50%
#Eval for threshold 0.35: DER 16.18%, MS 4.37%, FA 10.14%, SC 1.67%
#Eval for threshold 0.40: DER 15.25%, MS 5.33%, FA 8.12%, SC 1.80%
#Eval for threshold 0.45: DER 14.81%, MS 6.47%, FA 6.46%, SC 1.88%
#Eval for threshold 0.50: DER 14.71%, MS 7.77%, FA 5.06%, SC 1.88%
#Eval for threshold 0.55: DER 14.99%, MS 9.25%, FA 3.92%, SC 1.83%
#Eval for threshold 0.60: DER 15.63%, MS 10.98%, FA 2.95%, SC 1.70%
#Eval for threshold 0.70: DER 18.26%, MS 15.30%, FA 1.68%, SC 1.28%
#Eval for threshold 0.80: DER 22.91%, MS 21.22%, FA 0.94%, SC 0.75%
#2025-08-07 10:39:50,563 (infer2:255) INFO: currently, it will infer Eval set.
#
# Eval , collar=0.25
#Eval for threshold 0.20: DER 10.30%, MS 0.68%, FA 9.18%, SC 0.44%
#Eval for threshold 0.30: DER 6.89%, MS 1.14%, FA 5.25%, SC 0.50%
#Eval for threshold 0.35: DER 6.05%, MS 1.50%, FA 4.03%, SC 0.52%
#Eval for threshold 0.40: DER 5.49%, MS 1.90%, FA 3.03%, SC 0.55%
#Eval for threshold 0.45: DER 5.24%, MS 2.42%, FA 2.23%, SC 0.59% 
#Eval for threshold 0.50: DER 5.28%, MS 3.06%, FA 1.68%, SC 0.54% as report
#Eval for threshold 0.55: DER 5.44%, MS 3.65%, FA 1.30%, SC 0.48%
#Eval for threshold 0.60: DER 5.89%, MS 4.46%, FA 0.97%, SC 0.46%
#Eval for threshold 0.70: DER 7.46%, MS 6.54%, FA 0.57%, SC 0.34%
#Eval for threshold 0.80: DER 10.23%, MS 9.66%, FA 0.40%, SC 0.17%

# Test, collar=0.25
#Eval for threshold 0.20: DER 13.22%, MS 0.85%, FA 11.85%, SC 0.52%
#Eval for threshold 0.30: DER 9.53%, MS 1.55%, FA 7.19%, SC 0.79%
#Eval for threshold 0.35: DER 8.49%, MS 1.95%, FA 5.55%, SC 0.98%
#Eval for threshold 0.40: DER 7.77%, MS 2.43%, FA 4.21%, SC 1.13%
#Eval for threshold 0.45: DER 7.46%, MS 3.07%, FA 3.15%, SC 1.24%  
#Eval for threshold 0.50: DER 7.36%, MS 3.78%, FA 2.24%, SC 1.33% as report
#Eval for threshold 0.55: DER 7.56%, MS 4.70%, FA 1.51%, SC 1.34%
#Eval for threshold 0.60: DER 8.05%, MS 5.84%, FA 0.93%, SC 1.29%
#Eval for threshold 0.70: DER 10.14%, MS 8.87%, FA 0.31%, SC 0.96%
#Eval for threshold 0.80: DER 13.97%, MS 13.35%, FA 0.14%, SC 0.48%


if [ ${stage} -le 202 ] && [ ${stop_stage} -ge 202 ];then
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
    dataset_name="alimeeting" # dataset name
    speech_encoder_type="w2v-bert2"
    speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
    speech_encoder_config="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
    # for loading speaker embedding file
    spk_path=/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_epoch40_front_fix_seed_lr5e6_w2v-bert2_speech_encoder_single_backend_mamba2_multi_backend_transformer_rs_len4_shift2_d_state256
    mkdir -p $exp_dir
    data_dir="/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    rs_len=4
    segment_shift=2
    single_backend_type="mamba2"
    multi_backend_type="transformer"
    d_state=256
    num_transformer_layer=2
    CUDA_VISIABLE_DEVICES=0,1 \
    NCCL_ASYNC_ERROR_HANDLING=1\
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15715 \
   ts_vad2/train_accelerate_ddp2_debug2.py \
    --world-size 2 \
    --num-epochs 40\
    --start-epoch 14\
    --keep-last-k 20\
    --keep-last-epoch 20\
    --freeze-updates 4000\
    --grad-clip true\
    --lr 5e-6\
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

if [ ${stage} -le 203 ] && [ ${stop_stage} -ge 203 ];then
 exp_dir=/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_feature_dir_epoch40_front_fix_seed_lr5e6_w2v-bert2_speech_encoder_single_backend_mamba2_multi_backend_transformer_rs_len4_shift2_d_state256
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
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
 rttm_dir=/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 dataset_name="alimeeting" # dataset name
 speech_encoder_type="w2v-bert2"
 speech_encoder_path="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 speech_encoder_config="/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
 # for loading speaker embedding file
 spk_path=/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
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

#grep -r Eval logs/run_ts_vad2_aistation_stage203_lr5e6_infer_a800.log
#2025-08-07 09:17:28,171 (infer2:255) INFO: currently, it will infer Eval set.
#Eval for threshold 0.20: DER 27.00%, MS 2.16%, FA 23.24%, SC 1.59%
#Eval for threshold 0.30: DER 20.68%, MS 3.72%, FA 15.02%, SC 1.94%
#Eval for threshold 0.35: DER 18.91%, MS 4.64%, FA 12.19%, SC 2.08%
#Eval for threshold 0.40: DER 17.70%, MS 5.71%, FA 9.81%, SC 2.18%
#Eval for threshold 0.45: DER 17.04%, MS 6.93%, FA 7.83%, SC 2.28%
#Eval for threshold 0.50: DER 16.82%, MS 8.36%, FA 6.18%, SC 2.28%
#Eval for threshold 0.55: DER 16.94%, MS 9.91%, FA 4.84%, SC 2.19%
#Eval for threshold 0.60: DER 17.60%, MS 11.82%, FA 3.79%, SC 1.99%
#Eval for threshold 0.70: DER 20.20%, MS 16.51%, FA 2.21%, SC 1.48%
#Eval for threshold 0.80: DER 25.39%, MS 23.29%, FA 1.27%, SC 0.83%
#Eval for threshold 0.20: DER 28.35%, MS 2.32%, FA 24.47%, SC 1.56%
#Eval for threshold 0.30: DER 22.73%, MS 3.76%, FA 16.75%, SC 2.21%
#Eval for threshold 0.35: DER 20.88%, MS 4.69%, FA 13.73%, SC 2.46%
#Eval for threshold 0.40: DER 19.51%, MS 5.69%, FA 11.12%, SC 2.70%
#Eval for threshold 0.45: DER 18.60%, MS 6.84%, FA 8.90%, SC 2.86%
#Eval for threshold 0.50: DER 18.03%, MS 8.18%, FA 6.91%, SC 2.94%
#Eval for threshold 0.55: DER 17.96%, MS 9.85%, FA 5.18%, SC 2.92%
#Eval for threshold 0.60: DER 18.46%, MS 11.86%, FA 3.85%, SC 2.75%
#Eval for threshold 0.70: DER 21.07%, MS 16.93%, FA 2.04%, SC 2.10%
#Eval for threshold 0.80: DER 26.38%, MS 24.28%, FA 0.97%, SC 1.14%
#2025-08-07 09:26:53,765 (infer2:255) INFO: currently, it will infer Eval set.
#Eval for threshold 0.20: DER 15.77%, MS 0.91%, FA 14.19%, SC 0.67%
#Eval for threshold 0.30: DER 11.00%, MS 1.65%, FA 8.47%, SC 0.88%
#Eval for threshold 0.35: DER 9.71%, MS 2.10%, FA 6.65%, SC 0.96%
#Eval for threshold 0.40: DER 8.90%, MS 2.71%, FA 5.15%, SC 1.03%
#Eval for threshold 0.45: DER 8.47%, MS 3.36%, FA 3.93%, SC 1.18%
#Eval for threshold 0.50: DER 8.34%, MS 4.20%, FA 2.92%, SC 1.22% as report
#Eval for threshold 0.55: DER 8.52%, MS 5.19%, FA 2.12%, SC 1.22%
#Eval for threshold 0.60: DER 9.16%, MS 6.54%, FA 1.53%, SC 1.10%
#Eval for threshold 0.70: DER 11.36%, MS 9.84%, FA 0.76%, SC 0.76%
#Eval for threshold 0.80: DER 15.73%, MS 14.91%, FA 0.42%, SC 0.40%
#Eval for threshold 0.20: DER 19.68%, MS 1.07%, FA 17.92%, SC 0.70%
#Eval for threshold 0.30: DER 14.87%, MS 1.78%, FA 11.88%, SC 1.21%
#Eval for threshold 0.35: DER 13.26%, MS 2.26%, FA 9.52%, SC 1.48%
#Eval for threshold 0.40: DER 12.01%, MS 2.79%, FA 7.44%, SC 1.78%
#Eval for threshold 0.45: DER 11.12%, MS 3.46%, FA 5.60%, SC 2.06% 
#Eval for threshold 0.50: DER 10.50%, MS 4.24%, FA 4.01%, SC 2.24% as report
#Eval for threshold 0.55: DER 10.35%, MS 5.39%, FA 2.66%, SC 2.30%
#Eval for threshold 0.60: DER 10.67%, MS 6.78%, FA 1.69%, SC 2.20%
#Eval for threshold 0.70: DER 12.88%, MS 10.63%, FA 0.64%, SC 1.61%
#Eval for threshold 0.80: DER 17.39%, MS 16.48%, FA 0.19%, SC 0.72%
