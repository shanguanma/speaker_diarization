#!/usr/bin/env bash

stage=0
stop_stage=1000

. utils/parse_options.sh
#. path_for_speaker_diarization.sh
. path_for_dia_pt2.4.sh

## the note is from run_ddp_phase1.sh
# it doesn't add noise and rirs to train tsvad model, no grad-clip and no freeze update.
# speech encoder is cam++ , oracle target speaker embedding is from cam++ pretrain model
# this cam++ pretrain model is trained on cn-cnceleb and voxceleb dataset, checkpoint is from https://www.modelscope.cn/models/iic/speech_campplus_sv_zh_en_16k-common_advanced/files
## if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1
# exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad_ddp_phase1/exp
# CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12683 ts_vad_ddp_phase1/train_accelerate_ddp.py\
#    --world-size 2 \
#    --num-epochs 20\
#    --start-epoch 1\
#    --freeze-updates 0\
#    --master-port 12683\
#    --grad-clip false\
#    --exp-dir $exp_dir
#fi
#
#if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ];then
# exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad_ddp_phase1/exp
# model_file=$exp_dir/best-valid-der.pt
# rs_len=4
# segment_shift=1
# label_rate=25
# min_silence=0.32
# min_speech=0.0
# #infer_sets="Eval Test"
# infer_sets="Eval"
# rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
# sctk_tool_path="./SCTK-2.4.12"
# collar=0.25
# results_path=$exp_dir/
# for name in $infer_sets;do
#    results_path=$exp_dir/$name
#    mkdir -p $results_path
#
#  python3 ts_vad_ddp_phase1/infer.py \
#    --model-file $model_file\
#    --rs-len $rs_len\
#    --segment-shift $segment_shift\
#    --label-rate $label_rate\
#    --min-speech $min_speech\
#    --min-silence $min_silence\
#    --rttm-name alimeeting_${name}.rttm\
#    --rttm-dir $rttm_dir\
#    --sctk-tool-path $sctk_tool_path \
#    --collar $collar\
#    --results-path $results_path \
#    --split $name
#done
## Eval set
##model_file=$exp_dir/best-valid-der.pt
## cat logs/run_ddp_phase1_stage10_4.log
##Model DER:  0.13087824463709546
##Model ACC:  0.9547311285294958
##frame_len: 0.04!!
##100%|██████████| 25/25 [00:23<00:00,  1.06it/s]
##Eval for threshold 0.20: DER 7.06%, MS 1.10%, FA 5.57%, SC 0.39%
##
##Eval for threshold 0.30: DER 5.47%, MS 1.53%, FA 3.51%, SC 0.43%
##
##Eval for threshold 0.35: DER 5.10%, MS 1.79%, FA 2.85%, SC 0.46%
##
##Eval for threshold 0.40: DER 4.90%, MS 2.07%, FA 2.34%, SC 0.49%
##
##Eval for threshold 0.45: DER 4.78%, MS 2.39%, FA 1.91%, SC 0.47% as report
##
##Eval for threshold 0.50: DER 4.82%, MS 2.79%, FA 1.56%, SC 0.47%
##
##Eval for threshold 0.55: DER 4.95%, MS 3.22%, FA 1.28%, SC 0.45%
##
##Eval for threshold 0.60: DER 5.24%, MS 3.77%, FA 1.08%, SC 0.39%
##
##Eval for threshold 0.70: DER 6.10%, MS 5.00%, FA 0.75%, SC 0.34%
##
##Eval for threshold 0.80: DER 8.14%, MS 7.39%, FA 0.52%, SC 0.22%
#
### Test set
## model_file=$exp_dir/best-valid-der.pt
## cat logs/run_ddp_phase1_stage11.log
##Model DER:  0.13085100596695015
##Model ACC:  0.9506723245350728
##frame_len: 0.04!!
##100%|██████████| 60/60 [00:58<00:00,  1.02it/s]
##Eval for threshold 0.20: DER 9.58%, MS 1.09%, FA 7.58%, SC 0.91%
##
##Eval for threshold 0.30: DER 7.47%, MS 1.60%, FA 4.73%, SC 1.14%
##
##Eval for threshold 0.35: DER 6.93%, MS 1.87%, FA 3.82%, SC 1.23%
##
##Eval for threshold 0.40: DER 6.60%, MS 2.21%, FA 3.11%, SC 1.28%
##
##Eval for threshold 0.45: DER 6.38%, MS 2.58%, FA 2.47%, SC 1.32% as report
##
##Eval for threshold 0.50: DER 6.37%, MS 3.04%, FA 2.03%, SC 1.30%
##
##Eval for threshold 0.55: DER 6.48%, MS 3.57%, FA 1.66%, SC 1.24%
##
##Eval for threshold 0.60: DER 6.66%, MS 4.15%, FA 1.36%, SC 1.15%
##
##Eval for threshold 0.70: DER 7.54%, MS 5.71%, FA 0.88%, SC 0.95%
##
##Eval for threshold 0.80: DER 9.27%, MS 8.03%, FA 0.51%, SC 0.73%
#fi
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
   # it adds noise to train tsvad model , no grad-clip and no freeze update.
   # # speech encoder is cam++ , oracle target speaker embedding is from cam++ pretrain model
# this cam++ pretrain model is trained on cn-cnceleb and voxceleb dataset, checkpoint is from https://www.modelscope.cn/models/iic/speech_campplus_sv_zh_en_16k-common_advanced/files
   export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_unfreeze_with_musan
   CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --freeze-updates 0\
    --grad-clip false\
    --musan-path $musan_path \
    --master-port 12673\
    --exp-dir $exp_dir
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
    # it adds noise and rirs to train tsvad model , no grad-clip and no freeze update.
    # # speech encoder is cam++ , oracle target speaker embedding is from cam++ pretrain model
# this cam++ pretrain model is trained on cn-cnceleb and voxceleb dataset, checkpoint is from https://www.modelscope.cn/models/iic/speech_campplus_sv_zh_en_16k-common_advanced/files
   export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_unfreeze_with_musan_rirs
   CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12683 ts_vad2/train_accelerate_ddp.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --freeze-updates 0\
    --grad-clip false\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --master-port 12683\
    --exp-dir $exp_dir
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_unfreeze_with_musan
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 results_path=$exp_dir/
 for name in $infer_sets;do
    results_path=$exp_dir/$name
    mkdir -p $results_path

  python3 ts_vad2/infer.py \
    --model-file $model_file\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name alimeeting_${name}.rttm\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $collar\
    --results-path $results_path \
    --split $name
done
# cat logs/run_ts_vad2_stage3.log
# Eval set
# Model DER:  0.1241234652367966
#Model ACC:  0.9577892013147111
#frame_len: 0.04!!
#100%|██████████| 25/25 [00:23<00:00,  1.05it/s]
#Eval for threshold 0.20: DER 7.32%, MS 1.05%, FA 5.99%, SC 0.29%
#
#Eval for threshold 0.30: DER 5.47%, MS 1.55%, FA 3.61%, SC 0.31%
#
#Eval for threshold 0.35: DER 4.98%, MS 1.80%, FA 2.84%, SC 0.34%
#
#Eval for threshold 0.40: DER 4.76%, MS 2.09%, FA 2.32%, SC 0.35%
#
#Eval for threshold 0.45: DER 4.62%, MS 2.38%, FA 1.90%, SC 0.34%
#
#Eval for threshold 0.50: DER 4.58%, MS 2.75%, FA 1.51%, SC 0.32%
#
#Eval for threshold 0.55: DER 4.69%, MS 3.14%, FA 1.26%, SC 0.29%
#
#Eval for threshold 0.60: DER 4.88%, MS 3.63%, FA 0.99%, SC 0.26%
#
#Eval for threshold 0.70: DER 5.82%, MS 4.95%, FA 0.67%, SC 0.20%
#
#Eval for threshold 0.80: DER 7.53%, MS 6.95%, FA 0.46%, SC 0.12%
# Test set
#Model DER:  0.1236898679053932
#Model ACC:  0.954454092760533
#frame_len: 0.04!!
#100%|██████████| 60/60 [00:59<00:00,  1.02it/s]
#Eval for threshold 0.20: DER 8.94%, MS 1.16%, FA 7.25%, SC 0.52%
#
#Eval for threshold 0.30: DER 6.56%, MS 1.74%, FA 4.18%, SC 0.65%
#
#Eval for threshold 0.35: DER 6.04%, MS 2.06%, FA 3.29%, SC 0.68%
#
#Eval for threshold 0.40: DER 5.71%, MS 2.41%, FA 2.58%, SC 0.72%
#
#Eval for threshold 0.45: DER 5.47%, MS 2.78%, FA 1.98%, SC 0.71%
#
#Eval for threshold 0.50: DER 5.44%, MS 3.21%, FA 1.54%, SC 0.68%
#
#Eval for threshold 0.55: DER 5.59%, MS 3.76%, FA 1.19%, SC 0.63%
#
#Eval for threshold 0.60: DER 5.83%, MS 4.38%, FA 0.87%, SC 0.58%
#
#Eval for threshold 0.70: DER 6.83%, MS 5.88%, FA 0.48%, SC 0.46%
#
#Eval for threshold 0.80: DER 8.76%, MS 8.21%, FA 0.24%, SC 0.31%

fi
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_unfreeze_with_musan_rirs
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 results_path=$exp_dir/
 for name in $infer_sets;do
    results_path=$exp_dir/$name
    mkdir -p $results_path

  python3 ts_vad2/infer.py \
    --model-file $model_file\
    --rs-len $rs_len\
    --segment-shift $segment_shift\
    --label-rate $label_rate\
    --min-speech $min_speech\
    --min-silence $min_silence\
    --rttm-name alimeeting_${name}.rttm\
    --rttm-dir $rttm_dir\
    --sctk-tool-path $sctk_tool_path \
    --collar $collar\
    --results-path $results_path \
    --split $name
done
#  cat logs/run_ts_vad2_stage4.log
# Eval set
# Model DER:  0.12873960218721095
#Model ACC:  0.9565194062289926
#frame_len: 0.04!!
#100%|██████████| 25/25 [00:23<00:00,  1.05it/s]
#Eval for threshold 0.20: DER 7.89%, MS 1.05%, FA 6.59%, SC 0.25%
#
#Eval for threshold 0.30: DER 5.78%, MS 1.63%, FA 3.85%, SC 0.31%
#
#Eval for threshold 0.35: DER 5.18%, MS 1.89%, FA 2.96%, SC 0.33%
#
#Eval for threshold 0.40: DER 4.86%, MS 2.18%, FA 2.36%, SC 0.33%
#
#Eval for threshold 0.45: DER 4.70%, MS 2.52%, FA 1.87%, SC 0.31% # as report
#
#Eval for threshold 0.50: DER 4.70%, MS 2.93%, FA 1.50%, SC 0.27%
#
#Eval for threshold 0.55: DER 4.90%, MS 3.46%, FA 1.19%, SC 0.26%
#
#Eval for threshold 0.60: DER 5.19%, MS 3.99%, FA 0.97%, SC 0.23%
#
#Eval for threshold 0.70: DER 6.21%, MS 5.43%, FA 0.65%, SC 0.13%
#
#Eval for threshold 0.80: DER 8.01%, MS 7.47%, FA 0.45%, SC 0.09%


# Test set
#Model DER:  0.12125134137046271
#Model ACC:  0.955173911927526
#frame_len: 0.04!!
#100%|██████████| 60/60 [00:59<00:00,  1.01it/s]
#Eval for threshold 0.20: DER 8.02%, MS 1.16%, FA 6.36%, SC 0.50%
#
#Eval for threshold 0.30: DER 6.00%, MS 1.78%, FA 3.59%, SC 0.62%
#
#Eval for threshold 0.35: DER 5.54%, MS 2.14%, FA 2.76%, SC 0.65%
#
#Eval for threshold 0.40: DER 5.29%, MS 2.52%, FA 2.09%, SC 0.68%
#
#Eval for threshold 0.45: DER 5.19%, MS 2.93%, FA 1.56%, SC 0.70% # as report
#
#Eval for threshold 0.50: DER 5.29%, MS 3.44%, FA 1.17%, SC 0.68%
#
#Eval for threshold 0.55: DER 5.55%, MS 4.03%, FA 0.89%, SC 0.63%
#
#Eval for threshold 0.60: DER 5.90%, MS 4.66%, FA 0.67%, SC 0.57%
#
#Eval for threshold 0.70: DER 7.13%, MS 6.32%, FA 0.37%, SC 0.44%
#
#Eval for threshold 0.80: DER 9.22%, MS 8.73%, FA 0.19%, SC 0.30%
fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ];then
    # # it adds noise  to train tsvad model , no grad-clip and no freeze update.
    # # speech encoder is WavLM (only using cnn frontend and first 6 layers transformer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this WavLM is trained 60khours libri-light and 10k hours gigaspeech and 24k hours VoxPopuli,
    # checkpoint is from https://github.com/microsoft/unilm/tree/master/wavlm
   export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    speech_encoder_type="WavLM"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Base+.pt"
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_unfreeze_with_musan_wavlm
   CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --freeze-updates 0\
    --grad-clip false\
    --musan-path $musan_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --master-port 12673\
    --exp-dir $exp_dir
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ];then
    # # it adds noise and rirs to train tsvad model , no grad-clip and no freeze update.
    # # speech encoder is WavLM (only using cnn frontend and first 6 layers transformer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this WavLM is trained 60khours libri-light and 10k hours gigaspeech and 24k hours VoxPopuli,
    # checkpoint is from https://github.com/microsoft/unilm/tree/master/wavlm
   export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    speech_encoder_type="WavLM"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Base+.pt"
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_unfreeze_with_musan_rirs_wavlm
   CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12683 ts_vad2/train_accelerate_ddp2.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --freeze-updates 0\
    --grad-clip false\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --master-port 12683\
    --exp-dir $exp_dir
fi



if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_unfreeze_with_musan_wavlm
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 results_path=$exp_dir/
  # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="WavLM"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Base+.pt"
 for name in $infer_sets;do
    results_path=$exp_dir/$name
    mkdir -p $results_path

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
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path
done
# cat logs/run_ts_vad2_stage7.log
# Eval set
# Model DER:  0.15291383123550295
#Model ACC:  0.9464658629313634
#frame_len: 0.04!!
#100%|██████████| 25/25 [00:24<00:00,  1.03it/s]
#Eval for threshold 0.20: DER 8.74%, MS 1.52%, FA 6.43%, SC 0.78%
#
#Eval for threshold 0.30: DER 6.94%, MS 2.35%, FA 3.66%, SC 0.93%
#
#Eval for threshold 0.35: DER 6.54%, MS 2.82%, FA 2.69%, SC 1.03%
#
#Eval for threshold 0.40: DER 6.42%, MS 3.30%, FA 2.04%, SC 1.08%
#
#Eval for threshold 0.45: DER 6.51%, MS 3.86%, FA 1.54%, SC 1.12%
#
#Eval for threshold 0.50: DER 6.79%, MS 4.54%, FA 1.21%, SC 1.03%
#
#Eval for threshold 0.55: DER 7.21%, MS 5.33%, FA 0.99%, SC 0.90%
#
#Eval for threshold 0.60: DER 7.82%, MS 6.25%, FA 0.81%, SC 0.77%
#
#Eval for threshold 0.70: DER 9.80%, MS 8.72%, FA 0.56%, SC 0.52%
#
#Eval for threshold 0.80: DER 12.96%, MS 12.25%, FA 0.41%, SC 0.30%
# Test set
# Model DER:  0.15031020587596527
#Model ACC:  0.9441963726764806
#frame_len: 0.04!!
#100%|██████████| 60/60 [00:59<00:00,  1.01it/s]
#Eval for threshold 0.20: DER 11.06%, MS 1.49%, FA 8.36%, SC 1.21%
#
#Eval for threshold 0.30: DER 8.51%, MS 2.35%, FA 4.66%, SC 1.51%
#
#Eval for threshold 0.35: DER 7.95%, MS 2.84%, FA 3.45%, SC 1.66%
#
#Eval for threshold 0.40: DER 7.62%, MS 3.34%, FA 2.53%, SC 1.76%
#
#Eval for threshold 0.45: DER 7.53%, MS 3.95%, FA 1.76%, SC 1.82%
#
#Eval for threshold 0.50: DER 7.83%, MS 4.79%, FA 1.29%, SC 1.75%
#
#Eval for threshold 0.55: DER 8.39%, MS 5.84%, FA 0.99%, SC 1.56%
#
#Eval for threshold 0.60: DER 9.09%, MS 6.95%, FA 0.75%, SC 1.39%
#
#Eval for threshold 0.70: DER 11.03%, MS 9.56%, FA 0.42%, SC 1.05%
#
#Eval for threshold 0.80: DER 14.12%, MS 13.25%, FA 0.22%, SC 0.66%


fi
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_unfreeze_with_musan_rirs_wavlm
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 results_path=$exp_dir/
  # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="WavLM"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Base+.pt"
 for name in $infer_sets;do
    results_path=$exp_dir/$name
    mkdir -p $results_path

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
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path
done
#Eval set
# Model DER:  0.13855229689212145
#Model ACC:  0.9512012888010657
#frame_len: 0.04!!
#100%|██████████| 25/25 [00:24<00:00,  1.03it/s]
#Eval for threshold 0.20: DER 9.61%, MS 1.17%, FA 7.96%, SC 0.47%
#
#Eval for threshold 0.30: DER 6.98%, MS 1.80%, FA 4.53%, SC 0.64%
#
#Eval for threshold 0.35: DER 6.26%, MS 2.14%, FA 3.41%, SC 0.71%
#
#Eval for threshold 0.40: DER 5.84%, MS 2.50%, FA 2.53%, SC 0.80%
#
#Eval for threshold 0.45: DER 5.75%, MS 3.02%, FA 1.94%, SC 0.79%
#
#Eval for threshold 0.50: DER 5.86%, MS 3.68%, FA 1.48%, SC 0.69%
#
#Eval for threshold 0.55: DER 6.19%, MS 4.41%, FA 1.18%, SC 0.60%
#
#Eval for threshold 0.60: DER 6.69%, MS 5.20%, FA 0.97%, SC 0.52%
#
#Eval for threshold 0.70: DER 8.15%, MS 7.15%, FA 0.64%, SC 0.36%
#
#Eval for threshold 0.80: DER 10.84%, MS 10.16%, FA 0.44%, SC 0.24%
# Test set
#Model DER:  0.13810391543172545
#Model ACC:  0.9478695106461118
#frame_len: 0.04!!
#100%|██████████| 60/60 [00:59<00:00,  1.02it/s]
#Eval for threshold 0.20: DER 11.29%, MS 1.14%, FA 9.17%, SC 0.98%
#
#Eval for threshold 0.30: DER 8.37%, MS 1.82%, FA 5.20%, SC 1.35%
#
#Eval for threshold 0.35: DER 7.64%, MS 2.21%, FA 3.92%, SC 1.51%
#
#Eval for threshold 0.40: DER 7.19%, MS 2.65%, FA 2.90%, SC 1.64%
#
#Eval for threshold 0.45: DER 7.01%, MS 3.21%, FA 2.12%, SC 1.69%
#
#Eval for threshold 0.50: DER 7.12%, MS 3.93%, FA 1.55%, SC 1.64%
#
#Eval for threshold 0.55: DER 7.43%, MS 4.80%, FA 1.12%, SC 1.51%
#
#Eval for threshold 0.60: DER 7.98%, MS 5.78%, FA 0.85%, SC 1.35%
#
#Eval for threshold 0.70: DER 9.62%, MS 8.16%, FA 0.44%, SC 1.01%
#
#Eval for threshold 0.80: DER 12.28%, MS 11.38%, FA 0.22%, SC 0.68%



fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ];then
     # # it desen't adds noise and rirs to train tsvad model , no grad-clip and no freeze update.
    # # speech encoder is WavLM (only using cnn frontend and first 6 layers transformer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this WavLM is trained 60khours libri-light and 10k hours gigaspeech and 24k hours VoxPopuli,
    # checkpoint is from https://github.com/microsoft/unilm/tree/master/wavlm
   export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    speech_encoder_type="WavLM"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Base+.pt"
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_unfreeze_without_musan_rirs_wavlm
   CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12683 ts_vad2/train_accelerate_ddp2.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --freeze-updates 0\
    --grad-clip false\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --master-port 12683\
    --exp-dir $exp_dir
fi

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_unfreeze_without_musan_rirs_wavlm
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 results_path=$exp_dir/
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="WavLM"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Base+.pt"
 for name in $infer_sets;do
    results_path=$exp_dir/$name
    mkdir -p $results_path

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
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path
done
# cat logs/run_ts_vad2_stage10.log
# Eval Set
#Model DER:  0.14345025649794238
#Model ACC:  0.9482676142404353
#100%|██████████| 25/25 [00:23<00:00,  1.06it/s]
#Eval for threshold 0.20: DER 8.26%, MS 1.46%, FA 5.97%, SC 0.83%
#
#Eval for threshold 0.30: DER 6.68%, MS 2.07%, FA 3.68%, SC 0.93%
#
#Eval for threshold 0.35: DER 6.36%, MS 2.41%, FA 2.98%, SC 0.97%
#
#Eval for threshold 0.40: DER 6.18%, MS 2.78%, FA 2.41%, SC 0.99%
#
#Eval for threshold 0.45: DER 6.11%, MS 3.16%, FA 1.90%, SC 1.05%
#
#Eval for threshold 0.50: DER 6.17%, MS 3.69%, FA 1.46%, SC 1.02%
#
#Eval for threshold 0.55: DER 6.45%, MS 4.33%, FA 1.18%, SC 0.95%
#
#Eval for threshold 0.60: DER 6.90%, MS 5.00%, FA 1.00%, SC 0.90%
#
#Eval for threshold 0.70: DER 8.23%, MS 6.83%, FA 0.72%, SC 0.68%
#
#Eval for threshold 0.80: DER 10.65%, MS 9.75%, FA 0.48%, SC 0.43%

# Test Set
#Model DER:  0.157231605835709
#Model ACC:  0.9380840895002966
#100%|██████████| 60/60 [00:58<00:00,  1.02it/s]
#Eval for threshold 0.20: DER 12.98%, MS 1.43%, FA 9.57%, SC 1.98%
#
#Eval for threshold 0.30: DER 10.83%, MS 2.06%, FA 6.19%, SC 2.57%
#
#Eval for threshold 0.35: DER 10.20%, MS 2.38%, FA 4.98%, SC 2.83%
#
#Eval for threshold 0.40: DER 9.78%, MS 2.75%, FA 3.97%, SC 3.06%
#
#Eval for threshold 0.45: DER 9.51%, MS 3.21%, FA 3.11%, SC 3.19%
#
#Eval for threshold 0.50: DER 9.45%, MS 3.80%, FA 2.41%, SC 3.24%
#
#Eval for threshold 0.55: DER 9.65%, MS 4.60%, FA 1.96%, SC 3.09%
#
#Eval for threshold 0.60: DER 9.97%, MS 5.49%, FA 1.60%, SC 2.88%
#
#Eval for threshold 0.70: DER 11.19%, MS 7.79%, FA 1.00%, SC 2.40%
#
#Eval for threshold 0.80: DER 13.51%, MS 11.13%, FA 0.61%, SC 1.77%
fi


if [ ${stage} -le 20 ] && [ ${stop_stage} -ge 20 ];then

    # # it adds noise and rirs to train tsvad model ,grad-clip and freeze speech encoder 4000 update.
    # # speech encoder is WavLM (only using cnn frontend and first 6 layers transformer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this WavLM is trained 60khours libri-light and 10k hours gigaspeech and 24k hours VoxPopuli,
    # checkpoint is from https://github.com/microsoft/unilm/tree/master/wavlm
   export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    speech_encoder_type="WavLM"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Base+.pt"
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wavlm
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12683 ts_vad2/train_accelerate_ddp2.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --freeze-updates 4000\
    --grad-clip true\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --master-port 12683\
    --exp-dir $exp_dir
fi

if [ ${stage} -le 21 ] && [ ${stop_stage} -ge 21 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wavlm
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 results_path=$exp_dir/
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="WavLM"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Base+.pt"
 for name in $infer_sets;do
    results_path=$exp_dir/$name
    mkdir -p $results_path
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
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path
done
# Eval
#Model DER:  0.13427561017705303
#Model ACC:  0.9533562678994314
#frame_len: 0.04!!
#100%|██████████| 25/25 [00:23<00:00,  1.05it/s]
#Eval for threshold 0.20: DER 8.57%, MS 1.08%, FA 7.01%, SC 0.48%
#
#Eval for threshold 0.30: DER 6.45%, MS 1.68%, FA 4.21%, SC 0.56%
#
#Eval for threshold 0.35: DER 5.91%, MS 2.06%, FA 3.27%, SC 0.58%
#
#Eval for threshold 0.40: DER 5.58%, MS 2.42%, FA 2.52%, SC 0.64%
#
#Eval for threshold 0.45: DER 5.41%, MS 2.82%, FA 1.93%, SC 0.66%
#
#Eval for threshold 0.50: DER 5.51%, MS 3.35%, FA 1.53%, SC 0.63%
#
#Eval for threshold 0.55: DER 5.75%, MS 4.00%, FA 1.19%, SC 0.56%
#
#Eval for threshold 0.60: DER 6.11%, MS 4.67%, FA 0.96%, SC 0.47%
#
#Eval for threshold 0.70: DER 7.29%, MS 6.33%, FA 0.65%, SC 0.32%
#
#Eval for threshold 0.80: DER 9.52%, MS 8.93%, FA 0.44%, SC 0.15%
## Test set
#Model DER:  0.13585995600566883
#Model ACC:  0.9482740784522706
#frame_len: 0.04!!
#100%|██████████| 60/60 [00:59<00:00,  1.01it/s]
#Eval for threshold 0.20: DER 10.27%, MS 1.21%, FA 7.76%, SC 1.29%
#
#Eval for threshold 0.30: DER 8.00%, MS 1.93%, FA 4.42%, SC 1.65%
#
#Eval for threshold 0.35: DER 7.40%, MS 2.33%, FA 3.30%, SC 1.78%
#
#Eval for threshold 0.40: DER 7.09%, MS 2.77%, FA 2.46%, SC 1.86%
#
#Eval for threshold 0.45: DER 7.00%, MS 3.29%, FA 1.80%, SC 1.91%
#
#Eval for threshold 0.50: DER 7.12%, MS 3.97%, FA 1.31%, SC 1.84%
#
#Eval for threshold 0.55: DER 7.43%, MS 4.76%, FA 0.96%, SC 1.72%
#
#Eval for threshold 0.60: DER 7.87%, MS 5.61%, FA 0.73%, SC 1.53%
#
#Eval for threshold 0.70: DER 9.31%, MS 7.72%, FA 0.41%, SC 1.18%
#
#Eval for threshold 0.80: DER 11.75%, MS 10.73%, FA 0.21%, SC 0.82%
fi

## utils now(2024-8-30), the best setting
if [ ${stage} -le 22 ] && [ ${stop_stage} -ge 22 ];then

    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is WavLM (only using cnn frontend and first 6 layers transformer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this WavLM is trained 60khours libri-light and 10k hours gigaspeech and 24k hours VoxPopuli,
    # checkpoint is from https://github.com/microsoft/unilm/tree/master/wavlm
    # how to look for port ?
    # netstat -tuln
   export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    speech_encoder_type="WavLM"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Base+.pt"
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wavlm_epoch40
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  accelerate launch --main_process_port 12785 \
   ts_vad2/train_accelerate_ddp2.py \
    --world-size 2 \
    --num-epochs 40\
    --start-epoch 31\
    --freeze-updates 4000\
    --grad-clip true\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --exp-dir $exp_dir
fi



if [ ${stage} -le 23 ] && [ ${stop_stage} -ge 23 ];then

    # # it adds noise and rirs to train tsvad model , no grad-clip and no freeze update.
    # # speech encoder is WavLM (only using cnn frontend and first 6 layers transformer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this WavLM is trained 60khours libri-light and 10k hours gigaspeech and 24k hours VoxPopuli,
    # checkpoint is from https://github.com/microsoft/unilm/tree/master/wavlm
    # how to look for port ?
    # netstat -tuln
   export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    speech_encoder_type="WavLM"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Base+.pt"
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_nofreeze_with_musan_rirs_wavlm_epoch40
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  accelerate launch --main_process_port 12785 \
   ts_vad2/train_accelerate_ddp2.py \
    --world-size 2 \
    --num-epochs 40\
    --start-epoch 1\
    --freeze-updates 0\
    --grad-clip false\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --exp-dir $exp_dir
fi



if [ ${stage} -le 24 ] && [ ${stop_stage} -ge 24 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wavlm_epoch40
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 results_path=$exp_dir/
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="WavLM"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Base+.pt"
 for name in $infer_sets;do
    results_path=$exp_dir/$name
    mkdir -p $results_path
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
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path
done

# cat logs/run_ts_vad2_stage24.log
# Eval set
# Model DER:  0.13363037744756806
#Model ACC:  0.9537209631208543
#frame_len: 0.04!!
#100%|██████████| 25/25 [00:24<00:00,  1.04it/s]
#Eval for threshold 0.20: DER 7.77%, MS 1.11%, FA 6.17%, SC 0.49%
#
#Eval for threshold 0.30: DER 5.89%, MS 1.71%, FA 3.54%, SC 0.64%
#
#Eval for threshold 0.35: DER 5.47%, MS 2.08%, FA 2.74%, SC 0.65%
#
#Eval for threshold 0.40: DER 5.29%, MS 2.51%, FA 2.13%, SC 0.65%
#
#Eval for threshold 0.45: DER 5.24%, MS 2.93%, FA 1.64%, SC 0.67%
#
#Eval for threshold 0.50: DER 5.42%, MS 3.47%, FA 1.31%, SC 0.64%
#
#Eval for threshold 0.55: DER 5.71%, MS 4.08%, FA 1.06%, SC 0.58%
#
#Eval for threshold 0.60: DER 6.18%, MS 4.80%, FA 0.87%, SC 0.52%
#
#Eval for threshold 0.70: DER 7.42%, MS 6.44%, FA 0.61%, SC 0.37%
#
#Eval for threshold 0.80: DER 9.57%, MS 8.94%, FA 0.42%, SC 0.22%

# Test set
#Model DER:  0.12851162538326832
#Model ACC:  0.952500379550336
#frame_len: 0.04!!
#100%|██████████| 60/60 [01:00<00:00,  1.00s/it]
#Eval for threshold 0.20: DER 9.13%, MS 1.22%, FA 7.31%, SC 0.59%
#
#Eval for threshold 0.30: DER 6.83%, MS 1.94%, FA 4.13%, SC 0.76%
#
#Eval for threshold 0.35: DER 6.31%, MS 2.31%, FA 3.15%, SC 0.85%
#
#Eval for threshold 0.40: DER 6.02%, MS 2.73%, FA 2.36%, SC 0.93%
#
#Eval for threshold 0.45: DER 5.92%, MS 3.15%, FA 1.76%, SC 1.00%
#
#Eval for threshold 0.50: DER 6.07%, MS 3.78%, FA 1.32%, SC 0.97%
#
#Eval for threshold 0.55: DER 6.43%, MS 4.54%, FA 0.99%, SC 0.89%
#
#Eval for threshold 0.60: DER 6.90%, MS 5.36%, FA 0.74%, SC 0.80%
#
#Eval for threshold 0.70: DER 8.36%, MS 7.40%, FA 0.40%, SC 0.56%
#
#Eval for threshold 0.80: DER 10.98%, MS 10.41%, FA 0.21%, SC 0.35%
fi

if [ ${stage} -le 25 ] && [ ${stop_stage} -ge 25 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_nofreeze_with_musan_rirs_wavlm_epoch40
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 results_path=$exp_dir/
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="WavLM"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Base+.pt"
 for name in $infer_sets;do
    results_path=$exp_dir/$name
    mkdir -p $results_path
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
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path
done
# cat logs/run_ts_vad2_stage25.log
# Eval set
#Model DER:  0.1443624146642278
#Model ACC:  0.9481869565793317
#frame_len: 0.04!!
#100%|██████████| 25/25 [00:23<00:00,  1.06it/s]
#Eval for threshold 0.20: DER 9.39%, MS 1.16%, FA 7.43%, SC 0.80%
#
#Eval for threshold 0.30: DER 7.28%, MS 1.73%, FA 4.44%, SC 1.11%
#
#Eval for threshold 0.35: DER 6.72%, MS 2.13%, FA 3.34%, SC 1.25%
#
#Eval for threshold 0.40: DER 6.45%, MS 2.59%, FA 2.38%, SC 1.48%
#
#Eval for threshold 0.45: DER 6.38%, MS 3.19%, FA 1.69%, SC 1.50%
#
#Eval for threshold 0.50: DER 6.63%, MS 3.90%, FA 1.33%, SC 1.41%
#
#Eval for threshold 0.55: DER 7.08%, MS 4.82%, FA 1.04%, SC 1.22%
#
#Eval for threshold 0.60: DER 7.68%, MS 5.78%, FA 0.85%, SC 1.06%
#
#Eval for threshold 0.70: DER 9.25%, MS 7.99%, FA 0.56%, SC 0.70%
#
#Eval for threshold 0.80: DER 11.80%, MS 10.98%, FA 0.42%, SC 0.40%
# Test set
#Model DER:  0.14017582985020072
#Model ACC:  0.9471487996961511
#frame_len: 0.04!!
#100%|██████████| 60/60 [00:58<00:00,  1.02it/s]
#Eval for threshold 0.20: DER 10.70%, MS 1.23%, FA 8.09%, SC 1.38%
#
#Eval for threshold 0.30: DER 8.09%, MS 2.02%, FA 4.42%, SC 1.65%
#
#Eval for threshold 0.35: DER 7.58%, MS 2.48%, FA 3.38%, SC 1.73%
#
#Eval for threshold 0.40: DER 7.26%, MS 2.99%, FA 2.51%, SC 1.75%
#
#Eval for threshold 0.45: DER 7.17%, MS 3.60%, FA 1.84%, SC 1.73%
#
#Eval for threshold 0.50: DER 7.33%, MS 4.29%, FA 1.40%, SC 1.65%
#
#Eval for threshold 0.55: DER 7.66%, MS 5.10%, FA 1.05%, SC 1.51%
#
#Eval for threshold 0.60: DER 8.17%, MS 6.00%, FA 0.80%, SC 1.37%
#
#Eval for threshold 0.70: DER 9.75%, MS 8.24%, FA 0.44%, SC 1.07%
#
#Eval for threshold 0.80: DER 12.43%, MS 11.47%, FA 0.22%, SC 0.73%


fi
# (todo),because num_epochs=40, two gpus, per device  batch_size is 64, its total num_update steps are about  60000
# so I will update schedul lr with warm_up_step 6000,max_update_step 60000.
if [ ${stage} -le 26 ] && [ ${stop_stage} -ge 26 ];then

    # # it adds noise and rirs to train tsvad model , no grad-clip and no freeze update.
    # # speech encoder is WavLM (only using cnn frontend and first 6 layers transformer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this WavLM is trained 60khours libri-light and 10k hours gigaspeech and 24k hours VoxPopuli,
    # checkpoint is from https://github.com/microsoft/unilm/tree/master/wavlm
    # how to look for port ?
    # netstat -tuln
   export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    speech_encoder_type="WavLM"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Base+.pt"
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wavlm_epoch40_adapted_lr
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  freeze_updates=6000
  warmup_updates=6000
  max_updates=60000
  CUDA_VISIABLE_DEVICES=0,1 \
  accelerate launch --main_process_port 12785 \
   ts_vad2/train_accelerate_ddp2.py \
    --world-size 2 \
    --num-epochs 40\
    --start-epoch 1\
    --freeze-updates $freeze_updates\
    --warmup-updates $warmup_updates\
    --max-updates $max_updates\
    --grad-clip true\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --warmup-updates $warmup_updates\
    --max-updates $max_updates\
    --exp-dir $exp_dir
fi

if [ ${stage} -le 27 ] && [ ${stop_stage} -ge 27 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wavlm_epoch40_adapted_lr
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 results_path=$exp_dir/
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="WavLM"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Base+.pt"
 for name in $infer_sets;do
    results_path=$exp_dir/$name
    mkdir -p $results_path
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
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path
done
# cat logs/run_ts_vad2_stage27.log
# Eval set
#Model DER:  0.13300386041816997
#Model ACC:  0.9534930776886393
#100%|██████████| 25/25 [00:23<00:00,  1.06it/s]
#Eval for threshold 0.20: DER 8.35%, MS 1.20%, FA 6.66%, SC 0.50%
#
#Eval for threshold 0.30: DER 6.40%, MS 1.69%, FA 4.12%, SC 0.59%
#
#Eval for threshold 0.35: DER 5.87%, MS 1.95%, FA 3.25%, SC 0.66%
#
#Eval for threshold 0.40: DER 5.56%, MS 2.29%, FA 2.56%, SC 0.71%
#
#Eval for threshold 0.45: DER 5.39%, MS 2.66%, FA 1.97%, SC 0.77%
#
#Eval for threshold 0.50: DER 5.39%, MS 3.07%, FA 1.53%, SC 0.79%
#
#Eval for threshold 0.55: DER 5.56%, MS 3.62%, FA 1.21%, SC 0.74%
#
#Eval for threshold 0.60: DER 5.91%, MS 4.29%, FA 0.98%, SC 0.64%
#
#Eval for threshold 0.70: DER 6.97%, MS 5.87%, FA 0.65%, SC 0.45%
#
#Eval for threshold 0.80: DER 9.02%, MS 8.33%, FA 0.47%, SC 0.21%

# Test set
#Model DER:  0.1369911317899102
#Model ACC:  0.9474153639033434
#100%|██████████| 60/60 [00:58<00:00,  1.03it/s]
#Eval for threshold 0.20: DER 9.69%, MS 1.30%, FA 7.02%, SC 1.37%
#
#Eval for threshold 0.30: DER 7.72%, MS 1.96%, FA 4.06%, SC 1.70%
#
#Eval for threshold 0.35: DER 7.28%, MS 2.35%, FA 3.10%, SC 1.82%
#
#Eval for threshold 0.40: DER 7.05%, MS 2.81%, FA 2.38%, SC 1.86%
#
#Eval for threshold 0.45: DER 7.01%, MS 3.29%, FA 1.83%, SC 1.89%
#
#Eval for threshold 0.50: DER 7.15%, MS 3.88%, FA 1.41%, SC 1.87%
#
#Eval for threshold 0.55: DER 7.43%, MS 4.59%, FA 1.08%, SC 1.76%
#
#Eval for threshold 0.60: DER 7.83%, MS 5.40%, FA 0.81%, SC 1.63%
#
#Eval for threshold 0.70: DER 9.06%, MS 7.33%, FA 0.45%, SC 1.28%
#
#Eval for threshold 0.80: DER 11.20%, MS 10.05%, FA 0.25%, SC 0.91%

fi

# compared with stage22, stage28 will increase epoch num to 60
if [ ${stage} -le 28 ] && [ ${stop_stage} -ge 28 ];then

    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is WavLM (only using cnn frontend and first 6 layers transformer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this WavLM is trained 60khours libri-light and 10k hours gigaspeech and 24k hours VoxPopuli,
    # checkpoint is from https://github.com/microsoft/unilm/tree/master/wavlm
    # how to look for port ?
    # netstat -tuln
   export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    speech_encoder_type="WavLM"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Base+.pt"
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wavlm_epoch60
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  accelerate launch --main_process_port 12785 \
   ts_vad2/train_accelerate_ddp2.py \
    --world-size 2 \
    --num-epochs 60\
    --start-epoch 1\
    --freeze-updates 4000\
    --grad-clip true\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --exp-dir $exp_dir
fi

if [ ${stage} -le 29 ] && [ ${stop_stage} -ge 29 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wavlm_epoch60
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 results_path=$exp_dir/
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="WavLM"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Base+.pt"
 for name in $infer_sets;do
    results_path=$exp_dir/$name
    mkdir -p $results_path
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
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path
done
# cat logs/run_ts_vad2_stage29.log
# Eval set
#Model DER:  0.1308091947267245
#Model ACC:  0.9546808652585775
#100%|██████████| 25/25 [00:24<00:00,  1.02it/s]
#Eval for threshold 0.20: DER 8.46%, MS 0.97%, FA 7.11%, SC 0.38%
#
#Eval for threshold 0.30: DER 6.10%, MS 1.50%, FA 4.11%, SC 0.48%
#
#Eval for threshold 0.35: DER 5.55%, MS 1.82%, FA 3.23%, SC 0.50%
#
#Eval for threshold 0.40: DER 5.22%, MS 2.15%, FA 2.51%, SC 0.56%
#
#Eval for threshold 0.45: DER 5.09%, MS 2.54%, FA 2.00%, SC 0.56%
#
#Eval for threshold 0.50: DER 5.12%, MS 3.01%, FA 1.59%, SC 0.52%
#
#Eval for threshold 0.55: DER 5.29%, MS 3.51%, FA 1.28%, SC 0.49%
#
#Eval for threshold 0.60: DER 5.63%, MS 4.18%, FA 1.02%, SC 0.43%
#
#Eval for threshold 0.70: DER 6.82%, MS 5.84%, FA 0.69%, SC 0.29%
#
#Eval for threshold 0.80: DER 9.00%, MS 8.37%, FA 0.47%, SC 0.16%
# Test set
#Model DER:  0.12977988418712308
#Model ACC:  0.9518246387975466
#100%|██████████| 60/60 [00:59<00:00,  1.00it/s]
#Eval for threshold 0.20: DER 10.17%, MS 1.08%, FA 8.39%, SC 0.70%
#
#Eval for threshold 0.30: DER 7.46%, MS 1.75%, FA 4.78%, SC 0.94%
#
#Eval for threshold 0.35: DER 6.79%, MS 2.15%, FA 3.64%, SC 1.00%
#
#Eval for threshold 0.40: DER 6.39%, MS 2.60%, FA 2.75%, SC 1.04%
#
#Eval for threshold 0.45: DER 6.22%, MS 3.09%, FA 2.02%, SC 1.10%
#
#Eval for threshold 0.50: DER 6.28%, MS 3.69%, FA 1.49%, SC 1.10%
#
#Eval for threshold 0.55: DER 6.53%, MS 4.42%, FA 1.11%, SC 0.99%
#
#Eval for threshold 0.60: DER 6.96%, MS 5.21%, FA 0.84%, SC 0.91%
#
#Eval for threshold 0.70: DER 8.32%, MS 7.15%, FA 0.45%, SC 0.72%
#
#Eval for threshold 0.80: DER 10.96%, MS 10.27%, FA 0.23%, SC 0.45%
#
fi

# compared with stage22, stage30 will increase max_update to 60000
if [ ${stage} -le 30 ] && [ ${stop_stage} -ge 30 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is WavLM (only using cnn frontend and first 6 layers transformer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this WavLM is trained 60khours libri-light and 10k hours gigaspeech and 24k hours VoxPopuli,
    # checkpoint is from https://github.com/microsoft/unilm/tree/master/wavlm
    # how to look for port ?
    # netstat -tuln
   export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    speech_encoder_type="WavLM"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Base+.pt"
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wavlm_epoch40_maxupdate60k
    max_updates=60000
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  accelerate launch --main_process_port 12885 \
   ts_vad2/train_accelerate_ddp2.py \
    --world-size 2 \
    --num-epochs 40\
    --start-epoch 1\
    --freeze-updates 4000\
    --grad-clip true\
    --max-updates $max_updates\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --exp-dir $exp_dir
fi

if [ ${stage} -le 31 ] && [ ${stop_stage} -ge 31 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wavlm_epoch40_maxupdate60k
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 results_path=$exp_dir/
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="WavLM"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Base+.pt"
 for name in $infer_sets;do
    results_path=$exp_dir/$name
    mkdir -p $results_path
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
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path
done
# cat logs/run_ts_vad2_stage31.log
# Eval set
## Model DER:  0.13601807983172692
#Model ACC:  0.9524805364203999
#100%|██████████| 25/25 [00:24<00:00,  1.04it/s]
#Eval for threshold 0.20: DER 7.65%, MS 1.24%, FA 5.85%, SC 0.57%
#
#Eval for threshold 0.30: DER 6.10%, MS 1.83%, FA 3.57%, SC 0.69%
#
#Eval for threshold 0.35: DER 5.71%, MS 2.14%, FA 2.81%, SC 0.76%
#
#Eval for threshold 0.40: DER 5.43%, MS 2.48%, FA 2.15%, SC 0.79%
#
#Eval for threshold 0.45: DER 5.41%, MS 2.90%, FA 1.73%, SC 0.78%
#
#Eval for threshold 0.50: DER 5.49%, MS 3.32%, FA 1.37%, SC 0.80%
#
#Eval for threshold 0.55: DER 5.79%, MS 3.95%, FA 1.12%, SC 0.72%
#
#Eval for threshold 0.60: DER 6.19%, MS 4.66%, FA 0.90%, SC 0.63%
#
#Eval for threshold 0.70: DER 7.43%, MS 6.40%, FA 0.62%, SC 0.41%
#
#Eval for threshold 0.80: DER 9.62%, MS 8.90%, FA 0.47%, SC 0.25%
# Test set
#Model DER:  0.13744124505072802
#Model ACC:  0.9470390138582262
#100%|██████████| 60/60 [00:59<00:00,  1.01it/s]
#Eval for threshold 0.20: DER 9.34%, MS 1.49%, FA 6.12%, SC 1.73%
#
#Eval for threshold 0.30: DER 7.76%, MS 2.27%, FA 3.46%, SC 2.04%
#
#Eval for threshold 0.35: DER 7.42%, MS 2.68%, FA 2.64%, SC 2.11%
#
#Eval for threshold 0.40: DER 7.29%, MS 3.14%, FA 2.00%, SC 2.15%
#
#Eval for threshold 0.45: DER 7.26%, MS 3.63%, FA 1.47%, SC 2.16%
#
#Eval for threshold 0.50: DER 7.38%, MS 4.26%, FA 1.05%, SC 2.07%
#
#Eval for threshold 0.55: DER 7.72%, MS 5.01%, FA 0.79%, SC 1.91%
#
#Eval for threshold 0.60: DER 8.18%, MS 5.79%, FA 0.61%, SC 1.78%
#
#Eval for threshold 0.70: DER 9.54%, MS 7.72%, FA 0.35%, SC 1.47%

#Eval for threshold 0.80: DER 11.69%, MS 10.42%, FA 0.20%, SC 1.07%
fi
# compared with stage22, stage32 will use feature_grad_mult=0.0
if [ ${stage} -le 32 ] && [ ${stop_stage} -ge 32 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is WavLM (only using cnn frontend and first 6 layers transformer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this WavLM is trained 60khours libri-light and 10k hours gigaspeech and 24k hours VoxPopuli,
    # checkpoint is from https://github.com/microsoft/unilm/tree/master/wavlm
    # how to look for port ?
    # netstat -tuln
   export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    speech_encoder_type="WavLM"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Base+.pt"
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wavlm_epoch40_feature_grad_mult0.0
    max_updates=40000
    feature_grad_mult=0.0 # means that freezed cnn feature frontent of wavlm in training tsvad.

   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  accelerate launch --main_process_port 12885 \
   ts_vad2/train_accelerate_ddp2.py \
    --world-size 2 \
    --num-epochs 40\
    --start-epoch 1\
    --freeze-updates 4000\
    --grad-clip true\
    --max-updates $max_updates\
    --feature-grad-mult $feature_grad_mult\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --exp-dir $exp_dir
fi

if [ ${stage} -le 33 ] && [ ${stop_stage} -ge 33 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wavlm_epoch40_feature_grad_mult0.0
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 results_path=$exp_dir/
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="WavLM"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Base+.pt"
 for name in $infer_sets;do
    results_path=$exp_dir/$name
    mkdir -p $results_path
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
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path
done
# Eval set
# Model DER:  0.13080951872827587
#Model ACC:  0.9550436840552864
#100%|██████████| 25/25 [00:23<00:00,  1.06it/s]
#Eval for threshold 0.20: DER 8.26%, MS 0.98%, FA 6.98%, SC 0.30%
#
#Eval for threshold 0.30: DER 5.96%, MS 1.55%, FA 4.01%, SC 0.40%
#
#Eval for threshold 0.35: DER 5.49%, MS 1.90%, FA 3.16%, SC 0.44%
#
#Eval for threshold 0.40: DER 5.22%, MS 2.29%, FA 2.47%, SC 0.46%
#
#Eval for threshold 0.45: DER 5.07%, MS 2.67%, FA 1.95%, SC 0.46% as report
#
#Eval for threshold 0.50: DER 5.14%, MS 3.13%, FA 1.56%, SC 0.45%
#
#Eval for threshold 0.55: DER 5.24%, MS 3.60%, FA 1.21%, SC 0.42%
#
#Eval for threshold 0.60: DER 5.49%, MS 4.17%, FA 0.98%, SC 0.35%
#
#Eval for threshold 0.70: DER 6.62%, MS 5.71%, FA 0.64%, SC 0.27%
#
#Eval for threshold 0.80: DER 8.62%, MS 7.97%, FA 0.44%, SC 0.21%

# Test set
# Model DER:  0.1329731605525156
#Model ACC:  0.9501785662733724
#100%|██████████| 60/60 [00:58<00:00,  1.02it/s]
#Eval for threshold 0.20: DER 10.16%, MS 1.11%, FA 8.02%, SC 1.03%
#
#Eval for threshold 0.30: DER 7.74%, MS 1.84%, FA 4.66%, SC 1.24%
#
#Eval for threshold 0.35: DER 7.14%, MS 2.25%, FA 3.57%, SC 1.33%
#
#Eval for threshold 0.40: DER 6.76%, MS 2.70%, FA 2.68%, SC 1.38%
#
#Eval for threshold 0.45: DER 6.62%, MS 3.24%, FA 1.99%, SC 1.39% as report
#
#Eval for threshold 0.50: DER 6.74%, MS 3.92%, FA 1.51%, SC 1.32%
#
#Eval for threshold 0.55: DER 7.05%, MS 4.69%, FA 1.13%, SC 1.23%
#
#Eval for threshold 0.60: DER 7.46%, MS 5.49%, FA 0.82%, SC 1.15%
#
#Eval for threshold 0.70: DER 8.74%, MS 7.37%, FA 0.45%, SC 0.92%
#
#Eval for threshold 0.80: DER 11.05%, MS 10.19%, FA 0.22%, SC 0.64%
fi

# the same is as stage 22, I just want to verify if it is the best.
if [ ${stage} -le 34 ] && [ ${stop_stage} -ge 34 ];then

    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is WavLM (only using cnn frontend and first 6 layers transformer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this WavLM is trained 60khours libri-light and 10k hours gigaspeech and 24k hours VoxPopuli,
    # checkpoint is from https://github.com/microsoft/unilm/tree/master/wavlm
    # how to look for port ?
    # netstat -tuln
   export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    speech_encoder_type="WavLM"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Base+.pt"
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wavlm_epoch40_again
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  accelerate launch --main_process_port 12785 \
   ts_vad2/train_accelerate_ddp2.py \
    --world-size 2 \
    --num-epochs 40\
    --start-epoch 1\
    --freeze-updates 4000\
    --grad-clip true\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --exp-dir $exp_dir
fi


if [ ${stage} -le 35 ] && [ ${stop_stage} -ge 35 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wavlm_epoch40_again
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 #results_path=$exp_dir/
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="WavLM"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Base+.pt"
 for name in $infer_sets;do
    results_path=$exp_dir
    #mkdir -p $results_path
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
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path
done
# cat logs/run_ts_vad2_stage35.log
# Eval set
# Model DER:  0.13347579431409276
#Model ACC:  0.9538651180946319
#100%|██████████| 25/25 [00:23<00:00,  1.06it/s]
#Eval for threshold 0.20: DER 8.51%, MS 1.03%, FA 7.10%, SC 0.39%
#
#Eval for threshold 0.30: DER 6.26%, MS 1.62%, FA 4.20%, SC 0.43%
#
#Eval for threshold 0.35: DER 5.68%, MS 1.93%, FA 3.29%, SC 0.46%
#
#Eval for threshold 0.40: DER 5.35%, MS 2.29%, FA 2.56%, SC 0.50%
#
#Eval for threshold 0.45: DER 5.25%, MS 2.73%, FA 1.98%, SC 0.54%
#
#Eval for threshold 0.50: DER 5.30%, MS 3.21%, FA 1.58%, SC 0.50%
#
#Eval for threshold 0.55: DER 5.49%, MS 3.78%, FA 1.28%, SC 0.43%
#
#Eval for threshold 0.60: DER 5.92%, MS 4.51%, FA 1.01%, SC 0.41%
#
#Eval for threshold 0.70: DER 7.17%, MS 6.27%, FA 0.61%, SC 0.30%
#
#Eval for threshold 0.80: DER 9.34%, MS 8.69%, FA 0.45%, SC 0.20%
# Test set
# Model DER:  0.1294195762199424
#Model ACC:  0.9520336178533109
#100%|██████████| 60/60 [00:58<00:00,  1.02it/s]
#Eval for threshold 0.20: DER 9.51%, MS 1.15%, FA 7.72%, SC 0.64%
#
#Eval for threshold 0.30: DER 7.16%, MS 1.82%, FA 4.52%, SC 0.81%
#
#Eval for threshold 0.35: DER 6.58%, MS 2.19%, FA 3.47%, SC 0.91%
#
#Eval for threshold 0.40: DER 6.23%, MS 2.58%, FA 2.67%, SC 0.99%
#
#Eval for threshold 0.45: DER 6.08%, MS 3.03%, FA 1.99%, SC 1.06%
#
#Eval for threshold 0.50: DER 6.10%, MS 3.57%, FA 1.46%, SC 1.07%
#
#Eval for threshold 0.55: DER 6.32%, MS 4.26%, FA 1.08%, SC 0.98%
#
#Eval for threshold 0.60: DER 6.77%, MS 5.06%, FA 0.82%, SC 0.88%
#
#Eval for threshold 0.70: DER 8.08%, MS 6.99%, FA 0.45%, SC 0.65%
#
#Eval for threshold 0.80: DER 10.43%, MS 9.77%, FA 0.22%, SC 0.44%

fi

##  I will try to avg model to infer
if [ ${stage} -le 36 ] && [ ${stop_stage} -ge 36 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wavlm_epoch40_again
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="WavLM"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Base+.pt"
 for name in $infer_sets;do
    results_path=$exp_dir/avg_9_epochs
    mkdir -p $results_path
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
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --use-averaged-model true\
    --avg 9\
    --epoch 40\
    --exp-dir $exp_dir

done
# cat logs/run_ts_vad2_stage36.log
# Eval set
# Model DER:  0.13364739710118717
#Model ACC:  0.9537221429922941
#100%|██████████| 25/25 [00:24<00:00,  1.03it/s]
#Eval for threshold 0.20: DER 8.53%, MS 1.04%, FA 7.10%, SC 0.39%
#
#Eval for threshold 0.30: DER 6.26%, MS 1.63%, FA 4.18%, SC 0.45%
#
#Eval for threshold 0.35: DER 5.71%, MS 1.96%, FA 3.27%, SC 0.48%
#
#Eval for threshold 0.40: DER 5.38%, MS 2.32%, FA 2.55%, SC 0.51%
#
#Eval for threshold 0.45: DER 5.28%, MS 2.76%, FA 1.97%, SC 0.55%
#
#Eval for threshold 0.50: DER 5.31%, MS 3.24%, FA 1.54%, SC 0.53%
#
#Eval for threshold 0.55: DER 5.54%, MS 3.85%, FA 1.25%, SC 0.44%
#
#Eval for threshold 0.60: DER 5.99%, MS 4.58%, FA 1.00%, SC 0.41%
#
#Eval for threshold 0.70: DER 7.27%, MS 6.37%, FA 0.60%, SC 0.30%
#
#Eval for threshold 0.80: DER 9.49%, MS 8.85%, FA 0.45%, SC 0.19%
# Test set
# Model DER:  0.1292510118229078
#Model ACC:  0.9520492888004524
#100%|██████████| 60/60 [00:59<00:00,  1.01it/s]
#Eval for threshold 0.20: DER 9.45%, MS 1.17%, FA 7.62%, SC 0.66%
#
#Eval for threshold 0.30: DER 7.14%, MS 1.85%, FA 4.46%, SC 0.83%
#
#Eval for threshold 0.35: DER 6.56%, MS 2.22%, FA 3.41%, SC 0.93%
#
#Eval for threshold 0.40: DER 6.21%, MS 2.63%, FA 2.57%, SC 1.01%
#
#Eval for threshold 0.45: DER 6.07%, MS 3.07%, FA 1.93%, SC 1.07%
#
#Eval for threshold 0.50: DER 6.09%, MS 3.61%, FA 1.42%, SC 1.07%
#
#Eval for threshold 0.55: DER 6.37%, MS 4.33%, FA 1.05%, SC 0.98%
#
#Eval for threshold 0.60: DER 6.80%, MS 5.11%, FA 0.80%, SC 0.89%
#
#Eval for threshold 0.70: DER 8.12%, MS 7.04%, FA 0.43%, SC 0.65%
#
#Eval for threshold 0.80: DER 10.47%, MS 9.82%, FA 0.21%, SC 0.44%
fi

if [ ${stage} -le 37 ] && [ ${stop_stage} -ge 37 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wavlm_epoch40_again
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 #results_path=$exp_dir/avg_5_epochs
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="WavLM"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Base+.pt"
 for name in $infer_sets;do
    results_path=$exp_dir/avg_5_epochs
    mkdir -p $results_path
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
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --use-averaged-model true\
    --avg 5\
    --epoch 40\
    --exp-dir $exp_dir
  done
# cat logs/run_ts_vad2_stage37.log
# Eval set
  # Model DER:  0.13354768696965927
#Model ACC:  0.9537855081341381
#100%|██████████| 25/25 [00:23<00:00,  1.04it/s]
#Eval for threshold 0.20: DER 8.46%, MS 1.04%, FA 7.03%, SC 0.39%
#
#Eval for threshold 0.30: DER 6.22%, MS 1.63%, FA 4.14%, SC 0.44%
#
#Eval for threshold 0.35: DER 5.67%, MS 1.96%, FA 3.23%, SC 0.47%
#
#Eval for threshold 0.40: DER 5.37%, MS 2.33%, FA 2.53%, SC 0.50%
#
#Eval for threshold 0.45: DER 5.25%, MS 2.77%, FA 1.94%, SC 0.54%
#
#Eval for threshold 0.50: DER 5.30%, MS 3.25%, FA 1.54%, SC 0.51%
#
#Eval for threshold 0.55: DER 5.52%, MS 3.84%, FA 1.25%, SC 0.43%
#
#Eval for threshold 0.60: DER 5.98%, MS 4.57%, FA 1.00%, SC 0.41%
#
#Eval for threshold 0.70: DER 7.25%, MS 6.36%, FA 0.60%, SC 0.29%
#
#Eval for threshold 0.80: DER 9.46%, MS 8.82%, FA 0.45%, SC 0.19%
# Test set
#Model DER:  0.12912920191973873
#Model ACC:  0.9521279405661883
#100%|██████████| 60/60 [00:59<00:00,  1.01it/s]
#Eval for threshold 0.20: DER 9.39%, MS 1.18%, FA 7.56%, SC 0.65%
#
#Eval for threshold 0.30: DER 7.10%, MS 1.86%, FA 4.42%, SC 0.82%
#
#Eval for threshold 0.35: DER 6.52%, MS 2.22%, FA 3.37%, SC 0.93%
#
#Eval for threshold 0.40: DER 6.18%, MS 2.62%, FA 2.56%, SC 1.00%
#
#Eval for threshold 0.45: DER 6.07%, MS 3.08%, FA 1.93%, SC 1.06%
#
#Eval for threshold 0.50: DER 6.08%, MS 3.61%, FA 1.41%, SC 1.06%
#
#Eval for threshold 0.55: DER 6.34%, MS 4.33%, FA 1.05%, SC 0.97%
#
#Eval for threshold 0.60: DER 6.79%, MS 5.10%, FA 0.80%, SC 0.89%
#
#Eval for threshold 0.70: DER 8.08%, MS 7.02%, FA 0.43%, SC 0.64%
#
#Eval for threshold 0.80: DER 10.44%, MS 9.79%, FA 0.21%, SC 0.43%
fi

 if [ ${stage} -le 38 ] && [ ${stop_stage} -ge 38 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wavlm_epoch40_again
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25

 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="WavLM"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Base+.pt"
 for name in $infer_sets;do
    results_path=$exp_dir/avg_5_checkpoints
    mkdir -p $results_path
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
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --use-averaged-model true\
    --avg 5\
    --iter 61500\
    --exp-dir $exp_dir
done
#  cat logs/run_ts_vad2_stage38.log
# Eval set
# Model DER:  0.13388220345955945
#Model ACC:  0.9537156993805513
#100%|██████████| 25/25 [00:24<00:00,  1.04it/s]
#Eval for threshold 0.20: DER 8.36%, MS 1.08%, FA 6.90%, SC 0.39%
#
#Eval for threshold 0.30: DER 6.13%, MS 1.66%, FA 4.02%, SC 0.45%
#
#Eval for threshold 0.35: DER 5.62%, MS 2.03%, FA 3.12%, SC 0.48%
#
#Eval for threshold 0.40: DER 5.33%, MS 2.38%, FA 2.44%, SC 0.51%
#
#Eval for threshold 0.45: DER 5.24%, MS 2.83%, FA 1.88%, SC 0.53%
#
#Eval for threshold 0.50: DER 5.34%, MS 3.32%, FA 1.51%, SC 0.51%
#
#Eval for threshold 0.55: DER 5.55%, MS 3.91%, FA 1.21%, SC 0.43%
#
#Eval for threshold 0.60: DER 6.02%, MS 4.65%, FA 0.98%, SC 0.40%
#
#Eval for threshold 0.70: DER 7.29%, MS 6.43%, FA 0.58%, SC 0.28%
#
#Eval for threshold 0.80: DER 9.51%, MS 8.88%, FA 0.44%, SC 0.18%
## Test set
#Model DER:  0.12941048652149717
#Model ACC:  0.9520554897926959
#100%|██████████| 60/60 [00:59<00:00,  1.01it/s]
#Eval for threshold 0.20: DER 9.24%, MS 1.21%, FA 7.37%, SC 0.66%
#
#Eval for threshold 0.30: DER 7.05%, MS 1.90%, FA 4.32%, SC 0.83%
#
#Eval for threshold 0.35: DER 6.48%, MS 2.27%, FA 3.28%, SC 0.93%
#
#Eval for threshold 0.40: DER 6.17%, MS 2.68%, FA 2.50%, SC 1.00%
#
#Eval for threshold 0.45: DER 6.05%, MS 3.13%, FA 1.86%, SC 1.07%
#
#Eval for threshold 0.50: DER 6.11%, MS 3.68%, FA 1.37%, SC 1.07%
#
#Eval for threshold 0.55: DER 6.39%, MS 4.40%, FA 1.02%, SC 0.97%
#
#Eval for threshold 0.60: DER 6.85%, MS 5.18%, FA 0.78%, SC 0.88%
#
#Eval for threshold 0.70: DER 8.14%, MS 7.09%, FA 0.42%, SC 0.63%
#
#Eval for threshold 0.80: DER 10.49%, MS 9.86%, FA 0.21%, SC 0.43%
fi

if [ ${stage} -le 39 ] && [ ${stop_stage} -ge 39 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wavlm_epoch40_again
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25

 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="WavLM"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Base+.pt"
 for name in $infer_sets;do
    results_path=$exp_dir/avg_9_checkpoints
    mkdir -p $results_path
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
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --use-averaged-model true\
    --avg 9\
    --iter 61500\
    --exp-dir $exp_dir
done
# cat logs/run_ts_vad2_stage39.log
# Eval set
# Model DER:  0.13386643147570892
#Model ACC:  0.9537041821748825
#100%|██████████| 25/25 [00:24<00:00,  1.02it/s]
#Eval for threshold 0.20: DER 8.36%, MS 1.07%, FA 6.89%, SC 0.39%
#
#Eval for threshold 0.30: DER 6.15%, MS 1.67%, FA 4.04%, SC 0.45%
#
#Eval for threshold 0.35: DER 5.62%, MS 2.02%, FA 3.12%, SC 0.48%
#
#Eval for threshold 0.40: DER 5.34%, MS 2.38%, FA 2.45%, SC 0.51%
#
#Eval for threshold 0.45: DER 5.24%, MS 2.82%, FA 1.89%, SC 0.53%
#
#Eval for threshold 0.50: DER 5.34%, MS 3.33%, FA 1.50%, SC 0.51%
#
#Eval for threshold 0.55: DER 5.56%, MS 3.92%, FA 1.21%, SC 0.43%
#
#Eval for threshold 0.60: DER 6.04%, MS 4.67%, FA 0.98%, SC 0.39%
#
#Eval for threshold 0.70: DER 7.30%, MS 6.43%, FA 0.58%, SC 0.28%
#
#Eval for threshold 0.80: DER 9.54%, MS 8.92%, FA 0.45%, SC 0.18%
# Test set
#Model DER:  0.1293795325501427
#Model ACC:  0.952050733277538
#100%|██████████| 60/60 [00:59<00:00,  1.01it/s]
#Eval for threshold 0.20: DER 9.23%, MS 1.21%, FA 7.36%, SC 0.66%
#
#Eval for threshold 0.30: DER 7.05%, MS 1.90%, FA 4.31%, SC 0.83%
#
#Eval for threshold 0.35: DER 6.47%, MS 2.27%, FA 3.26%, SC 0.94%
#
#Eval for threshold 0.40: DER 6.17%, MS 2.68%, FA 2.49%, SC 1.00%
#
#Eval for threshold 0.45: DER 6.06%, MS 3.13%, FA 1.86%, SC 1.07%
#
#Eval for threshold 0.50: DER 6.12%, MS 3.69%, FA 1.36%, SC 1.06%
#
#Eval for threshold 0.55: DER 6.39%, MS 4.40%, FA 1.02%, SC 0.98%
#
#Eval for threshold 0.60: DER 6.84%, MS 5.17%, FA 0.78%, SC 0.89%
#
#Eval for threshold 0.70: DER 8.16%, MS 7.11%, FA 0.42%, SC 0.63%
#
#Eval for threshold 0.80: DER 10.50%, MS 9.87%, FA 0.21%, SC 0.43%
fi
##(2024-9-4) Duo Ma note:
# Based on previous training experience
# Regardless of whether model averaging is used or not,
# in the speaker log system task, the pure transformer structure has no effect on performance.
# In the hybrid CNN and transformer structure, the average model may deteriorate.
# Therefore, it is finally decided to choose the model with the smallest vad der as the test model.



## To compare with stage 23 or stage 22, stage40 will use wavlm-larger pretrain model as speech encoder
## And a small change, I now fix the random seed of training to 1337,
# which comes from fairseq1, and the random seed no longer changes depending on the change of training epoch.
if [ ${stage} -le 40 ] && [ ${stop_stage} -ge 40 ];then

    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is WavLM (only using cnn frontend and first 6 layers transformer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this WavLM is trained 60khours libri-light and 10k hours gigaspeech and 24k hours VoxPopuli,
    # checkpoint is from https://github.com/microsoft/unilm/tree/master/wavlm
    # how to look for port ?
    # netstat -tuln
   export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    speech_encoder_type="WavLM"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Large.pt"
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wavlm_large_epoch40
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  accelerate launch --main_process_port 12785 \
   ts_vad2/train_accelerate_ddp2.py \
    --world-size 2 \
    --num-epochs 40\
    --start-epoch 1\
    --freeze-updates 4000\
    --grad-clip true\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --exp-dir $exp_dir
fi

if [ ${stage} -le 41 ] && [ ${stop_stage} -ge 41 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wavlm_large_epoch40
 model_file=$exp_dir/best-valid-der.pt
 #model_file=$exp_dir/best-valid-loss.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="WavLM"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Large.pt"
 for name in $infer_sets;do
    results_path=$exp_dir
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
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path
done
#  cat logs/run_ts_vad2_stage41.log
# model_file=$exp_dir/best-valid-der.pt
# Eval set
# Model DER:  0.1257801685813953
#Model ACC:  0.9571268127001323
#100%|██████████| 25/25 [00:24<00:00,  1.03it/s]
#Eval for threshold 0.20: DER 7.84%, MS 0.89%, FA 6.63%, SC 0.32%
#
#Eval for threshold 0.30: DER 5.73%, MS 1.37%, FA 3.95%, SC 0.40%
#
#Eval for threshold 0.35: DER 5.24%, MS 1.65%, FA 3.15%, SC 0.45%
#
#Eval for threshold 0.40: DER 4.88%, MS 1.95%, FA 2.48%, SC 0.45%
#
#Eval for threshold 0.45: DER 4.74%, MS 2.31%, FA 2.01%, SC 0.43%
#
#Eval for threshold 0.50: DER 4.72%, MS 2.71%, FA 1.59%, SC 0.42%
#
#Eval for threshold 0.55: DER 4.84%, MS 3.13%, FA 1.30%, SC 0.41%
#
#Eval for threshold 0.60: DER 5.06%, MS 3.64%, FA 1.05%, SC 0.37%
#
#Eval for threshold 0.70: DER 6.00%, MS 5.00%, FA 0.71%, SC 0.29%
#
#Eval for threshold 0.80: DER 7.85%, MS 7.16%, FA 0.49%, SC 0.20%

# Test set
# Model DER:  0.11652162698188549
#Model ACC:  0.9578242702816648
#100%|██████████| 60/60 [00:59<00:00,  1.01it/s]
#Eval for threshold 0.20: DER 7.99%, MS 0.94%, FA 6.70%, SC 0.35%
#
#Eval for threshold 0.30: DER 5.84%, MS 1.50%, FA 3.91%, SC 0.43%
#
#Eval for threshold 0.35: DER 5.32%, MS 1.81%, FA 3.04%, SC 0.47%
#
#Eval for threshold 0.40: DER 4.99%, MS 2.15%, FA 2.33%, SC 0.51%
#
#Eval for threshold 0.45: DER 4.86%, MS 2.52%, FA 1.79%, SC 0.54%
#
#Eval for threshold 0.50: DER 4.87%, MS 2.96%, FA 1.36%, SC 0.54%
#
#Eval for threshold 0.55: DER 5.07%, MS 3.49%, FA 1.04%, SC 0.54%
#
#Eval for threshold 0.60: DER 5.42%, MS 4.09%, FA 0.81%, SC 0.52%
#
#Eval for threshold 0.70: DER 6.54%, MS 5.65%, FA 0.47%, SC 0.42%
#
#Eval for threshold 0.80: DER 8.55%, MS 8.03%, FA 0.25%, SC 0.27%
fi

if [ ${stage} -le 42 ] && [ ${stop_stage} -ge 42 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wavlm_large_epoch40
 model_file=$exp_dir/best-valid-loss.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="WavLM"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Large.pt"
 for name in $infer_sets;do
    results_path=$exp_dir/best-valid-loss
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
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path
done
# cat logs/run_ts_vad2_stage42.log
# Eval set
# Model DER:  0.1257801685813953
#Model ACC:  0.9571268127001323
#100%|██████████| 25/25 [00:24<00:00,  1.02it/s]
#Eval for threshold 0.20: DER 7.84%, MS 0.89%, FA 6.63%, SC 0.32%
#
#Eval for threshold 0.30: DER 5.73%, MS 1.37%, FA 3.95%, SC 0.40%
#
#Eval for threshold 0.35: DER 5.24%, MS 1.65%, FA 3.15%, SC 0.45%
#
#Eval for threshold 0.40: DER 4.88%, MS 1.95%, FA 2.48%, SC 0.45%
#
#Eval for threshold 0.45: DER 4.74%, MS 2.31%, FA 2.01%, SC 0.43%
#
#Eval for threshold 0.50: DER 4.72%, MS 2.71%, FA 1.59%, SC 0.42%
#
#Eval for threshold 0.55: DER 4.84%, MS 3.13%, FA 1.30%, SC 0.41%
#
#Eval for threshold 0.60: DER 5.06%, MS 3.64%, FA 1.05%, SC 0.37%
#
#Eval for threshold 0.70: DER 6.00%, MS 5.00%, FA 0.71%, SC 0.29%
#
#Eval for threshold 0.80: DER 7.85%, MS 7.16%, FA 0.49%, SC 0.20%
# Test set
## Model DER:  0.11652162698188549
#Model ACC:  0.9578242702816648
#100%|██████████| 60/60 [00:59<00:00,  1.01it/s]
#Eval for threshold 0.20: DER 7.99%, MS 0.94%, FA 6.70%, SC 0.35%
#
#Eval for threshold 0.30: DER 5.84%, MS 1.50%, FA 3.91%, SC 0.43%
#
#Eval for threshold 0.35: DER 5.32%, MS 1.81%, FA 3.04%, SC 0.47%
#
#Eval for threshold 0.40: DER 4.99%, MS 2.15%, FA 2.33%, SC 0.51%
#
#Eval for threshold 0.45: DER 4.86%, MS 2.52%, FA 1.79%, SC 0.54%
#
#Eval for threshold 0.50: DER 4.87%, MS 2.96%, FA 1.36%, SC 0.54%
#
#Eval for threshold 0.55: DER 5.07%, MS 3.49%, FA 1.04%, SC 0.54%
#
#Eval for threshold 0.60: DER 5.42%, MS 4.09%, FA 0.81%, SC 0.52%
#
#Eval for threshold 0.70: DER 6.54%, MS 5.65%, FA 0.47%, SC 0.42%
#
#Eval for threshold 0.80: DER 8.55%, MS 8.03%, FA 0.25%, SC 0.27%
fi

## To compare with stage40, stage43 will use whisper-large-v2 pretrain model as speech encoder
## it is not working(2024-9-8) todo fix it.
if [ ${stage} -le 43 ] && [ ${stop_stage} -ge 43 ];then

    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is whisper encoder of whisper-large-v2 (only using 16-23 layer and first cnn layer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this whisper-large-v2 is trained on 680k hours of labelled data,
    # checkpoint is from https://huggingface.co/openai/whisper-large-v2/tree/main
    # how to look for port ?
    # netstat -tuln
   export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    speech_encoder_type="whisper"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/whisper/whisper-large-v2.pt"
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_whisper_large-v2_epoch40
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  accelerate launch --main_process_port 12785 \
   ts_vad2/train_accelerate_ddp2.py \
    --world-size 2 \
    --num-epochs 40\
    --start-epoch 1\
    --freeze-updates 4000\
    --grad-clip true\
    --lr 1e-5\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --exp-dir $exp_dir
fi
# campared with stage 40, I will average model in train stage,
# this trick is from icefall.
if [ ${stage} -le 45 ] && [ ${stop_stage} -ge 45 ];then

    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is WavLM (only using cnn frontend and first 6 layers transformer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this WavLM is trained 60khours libri-light and 10k hours gigaspeech and 24k hours VoxPopuli,
    # checkpoint is from https://github.com/microsoft/unilm/tree/master/wavlm
    # how to look for port ?
    # netstat -tuln
   export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES

    # for loading pretrain model weigt
    speech_encoder_type="WavLM"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Large.pt"
    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"

    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wavlm_large_using_average_model_epoch40
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  accelerate launch --main_process_port 12885 \
   ts_vad2/train_accelerate_ddp2.py \
    --world-size 2 \
    --num-epochs 40\
    --start-epoch 1\
    --freeze-updates 4000\
    --grad-clip true\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --train-on-average true\
    --exp-dir $exp_dir
fi
if [ ${stage} -le 46 ] && [ ${stop_stage} -ge 46 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wavlm_large_using_average_model_epoch40
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="WavLM"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Large.pt"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

 for name in $infer_sets;do
    results_path=$exp_dir
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
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --data-dir $data_dir
done
# Eval set
## Model DER:  0.12400508346290659
#Model ACC:  0.9578473174768353
#100%|██████████| 25/25 [00:12<00:00,  1.95it/s]
#Eval for threshold 0.20: DER 7.48%, MS 0.93%, FA 6.27%, SC 0.27%
#
#Eval for threshold 0.30: DER 5.46%, MS 1.42%, FA 3.71%, SC 0.34%
#
#Eval for threshold 0.35: DER 5.00%, MS 1.68%, FA 2.98%, SC 0.34%
#
#Eval for threshold 0.40: DER 4.61%, MS 1.95%, FA 2.31%, SC 0.36%
#
#Eval for threshold 0.45: DER 4.44%, MS 2.24%, FA 1.81%, SC 0.40%
#
#Eval for threshold 0.50: DER 4.46%, MS 2.61%, FA 1.46%, SC 0.39%
#
#Eval for threshold 0.55: DER 4.60%, MS 3.06%, FA 1.18%, SC 0.36%
#
#Eval for threshold 0.60: DER 4.82%, MS 3.54%, FA 0.92%, SC 0.36%
#
#Eval for threshold 0.70: DER 5.77%, MS 4.85%, FA 0.64%, SC 0.27%
#
#Eval for threshold 0.80: DER 7.55%, MS 6.95%, FA 0.45%, SC 0.14%
# Test set
## Model DER:  0.11902468761900581
#Model ACC:  0.9569139207415561
#100%|██████████| 60/60 [00:31<00:00,  1.92it/s]
#Eval for threshold 0.20: DER 8.79%, MS 0.91%, FA 7.51%, SC 0.37%
#
#Eval for threshold 0.30: DER 6.42%, MS 1.45%, FA 4.52%, SC 0.46%
#
#Eval for threshold 0.35: DER 5.79%, MS 1.73%, FA 3.55%, SC 0.51%
#
#Eval for threshold 0.40: DER 5.38%, MS 2.06%, FA 2.78%, SC 0.54%
#
#Eval for threshold 0.45: DER 5.17%, MS 2.44%, FA 2.14%, SC 0.59%
#
#Eval for threshold 0.50: DER 5.10%, MS 2.87%, FA 1.65%, SC 0.58%
#
#Eval for threshold 0.55: DER 5.24%, MS 3.41%, FA 1.28%, SC 0.55%
#
#Eval for threshold 0.60: DER 5.54%, MS 4.02%, FA 0.99%, SC 0.53%
#
#Eval for threshold 0.70: DER 6.58%, MS 5.60%, FA 0.56%, SC 0.41%
#
#Eval for threshold 0.80: DER 8.54%, MS 7.97%, FA 0.27%, SC 0.29%

fi



## for compared stage 34, stage48 doesn't use add noise and rirs to train tsvad.
if [ ${stage} -le 48 ] && [ ${stop_stage} -ge 48 ];then

    # # it doesn't add noise and rirs to train tsvad model , uses grad-clip and freeze update.
    # # speech encoder is WavLM (only using cnn frontend and first 6 layers transformer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this WavLM is trained 60khours libri-light and 10k hours gigaspeech and 24k hours VoxPopuli,
    # checkpoint is from https://github.com/microsoft/unilm/tree/master/wavlm
    # how to look for port ?
    # netstat -tuln
   export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=""
    rir_path=""
    speech_encoder_type="WavLM"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Base+.pt"
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_without_musan_rirs_wavlm_epoch40
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  accelerate launch --main_process_port 12885 \
   ts_vad2/train_accelerate_ddp2.py \
    --world-size 2 \
    --num-epochs 40\
    --start-epoch 1\
    --freeze-updates 4000\
    --grad-clip true\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --exp-dir $exp_dir
fi

if [ ${stage} -le 49 ] && [ ${stop_stage} -ge 49 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_without_musan_rirs_wavlm_epoch40
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="WavLM"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Base+.pt"

 for name in $infer_sets;do
    results_path=$exp_dir/$name
    mkdir -p $results_path
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
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path
done
# Eval set
#Model DER:  0.1478424212820199
#Model ACC:  0.9450214575913956
#100%|██████████| 25/25 [00:23<00:00,  1.05it/s]
#Eval for threshold 0.20: DER 9.73%, MS 1.30%, FA 7.14%, SC 1.29%
#
#Eval for threshold 0.30: DER 7.87%, MS 1.89%, FA 4.29%, SC 1.69%
#
#Eval for threshold 0.35: DER 7.47%, MS 2.26%, FA 3.32%, SC 1.89%
#
#Eval for threshold 0.40: DER 7.18%, MS 2.61%, FA 2.54%, SC 2.03%
#
#Eval for threshold 0.45: DER 7.09%, MS 3.08%, FA 1.91%, SC 2.09%
#
#Eval for threshold 0.50: DER 7.13%, MS 3.63%, FA 1.46%, SC 2.04%
#
#Eval for threshold 0.55: DER 7.39%, MS 4.35%, FA 1.14%, SC 1.90%
#
#Eval for threshold 0.60: DER 7.88%, MS 5.24%, FA 0.95%, SC 1.69%
#
#Eval for threshold 0.70: DER 9.16%, MS 7.20%, FA 0.68%, SC 1.29%
#
#Eval for threshold 0.80: DER 11.51%, MS 10.13%, FA 0.50%, SC 0.89%
# Test set
#Model DER:  0.1466827737660315
#Model ACC:  0.9437713543089978
#100%|██████████| 60/60 [00:58<00:00,  1.02it/s]
#Eval for threshold 0.20: DER 11.64%, MS 1.28%, FA 8.97%, SC 1.38%
#
#Eval for threshold 0.30: DER 9.55%, MS 1.92%, FA 5.80%, SC 1.83%
#
#Eval for threshold 0.35: DER 8.97%, MS 2.29%, FA 4.71%, SC 1.97%
#
#Eval for threshold 0.40: DER 8.63%, MS 2.72%, FA 3.82%, SC 2.09%
#
#Eval for threshold 0.45: DER 8.47%, MS 3.23%, FA 3.13%, SC 2.11%
#
#Eval for threshold 0.50: DER 8.51%, MS 3.88%, FA 2.55%, SC 2.07%
#
#Eval for threshold 0.55: DER 8.74%, MS 4.65%, FA 2.15%, SC 1.93%
#
#Eval for threshold 0.60: DER 9.17%, MS 5.57%, FA 1.83%, SC 1.77%
#
#Eval for threshold 0.70: DER 10.34%, MS 7.62%, FA 1.29%, SC 1.43%
#
#Eval for threshold 0.80: DER 12.43%, MS 10.53%, FA 0.86%, SC 1.04%
fi
# extract target speaker embedding using opensource checkpoint from wespeaker
if [ ${stage} -le 60 ] && [ ${stop_stage} -ge 60 ];then
   echo "generate eval(dev) speaker embedding"
   feature_name=ecapa_tdnn_on_voxceleb_wespeaker_feature_dir
   dest_dir=/mntcephfs/lab_data/maduo/model_hub/
   pretrained_model=/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/en/wespeaker/voxceleb_ECAPA1024_LM/avg_model.pt
   model_name="ECAPA_TDNN_GLOB_c1024"
   subsets="Eval Test Train"
   for name in $subsets;do
    if [ $name = "Train" ];then
     echo "extract train target speaker embedding"
     # 提取embedding
     input_dir=/mntcephfs/lab_data/maduo/datasets/alimeeting/${name}_Ali_far/target_audio/
     wav_path=$input_dir/wavs.txt
    else
        echo "extract $name target speaker embedding"
        # 提取embedding
     input_dir=/mntcephfs/lab_data/maduo/datasets/alimeeting/${name}_Ali/${name}_Ali_far/target_audio/
     wav_path=$input_dir/wavs.txt
    fi
    save_dir=$dest_dir/ts_vad/spk_embed/alimeeting/SpeakerEmbedding/$name/$feature_name
    python3 ts_vad2/generate_chunk_speaker_embedding_from_wespeaker_for_diarization.py\
           --pretrained_model $pretrained_model\
           --model_name $model_name\
           --wavs $wav_path\
           --save_dir $save_dir
   done
fi

if [ ${stage} -le 61 ] && [ ${stop_stage} -ge 61 ];then

    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # speech encoder is WavLM (only using cnn frontend and first 6 layers transformer) ,
    #  oracle target speaker embedding is from ecapa_tdnn pretrain model from wespeaker
    # this WavLM is trained 60khours libri-light and 10k hours gigaspeech and 24k hours VoxPopuli,
    # how to look for port ?
    # netstat -tuln
   export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="WavLM"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Large.pt"
    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="ecapa_tdnn_on_voxceleb_wespeaker_feature_dir"

    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_ecapa_tdnn_1024_wespeaker_epoch40
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  accelerate launch --main_process_port 12785 \
   ts_vad2/train_accelerate_ddp2.py \
    --world-size 2 \
    --num-epochs 40\
    --start-epoch 1\
    --freeze-updates 4000\
    --grad-clip true\
    --lr 2e-4\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --exp-dir $exp_dir
fi

if [ ${stage} -le 62 ] && [ ${stop_stage} -ge 62 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_ecapa_tdnn_1024_wespeaker_epoch40
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 results_path=$exp_dir/
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="WavLM"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Large.pt"
 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="ecapa_tdnn_on_voxceleb_wespeaker_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

 for name in $infer_sets;do
    results_path=$exp_dir/$name
    mkdir -p $results_path
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
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --data-dir $data_dir
done
#Eval set
#Model DER:  0.14259121370453978
#Model ACC:  0.9492472824263307
#100%|██████████| 25/25 [00:13<00:00,  1.91it/s]
#Eval for threshold 0.20: DER 9.51%, MS 0.83%, FA 8.09%, SC 0.59%
#
#Eval for threshold 0.30: DER 6.88%, MS 1.39%, FA 4.71%, SC 0.78%
#
#Eval for threshold 0.35: DER 6.24%, MS 1.69%, FA 3.71%, SC 0.84%
#
#Eval for threshold 0.40: DER 5.82%, MS 2.08%, FA 2.86%, SC 0.88%
#
#Eval for threshold 0.45: DER 5.69%, MS 2.55%, FA 2.24%, SC 0.89%
#
#Eval for threshold 0.50: DER 5.67%, MS 3.08%, FA 1.72%, SC 0.87%
#
#Eval for threshold 0.55: DER 5.90%, MS 3.72%, FA 1.35%, SC 0.83%
#
#Eval for threshold 0.60: DER 6.25%, MS 4.44%, FA 1.04%, SC 0.77%
#
#Eval for threshold 0.70: DER 7.71%, MS 6.47%, FA 0.67%, SC 0.57%
#
#Eval for threshold 0.80: DER 10.49%, MS 9.65%, FA 0.46%, SC 0.38%
#
#Test set
#Model DER:  0.14119585123726083
#Model ACC:  0.945112619276462
#100%|██████████| 60/60 [00:31<00:00,  1.90it/s]
#Eval for threshold 0.20: DER 11.31%, MS 0.95%, FA 8.97%, SC 1.39%
#
#Eval for threshold 0.30: DER 8.64%, MS 1.61%, FA 5.21%, SC 1.82%
#
#Eval for threshold 0.35: DER 7.85%, MS 2.00%, FA 3.82%, SC 2.03%
#
#Eval for threshold 0.40: DER 7.41%, MS 2.47%, FA 2.78%, SC 2.17%
#
#Eval for threshold 0.45: DER 7.23%, MS 3.08%, FA 2.00%, SC 2.15%
#
#Eval for threshold 0.50: DER 7.39%, MS 3.88%, FA 1.47%, SC 2.04%
#
#Eval for threshold 0.55: DER 7.75%, MS 4.80%, FA 1.11%, SC 1.84%
#
#Eval for threshold 0.60: DER 8.27%, MS 5.82%, FA 0.83%, SC 1.62%
#
#Eval for threshold 0.70: DER 9.77%, MS 8.16%, FA 0.43%, SC 1.18%
#
#Eval for threshold 0.80: DER 12.38%, MS 11.42%, FA 0.21%, SC 0.75%
fi

if [ ${stage} -le 63 ] && [ ${stop_stage} -ge 63 ];then

    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # speech encoder is WavLM (only using cnn frontend and first 6 layers transformer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this WavLM is trained 60khours libri-light and 10k hours gigaspeech and 24k hours VoxPopuli,
    # how to look for port ?
    # netstat -tuln
   export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="WavLM"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Large.pt"
    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="ecapa_tdnn_on_voxceleb_wespeaker_feature_dir"

    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_ecapa_tdnn_1024_wespeaker_using_average_model_epoch40
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  accelerate launch --main_process_port 12785 \
   ts_vad2/train_accelerate_ddp2.py \
    --world-size 2 \
    --num-epochs 40\
    --start-epoch 1\
    --freeze-updates 4000\
    --grad-clip true\
    --lr 2e-4\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --train-on-average true\
    --exp-dir $exp_dir
fi

if [ ${stage} -le 64 ] && [ ${stop_stage} -ge 64 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_ecapa_tdnn_1024_wespeaker_using_average_model_epoch40
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 results_path=$exp_dir/
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="WavLM"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Large.pt"
 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="ecapa_tdnn_on_voxceleb_wespeaker_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

 for name in $infer_sets;do
    results_path=$exp_dir/$name
    mkdir -p $results_path
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
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --data-dir $data_dir
done
# Eval set
#Model DER:  0.15122636784399102
#Model ACC:  0.9437359562033952
#100%|██████████| 25/25 [00:12<00:00,  1.94it/s]
#Eval for threshold 0.20: DER 12.10%, MS 0.78%, FA 10.46%, SC 0.86%
#
#Eval for threshold 0.30: DER 8.74%, MS 1.30%, FA 6.19%, SC 1.25%
#
#Eval for threshold 0.35: DER 7.76%, MS 1.60%, FA 4.80%, SC 1.36%
#
#Eval for threshold 0.40: DER 7.19%, MS 1.99%, FA 3.72%, SC 1.49%
#
#Eval for threshold 0.45: DER 6.86%, MS 2.41%, FA 2.87%, SC 1.58%
#
#Eval for threshold 0.50: DER 6.66%, MS 2.86%, FA 2.17%, SC 1.64%
#
#Eval for threshold 0.55: DER 6.83%, MS 3.50%, FA 1.70%, SC 1.63%
#
#Eval for threshold 0.60: DER 7.13%, MS 4.31%, FA 1.31%, SC 1.51%
#
#Eval for threshold 0.70: DER 8.38%, MS 6.32%, FA 0.76%, SC 1.30%
#
#Eval for threshold 0.80: DER 10.74%, MS 9.34%, FA 0.50%, SC 0.90%
# Test set
# Model DER:  0.14451955771271002
#Model ACC:  0.9436771640213649
#100%|██████████| 60/60 [00:31<00:00,  1.91it/s]
#Eval for threshold 0.20: DER 12.00%, MS 0.91%, FA 9.48%, SC 1.61%
#
#Eval for threshold 0.30: DER 9.19%, MS 1.54%, FA 5.58%, SC 2.07%
#
#Eval for threshold 0.35: DER 8.39%, MS 1.92%, FA 4.16%, SC 2.31%
#
#Eval for threshold 0.40: DER 7.80%, MS 2.36%, FA 2.97%, SC 2.47%
#
#Eval for threshold 0.45: DER 7.53%, MS 2.94%, FA 2.06%, SC 2.53%
#
#Eval for threshold 0.50: DER 7.57%, MS 3.66%, FA 1.45%, SC 2.46%
#
#Eval for threshold 0.55: DER 7.90%, MS 4.55%, FA 1.06%, SC 2.29%
#
#Eval for threshold 0.60: DER 8.42%, MS 5.60%, FA 0.76%, SC 2.06%
#
#Eval for threshold 0.70: DER 10.11%, MS 8.21%, FA 0.38%, SC 1.51%
#
#Eval for threshold 0.80: DER 12.67%, MS 11.47%, FA 0.20%, SC 1.01%
fi
# compared with stage40, I will use all transformer of wavlm and weight sum them as wavlm output feature.
# and freeze encoder 40k steps, because freeze encoder 4k steps, it will not be able to train properly
# the trick is same s3prl finetune downstream task
# #I move fixed_random_seed to the training entrance(main()) from train_one_epoch()  to ensure repeatable training.
if [ ${stage} -le 65 ] && [ ${stop_stage} -ge 65 ];then

    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is WavLM (only using cnn frontend and first 6 layers transformer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this WavLM is trained 60khours libri-light and 10k hours gigaspeech and 24k hours VoxPopuli,
    # checkpoint is from https://github.com/microsoft/unilm/tree/master/wavlm
    # how to look for port ?
    # netstat -tuln
   export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES

    # for loading pretrain model weigt
    speech_encoder_type="WavLM_weight_sum"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Large.pt"
    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"

    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wavlm_large_weight_sum_epoch40
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  accelerate launch --main_process_port 12885 \
   ts_vad2/train_accelerate_ddp2.py \
    --world-size 2 \
    --num-epochs 40\
    --start-epoch 1\
    --freeze-updates 40000\
    --grad-clip true\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --train-on-average false\
    --exp-dir $exp_dir
fi

if [ ${stage} -le 66 ] && [ ${stop_stage} -ge 66 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wavlm_large_weight_sum_epoch40
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="WavLM_weight_sum"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Large.pt"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

 for name in $infer_sets;do
    results_path=$exp_dir
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
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --data-dir $data_dir
done
# Eval set
#Model DER:  0.15426194355516923
#Model ACC:  0.9456594985054682
#100%|██████████| 25/25 [00:13<00:00,  1.90it/s]
#Eval for threshold 0.20: DER 13.10%, MS 1.24%, FA 11.30%, SC 0.56%
#
#Eval for threshold 0.30: DER 8.81%, MS 1.92%, FA 6.19%, SC 0.71%
#
#Eval for threshold 0.35: DER 7.82%, MS 2.36%, FA 4.67%, SC 0.80%
#
#Eval for threshold 0.40: DER 7.26%, MS 2.89%, FA 3.45%, SC 0.92%
#
#Eval for threshold 0.45: DER 7.06%, MS 3.50%, FA 2.59%, SC 0.98%
#
#Eval for threshold 0.50: DER 7.17%, MS 4.33%, FA 1.83%, SC 1.01%
#
#Eval for threshold 0.55: DER 7.46%, MS 5.19%, FA 1.33%, SC 0.94%
#
#Eval for threshold 0.60: DER 8.15%, MS 6.36%, FA 0.97%, SC 0.82%
#
#Eval for threshold 0.70: DER 10.31%, MS 9.26%, FA 0.56%, SC 0.49%
#
#Eval for threshold 0.80: DER 13.64%, MS 13.02%, FA 0.37%, SC 0.24%
# Test set
#Model DER:  0.14458976437820859
#Model ACC:  0.9461460578900024
#100%|██████████| 60/60 [00:31<00:00,  1.91it/s]
#Eval for threshold 0.20: DER 11.82%, MS 1.34%, FA 9.70%, SC 0.78%
#
#Eval for threshold 0.30: DER 8.64%, MS 2.24%, FA 5.33%, SC 1.07%
#
#Eval for threshold 0.35: DER 7.90%, MS 2.76%, FA 3.99%, SC 1.15%
#
#Eval for threshold 0.40: DER 7.45%, MS 3.30%, FA 2.94%, SC 1.21%
#
#Eval for threshold 0.45: DER 7.35%, MS 3.98%, FA 2.11%, SC 1.26%
#
#Eval for threshold 0.50: DER 7.45%, MS 4.73%, FA 1.49%, SC 1.23%
#
#Eval for threshold 0.55: DER 7.82%, MS 5.65%, FA 1.04%, SC 1.13%
#
#Eval for threshold 0.60: DER 8.43%, MS 6.73%, FA 0.73%, SC 0.97%
#
#Eval for threshold 0.70: DER 10.22%, MS 9.20%, FA 0.35%, SC 0.68%
#
#Eval for threshold 0.80: DER 13.27%, MS 12.71%, FA 0.16%, SC 0.40%



fi
# compared with stage 40,
## # and freeze encoder 60k steps, because freeze encoder 4k steps, it will not be able to train properly
# the trick is same s3prl finetune downstream task
#I move fixed_random_seed to the training entrance(main()) from train_one_epoch()  to ensure repeatable training.
if [ ${stage} -le 67 ] && [ ${stop_stage} -ge 67 ];then

    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is WavLM (only using cnn frontend and first 6 layers transformer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this WavLM is trained 60khours libri-light and 10k hours gigaspeech and 24k hours VoxPopuli,
    # checkpoint is from https://github.com/microsoft/unilm/tree/master/wavlm
    # how to look for port ?
    # netstat -tuln
   export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES

    # for loading pretrain model weigt
    speech_encoder_type="WavLM_weight_sum"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Large.pt"
    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"

    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wavlm_large_weight_sum_epoch40_again
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  accelerate launch --main_process_port 12985 \
   ts_vad2/train_accelerate_ddp2.py \
    --world-size 2 \
    --num-epochs 40\
    --start-epoch 1\
    --freeze-updates 60000\
    --grad-clip true\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --train-on-average false\
    --exp-dir $exp_dir
fi

if [ ${stage} -le 68 ] && [ ${stop_stage} -ge 68 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wavlm_large_weight_sum_epoch40_again
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="WavLM_weight_sum"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Large.pt"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

 for name in $infer_sets;do
    results_path=$exp_dir
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
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --data-dir $data_dir
done
# Eval set
##Model DER:  0.15426194355516923
#Model ACC:  0.9456594985054682
#100%|██████████| 25/25 [00:13<00:00,  1.92it/s]
#Eval for threshold 0.20: DER 13.10%, MS 1.24%, FA 11.30%, SC 0.56%
#
#Eval for threshold 0.30: DER 8.81%, MS 1.92%, FA 6.19%, SC 0.71%
#
#Eval for threshold 0.35: DER 7.82%, MS 2.36%, FA 4.67%, SC 0.80%
#
#Eval for threshold 0.40: DER 7.26%, MS 2.89%, FA 3.45%, SC 0.92%
#
#Eval for threshold 0.45: DER 7.06%, MS 3.50%, FA 2.59%, SC 0.98%
#
#Eval for threshold 0.50: DER 7.17%, MS 4.33%, FA 1.83%, SC 1.01%
#
#Eval for threshold 0.55: DER 7.46%, MS 5.19%, FA 1.33%, SC 0.94%
#
#Eval for threshold 0.60: DER 8.15%, MS 6.36%, FA 0.97%, SC 0.82%
#
#Eval for threshold 0.70: DER 10.31%, MS 9.26%, FA 0.56%, SC 0.49%
#
#Eval for threshold 0.80: DER 13.64%, MS 13.02%, FA 0.37%, SC 0.24%
## Test set
#Model DER:  0.14458976437820859
#Model ACC:  0.9461460578900024
#100%|██████████| 60/60 [00:31<00:00,  1.92it/s]
#Eval for threshold 0.20: DER 11.82%, MS 1.34%, FA 9.70%, SC 0.78%
#
#Eval for threshold 0.30: DER 8.64%, MS 2.24%, FA 5.33%, SC 1.07%
#
#Eval for threshold 0.35: DER 7.90%, MS 2.76%, FA 3.99%, SC 1.15%
#
#Eval for threshold 0.40: DER 7.45%, MS 3.30%, FA 2.94%, SC 1.21%
#
#Eval for threshold 0.45: DER 7.35%, MS 3.98%, FA 2.11%, SC 1.26%
#
#Eval for threshold 0.50: DER 7.45%, MS 4.73%, FA 1.49%, SC 1.23%
#
#Eval for threshold 0.55: DER 7.82%, MS 5.65%, FA 1.04%, SC 1.13%
#
#Eval for threshold 0.60: DER 8.43%, MS 6.73%, FA 0.73%, SC 0.97%
#
#Eval for threshold 0.70: DER 10.22%, MS 9.20%, FA 0.35%, SC 0.68%
#
#Eval for threshold 0.80: DER 13.27%, MS 12.71%, FA 0.16%, SC 0.40%
fi
# compared with stage 65, I will load stage best checkpooint and finetune it without freeze encoder of wavlm
if [ ${stage} -le  75 ] && [ ${stop_stage} -ge 75 ];then

    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is WavLM (only using cnn frontend and first 6 layers transformer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this WavLM is trained 60khours libri-light and 10k hours gigaspeech and 24k hours VoxPopuli,
    # checkpoint is from https://github.com/microsoft/unilm/tree/master/wavlm
    # how to look for port ?
    # netstat -tuln

    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES

    # for loading pretrain model weigt
    speech_encoder_type="WavLM_weight_sum"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Large.pt"
    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
    dest_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wavlm_large_weight_sum_epoch40
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wavlm_large_weight_sum_epoch40_contine_ft
    mkdir -p $exp_dir
    cp -r $dest_dir/best-valid-der.pt $exp_dir/
    mv $exp_dir/best-valid-der.pt $exp_dir/epoch-0.pt
    #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  accelerate launch --main_process_port 12885 \
   ts_vad2/train_accelerate_ddp2.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --freeze-updates 0\
    --grad-clip false\
    --lr 1e-5\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --do-finetune true\
    --finetune-ckpt $exp_dir/epoch-0.pt\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --train-on-average false\
    --exp-dir $exp_dir
fi
if [ ${stage} -le 76 ] && [ ${stop_stage} -ge 76 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wavlm_large_weight_sum_epoch40_contine_ft
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="WavLM_weight_sum"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Large.pt"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

 for name in $infer_sets;do
    results_path=$exp_dir
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
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
     --data-dir $data_dir
done
#Eval set
#Model DER:  0.13729885636098363
#Model ACC:  0.952242195939484
#100%|██████████| 25/25 [00:14<00:00,  1.69it/s]
#Eval for threshold 0.20: DER 9.55%, MS 1.04%, FA 8.10%, SC 0.42%
#
#Eval for threshold 0.30: DER 6.86%, MS 1.64%, FA 4.68%, SC 0.53%
#
#Eval for threshold 0.35: DER 6.18%, MS 2.00%, FA 3.57%, SC 0.61%
#
#Eval for threshold 0.40: DER 5.75%, MS 2.38%, FA 2.69%, SC 0.68%
#
#Eval for threshold 0.45: DER 5.59%, MS 2.83%, FA 2.03%, SC 0.73%
#
#Eval for threshold 0.50: DER 5.72%, MS 3.49%, FA 1.47%, SC 0.76%
#
#Eval for threshold 0.55: DER 6.01%, MS 4.19%, FA 1.14%, SC 0.68%
#
#Eval for threshold 0.60: DER 6.52%, MS 5.07%, FA 0.88%, SC 0.57%
#
#Eval for threshold 0.70: DER 8.05%, MS 7.16%, FA 0.57%, SC 0.32%
#
#Eval for threshold 0.80: DER 10.67%, MS 10.10%, FA 0.42%, SC 0.16%
#Test set
#Model DER:  0.1320156284307031
#Model ACC:  0.9503373141620844
#100%|██████████| 60/60 [00:36<00:00,  1.66it/s]
#Eval for threshold 0.20: DER 9.79%, MS 1.13%, FA 7.90%, SC 0.76%
#
#Eval for threshold 0.30: DER 7.50%, MS 1.81%, FA 4.61%, SC 1.08%
#
#Eval for threshold 0.35: DER 6.98%, MS 2.21%, FA 3.56%, SC 1.20%
#
#Eval for threshold 0.40: DER 6.65%, MS 2.70%, FA 2.66%, SC 1.30%
#
#Eval for threshold 0.45: DER 6.56%, MS 3.24%, FA 1.93%, SC 1.39%
#
#Eval for threshold 0.50: DER 6.71%, MS 3.90%, FA 1.37%, SC 1.44%
#
#Eval for threshold 0.55: DER 7.05%, MS 4.73%, FA 0.99%, SC 1.34%
#
#Eval for threshold 0.60: DER 7.56%, MS 5.62%, FA 0.70%, SC 1.24%
#
#Eval for threshold 0.70: DER 9.04%, MS 7.78%, FA 0.34%, SC 0.92%
#
#Eval for threshold 0.80: DER 11.51%, MS 10.77%, FA 0.17%, SC 0.58%
fi

# (todo)
# compared with stage 40(util now(2024/9/14, it is sota),
# #I move fixed_random_seed to the training entrance(main()) from train_one_epoch()  to ensure repeatable training.
if [ ${stage} -le 77 ] && [ ${stop_stage} -ge 77 ];then

    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is WavLM(i.e. wavlm_large) (only using cnn frontend and first 6 layers transformer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this WavLM is trained 60khours libri-light and 10k hours gigaspeech and 24k hours VoxPopuli,
    # checkpoint is from https://github.com/microsoft/unilm/tree/master/wavlm
    # how to look for port ?
    # netstat -tuln
   export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    speech_encoder_type="WavLM"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Large.pt"
    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wavlm_large_epoch40_front_fix_seed
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wavlm_large_epoch40_front_fix_seed_again
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  accelerate launch --main_process_port 12985 \
   ts_vad2/train_accelerate_ddp2.py \
    --world-size 2 \
    --num-epochs 40\
    --start-epoch 1\
    --freeze-updates 4000\
    --grad-clip true\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --select-encoder-layer-nums 6\
    --exp-dir $exp_dir
fi
if [ ${stage} -le 78 ] && [ ${stop_stage} -ge 78 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wavlm_large_epoch40_front_fix_seed
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="WavLM"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Large.pt"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

 for name in $infer_sets;do
    results_path=$exp_dir
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
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --select-encoder-layer-nums 6\
    --data-dir $data_dir
done
#Eval set
#Model DER:  0.12804696686150424
#Model ACC:  0.9560720149702808
#100%|██████████| 25/25 [00:11<00:00,  2.09it/s]
#Eval for threshold 0.20: DER 8.26%, MS 0.94%, FA 6.96%, SC 0.36%
#
#Eval for threshold 0.30: DER 5.88%, MS 1.40%, FA 4.02%, SC 0.46%
#
#Eval for threshold 0.35: DER 5.25%, MS 1.66%, FA 3.09%, SC 0.50%
#
#Eval for threshold 0.40: DER 4.88%, MS 1.95%, FA 2.39%, SC 0.53%
#
#Eval for threshold 0.45: DER 4.77%, MS 2.30%, FA 1.92%, SC 0.56% as report
#
#Eval for threshold 0.50: DER 4.81%, MS 2.75%, FA 1.54%, SC 0.52%
#
#Eval for threshold 0.55: DER 4.95%, MS 3.21%, FA 1.23%, SC 0.51%
#
#Eval for threshold 0.60: DER 5.25%, MS 3.81%, FA 0.97%, SC 0.47%
#
#Eval for threshold 0.70: DER 6.20%, MS 5.21%, FA 0.65%, SC 0.34%
#
#Eval for threshold 0.80: DER 8.31%, MS 7.63%, FA 0.46%, SC 0.22%
#Test set
#Model DER:  0.11750170965157429
#Model ACC:  0.9574915917894956
#100%|██████████| 60/60 [00:28<00:00,  2.09it/s]
#Eval for threshold 0.20: DER 8.06%, MS 1.07%, FA 6.62%, SC 0.37%
#
#Eval for threshold 0.30: DER 5.87%, MS 1.65%, FA 3.77%, SC 0.46%
#
#Eval for threshold 0.35: DER 5.35%, MS 1.95%, FA 2.90%, SC 0.50%
#
#Eval for threshold 0.40: DER 5.02%, MS 2.26%, FA 2.23%, SC 0.53%
#
#Eval for threshold 0.45: DER 4.93%, MS 2.65%, FA 1.71%, SC 0.57% as report
#
#Eval for threshold 0.50: DER 4.99%, MS 3.13%, FA 1.29%, SC 0.57%
#
#Eval for threshold 0.55: DER 5.19%, MS 3.66%, FA 0.99%, SC 0.53%
#
#Eval for threshold 0.60: DER 5.53%, MS 4.29%, FA 0.74%, SC 0.51%
#
#Eval for threshold 0.70: DER 6.71%, MS 5.95%, FA 0.39%, SC 0.37%
#
#Eval for threshold 0.80: DER 8.69%, MS 8.24%, FA 0.19%, SC 0.25%

fi
# compared with stage 77, I will use first 12 transformer layer of wavlm in tsvad model.
if [ ${stage} -le 79 ] && [ ${stop_stage} -ge 79 ];then

    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is WavLM(i.e. wavlm_large) (only using cnn frontend and first 12 layers transformer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this WavLM is trained 60khours libri-light and 10k hours gigaspeech and 24k hours VoxPopuli,
    # checkpoint is from https://github.com/microsoft/unilm/tree/master/wavlm
    # how to look for port ?
    # netstat -tuln
   export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    speech_encoder_type="WavLM"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Large.pt"
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wavlm_large_epoch40_front_fix_seed_12layers
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  accelerate launch --main_process_port 12785 \
   ts_vad2/train_accelerate_ddp2.py \
    --world-size 2 \
    --num-epochs 40\
    --start-epoch 1\
    --freeze-updates 4000\
    --grad-clip true\
    --select-encoder-layer-nums 12\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --exp-dir $exp_dir
fi

if [ ${stage} -le 80 ] && [ ${stop_stage} -ge 80 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wavlm_large_epoch40_front_fix_seed_12layers
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="WavLM"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Large.pt"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

 for name in $infer_sets;do
    results_path=$exp_dir
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
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --select-encoder-layer-nums 12\
    --data-dir $data_dir
done
# Eval set
# Model DER:  0.12481599727276411
#Model ACC:  0.9573069769409085
#100%|██████████| 25/25 [00:15<00:00,  1.66it/s]
#Eval for threshold 0.20: DER 7.85%, MS 0.92%, FA 6.62%, SC 0.31%
#
#Eval for threshold 0.30: DER 5.61%, MS 1.41%, FA 3.81%, SC 0.39%
#
#Eval for threshold 0.35: DER 5.07%, MS 1.69%, FA 2.95%, SC 0.43%
#
#Eval for threshold 0.40: DER 4.80%, MS 2.01%, FA 2.36%, SC 0.43%
#
#Eval for threshold 0.45: DER 4.63%, MS 2.34%, FA 1.84%, SC 0.44%
#
#Eval for threshold 0.50: DER 4.65%, MS 2.75%, FA 1.45%, SC 0.45%
#
#Eval for threshold 0.55: DER 4.86%, MS 3.27%, FA 1.16%, SC 0.43%
#
#Eval for threshold 0.60: DER 5.13%, MS 3.78%, FA 0.96%, SC 0.39%
#
#Eval for threshold 0.70: DER 6.16%, MS 5.19%, FA 0.64%, SC 0.33%
#
#Eval for threshold 0.80: DER 8.10%, MS 7.46%, FA 0.44%, SC 0.20%
# Test set
#Model DER:  0.12199758287725823
#Model ACC:  0.9549705089387199
#100%|██████████| 60/60 [00:37<00:00,  1.58it/s]
#Eval for threshold 0.20: DER 8.53%, MS 0.97%, FA 6.89%, SC 0.67%
#
#Eval for threshold 0.30: DER 6.41%, MS 1.54%, FA 4.00%, SC 0.87%
#
#Eval for threshold 0.35: DER 5.94%, MS 1.90%, FA 3.10%, SC 0.94%
#
#Eval for threshold 0.40: DER 5.65%, MS 2.28%, FA 2.37%, SC 0.99%
#
#Eval for threshold 0.45: DER 5.53%, MS 2.71%, FA 1.78%, SC 1.05%
#
#Eval for threshold 0.50: DER 5.61%, MS 3.25%, FA 1.33%, SC 1.03%
#
#Eval for threshold 0.55: DER 5.88%, MS 3.91%, FA 1.02%, SC 0.95%
#
#Eval for threshold 0.60: DER 6.23%, MS 4.61%, FA 0.78%, SC 0.84%
#
#Eval for threshold 0.70: DER 7.41%, MS 6.35%, FA 0.41%, SC 0.65%
#
#Eval for threshold 0.80: DER 9.65%, MS 8.97%, FA 0.23%, SC 0.45%
fi

# compared with stage77, I will more better lr than fixed max_update=40k.
if [ ${stage} -le 81 ] && [ ${stop_stage} -ge 81 ];then

    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is WavLM(i.e. wavlm_large) (only using cnn frontend and first 6 layers transformer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this WavLM is trained 60khours libri-light and 10k hours gigaspeech and 24k hours VoxPopuli,
    # checkpoint is from https://github.com/microsoft/unilm/tree/master/wavlm
    # how to look for port ?
    # netstat -tuln
  export NCCL_DEBUG=INFO
  export PYTHONFAULTHANDLER=1
  musan_path=/mntcephfs/lee_dataset/asr/musan
  rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
  speech_encoder_type="WavLM"
  speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Large.pt"
  exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wavlm_large_epoch40_front_fix_seed_with_nums_steps

  # for loading speaker embedding file
  spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
  speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
  data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
  #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \

  CUDA_VISIABLE_DEVICES=0,1 \
  accelerate launch --main_process_port 12785 \
   ts_vad2/train_accelerate_ddp2_with_num_steps.py \
    --world-size 2 \
    --num-epochs 40\
    --start-epoch 1\
    --freeze-updates 4000\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --grad-clip true\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --select-encoder-layer-nums 6\
    --exp-dir $exp_dir\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --data-dir $data_dir
fi

if [ ${stage} -le 82 ] && [ ${stop_stage} -ge 82 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wavlm_large_epoch40_front_fix_seed_with_nums_steps
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="WavLM"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Large.pt"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

 for name in $infer_sets;do
    results_path=$exp_dir
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
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --select-encoder-layer-nums 6\
    --data-dir $data_dir
done
# Eval set
#Model DER:  0.13140031794694756
#Model ACC:  0.9539099822176674
#100%|██████████| 25/25 [00:12<00:00,  1.96it/s]
#Eval for threshold 0.20: DER 8.03%, MS 1.13%, FA 6.31%, SC 0.58%
#
#Eval for threshold 0.30: DER 6.11%, MS 1.71%, FA 3.69%, SC 0.71%
#
#Eval for threshold 0.35: DER 5.60%, MS 1.99%, FA 2.85%, SC 0.76%
#
#Eval for threshold 0.40: DER 5.41%, MS 2.37%, FA 2.26%, SC 0.79%
#
#Eval for threshold 0.45: DER 5.28%, MS 2.72%, FA 1.75%, SC 0.81%
#
#Eval for threshold 0.50: DER 5.41%, MS 3.25%, FA 1.40%, SC 0.77%
#
#Eval for threshold 0.55: DER 5.65%, MS 3.85%, FA 1.07%, SC 0.73%
#
#Eval for threshold 0.60: DER 6.08%, MS 4.53%, FA 0.89%, SC 0.65%
#
#Eval for threshold 0.70: DER 7.36%, MS 6.28%, FA 0.59%, SC 0.48%
#
#Eval for threshold 0.80: DER 9.55%, MS 8.81%, FA 0.45%, SC 0.30%
#
#Test set
#Model DER:  0.12817452423061232
#Model ACC:  0.9520175775453524
#100%|██████████| 60/60 [00:30<00:00,  1.95it/s]
#Eval for threshold 0.20: DER 8.75%, MS 1.29%, FA 6.67%, SC 0.79%
#
#Eval for threshold 0.30: DER 6.78%, MS 1.94%, FA 3.79%, SC 1.05%
#
#Eval for threshold 0.35: DER 6.34%, MS 2.34%, FA 2.86%, SC 1.14%
#
#Eval for threshold 0.40: DER 6.11%, MS 2.75%, FA 2.14%, SC 1.22%
#
#Eval for threshold 0.45: DER 6.05%, MS 3.21%, FA 1.58%, SC 1.25%
#
#Eval for threshold 0.50: DER 6.23%, MS 3.84%, FA 1.16%, SC 1.22%
#
#Eval for threshold 0.55: DER 6.62%, MS 4.60%, FA 0.92%, SC 1.10%
#
#Eval for threshold 0.60: DER 7.11%, MS 5.42%, FA 0.69%, SC 1.00%
#
#Eval for threshold 0.70: DER 8.57%, MS 7.44%, FA 0.37%, SC 0.76%
#
#Eval for threshold 0.80: DER 10.92%, MS 10.23%, FA 0.18%, SC 0.51%

fi

# extract target speaker embedding using samresnet checkpoint from wespeaker
if [ ${stage} -le 83 ] && [ ${stop_stage} -ge 83 ];then
   echo "generate eval(dev) speaker embedding"
   feature_name=simam_resnet34_on_multilingual_and_ft_on_voxceleb_wespeaker_feature_dir
   dest_dir=/mntcephfs/lab_data/maduo/model_hub/
   pretrained_model=/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/multilingual/wespeaker/voxblink2_samresnet34_ft/avg_model.pt
   model_name="SimAM_ResNet34_ASP"
   subsets="Eval Test Train"
   for name in $subsets;do
    if [ $name = "Train" ];then
     echo "extract train target speaker embedding"
     # 提取embedding
     input_dir=/mntcephfs/lab_data/maduo/datasets/alimeeting/${name}_Ali_far/target_audio/
     wav_path=$input_dir/wavs.txt
    else
        echo "extract $name target speaker embedding"
        # 提取embedding
     input_dir=/mntcephfs/lab_data/maduo/datasets/alimeeting/${name}_Ali/${name}_Ali_far/target_audio/
     wav_path=$input_dir/wavs.txt
    fi
    save_dir=$dest_dir/ts_vad/spk_embed/alimeeting/SpeakerEmbedding/$name/$feature_name
    python3 ts_vad2/generate_chunk_speaker_embedding_from_wespeaker_for_diarization.py\
           --pretrained_model $pretrained_model\
           --model_name $model_name\
           --wavs $wav_path\
           --save_dir $save_dir
   done
fi


# compared with stage77, I use simam_resnet34_ft checkpoint to get speaker embedding.
if [ ${stage} -le 84 ] && [ ${stop_stage} -ge 84 ];then

    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is WavLM(i.e. wavlm_large) (only using cnn frontend and first 6 layers transformer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this WavLM is trained 60khours libri-light and 10k hours gigaspeech and 24k hours VoxPopuli,
    # checkpoint is from https://github.com/microsoft/unilm/tree/master/wavlm
    # how to look for port ?
    # netstat -tuln
  export NCCL_DEBUG=INFO
  export PYTHONFAULTHANDLER=1
  musan_path=/mntcephfs/lee_dataset/asr/musan
  rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
  speech_encoder_type="WavLM"
  speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Large.pt"
  exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wavlm_large_epoch40_front_fix_seed_with_simam_resnet34_ft

   # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="simam_resnet34_on_multilingual_and_ft_on_voxceleb_wespeaker_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
 speaker_embed_dim=256
 #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \

   #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  accelerate launch --main_process_port 12885 \
   ts_vad2/train_accelerate_ddp2.py \
    --world-size 2 \
    --num-epochs 40\
    --start-epoch 1\
    --freeze-updates 4000\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --grad-clip true\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --select-encoder-layer-nums 6\
    --exp-dir $exp_dir\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --data-dir $data_dir\
    --speaker-embed-dim $speaker_embed_dim
fi


if [ ${stage} -le 85 ] && [ ${stop_stage} -ge 85 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wavlm_large_epoch40_front_fix_seed_with_simam_resnet34_ft
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="WavLM"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Large.pt"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="simam_resnet34_on_multilingual_and_ft_on_voxceleb_wespeaker_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
 speaker_embed_dim=256
 for name in $infer_sets;do
    results_path=$exp_dir
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
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --select-encoder-layer-nums 6\
    --speaker-embed-dim $speaker_embed_dim\
    --data-dir $data_dir
done
# cat logs/run_ts_vad2_stage84-85.log
# Eval set
# Model DER:  0.13917119020703766
#Model ACC:  0.9497854888844821
#100%|██████████| 25/25 [00:16<00:00,  1.55it/s]
#Eval for threshold 0.20: DER 9.93%, MS 0.91%, FA 8.39%, SC 0.62%
#
#Eval for threshold 0.30: DER 7.07%, MS 1.39%, FA 4.89%, SC 0.79%
#
#Eval for threshold 0.35: DER 6.44%, MS 1.69%, FA 3.86%, SC 0.89%
#
#Eval for threshold 0.40: DER 5.97%, MS 2.00%, FA 3.02%, SC 0.96%
#
#Eval for threshold 0.45: DER 5.70%, MS 2.36%, FA 2.28%, SC 1.07%
#
#Eval for threshold 0.50: DER 5.64%, MS 2.82%, FA 1.74%, SC 1.08%
#
#Eval for threshold 0.55: DER 5.86%, MS 3.46%, FA 1.34%, SC 1.06%
#
#Eval for threshold 0.60: DER 6.15%, MS 4.10%, FA 1.04%, SC 1.02%
#
#Eval for threshold 0.70: DER 7.39%, MS 5.89%, FA 0.73%, SC 0.77%
#
#Eval for threshold 0.80: DER 9.77%, MS 8.75%, FA 0.49%, SC 0.54%
## Test set
#Model DER:  0.13868180352644086
#Model ACC:  0.9446762738146846
#100%|██████████| 60/60 [00:38<00:00,  1.54it/s]
#Eval for threshold 0.20: DER 10.52%, MS 1.04%, FA 7.61%, SC 1.88%
#
#Eval for threshold 0.30: DER 8.36%, MS 1.63%, FA 4.41%, SC 2.31%
#
#Eval for threshold 0.35: DER 7.85%, MS 1.97%, FA 3.42%, SC 2.46%
#
#Eval for threshold 0.40: DER 7.51%, MS 2.35%, FA 2.58%, SC 2.58%
#
#Eval for threshold 0.45: DER 7.34%, MS 2.82%, FA 1.83%, SC 2.70%
#
#Eval for threshold 0.50: DER 7.47%, MS 3.48%, FA 1.26%, SC 2.73%
#
#Eval for threshold 0.55: DER 7.78%, MS 4.25%, FA 0.93%, SC 2.61%
#
#Eval for threshold 0.60: DER 8.26%, MS 5.11%, FA 0.68%, SC 2.46%
#
#Eval for threshold 0.70: DER 9.67%, MS 7.19%, FA 0.36%, SC 2.12%
#
#Eval for threshold 0.80: DER 12.16%, MS 10.37%, FA 0.19%, SC 1.61%

fi
# extract target speaker embedding using samresnet checkpoint(without voxceleb finetune) from wespeaker
if [ ${stage} -le 86 ] && [ ${stop_stage} -ge 86 ];then
   echo "generate eval(dev) speaker embedding"
   feature_name=simam_resnet34_on_multilingual_wespeaker_feature_dir
   dest_dir=/mntcephfs/lab_data/maduo/model_hub/
   pretrained_model=/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/multilingual/wespeaker/voxblink2_samresnet34/avg_model.pt
   model_name="SimAM_ResNet34_ASP"
   subsets="Eval Test Train"
   for name in $subsets;do
    if [ $name = "Train" ];then
     echo "extract train target speaker embedding"
     # 提取embedding
     input_dir=/mntcephfs/lab_data/maduo/datasets/alimeeting/${name}_Ali_far/target_audio/
     wav_path=$input_dir/wavs.txt
    else
        echo "extract $name target speaker embedding"
        # 提取embedding
     input_dir=/mntcephfs/lab_data/maduo/datasets/alimeeting/${name}_Ali/${name}_Ali_far/target_audio/
     wav_path=$input_dir/wavs.txt
    fi
    save_dir=$dest_dir/ts_vad/spk_embed/alimeeting/SpeakerEmbedding/$name/$feature_name
    python3 ts_vad2/generate_chunk_speaker_embedding_from_wespeaker_for_diarization.py\
           --pretrained_model $pretrained_model\
           --model_name $model_name\
           --wavs $wav_path\
           --save_dir $save_dir
   done
fi

# compared with stage84, I use simam_resnet34 checkpoint to get speaker embedding.
if [ ${stage} -le 87 ] && [ ${stop_stage} -ge 87 ];then

    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is WavLM(i.e. wavlm_large) (only using cnn frontend and first 6 layers transformer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this WavLM is trained 60khours libri-light and 10k hours gigaspeech and 24k hours VoxPopuli,
    # checkpoint is from https://github.com/microsoft/unilm/tree/master/wavlm
    # how to look for port ?
    # netstat -tuln
  export NCCL_DEBUG=INFO
  export PYTHONFAULTHANDLER=1
  musan_path=/mntcephfs/lee_dataset/asr/musan
  rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
  speech_encoder_type="WavLM"
  speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Large.pt"
  exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wavlm_large_epoch40_front_fix_seed_with_simam_resnet34

   # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="simam_resnet34_on_multilingual_wespeaker_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
 speaker_embed_dim=256
 #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \

   #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  accelerate launch --main_process_port 12885 \
   ts_vad2/train_accelerate_ddp2.py \
    --world-size 2 \
    --num-epochs 40\
    --start-epoch 1\
    --freeze-updates 4000\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --grad-clip true\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --select-encoder-layer-nums 6\
    --exp-dir $exp_dir\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --data-dir $data_dir\
    --speaker-embed-dim $speaker_embed_dim
fi

if [ ${stage} -le 88 ] && [ ${stop_stage} -ge 88 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wavlm_large_epoch40_front_fix_seed_with_simam_resnet34
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="WavLM"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Large.pt"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="simam_resnet34_on_multilingual_and_ft_on_voxceleb_wespeaker_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
 speaker_embed_dim=256
 for name in $infer_sets;do
    results_path=$exp_dir
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
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --spk-path $spk_path\
     --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --select-encoder-layer-nums 6\
    --speaker-embed-dim $speaker_embed_dim\
    --data-dir $data_dir
done
fi
# Eval set
# Model DER:  0.4101110779746862
#Model ACC:  0.8660732893205443
#100%|██████████| 25/25 [00:12<00:00,  1.95it/s]
#Eval for threshold 0.20: DER 28.62%, MS 21.28%, FA 4.66%, SC 2.68%
#
#Eval for threshold 0.30: DER 30.37%, MS 24.85%, FA 2.79%, SC 2.73%
#
#Eval for threshold 0.35: DER 31.47%, MS 26.56%, FA 2.13%, SC 2.77%
#
#Eval for threshold 0.40: DER 32.61%, MS 28.21%, FA 1.62%, SC 2.78%
#
#Eval for threshold 0.45: DER 33.89%, MS 29.93%, FA 1.22%, SC 2.74%
#
#Eval for threshold 0.50: DER 35.44%, MS 31.85%, FA 0.95%, SC 2.64%
#
#Eval for threshold 0.55: DER 37.05%, MS 33.81%, FA 0.72%, SC 2.51%
#
#Eval for threshold 0.60: DER 38.79%, MS 35.91%, FA 0.57%, SC 2.32%
#
#Eval for threshold 0.70: DER 42.64%, MS 40.35%, FA 0.39%, SC 1.90%
#
#Eval for threshold 0.80: DER 47.73%, MS 46.06%, FA 0.30%, SC 1.37%
#Test set
#Model DER:  0.5764384459282696
#Model ACC:  0.8213774607251713
#100%|██████████| 60/60 [00:39<00:00,  1.53it/s]
#Eval for threshold 0.20: DER 46.50%, MS 40.10%, FA 3.97%, SC 2.43%
#
#Eval for threshold 0.30: DER 49.28%, MS 44.55%, FA 2.12%, SC 2.60%
#
#Eval for threshold 0.35: DER 50.66%, MS 46.52%, FA 1.52%, SC 2.62%
#
#Eval for threshold 0.40: DER 52.03%, MS 48.36%, FA 1.05%, SC 2.61%
#
##Eval for threshold 0.45: DER 53.46%, MS 50.25%, FA 0.67%, SC 2.54%
#
#Eval for threshold 0.50: DER 55.06%, MS 52.24%, FA 0.43%, SC 2.40%
#
#Eval for threshold 0.55: DER 56.68%, MS 54.19%, FA 0.28%, SC 2.22%
#
#Eval for threshold 0.60: DER 58.40%, MS 56.19%, FA 0.19%, SC 2.01%
#
#Eval for threshold 0.70: DER 62.12%, MS 60.51%, FA 0.06%, SC 1.55%
#
#Eval for threshold 0.80: DER 66.15%, MS 65.12%, FA 0.02%, SC 1.01%


# compared with stage65, I will apply layenorm on the weight sum of all transformer layer feature in pretrain wavlm model
if [ ${stage} -le 90 ] && [ ${stop_stage} -ge 90 ];then

    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    #  oracle target speaker embedding is from cam++ pretrain model
    # this WavLM is trained 60khours libri-light and 10k hours gigaspeech and 24k hours VoxPopuli,
    # checkpoint is from https://github.com/microsoft/unilm/tree/master/wavlm
    # how to look for port ?
    # netstat -tuln
   export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES

    # for loading pretrain model weigt
    speech_encoder_type="WavLM_weight_sum"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Large.pt"
    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"

    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wavlm_large_weight_sum_layernorm_epoch40
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  accelerate launch --main_process_port 12785 \
   ts_vad2/train_accelerate_ddp2.py \
    --world-size 2 \
    --num-epochs 40\
    --start-epoch 1\
    --freeze-updates 40000\
    --grad-clip true\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --train-on-average false\
    --wavlm-fuse-feat-post-norm true \
    --exp-dir $exp_dir
fi
if [ ${stage} -le 91 ] && [ ${stop_stage} -ge 91 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wavlm_large_weight_sum_layernorm_epoch40
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="WavLM_weight_sum"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Large.pt"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

 for name in $infer_sets;do
    results_path=$exp_dir
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
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm true \
    --data-dir $data_dir
done
fi
# Eval set
#Model DER:  0.15236881400811017
#Model ACC:  0.9465887123352167
#100%|██████████| 25/25 [00:15<00:00,  1.58it/s]
#Eval for threshold 0.20: DER 12.10%, MS 1.16%, FA 10.42%, SC 0.52%
#
#Eval for threshold 0.30: DER 8.44%, MS 1.92%, FA 5.79%, SC 0.72%
#
#Eval for threshold 0.35: DER 7.55%, MS 2.36%, FA 4.32%, SC 0.87%
#
#Eval for threshold 0.40: DER 7.02%, MS 2.85%, FA 3.21%, SC 0.95%
#
#Eval for threshold 0.45: DER 6.81%, MS 3.39%, FA 2.40%, SC 1.02%
#
#Eval for threshold 0.50: DER 6.82%, MS 4.11%, FA 1.67%, SC 1.04%
#
#Eval for threshold 0.55: DER 7.22%, MS 5.05%, FA 1.26%, SC 0.91%
#
#Eval for threshold 0.60: DER 7.74%, MS 6.02%, FA 0.94%, SC 0.78%
#
#Eval for threshold 0.70: DER 9.65%, MS 8.63%, FA 0.55%, SC 0.48%
#
#Eval for threshold 0.80: DER 12.71%, MS 12.11%, FA 0.39%, SC 0.22%
#
# Test set
#Model DER:  0.14538464456685316
#Model ACC:  0.9455950203350078
#100%|██████████| 60/60 [00:37<00:00,  1.58it/s]
#Eval for threshold 0.20: DER 12.35%, MS 1.23%, FA 10.37%, SC 0.75%
#
#Eval for threshold 0.30: DER 8.89%, MS 2.05%, FA 5.83%, SC 1.01%
#
#Eval for threshold 0.35: DER 8.05%, MS 2.53%, FA 4.39%, SC 1.13%
#
#Eval for threshold 0.40: DER 7.55%, MS 3.06%, FA 3.25%, SC 1.23%
#
#Eval for threshold 0.45: DER 7.37%, MS 3.68%, FA 2.36%, SC 1.33%
#
#Eval for threshold 0.50: DER 7.46%, MS 4.38%, FA 1.71%, SC 1.37%
#
#Eval for threshold 0.55: DER 7.82%, MS 5.29%, FA 1.20%, SC 1.33%
#
#Eval for threshold 0.60: DER 8.43%, MS 6.33%, FA 0.86%, SC 1.24%
#
#Eval for threshold 0.70: DER 10.26%, MS 8.92%, FA 0.41%, SC 0.93%
#
#Eval for threshold 0.80: DER 13.31%, MS 12.53%, FA 0.17%, SC 0.60%
if [ ${stage} -le  92 ] && [ ${stop_stage} -ge 92 ];then

    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is WavLM (only using cnn frontend and first 6 layers transformer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this WavLM is trained 60khours libri-light and 10k hours gigaspeech and 24k hours VoxPopuli,
    # checkpoint is from https://github.com/microsoft/unilm/tree/master/wavlm
    # how to look for port ?
    # netstat -tuln

    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES

    # for loading pretrain model weigt
    speech_encoder_type="WavLM_weight_sum"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Large.pt"
    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
    dest_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wavlm_large_weight_sum_layernorm_epoch40
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wavlm_large_weight_sum_layernorm_epoch40_contine_ft
    mkdir -p $exp_dir
    cp -r $dest_dir/best-valid-der.pt $exp_dir/
    mv $exp_dir/best-valid-der.pt $exp_dir/epoch-0.pt
    #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  accelerate launch --main_process_port 12785 \
   ts_vad2/train_accelerate_ddp2.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --freeze-updates 0\
    --grad-clip false\
    --lr 1e-5\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --do-finetune true\
    --finetune-ckpt $exp_dir/epoch-0.pt\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --train-on-average false\
    --wavlm-fuse-feat-post-norm true \
    --exp-dir $exp_dir
fi
if [ ${stage} -le 93 ] && [ ${stop_stage} -ge 93 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wavlm_large_weight_sum_layernorm_epoch40_contine_ft
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="WavLM_weight_sum"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Large.pt"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

 for name in $infer_sets;do
    results_path=$exp_dir
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
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm true \
    --data-dir $data_dir
done
fi
# Eval set
#Model DER:  0.13667159449237004
#Model ACC:  0.9525711127973927
#100%|██████████| 25/25 [00:17<00:00,  1.41it/s]
#Eval for threshold 0.20: DER 9.62%, MS 1.09%, FA 8.01%, SC 0.52%
#
#Eval for threshold 0.30: DER 6.91%, MS 1.73%, FA 4.51%, SC 0.67%
#
#Eval for threshold 0.35: DER 6.22%, MS 2.07%, FA 3.42%, SC 0.73%
#
#Eval for threshold 0.40: DER 5.79%, MS 2.43%, FA 2.56%, SC 0.80%
#
#Eval for threshold 0.45: DER 5.67%, MS 2.87%, FA 1.96%, SC 0.84%
#
#Eval for threshold 0.50: DER 5.68%, MS 3.36%, FA 1.51%, SC 0.81%
#
#Eval for threshold 0.55: DER 5.87%, MS 3.99%, FA 1.14%, SC 0.74%
#
#Eval for threshold 0.60: DER 6.24%, MS 4.65%, FA 0.91%, SC 0.68%
#
#Eval for threshold 0.70: DER 7.57%, MS 6.50%, FA 0.60%, SC 0.47%
#
#Eval for threshold 0.80: DER 10.10%, MS 9.41%, FA 0.41%, SC 0.27%
#
# Test set
#Model DER:  0.1309318378824793
#Model ACC:  0.950629249149897
#100%|██████████| 60/60 [00:38<00:00,  1.56it/s]
#Eval for threshold 0.20: DER 9.79%, MS 1.07%, FA 7.92%, SC 0.80%
#
#Eval for threshold 0.30: DER 7.34%, MS 1.73%, FA 4.53%, SC 1.08%
#
#Eval for threshold 0.35: DER 6.79%, MS 2.16%, FA 3.44%, SC 1.19%
#
#Eval for threshold 0.40: DER 6.47%, MS 2.59%, FA 2.59%, SC 1.28%
#
#Eval for threshold 0.45: DER 6.33%, MS 3.05%, FA 1.92%, SC 1.36%
#
#Eval for threshold 0.50: DER 6.41%, MS 3.67%, FA 1.40%, SC 1.34%
#
#Eval for threshold 0.55: DER 6.78%, MS 4.42%, FA 1.06%, SC 1.29%
#
#Eval for threshold 0.60: DER 7.28%, MS 5.32%, FA 0.77%, SC 1.20%
#
#Eval for threshold 0.70: DER 8.71%, MS 7.34%, FA 0.39%, SC 0.97%
#
#Eval for threshold 0.80: DER 11.15%, MS 10.30%, FA 0.18%, SC 0.67%

# compared with stage 77(it is sota of wavlm method)
# I used more better ssl model(i.e. w2v-bert2.0, it is from sameless model)
## note: Both der and loss are best when epoch is equal to 10, after epoch=10, loss is nan, DER=1.
if [ ${stage} -le 95 ] && [ ${stop_stage} -ge 95 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is wav-bert2.0 (only using position embedding and first 6 layers conformer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this w2v-bert2.0 is trained Languages: 143+ , Size: 4.5M hours (it is from this paper https://arxiv.org/pdf/2312.05187)
    # checkpoint is from https://huggingface.co/facebook/w2v-bert-2.0/tree/main
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="w2v-bert2"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
    speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_w2v-bert2.0_first6layer_epoch40_front_fix_seed
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 12815 \
   ts_vad2/train_accelerate_ddp2_debug.py \
    --world-size 2 \
    --num-epochs 40\
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
    --speech-encoder-config $speech_encoder_config\
    --select-encoder-layer-nums 6\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --exp-dir $exp_dir
fi

if [ ${stage} -le 96 ] && [ ${stop_stage} -ge 96 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_w2v-bert2.0_first6layer_epoch40_front_fix_seed
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="w2v-bert2"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

 for name in $infer_sets;do
    results_path=$exp_dir
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
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --speech-encoder-config $speech_encoder_config\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir
done
#  cat logs/run_ts_vad2_stage96.log
## Eval set
## Model DER:  0.17737152230217448
#Model ACC:  0.9359282900355866
#100%|██████████| 25/25 [00:23<00:00,  1.05it/s]
#Eval for threshold 0.20: DER 17.49%, MS 0.98%, FA 15.82%, SC 0.69%
#
#Eval for threshold 0.30: DER 13.22%, MS 1.73%, FA 10.32%, SC 1.16%
#
#Eval for threshold 0.35: DER 11.90%, MS 2.24%, FA 8.19%, SC 1.47%
#
#Eval for threshold 0.40: DER 10.79%, MS 2.81%, FA 6.20%, SC 1.78%
#
#Eval for threshold 0.45: DER 10.10%, MS 3.62%, FA 4.55%, SC 1.94%
#
#Eval for threshold 0.50: DER 9.82%, MS 4.60%, FA 3.22%, SC 2.00% as report
#
#Eval for threshold 0.55: DER 10.13%, MS 5.98%, FA 2.30%, SC 1.85%
#
#Eval for threshold 0.60: DER 10.81%, MS 7.56%, FA 1.67%, SC 1.58%
#
#Eval for threshold 0.70: DER 12.95%, MS 11.14%, FA 0.81%, SC 0.99%
#
#Eval for threshold 0.80: DER 16.35%, MS 15.47%, FA 0.41%, SC 0.47%
## Test set
#Model DER:  0.16273131540209776
#Model ACC:  0.9399814543127758
#100%|██████████| 60/60 [00:58<00:00,  1.02it/s]
#Eval for threshold 0.20: DER 17.65%, MS 1.11%, FA 15.93%, SC 0.61%
#
#Eval for threshold 0.30: DER 12.58%, MS 1.95%, FA 9.66%, SC 0.97%
#
#Eval for threshold 0.35: DER 11.04%, MS 2.47%, FA 7.45%, SC 1.12%
#
#Eval for threshold 0.40: DER 9.98%, MS 3.11%, FA 5.63%, SC 1.24%
#
#Eval for threshold 0.45: DER 9.31%, MS 3.87%, FA 4.09%, SC 1.34%
#
#Eval for threshold 0.50: DER 9.13%, MS 4.86%, FA 2.93%, SC 1.34% as report
#
#Eval for threshold 0.55: DER 9.26%, MS 6.03%, FA 1.91%, SC 1.32%
#
#Eval for threshold 0.60: DER 9.90%, MS 7.47%, FA 1.25%, SC 1.17%
#
#Eval for threshold 0.70: DER 12.34%, MS 11.07%, FA 0.45%, SC 0.82%
#
#Eval for threshold 0.80: DER 16.85%, MS 16.29%, FA 0.15%, SC 0.41%

fi

## I don't run this experiment
if [ ${stage} -le 97 ] && [ ${stop_stage} -ge 97 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is w2v-bert2.0 (only using position embedding and first 6 layers conformer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this w2v-bert2.0 is trained Languages: 143+ , Size: 4.5M hours (it is from this paper https://arxiv.org/pdf/2312.05187)
    # checkpoint is from https://huggingface.co/facebook/w2v-bert-2.0/tree/main
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="w2v-bert2"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
    speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"

    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed_lr1e5
    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_w2v-bert2.0_first12layers_epoch40_front_fix_seed
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  accelerate launch --main_process_port 12785 \
   ts_vad2/train_accelerate_ddp2_debug.py \
    --world-size 2 \
    --num-epochs 40\
    --start-epoch 1\
    --keep-last-k 1\
    --keep-last-epoch 1\
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
    --exp-dir $exp_dir
fi

if [ ${stage} -le 98 ] && [ ${stop_stage} -ge 98 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed_lr1e5
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="w2v-bert2"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

 for name in $infer_sets;do
    results_path=$exp_dir
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
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --speech-encoder-config $speech_encoder_config\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir
done
fi

if [ ${stage} -le 100 ] && [ ${stop_stage} -ge 100 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is hubert large Chinese model (only cnn front and first 6 layers conformer) ,
    # this checkpoint is from https://huggingface.co/TencentGameMate/chinese-hubert-large/tree/main
    #  oracle target speaker embedding is from cam++ pretrain model

    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="hubert"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/Chinese/hubert/pytorch_model.bin"
    speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/Chinese/hubert/config.json"

    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_hubert_large_Chinese_first6layer_epoch40_front_fix_seed
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 12915 \
   ts_vad2/train_accelerate_ddp2_debug.py \
    --world-size 2 \
    --num-epochs 40\
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
    --speech-encoder-config $speech_encoder_config\
     --select-encoder-layer-nums 6\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --exp-dir $exp_dir
fi
if [ ${stage} -le 101 ] && [ ${stop_stage} -ge 101 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_hubert_large_Chinese_first6layer_epoch40_front_fix_seed/
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="hubert"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/Chinese/hubert/pytorch_model.bin"
 speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/Chinese/hubert/config.json"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

 for name in $infer_sets;do
    results_path=$exp_dir
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
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --speech-encoder-config $speech_encoder_config\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir
done
# sbatch --nodes 1 --gres=gpu:1  --cpus-per-gpu=8  --ntasks-per-node 1  -p p-V100 -A t00120220002 -o logs/run_ts_vad2_stage101.log run_ts_vad2.sh --stage 101 --stop-stage 101
#cat logs/run_ts_vad2_stage101.log
# Eval set
#Model DER:  0.13132554975427957
#Model ACC:  0.9552208505549664
#100%|██████████| 25/25 [00:23<00:00,  1.05it/s]
#Eval for threshold 0.20: DER 8.05%, MS 0.97%, FA 6.71%, SC 0.37%
#
#Eval for threshold 0.30: DER 5.95%, MS 1.56%, FA 3.95%, SC 0.44%
#
#Eval for threshold 0.35: DER 5.40%, MS 1.88%, FA 3.06%, SC 0.46%
#
#Eval for threshold 0.40: DER 5.13%, MS 2.23%, FA 2.44%, SC 0.46%
#
#Eval for threshold 0.45: DER 4.99%, MS 2.64%, FA 1.90%, SC 0.45% as report
#
#Eval for threshold 0.50: DER 5.02%, MS 3.09%, FA 1.50%, SC 0.44%
#
#Eval for threshold 0.55: DER 5.24%, MS 3.63%, FA 1.21%, SC 0.40%
#
#Eval for threshold 0.60: DER 5.64%, MS 4.31%, FA 0.97%, SC 0.35%
#
#Eval for threshold 0.70: DER 6.82%, MS 5.90%, FA 0.67%, SC 0.25%
#
#Eval for threshold 0.80: DER 8.72%, MS 8.09%, FA 0.47%, SC 0.16%
## Test set
#Model DER:  0.12673966865862538
#Model ACC:  0.9535556432571402
#100%|██████████| 60/60 [00:58<00:00,  1.03it/s]
#Eval for threshold 0.20: DER 8.54%, MS 1.11%, FA 6.77%, SC 0.67%
#
#Eval for threshold 0.30: DER 6.53%, MS 1.81%, FA 3.88%, SC 0.84%
#
#Eval for threshold 0.35: DER 6.08%, MS 2.18%, FA 2.99%, SC 0.91%
#
#Eval for threshold 0.40: DER 5.83%, MS 2.59%, FA 2.27%, SC 0.97%
#
#Eval for threshold 0.45: DER 5.82%, MS 3.13%, FA 1.71%, SC 0.98% as report
#
#Eval for threshold 0.50: DER 5.93%, MS 3.70%, FA 1.29%, SC 0.93%
#
#Eval for threshold 0.55: DER 6.19%, MS 4.38%, FA 0.97%, SC 0.84%
#
#Eval for threshold 0.60: DER 6.61%, MS 5.13%, FA 0.72%, SC 0.75%
#
#Eval for threshold 0.70: DER 7.87%, MS 6.95%, FA 0.37%, SC 0.56%
#
#Eval for threshold 0.80: DER 10.13%, MS 9.58%, FA 0.18%, SC 0.37%
fi

if [ ${stage} -le 102 ] && [ ${stop_stage} -ge 102 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is wav-bert2.0 (only using position embedding and first 6 layers conformer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this w2v-bert2.0 is trained Languages: 143+ , Size: 4.5M hours (it is from this paper https://arxiv.org/pdf/2312.05187)
    # checkpoint is from https://huggingface.co/facebook/w2v-bert-2.0/tree/main
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="w2v-bert2"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
    speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_w2v-bert2.0_first6layer_epoch40_front_fix_seed_lr2e5
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 12815 \
   ts_vad2/train_accelerate_ddp2_debug.py \
    --world-size 2 \
    --num-epochs 40\
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
    --exp-dir $exp_dir
fi

if [ ${stage} -le 103 ] && [ ${stop_stage} -ge 103 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_w2v-bert2.0_first6layer_epoch40_front_fix_seed_lr2e5
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="w2v-bert2"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

 for name in $infer_sets;do
    results_path=$exp_dir
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
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --speech-encoder-config $speech_encoder_config\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir
done
#cat logs/run_ts_vad2_stage102-103.log
#Eval set
#Model DER:  0.13836935899963873
#Model ACC:  0.9522087995615252
#100%|██████████| 25/25 [00:23<00:00,  1.04it/s]
#Eval for threshold 0.20: DER 9.32%, MS 0.95%, FA 7.94%, SC 0.42%
#
#Eval for threshold 0.30: DER 6.36%, MS 1.73%, FA 4.17%, SC 0.46%
#
#Eval for threshold 0.35: DER 5.74%, MS 2.15%, FA 3.08%, SC 0.50%
#
#Eval for threshold 0.40: DER 5.45%, MS 2.70%, FA 2.23%, SC 0.52%
#
#Eval for threshold 0.45: DER 5.51%, MS 3.39%, FA 1.61%, SC 0.50% as report
#
#Eval for threshold 0.50: DER 5.79%, MS 4.12%, FA 1.18%, SC 0.49%
#
#Eval for threshold 0.55: DER 6.39%, MS 5.09%, FA 0.90%, SC 0.40%
#
#Eval for threshold 0.60: DER 7.04%, MS 6.01%, FA 0.70%, SC 0.33%
#
#Eval for threshold 0.70: DER 9.28%, MS 8.63%, FA 0.45%, SC 0.20%
#
#Eval for threshold 0.80: DER 12.80%, MS 12.39%, FA 0.33%, SC 0.09%
#
#Test set
#Model DER:  0.1392326635549102
#Model ACC:  0.9493199249227983
#100%|██████████| 60/60 [00:58<00:00,  1.02it/s]
#Eval for threshold 0.20: DER 10.11%, MS 1.26%, FA 8.41%, SC 0.44%
#
#Eval for threshold 0.30: DER 7.35%, MS 2.20%, FA 4.49%, SC 0.66%
#
#Eval for threshold 0.35: DER 6.72%, MS 2.78%, FA 3.16%, SC 0.77%
#
#Eval for threshold 0.40: DER 6.45%, MS 3.39%, FA 2.16%, SC 0.89%
#
#Eval for threshold 0.45: DER 6.56%, MS 4.19%, FA 1.41%, SC 0.96% as report
#
#Eval for threshold 0.50: DER 6.98%, MS 5.17%, FA 0.91%, SC 0.90%
#
#Eval for threshold 0.55: DER 7.79%, MS 6.42%, FA 0.59%, SC 0.78%
#
#Eval for threshold 0.60: DER 8.77%, MS 7.79%, FA 0.36%, SC 0.63%
#
#Eval for threshold 0.70: DER 11.46%, MS 10.95%, FA 0.17%, SC 0.34%
#
#Eval for threshold 0.80: DER 15.66%, MS 15.41%, FA 0.09%, SC 0.16%
fi
## util now(2024-10-22), it is sota.
if [ ${stage} -le 104 ] && [ ${stop_stage} -ge 104 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is wav-bert2.0 (only using position embedding and first 6 layers conformer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this w2v-bert2.0 is trained Languages: 143+ , Size: 4.5M hours (it is from this paper https://arxiv.org/pdf/2312.05187)
    # checkpoint is from https://huggingface.co/facebook/w2v-bert-2.0/tree/main
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="w2v-bert2"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
    speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_w2v-bert2.0_first6layer_epoch40_front_fix_seed_lr1e4
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 12815 \
   ts_vad2/train_accelerate_ddp2_debug.py \
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
    --exp-dir $exp_dir
fi

if [ ${stage} -le 105 ] && [ ${stop_stage} -ge 105 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_w2v-bert2.0_first6layer_epoch40_front_fix_seed_lr1e4
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="w2v-bert2"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

 for name in $infer_sets;do
    results_path=$exp_dir
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
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --speech-encoder-config $speech_encoder_config\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir
done
# logs/run_ts_vad2_stage103-104.log # stage name is not corrected,
#  sbatch --nodes 1 --gres=gpu:2  --cpus-per-gpu=8  --ntasks-per-node 2  -p p-V100 -A t00120220002 -o logs/run_ts_vad2_stage103-104.log run_ts_vad2.sh --stage 104 --stop-stage 105
# Eval set
# Model DER:  0.11886675610010529
#Model ACC:  0.9600840943017613
#100%|██████████| 25/25 [00:23<00:00,  1.04it/s]
#Eval for threshold 0.20: DER 8.32%, MS 0.75%, FA 7.33%, SC 0.23%
#
#Eval for threshold 0.30: DER 5.62%, MS 1.16%, FA 4.23%, SC 0.22%
#
#Eval for threshold 0.35: DER 5.01%, MS 1.40%, FA 3.39%, SC 0.22%
#
#Eval for threshold 0.40: DER 4.55%, MS 1.68%, FA 2.64%, SC 0.24%
#
#Eval for threshold 0.45: DER 4.35%, MS 1.97%, FA 2.12%, SC 0.26%
#
#Eval for threshold 0.50: DER 4.18%, MS 2.26%, FA 1.67%, SC 0.26% as report
#
#Eval for threshold 0.55: DER 4.21%, MS 2.66%, FA 1.32%, SC 0.23%
#
#Eval for threshold 0.60: DER 4.34%, MS 3.09%, FA 1.04%, SC 0.20%
#
#Eval for threshold 0.70: DER 5.15%, MS 4.32%, FA 0.67%, SC 0.16%
#
#Eval for threshold 0.80: DER 6.85%, MS 6.31%, FA 0.47%, SC 0.08%
## Test set
## Model DER:  0.11411711655964628
#Model ACC:  0.9584284074520223
#100%|██████████| 60/60 [00:58<00:00,  1.02it/s]
#Eval for threshold 0.20: DER 8.79%, MS 0.87%, FA 7.50%, SC 0.43%
#
#Eval for threshold 0.30: DER 5.98%, MS 1.35%, FA 4.11%, SC 0.51%
#
#Eval for threshold 0.35: DER 5.24%, MS 1.65%, FA 3.02%, SC 0.56%
#
#Eval for threshold 0.40: DER 4.87%, MS 2.00%, FA 2.25%, SC 0.61%
#
#Eval for threshold 0.45: DER 4.73%, MS 2.42%, FA 1.69%, SC 0.62%
#
#Eval for threshold 0.50: DER 4.75%, MS 2.91%, FA 1.25%, SC 0.59% as report
#
#Eval for threshold 0.55: DER 5.02%, MS 3.51%, FA 0.97%, SC 0.54%
#
#Eval for threshold 0.60: DER 5.40%, MS 4.20%, FA 0.72%, SC 0.48%
#
#Eval for threshold 0.70: DER 6.67%, MS 5.89%, FA 0.40%, SC 0.37%
#
#Eval for threshold 0.80: DER 8.80%, MS 8.35%, FA 0.20%, SC 0.25%
fi

if [ ${stage} -le 106 ] && [ ${stop_stage} -ge 106 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is hubert large Chinese model (only cnn front and first 6 layers conformer) ,
    # this checkpoint is from https://huggingface.co/TencentGameMate/chinese-hubert-large/tree/main
    #  oracle target speaker embedding is from cam++ pretrain model

    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="hubert"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/Chinese/hubert/pytorch_model.bin"
    speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/Chinese/hubert/config.json"

    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_hubert_large_Chinese_first6layer_epoch40_front_fix_seed_lr1e4
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 12915 \
   ts_vad2/train_accelerate_ddp2_debug.py \
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
    --exp-dir $exp_dir
fi
if [ ${stage} -le 107 ] && [ ${stop_stage} -ge 107 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_hubert_large_Chinese_first6layer_epoch40_front_fix_seed_lr1e4
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="hubert"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/Chinese/hubert/pytorch_model.bin"
 speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/Chinese/hubert/config.json"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

 for name in $infer_sets;do
    results_path=$exp_dir
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
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --speech-encoder-config $speech_encoder_config\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir
done
#sbatch --nodes 1 --gres=gpu:2  --cpus-per-gpu=8  --ntasks-per-node 2  -p p-V100 -A t00120220002 -o logs/run_ts_vad2_stage106-107.log run_ts_vad2.sh --stage 106 --stop-stage 107
# Eval set
#Model DER:  0.13218826431246347
#Model ACC:  0.955593959592309
#100%|██████████| 25/25 [00:24<00:00,  1.04it/s]
#Eval for threshold 0.20: DER 9.92%, MS 0.89%, FA 8.76%, SC 0.26%
#
#Eval for threshold 0.30: DER 6.60%, MS 1.53%, FA 4.80%, SC 0.27%
#
#Eval for threshold 0.35: DER 5.82%, MS 1.83%, FA 3.69%, SC 0.29%
#
#Eval for threshold 0.40: DER 5.45%, MS 2.25%, FA 2.89%, SC 0.31%
#
#Eval for threshold 0.45: DER 5.17%, MS 2.65%, FA 2.21%, SC 0.31%
#
#Eval for threshold 0.50: DER 5.15%, MS 3.10%, FA 1.74%, SC 0.31%
#
#Eval for threshold 0.55: DER 5.33%, MS 3.64%, FA 1.37%, SC 0.33%
#
#Eval for threshold 0.60: DER 5.58%, MS 4.25%, FA 1.06%, SC 0.28%
#
#Eval for threshold 0.70: DER 6.54%, MS 5.74%, FA 0.62%, SC 0.18%
#
#Eval for threshold 0.80: DER 8.56%, MS 8.03%, FA 0.44%, SC 0.10%
#Test set
#Model DER:  0.12737386399455694
#Model ACC:  0.9533405800952262
#100%|██████████| 60/60 [00:58<00:00,  1.02it/s]
#Eval for threshold 0.20: DER 9.38%, MS 1.08%, FA 7.77%, SC 0.54%
#
#Eval for threshold 0.30: DER 6.84%, MS 1.80%, FA 4.34%, SC 0.70%
#
#Eval for threshold 0.35: DER 6.25%, MS 2.19%, FA 3.30%, SC 0.76%
#
#Eval for threshold 0.40: DER 5.94%, MS 2.64%, FA 2.48%, SC 0.82%
#
#Eval for threshold 0.45: DER 5.89%, MS 3.17%, FA 1.87%, SC 0.85%
#
#Eval for threshold 0.50: DER 5.99%, MS 3.79%, FA 1.36%, SC 0.84%
#
#Eval for threshold 0.55: DER 6.27%, MS 4.49%, FA 1.02%, SC 0.76%
#
#Eval for threshold 0.60: DER 6.70%, MS 5.25%, FA 0.78%, SC 0.67%
#
#Eval for threshold 0.70: DER 8.03%, MS 7.12%, FA 0.40%, SC 0.51%
#
#Eval for threshold 0.80: DER 10.31%, MS 9.79%, FA 0.18%, SC 0.34%

fi
#(todo run)
if [ ${stage} -le 108 ] && [ ${stop_stage} -ge 108 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is mhubert-147 model (only cnn front and first 6 layers transformer) ,
    # the model trained on 90K hours of clean, open-license data
    # paper is from https://arxiv.org/pdf/2406.06371
    # this checkpoint is from https://huggingface.co/utter-project/mHuBERT-147/tree/main
    #  oracle target speaker embedding is from cam++ pretrain model
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="hubert"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/hubert/mhubert-147/pytorch_model.bin"
    speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/hubert/mhubert-147/config.json"

    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_mhubert-147_first6layer_epoch40_front_fix_seed_lr2e4
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 12915 \
   ts_vad2/train_accelerate_ddp2_debug.py \
    --world-size 2 \
    --num-epochs 40\
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
    --speech-encoder-config $speech_encoder_config\
    --select-encoder-layer-nums 6\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --exp-dir $exp_dir
fi

if [ ${stage} -le 109 ] && [ ${stop_stage} -ge 109 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_mhubert-147_first6layer_epoch40_front_fix_seed_lr2e4
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="hubert"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/hubert/mhubert-147/pytorch_model.bin"
 speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/hubert/mhubert-147/config.json"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

 for name in $infer_sets;do
    results_path=$exp_dir
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
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --speech-encoder-config $speech_encoder_config\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir
done
# Eval set
# Model DER:  0.14093660596744934
#Model ACC:  0.9515578817867807
#100%|██████████| 25/25 [00:24<00:00,  1.03it/s]
#Eval for threshold 0.20: DER 8.92%, MS 1.11%, FA 7.41%, SC 0.40%
#
#Eval for threshold 0.30: DER 6.20%, MS 1.84%, FA 3.89%, SC 0.47%
#
#Eval for threshold 0.35: DER 5.70%, MS 2.29%, FA 2.91%, SC 0.50%
#
#Eval for threshold 0.40: DER 5.46%, MS 2.78%, FA 2.15%, SC 0.52%
#
#Eval for threshold 0.45: DER 5.41%, MS 3.29%, FA 1.59%, SC 0.53% as report
#
#Eval for threshold 0.50: DER 5.55%, MS 3.85%, FA 1.20%, SC 0.50%
#
#Eval for threshold 0.55: DER 5.92%, MS 4.62%, FA 0.88%, SC 0.43%
#
#Eval for threshold 0.60: DER 6.58%, MS 5.54%, FA 0.72%, SC 0.32%
#
#Eval for threshold 0.70: DER 8.35%, MS 7.66%, FA 0.54%, SC 0.15%
#
#Eval for threshold 0.80: DER 11.17%, MS 10.68%, FA 0.40%, SC 0.09%
#
## Test set
#Model DER:  0.14141441158563522
#Model ACC:  0.9465136312733412
#100%|██████████| 60/60 [00:58<00:00,  1.03it/s]
#Eval for threshold 0.20: DER 10.40%, MS 1.37%, FA 7.84%, SC 1.19%
#
#Eval for threshold 0.30: DER 7.95%, MS 2.21%, FA 4.19%, SC 1.55%
#
#Eval for threshold 0.35: DER 7.51%, MS 2.69%, FA 3.16%, SC 1.67%
#
#Eval for threshold 0.40: DER 7.22%, MS 3.21%, FA 2.27%, SC 1.74%
#
#Eval for threshold 0.45: DER 7.17%, MS 3.80%, FA 1.61%, SC 1.76% as report
#
#Eval for threshold 0.50: DER 7.37%, MS 4.52%, FA 1.14%, SC 1.71%
#
#Eval for threshold 0.55: DER 7.79%, MS 5.43%, FA 0.78%, SC 1.57%
#
#Eval for threshold 0.60: DER 8.44%, MS 6.47%, FA 0.55%, SC 1.41%
#
#Eval for threshold 0.70: DER 10.29%, MS 8.94%, FA 0.27%, SC 1.08%
#
#Eval for threshold 0.80: DER 13.10%, MS 12.23%, FA 0.13%, SC 0.75%
fi

if [ ${stage} -le 110 ] && [ ${stop_stage} -ge 110 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is mhubert-147 model (only cnn front and first 6 layers transformer) ,
    # the model trained on 90K hours of clean, open-license data
    # paper is from https://arxiv.org/pdf/2406.06371
    # this checkpoint is from https://huggingface.co/utter-project/mHuBERT-147/tree/main
    #  oracle target speaker embedding is from cam++ pretrain model
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="hubert"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/hubert/mhubert-147/pytorch_model.bin"
    speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/hubert/mhubert-147/config.json"

    feature_grad_mult=0.0 # means that freezed cnn feature frontent of mhubert-147 in training tsvad.
    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_mhubert-147_first6layer_epoch40_front_fix_seed_lr2e4_feature_grad_mult0.0
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 12315 \
   ts_vad2/train_accelerate_ddp2_debug.py \
    --world-size 2 \
    --num-epochs 40\
    --start-epoch 1\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-updates 4000\
    --grad-clip true\
    --lr 2e-4\
    --feature-grad-mult $feature_grad_mult\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --speech-encoder-config $speech_encoder_config\
    --select-encoder-layer-nums 6\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --exp-dir $exp_dir
fi


if [ ${stage} -le 111 ] && [ ${stop_stage} -ge 111 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_mhubert-147_first6layer_epoch40_front_fix_seed_lr2e4_feature_grad_mult0.0
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="hubert"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/hubert/mhubert-147/pytorch_model.bin"
 speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/hubert/mhubert-147/config.json"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

 for name in $infer_sets;do
    results_path=$exp_dir
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
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --speech-encoder-config $speech_encoder_config\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir
done
# cat logs/run_ts_vad2_stage110-111.log
#Eval set
#Model DER:  0.1432355773523287
#Model ACC:  0.9502530646380765
#100%|██████████| 25/25 [00:24<00:00,  1.02it/s]
#Eval for threshold 0.20: DER 10.23%, MS 0.91%, FA 8.86%, SC 0.46%
#
#Eval for threshold 0.30: DER 7.14%, MS 1.53%, FA 5.03%, SC 0.59%
#
#Eval for threshold 0.35: DER 6.41%, MS 1.93%, FA 3.84%, SC 0.63%
#
#Eval for threshold 0.40: DER 5.92%, MS 2.38%, FA 2.86%, SC 0.68%
#
#Eval for threshold 0.45: DER 5.73%, MS 2.93%, FA 2.11%, SC 0.69%
#
#Eval for threshold 0.50: DER 5.80%, MS 3.57%, FA 1.59%, SC 0.63%
#
#Eval for threshold 0.55: DER 6.11%, MS 4.33%, FA 1.22%, SC 0.57%
#
#Eval for threshold 0.60: DER 6.68%, MS 5.25%, FA 0.96%, SC 0.47%
#
#Eval for threshold 0.70: DER 8.27%, MS 7.42%, FA 0.59%, SC 0.26%
#
#Eval for threshold 0.80: DER 10.97%, MS 10.41%, FA 0.43%, SC 0.13%
## Test set
#Model DER:  0.14152335355350387
#Model ACC:  0.9459266323985126
#100%|██████████| 60/60 [01:00<00:00,  1.01s/it]
#Eval for threshold 0.20: DER 10.82%, MS 1.27%, FA 8.29%, SC 1.26%
#
#Eval for threshold 0.30: DER 8.28%, MS 2.05%, FA 4.60%, SC 1.64%
#
#Eval for threshold 0.35: DER 7.68%, MS 2.50%, FA 3.42%, SC 1.77%
#
#Eval for threshold 0.40: DER 7.35%, MS 3.02%, FA 2.48%, SC 1.85%
#
#Eval for threshold 0.45: DER 7.31%, MS 3.64%, FA 1.80%, SC 1.87%
#
#Eval for threshold 0.50: DER 7.48%, MS 4.39%, FA 1.30%, SC 1.79%
#
#Eval for threshold 0.55: DER 7.84%, MS 5.23%, FA 0.94%, SC 1.67%
#
#Eval for threshold 0.60: DER 8.37%, MS 6.18%, FA 0.68%, SC 1.51%
#
#Eval for threshold 0.70: DER 10.00%, MS 8.46%, FA 0.33%, SC 1.21%
#
#Eval for threshold 0.80: DER 12.68%, MS 11.67%, FA 0.17%, SC 0.85%
fi


# campared with stage104-105, stage112 will increase lr from 1e-4 to 15e-5
if [ ${stage} -le 112 ] && [ ${stop_stage} -ge 112 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is wav-bert2.0 (only using position embedding and first 6 layers conformer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this w2v-bert2.0 is trained Languages: 143+ , Size: 4.5M hours (it is from this paper https://arxiv.org/pdf/2312.05187)
    # checkpoint is from https://huggingface.co/facebook/w2v-bert-2.0/tree/main
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="w2v-bert2"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
    speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_w2v-bert2.0_first6layer_epoch40_front_fix_seed_lr15e5
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 12615 \
   ts_vad2/train_accelerate_ddp2_debug.py \
    --world-size 2 \
    --num-epochs 40\
    --start-epoch 1\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-updates 4000\
    --grad-clip true\
    --lr 15e-5\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --speech-encoder-config $speech_encoder_config\
    --select-encoder-layer-nums 6\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --exp-dir $exp_dir
fi

if [ ${stage} -le 113 ] && [ ${stop_stage} -ge 113 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_w2v-bert2.0_first6layer_epoch40_front_fix_seed_lr15e5
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="w2v-bert2"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

 for name in $infer_sets;do
    results_path=$exp_dir
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
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --speech-encoder-config $speech_encoder_config\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir
done
# sbatch --nodes 1 --gres=gpu:2  --cpus-per-gpu=8  --ntasks-per-node 2  -p p-V100 -A t00120220002 -o logs/run_ts_vad2_stage112-113.log run_ts_vad2.sh --stage 112 --stop-stage 113
#cat logs/run_ts_vad2_stage112-113.log
##Eval set
#Model DER:  0.12430747712974934
#Model ACC:  0.9575508656813799
#100%|██████████| 25/25 [00:23<00:00,  1.05it/s]
#Eval for threshold 0.20: DER 9.73%, MS 0.73%, FA 8.68%, SC 0.31%
#
#Eval for threshold 0.30: DER 6.54%, MS 1.14%, FA 5.01%, SC 0.39%
#
#Eval for threshold 0.35: DER 5.72%, MS 1.31%, FA 3.97%, SC 0.44%
#
#Eval for threshold 0.40: DER 5.12%, MS 1.54%, FA 3.09%, SC 0.49%
#
#Eval for threshold 0.45: DER 4.75%, MS 1.81%, FA 2.41%, SC 0.52%
#
#Eval for threshold 0.50: DER 4.58%, MS 2.17%, FA 1.86%, SC 0.55% as report
#
#Eval for threshold 0.55: DER 4.56%, MS 2.59%, FA 1.44%, SC 0.54%
#
#Eval for threshold 0.60: DER 4.76%, MS 3.14%, FA 1.11%, SC 0.50%
#
#Eval for threshold 0.70: DER 5.63%, MS 4.50%, FA 0.73%, SC 0.40%
#
#Eval for threshold 0.80: DER 7.45%, MS 6.71%, FA 0.49%, SC 0.25%
#
#Test set
#Model DER:  0.11875953051893867
#Model ACC:  0.9566520109394607
#100%|██████████| 60/60 [00:57<00:00,  1.04it/s]
#Eval for threshold 0.20: DER 10.02%, MS 0.76%, FA 8.86%, SC 0.40%
#
#Eval for threshold 0.30: DER 6.56%, MS 1.27%, FA 4.72%, SC 0.57%
#
#Eval for threshold 0.35: DER 5.76%, MS 1.58%, FA 3.53%, SC 0.65%
#
#Eval for threshold 0.40: DER 5.22%, MS 1.95%, FA 2.56%, SC 0.72%
#
#Eval for threshold 0.45: DER 4.96%, MS 2.39%, FA 1.85%, SC 0.72%
#
#Eval for threshold 0.50: DER 4.99%, MS 2.95%, FA 1.32%, SC 0.72% as report
#
#Eval for threshold 0.55: DER 5.24%, MS 3.63%, FA 0.95%, SC 0.66%
#
#Eval for threshold 0.60: DER 5.63%, MS 4.37%, FA 0.70%, SC 0.56%
#
#Eval for threshold 0.70: DER 7.02%, MS 6.28%, FA 0.36%, SC 0.38%
#
#Eval for threshold 0.80: DER 9.55%, MS 9.14%, FA 0.18%, SC 0.22%
fi

# campared with stage104-105, stage114-115 will use average checkpoint in train stage.
if [ ${stage} -le 114 ] && [ ${stop_stage} -ge 114 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is wav-bert2.0 (only using position embedding and first 6 layers conformer) ,
    #  oracle target speaker embedding is from cam++ pretrain model
    # this w2v-bert2.0 is trained Languages: 143+ , Size: 4.5M hours (it is from this paper https://arxiv.org/pdf/2312.05187)
    # checkpoint is from https://huggingface.co/facebook/w2v-bert-2.0/tree/main
    # how to look for port ?
    # netstat -tuln
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="w2v-bert2"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
    speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_w2v-bert2.0_first6layer_epoch40_front_fix_seed_lr1e4_average_model_training
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 12615 \
   ts_vad2/train_accelerate_ddp2_debug.py \
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
    --train-on-average true\
    --exp-dir $exp_dir
fi


if [ ${stage} -le 115 ] && [ ${stop_stage} -ge 115 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_w2v-bert2.0_first6layer_epoch40_front_fix_seed_lr1e4_average_model_training
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="w2v-bert2"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
 speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

 for name in $infer_sets;do
    results_path=$exp_dir
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
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
     --speech-encoder-path $speech_encoder_path \
    --speech-encoder-config $speech_encoder_config\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir
done
# cat logs/run_ts_vad2_stage114-115.log
#Eval set
#Model DER:  0.11873613522306012
#Model ACC:  0.9598422372399227
#100%|██████████| 25/25 [00:23<00:00,  1.05it/s]
#Eval for threshold 0.20: DER 8.07%, MS 0.82%, FA 7.05%, SC 0.21%
#
#Eval for threshold 0.30: DER 5.47%, MS 1.28%, FA 3.98%, SC 0.22%
#
#Eval for threshold 0.35: DER 4.91%, MS 1.55%, FA 3.12%, SC 0.24%
#
#Eval for threshold 0.40: DER 4.46%, MS 1.82%, FA 2.38%, SC 0.26%
#
#Eval for threshold 0.45: DER 4.23%, MS 2.12%, FA 1.86%, SC 0.25%
#
#Eval for threshold 0.50: DER 4.19%, MS 2.49%, FA 1.42%, SC 0.28% as report
#
#Eval for threshold 0.55: DER 4.25%, MS 2.91%, FA 1.09%, SC 0.25%
#
#Eval for threshold 0.60: DER 4.52%, MS 3.41%, FA 0.89%, SC 0.21%
#
#Eval for threshold 0.70: DER 5.53%, MS 4.78%, FA 0.60%, SC 0.15%
#
#Eval for threshold 0.80: DER 7.45%, MS 6.94%, FA 0.43%, SC 0.08%
#Test set
#Model DER:  0.11521851974154258
#Model ACC:  0.9581827249869384
#100%|██████████| 60/60 [00:58<00:00,  1.03it/s]
#Eval for threshold 0.20: DER 8.08%, MS 0.96%, FA 6.74%, SC 0.38%
#
#Eval for threshold 0.30: DER 5.57%, MS 1.50%, FA 3.56%, SC 0.51%
#
#Eval for threshold 0.35: DER 4.98%, MS 1.81%, FA 2.63%, SC 0.54%
#
#Eval for threshold 0.40: DER 4.70%, MS 2.19%, FA 1.94%, SC 0.58%
#
#Eval for threshold 0.45: DER 4.65%, MS 2.63%, FA 1.43%, SC 0.60%
#
#Eval for threshold 0.50: DER 4.79%, MS 3.15%, FA 1.07%, SC 0.57% as report
#
#Eval for threshold 0.55: DER 5.15%, MS 3.80%, FA 0.82%, SC 0.52%
#
#Eval for threshold 0.60: DER 5.65%, MS 4.57%, FA 0.62%, SC 0.46%
#
#Eval for threshold 0.70: DER 7.03%, MS 6.35%, FA 0.34%, SC 0.34%
#
#Eval for threshold 0.80: DER 9.31%, MS 8.94%, FA 0.18%, SC 0.19%



fi
## compared with stage108-109, stage116-117 will change lr=2e-4 to lr=1e-4
if [ ${stage} -le 116 ] && [ ${stop_stage} -ge 116 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is mhubert-147 model (only cnn front and first 6 layers transformer) ,
    # the model trained on 90K hours of clean, open-license data
    # paper is from https://arxiv.org/pdf/2406.06371
    # this checkpoint is from https://huggingface.co/utter-project/mHuBERT-147/tree/main
    #  oracle target speaker embedding is from cam++ pretrain model
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="hubert"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/hubert/mhubert-147/pytorch_model.bin"
    speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/hubert/mhubert-147/config.json"

    #feature_grad_mult=0.0 # means that freezed cnn feature frontent of mhubert-147 in training tsvad.
    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_mhubert-147_first6layer_epoch40_front_fix_seed_lr1e4
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 12915 \
   ts_vad2/train_accelerate_ddp2_debug.py \
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
    --exp-dir $exp_dir
fi

if [ ${stage} -le 117 ] && [ ${stop_stage} -ge 117 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_mhubert-147_first6layer_epoch40_front_fix_seed_lr1e4
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="hubert"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/hubert/mhubert-147/pytorch_model.bin"
 speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/hubert/mhubert-147/config.json"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

 for name in $infer_sets;do
    results_path=$exp_dir
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
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --speech-encoder-config $speech_encoder_config\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir
done
# cat logs/run_ts_vad2_stage116-117.log
# Eval set
# Model DER:  0.14322045230736494
#Model ACC:  0.9504141803043209
#100%|██████████| 25/25 [00:24<00:00,  1.04it/s]
#Eval for threshold 0.20: DER 12.59%, MS 0.94%, FA 11.23%, SC 0.42%
#
#Eval for threshold 0.30: DER 7.96%, MS 1.58%, FA 5.86%, SC 0.52%
#
#Eval for threshold 0.35: DER 6.88%, MS 1.98%, FA 4.30%, SC 0.60%
#
#Eval for threshold 0.40: DER 6.21%, MS 2.42%, FA 3.16%, SC 0.63%
#
#Eval for threshold 0.45: DER 5.90%, MS 2.96%, FA 2.35%, SC 0.59% as report
#
#Eval for threshold 0.50: DER 5.92%, MS 3.61%, FA 1.77%, SC 0.54%
#
#Eval for threshold 0.55: DER 6.20%, MS 4.36%, FA 1.36%, SC 0.48%
#
#Eval for threshold 0.60: DER 6.73%, MS 5.30%, FA 1.04%, SC 0.40%
#
#Eval for threshold 0.70: DER 8.26%, MS 7.41%, FA 0.63%, SC 0.22%
#
#Eval for threshold 0.80: DER 11.13%, MS 10.56%, FA 0.45%, SC 0.13%
## Test set
#Model DER:  0.1408388483402552
#Model ACC:  0.947298832278388
#100%|██████████| 60/60 [00:58<00:00,  1.03it/s]
#Eval for threshold 0.20: DER 13.26%, MS 1.04%, FA 11.35%, SC 0.86%
#
#Eval for threshold 0.30: DER 9.10%, MS 1.81%, FA 6.16%, SC 1.13%
#
#Eval for threshold 0.35: DER 8.00%, MS 2.21%, FA 4.53%, SC 1.25%
#
#Eval for threshold 0.40: DER 7.35%, MS 2.70%, FA 3.30%, SC 1.34%
#
#Eval for threshold 0.45: DER 7.02%, MS 3.26%, FA 2.37%, SC 1.39% as report
#
#Eval for threshold 0.50: DER 7.07%, MS 3.97%, FA 1.71%, SC 1.40%
#
#Eval for threshold 0.55: DER 7.38%, MS 4.83%, FA 1.24%, SC 1.31%
#
#Eval for threshold 0.60: DER 7.96%, MS 5.86%, FA 0.89%, SC 1.21%
#
#Eval for threshold 0.70: DER 9.80%, MS 8.41%, FA 0.45%, SC 0.93%
#
#Eval for threshold 0.80: DER 12.85%, MS 12.06%, FA 0.18%, SC 0.61%

fi
# compared with stage104-105(util now(2024-10-31), it is sota),
# stage118-119, I will use more bigger model(mms-1b) SSL model(it is wav2vec2.0 network)
if [ ${stage} -le 118 ] && [ ${stop_stage} -ge 118 ];then
    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
    # # speech encoder is mms-1b model (only cnn front and first 6 layers transformer) ,
    # the model trained on 500K hours wavform data
    # paper is from https://arxiv.org/abs/2305.13516
    # this checkpoint is from https://huggingface.co/facebook/mms-1b/tree/main
    #  oracle target speaker embedding is from cam++ pretrain model
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    # for loading pretrain model weigt
    speech_encoder_type="wav2vec2"
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/mms/mms-1b/pytorch_model.bin"
    speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/mms/mms-1b/config.json"

    #feature_grad_mult=0.0 # means that freezed cnn feature frontent of mhubert-147 in training tsvad.
    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_mms-1b_first6layer_epoch40_front_fix_seed_lr1e4
   #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 12975 \
   ts_vad2/train_accelerate_ddp2_debug.py \
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
    --exp-dir $exp_dir
fi

if [ ${stage} -le 119 ] && [ ${stop_stage} -ge 119 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_mms-1b_first6layer_epoch40_front_fix_seed_lr1e4
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="wav2vec2"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/mms/mms-1b/pytorch_model.bin"
 speech_encoder_config="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/mms/mms-1b/config.json"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_en_zh_advanced_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

 for name in $infer_sets;do
    results_path=$exp_dir
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
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --speech-encoder-config $speech_encoder_config\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir
done
fi


## util now(2024-10-22), it is sota in huawei testset.
if [ ${stage} -le 120 ] && [ ${stop_stage} -ge 120 ];then
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

    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4
    data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

    #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 12815 \
   ts_vad2/train_accelerate_ddp2_debug2.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --keep-last-k 1\
    --keep-last-epoch 1\
    --freeze-updates 0\
    --grad-clip false\
    --lr 2e-4\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path\
    --select-encoder-layer-nums 6\
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --exp-dir $exp_dir\
    --data-dir $data_dir
fi

if [ ${stage} -le 121 ] && [ ${stop_stage} -ge 121 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

 for name in $infer_sets;do
    results_path=$exp_dir
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
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir

done
#sbatch --nodes 1 --gres=gpu:2  --cpus-per-gpu=6  --ntasks-per-node 2  -p p-A100 -A t00120220002 -o logs/run_ts_vad2_stage120-121_A100_2.log run_ts_vad2.sh --stage 120 --stop-stage 121
#Submitted batch job 203805
#cat  logs/run_ts_vad2_stage120-121_A100_2.log
#Eval set
#Model DER:  0.12561471382181064
#Model ACC:  0.956613160500625
#2024-11-17 02:30:18,621 (infer2:84) INFO: frame_len: 0.04!!
#100%|██████████| 25/25 [00:15<00:00,  1.59it/s]
#Eval for threshold 0.20: DER 8.17%, MS 0.87%, FA 7.01%, SC 0.29%
#
#Eval for threshold 0.30: DER 5.93%, MS 1.41%, FA 4.15%, SC 0.37%
#
#Eval for threshold 0.35: DER 5.40%, MS 1.67%, FA 3.31%, SC 0.41%
#
#Eval for threshold 0.40: DER 5.11%, MS 2.01%, FA 2.68%, SC 0.43%
#
#Eval for threshold 0.45: DER 4.92%, MS 2.39%, FA 2.10%, SC 0.42%
#
#Eval for threshold 0.50: DER 4.90%, MS 2.81%, FA 1.66%, SC 0.43% as report
#
#Eval for threshold 0.55: DER 4.96%, MS 3.28%, FA 1.29%, SC 0.39%
#
#Eval for threshold 0.60: DER 5.22%, MS 3.83%, FA 1.04%, SC 0.35%
#
#Eval for threshold 0.70: DER 6.06%, MS 5.15%, FA 0.64%, SC 0.27%
#
#Eval for threshold 0.80: DER 7.79%, MS 7.19%, FA 0.44%, SC 0.16%
#

## collar=0.0, Eval set
# SCTK-2.4.12/src/md-eval/md-eval.pl -c 0.0  -r /mntcephfs/lab_data/maduo/model_hub/ts_vad/alimeeting_eval.rttm -s /mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4/Eval/res_rttm_0.5
# 12.77/6.77/4.80/1.20


#Test set
#Model DER:  0.1175008144605523
#Model ACC:  0.9568169925868446
#2024-11-17 02:39:27,289 (infer2:84) INFO: frame_len: 0.04!!
#100%|██████████| 60/60 [00:38<00:00,  1.56it/s]
#Eval for threshold 0.20: DER 8.60%, MS 1.00%, FA 7.19%, SC 0.42%
#
#Eval for threshold 0.30: DER 6.29%, MS 1.60%, FA 4.21%, SC 0.49%
#
#Eval for threshold 0.35: DER 5.66%, MS 1.91%, FA 3.24%, SC 0.52%
#
#Eval for threshold 0.40: DER 5.35%, MS 2.25%, FA 2.56%, SC 0.54%
#
#Eval for threshold 0.45: DER 5.17%, MS 2.66%, FA 1.97%, SC 0.54%
#
#Eval for threshold 0.50: DER 5.18%, MS 3.12%, FA 1.53%, SC 0.54% as report
#
#Eval for threshold 0.55: DER 5.30%, MS 3.64%, FA 1.15%, SC 0.51%
#
#Eval for threshold 0.60: DER 5.56%, MS 4.26%, FA 0.82%, SC 0.48%
#
#Eval for threshold 0.70: DER 6.68%, MS 5.89%, FA 0.43%, SC 0.36%
#
#Eval for threshold 0.80: DER 8.70%, MS 8.23%, FA 0.21%, SC 0.27%

# collar=0.0, Test set
# SCTK-2.4.12/src/md-eval/md-eval.pl -c 0.0  -r /mntcephfs/lab_data/maduo/model_hub/ts_vad/alimeeting_test.rttm -s /mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4/Test/res_rttm_0.5
# 12.80/6.64/4.86/1.30
fi

# debug
if [ ${stage} -le 122 ] && [ ${stop_stage} -ge 122 ];then
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

    # for loading speaker embedding file
    spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
    speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"

    #exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_wav-bert2.0_epoch40_front_fix_seed
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_debug
    data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

    #CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
  #accelerate launch --debug --multi_gpu --mixed_precision=fp16 --num_processes=2  --main_process_port=12673 ts_vad2/train_accelerate_ddp2.py \
  CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 12915 \
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
     --data-dir $data_dir
fi

if [ ${stage} -le 123 ] && [ ${stop_stage} -ge 123 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_debug
 model_file=$exp_dir/best-valid-der.pt
 rs_len=4
 segment_shift=1
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar=0.25
 #collar=0.0
 # it is used to instance speech encoder of tsvad model base on different pretrain speaker model.
 speech_encoder_type="CAM++"
 speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"

 # for loading speaker embedding file
 spk_path=/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding # store speaker embedding directory
 speaker_embedding_name_dir="cam++_zh-cn_200k_feature_dir"
 data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path

 for name in $infer_sets;do
    results_path=$exp_dir
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
    --collar $collar\
    --results-path $results_path \
    --split $name\
    --speech-encoder-type $speech_encoder_type\
    --speech-encoder-path $speech_encoder_path \
    --spk-path $spk_path\
    --speaker-embedding-name-dir $speaker_embedding_name_dir\
    --wavlm-fuse-feat-post-norm false \
    --data-dir $data_dir
done

#sbatch --nodes 1 --gres=gpu:2  --cpus-per-gpu=6  --ntasks-per-node 2  -p p-A100 -A t00120220002 -o logs/run_ts_vad2_stage122-123_A100.log run_ts_vad2.sh --stage 122 --stop-stage 123
#Submitted batch job 203798
#
#cat logs/run_ts_vad2_stage122-123_A100.log
#Eval set
#Model DER:  0.1260389122400934
#Model ACC:  0.9566343205242521
#2024-11-16 21:18:05,728 (infer2:84) INFO: frame_len: 0.04!!
#100%|██████████| 25/25 [00:15<00:00,  1.61it/s]
#Eval for threshold 0.20: DER 8.06%, MS 0.94%, FA 6.82%, SC 0.30%
#
#Eval for threshold 0.30: DER 5.84%, MS 1.46%, FA 3.98%, SC 0.40%
#
#Eval for threshold 0.35: DER 5.33%, MS 1.77%, FA 3.16%, SC 0.40%
#
#Eval for threshold 0.40: DER 5.05%, MS 2.14%, FA 2.52%, SC 0.39%
#
#Eval for threshold 0.45: DER 4.92%, MS 2.54%, FA 2.00%, SC 0.38% as report
#
#Eval for threshold 0.50: DER 4.91%, MS 2.95%, FA 1.57%, SC 0.38%
#
#Eval for threshold 0.55: DER 4.98%, MS 3.42%, FA 1.22%, SC 0.34%
#
#Eval for threshold 0.60: DER 5.22%, MS 3.93%, FA 0.98%, SC 0.31%
#
#Eval for threshold 0.70: DER 6.20%, MS 5.31%, FA 0.66%, SC 0.23%
#
#Eval for threshold 0.80: DER 7.93%, MS 7.35%, FA 0.44%, SC 0.14%
#
# # collar=0.0, Eval set
# SCTK-2.4.12/src/md-eval/md-eval.pl -c 0.0  -r /mntcephfs/lab_data/maduo/model_hub/ts_vad/alimeeting_eval.rttm -s /mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_debug/Eval/res_rttm_0.45
# 12.87/6.09/5.60/1.18

#Test set
#Model DER:  0.11799974171623803
#Model ACC:  0.9564398575727613
#2024-11-16 21:23:13,290 (infer2:84) INFO: frame_len: 0.04!!
#100%|██████████| 60/60 [00:37<00:00,  1.59it/s]
#Eval for threshold 0.20: DER 8.37%, MS 1.08%, FA 6.88%, SC 0.41%
#
#Eval for threshold 0.30: DER 6.18%, MS 1.68%, FA 4.01%, SC 0.49%
#
#Eval for threshold 0.35: DER 5.58%, MS 2.01%, FA 3.06%, SC 0.51%
#
#Eval for threshold 0.40: DER 5.29%, MS 2.37%, FA 2.37%, SC 0.55%
#
#Eval for threshold 0.45: DER 5.15%, MS 2.78%, FA 1.81%, SC 0.56% as report
#
#Eval for threshold 0.50: DER 5.19%, MS 3.26%, FA 1.37%, SC 0.56%
#
#Eval for threshold 0.55: DER 5.39%, MS 3.81%, FA 1.05%, SC 0.53%
#
#Eval for threshold 0.60: DER 5.72%, MS 4.44%, FA 0.79%, SC 0.49%
#
#Eval for threshold 0.70: DER 6.76%, MS 5.98%, FA 0.41%, SC 0.36%
#
#Eval for threshold 0.80: DER 8.63%, MS 8.19%, FA 0.20%, SC 0.23%

# collar=0.0, Test set
# SCTK-2.4.12/src/md-eval/md-eval.pl -c 0.0  -r /mntcephfs/lab_data/maduo/model_hub/ts_vad/alimeeting_test.rttm -s /mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr2e4_debug/Test/res_rttm_0.45
# 12.93/5.99/5.58/1.36
fi

# stage124-127 is running hltsz it is at run_ts_vad2_hltsz.sh of egs/alimeeting
#compared with stage122-123, stage124-125 will use mamba2 to replace transformer
if [ ${stage} -le 124 ] && [ ${stop_stage} -ge 124 ];then
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
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr1e4_single_backend_mamba2_multi_backend_transformer_ts_len6_d_state_128
    data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
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
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr1e4_single_backend_mamba2_multi_backend_transformer_ts_len6_d_state_128
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
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0
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


#compared with stage124-125, stage126-127 d_state will reduce from 128 to 64
if [ ${stage} -le 126 ] && [ ${stop_stage} -ge 126 ];then
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
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr1e4_single_backend_mamba2_multi_backend_transformer_ts_len6_d_state_64
    data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
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
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr1e4_single_backend_mamba2_multi_backend_transformer_ts_len6_d_state_64
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
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0
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


#compared with stage124-125  stage128-129 d_state will increase from 128 to 256
if [ ${stage} -le 128 ] && [ ${stop_stage} -ge 128 ];then
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
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr1e4_single_backend_mamba2_multi_backend_transformer_ts_len6_d_state_256
    data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    rs_len=6
    segment_shift=2
    single_backend_type="mamba2"
    multi_backend_type="transformer"
    d_state=256
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

if [ ${stage} -le 129 ] && [ ${stop_stage} -ge 129 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr1e4_single_backend_mamba2_multi_backend_transformer_ts_len6_d_state_256
 model_file=$exp_dir/best-valid-der.pt
 rs_len=6
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
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0
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
#grep -r Eval logs/run_ts_vad2_stage129_A100.log
# collar=0.0
# dev of alimeeting
#Eval for threshold 0.20: DER 20.79%, MS 2.19%, FA 17.61%, SC 0.99%
#Eval for threshold 0.30: DER 16.41%, MS 3.39%, FA 11.81%, SC 1.21%
#Eval for threshold 0.35: DER 15.12%, MS 4.06%, FA 9.80%, SC 1.27%
#Eval for threshold 0.40: DER 14.18%, MS 4.77%, FA 8.10%, SC 1.31%
#Eval for threshold 0.45: DER 13.56%, MS 5.56%, FA 6.72%, SC 1.28%
#Eval for threshold 0.50: DER 13.25%, MS 6.43%, FA 5.57%, SC 1.26%
#Eval for threshold 0.55: DER 13.17%, MS 7.36%, FA 4.63%, SC 1.19%
#Eval for threshold 0.60: DER 13.31%, MS 8.45%, FA 3.76%, SC 1.10%
#Eval for threshold 0.70: DER 14.46%, MS 11.06%, FA 2.55%, SC 0.85%
#Eval for threshold 0.80: DER 16.77%, MS 14.64%, FA 1.61%, SC 0.52%

# test of alimeeting
#Eval for threshold 0.20: DER 20.84%, MS 2.05%, FA 17.78%, SC 1.00%
#Eval for threshold 0.30: DER 16.28%, MS 3.32%, FA 11.73%, SC 1.23%
#Eval for threshold 0.35: DER 15.03%, MS 4.02%, FA 9.69%, SC 1.32%
#Eval for threshold 0.40: DER 14.16%, MS 4.79%, FA 8.01%, SC 1.35%
#Eval for threshold 0.45: DER 13.65%, MS 5.67%, FA 6.63%, SC 1.35%
#Eval for threshold 0.50: DER 13.45%, MS 6.62%, FA 5.48%, SC 1.35%
#Eval for threshold 0.55: DER 13.45%, MS 7.65%, FA 4.47%, SC 1.33%
#Eval for threshold 0.60: DER 13.77%, MS 8.86%, FA 3.63%, SC 1.28%
#Eval for threshold 0.70: DER 15.06%, MS 11.74%, FA 2.26%, SC 1.06%
#Eval for threshold 0.80: DER 17.84%, MS 15.82%, FA 1.30%, SC 0.73%
#
# collar=0.25
# dev of alimeeting
#Eval for threshold 0.20: DER 9.62%, MS 0.77%, FA 8.58%, SC 0.27%
#Eval for threshold 0.30: DER 6.83%, MS 1.26%, FA 5.21%, SC 0.36%
#Eval for threshold 0.35: DER 6.03%, MS 1.52%, FA 4.10%, SC 0.41%
#Eval for threshold 0.40: DER 5.53%, MS 1.89%, FA 3.23%, SC 0.41%
#Eval for threshold 0.45: DER 5.12%, MS 2.23%, FA 2.49%, SC 0.40%
#Eval for threshold 0.50: DER 4.99%, MS 2.67%, FA 1.94%, SC 0.38%
#Eval for threshold 0.55: DER 5.04%, MS 3.15%, FA 1.54%, SC 0.35%
#Eval for threshold 0.60: DER 5.24%, MS 3.72%, FA 1.17%, SC 0.35%
#Eval for threshold 0.70: DER 6.14%, MS 5.16%, FA 0.72%, SC 0.25%
#Eval for threshold 0.80: DER 7.87%, MS 7.31%, FA 0.46%, SC 0.10%

# test of alimeeting
#Eval for threshold 0.20: DER 10.22%, MS 0.88%, FA 8.97%, SC 0.37%
#Eval for threshold 0.30: DER 7.09%, MS 1.50%, FA 5.12%, SC 0.47%
#Eval for threshold 0.35: DER 6.29%, MS 1.83%, FA 3.96%, SC 0.50%
#Eval for threshold 0.40: DER 5.77%, MS 2.24%, FA 3.03%, SC 0.51%
#Eval for threshold 0.45: DER 5.49%, MS 2.70%, FA 2.29%, SC 0.51%
#Eval for threshold 0.50: DER 5.48%, MS 3.22%, FA 1.74%, SC 0.51%
#Eval for threshold 0.55: DER 5.58%, MS 3.79%, FA 1.29%, SC 0.51%
#Eval for threshold 0.60: DER 5.90%, MS 4.46%, FA 0.94%, SC 0.50%
#Eval for threshold 0.70: DER 6.97%, MS 6.12%, FA 0.45%, SC 0.40%
#Eval for threshold 0.80: DER 9.08%, MS 8.65%, FA 0.19%, SC 0.24%



#compared with stage128-129,stage130-131 d_state will increase from 256 to 512
if [ ${stage} -le 130 ] && [ ${stop_stage} -ge 130 ];then
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
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr1e4_single_backend_mamba2_multi_backend_transformer_ts_len6_d_state_512
    data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    rs_len=6
    segment_shift=2
    single_backend_type="mamba2"
    multi_backend_type="transformer"
    d_state=512
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
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr1e4_single_backend_mamba2_multi_backend_transformer_ts_len6_d_state_512
 model_file=$exp_dir/best-valid-der.pt
 rs_len=6
 segment_shift=1
 single_backend_type="mamba2"
 multi_backend_type="transformer"
 d_state=512
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0
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




#compared with stage128-129,stage132-133 rs_len will reduce from 6 to 4
if [ ${stage} -le 132 ] && [ ${stop_stage} -ge 132 ];then
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
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr1e4_single_backend_mamba2_multi_backend_transformer_ts_len4_d_state_256
    data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    rs_len=4
    segment_shift=2
    single_backend_type="mamba2"
    multi_backend_type="transformer"
    d_state=256
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

if [ ${stage} -le 133 ] && [ ${stop_stage} -ge 133 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr1e4_single_backend_mamba2_multi_backend_transformer_ts_len4_d_state_256
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
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0
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



#compared with stage128-129,stage134-135 d_state will increase from 256 to 384
if [ ${stage} -le 134 ] && [ ${stop_stage} -ge 134 ];then
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
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr1e4_single_backend_mamba2_multi_backend_transformer_ts_len6_d_state_384
    data_dir="/mntcephfs/lab_data/maduo/datasets/alimeeting" # oracle target audio , mix audio and labels path
    rs_len=6
    segment_shift=2
    single_backend_type="mamba2"
    multi_backend_type="transformer"
    d_state=384
    num_transformer_layer=2
    CUDA_VISIABLE_DEVICES=0,1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --main_process_port 15515 \
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

if [ ${stage} -le 135 ] && [ ${stop_stage} -ge 135 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_freeze_with_musan_rirs_cam++_zh_200k_epoch20_front_fix_seed_lr1e4_single_backend_mamba2_multi_backend_transformer_ts_len6_d_state_384
 model_file=$exp_dir/best-valid-der.pt
 rs_len=6
 segment_shift=1
 single_backend_type="mamba2"
 multi_backend_type="transformer"
 d_state=384
 num_transformer_layer=2
 label_rate=25
 min_silence=0.32
 min_speech=0.0
 infer_sets="Eval Test"
 #infer_sets="Test"
 rttm_dir=/mntcephfs/lab_data/maduo/model_hub/ts_vad
 sctk_tool_path="./SCTK-2.4.12"
 collar="0.0 0.25"
 #collar=0.0
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
