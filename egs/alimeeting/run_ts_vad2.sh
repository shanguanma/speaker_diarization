#!/usr/bin/env bash

stage=0
stop_stage=1000

. utils/parse_options.sh
. path_for_speaker_diarization.sh


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
#Eval for threshold 0.45: DER 5.07%, MS 2.67%, FA 1.95%, SC 0.46%
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
#Eval for threshold 0.45: DER 6.62%, MS 3.24%, FA 1.99%, SC 1.39%
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




## for compared stage 34, stage48 doesn't use add noise and rirs to train tsvad.
if [ ${stage} -le 48 ] && [ ${stop_stage} -ge 48 ];then

    # # it adds noise and rirs to train tsvad model , grad-clip and freeze update.
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
# /mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wavlm/WavLM-Large.pt


