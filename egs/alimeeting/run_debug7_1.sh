#!/usr/bin/env bash

stage=0
stop_stage=1000

. utils/parse_options.sh
. path_for_speaker_diarization.sh

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad/exp7_1
 python3 ts_vad/train7_1.py\
    --world-size 1 \
    --num-epochs 30\
    --start-epoch 30\
    --use-fp16 1\
    --exp-dir $exp_dir
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad/exp7_2
 python3 ts_vad/train7_1.py\
    --world-size 1 \
    --num-epochs 60\
    --start-epoch 1\
    --use-fp16 1\
    --exp-dir $exp_dir
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad/exp7
 python3 ts_vad/train7.py\
    --world-size 1 \
    --num-epochs 30\
    --start-epoch 30\
    --use-fp16 1\
    --exp-dir $exp_dir
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad/exp7_2_avg
 python3 ts_vad/train7_2.py\
    --world-size 1 \
    --num-epochs 60\
    --start-epoch 1\
    --use-fp16 1\
    --exp-dir $exp_dir
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad/exp6_1_a
 python3 ts_vad/train6_1.py\
    --world-size 1 \
    --num-epochs 20\
    --start-epoch 1\
    --use-fp16 1\
    --exp-dir $exp_dir
fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad/exp6_2
 python3 ts_vad/train6_2.py\
    --world-size 1 \
    --num-epochs 20\
    --start-epoch 1\
    --use-fp16 1\
    --exp-dir $exp_dir
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad/exp7_2_baselr_3e-5
 base_lr=3e-5
 python3 ts_vad/train7_1.py\
    --world-size 1 \
    --num-epochs 60\
    --start-epoch 1\
    --use-fp16 1\
    --exp-dir $exp_dir\
    --base-lr $base_lr
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad/exp6_3
 #python3 ts_vad/train6_3.py\
  python3 ts_vad/train_fairseq2_style.py\
    --world-size 1 \
    --num-epochs 20\
    --start-epoch 1\
    --use-fp16 1\
    --exp-dir $exp_dir
fi
#(todo run,maybe better)
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad/exp7_2_baselr_5e-5
 base_lr=5e-5
 python3 ts_vad/train7_1.py\
    --world-size 1 \
    --num-epochs 60\
    --start-epoch 1\
    --use-fp16 1\
    --exp-dir $exp_dir\
    --base-lr $base_lr
fi

## best
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad/exp6_3
 #python3 ts_vad/train6_3.py\
  python3 ts_vad/train_fairseq2_style.py\
    --world-size 1 \
    --num-epochs 20\
    --start-epoch 1\
    --use-fp16 1\
    --exp-dir $exp_dir
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad/exp6_3
 #python3 ts_vad/train6_3.py\
  python3 ts_vad/train_fairseq2_style.py\
    --world-size 1 \
    --num-epochs 20\
    --start-epoch 1\
    --use-fp16 1\
    --exp-dir $exp_dir
fi


if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ];then
 exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad/exp7_debug_multi_gpu
 python3 ts_vad/train7.py\
    --world-size 2 \
    --num-epochs 30\
    --start-epoch 1\
    --use-fp16 1\
    --exp-dir $exp_dir
fi

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ];then
    . path_for_dia_pt2.4.sh
  #python3 FSDP_mnist.py
  export CUDA_LAUNCH_BLOCKING=1
  #TORCH_DISTRIBUTED_DEBUG=DETAIL python3 ts_vad1/FSDP_mnist_scaler.py
  TORCH_DISTRIBUTED_DEBUG=DETAIL python3 ts_vad1/DDP_mnist_scaler.py
fi
