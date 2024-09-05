#!/usr/bin/env bash

. path_for_speaker_diarization.sh
exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad/exp6
#python3 ts_vad/train6.py\
 python3 ts_vad/train6_1.py\
    --world-size 1 \
    --num-epochs 30\
    --start-epoch 1\
    --use-fp16 1\
    --exp-dir $exp_dir
