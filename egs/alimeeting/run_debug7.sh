#!/usr/bin/env bash

. path_for_speaker_diarization.sh
exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad/exp7
python3 ts_vad/train7.py\
    --world-size 1 \
    --num-epochs 30\
    --start-epoch 12\
    --use-fp16 1\
    --exp-dir $exp_dir
