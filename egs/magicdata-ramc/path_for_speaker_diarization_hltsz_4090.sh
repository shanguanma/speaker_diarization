#!/bin/bash

#. "/data1/home/maduo/miniconda3/etc/profile.d/conda.sh"
#. "/mntnfs/lee_data1/maduo/anaconda3/etc/profile.d/conda.sh"
#. "/share/workspace/maduo/miniconda3/etc/profile.d/conda.sh"

#conda activate cu118_py311_dia # python=3.11 torch=2.5.1

source /share/workspace/maduo/codebase/speaker_diarization/uv_env/bin/activate
export PYTHONPATH=/share/workspace/maduo/codebase/speaker_diarization:$PYTHONPATH
