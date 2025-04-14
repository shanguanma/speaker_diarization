#!/bin/bash

.  "/maduo/miniconda3/etc/profile.d/conda.sh"
#. "/mntnfs/lee_data1/maduo/anaconda3/etc/profile.d/conda.sh"
conda activate dia_cuda11.8_py311 # python=3.11 torch=2.4
export PYTHONPATH=/maduo/codebase/speaker_diarization:$PYTHONPATH
