#!/bin/bash

. "/cm/shared/apps/anaconda3/etc/profile.d/conda.sh"
#. "/mntnfs/lee_data1/maduo/anaconda3/etc/profile.d/conda.sh"
conda activate dia_pt2.4 # python=3.11 torch=2.4
export PYTHONPATH=/mntnfs/lee_data1/maduo/codebase/speaker_diarization:$PYTHONPATH
# python3
#>>> import torch
#>>> torch.__version__
#'2.4.0+cu118'
#>>> import torchaudio
#>>> torchaudio.__version__
#'2.4.0+cu118'
#>>> import numpy as np
#>>> np.__version__
#'2.0.1'
