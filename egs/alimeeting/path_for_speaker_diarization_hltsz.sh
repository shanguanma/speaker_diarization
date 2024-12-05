#!/bin/bash

. "/home/anaconda3/etc/profile.d/conda.sh"
#. "/mntnfs/lee_data1/maduo/anaconda3/etc/profile.d/conda.sh"
conda activate speaker_diarization # python=3.11 torch=2.1.2 torchaudio=2.1.2
export PYTHONPATH=/workspace/maduo/codebase/speaker_diarization:$PYTHONPATH
#>>> import torch
#>>> torch.__version__
#'2.1.2+cu118'
#>>> import modelscope
#>>> modelscope.__version__
#'1.17.1'
#>>> import torchaudio
#>>> torchaudio.__version__
#'2.1.2+cu118'
#>>> import numpy as np
#>>> np.__version__
#'1.23.0'
