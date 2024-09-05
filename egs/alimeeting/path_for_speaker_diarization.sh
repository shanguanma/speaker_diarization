#!/bin/bash

. "/cm/shared/apps/anaconda3/etc/profile.d/conda.sh"
#. "/mntnfs/lee_data1/maduo/anaconda3/etc/profile.d/conda.sh"
conda activate speaker_diarization # python=3.11 torch=2.1.1 torchaudio=2.1.1
export PYTHONPATH=/mntnfs/lee_data1/maduo/codebase/speaker_diarization:$PYTHONPATH
# python3
#Python 3.11.9 (main, Apr 19 2024, 16:48:06) [GCC 11.2.0] on linux
#Type "help", "copyright", "credits" or "license" for more information.
#>>> import torch
#>>> torch.__version__
#'2.1.1+cu118'
#>>> import numpy as np
#>>> np.__version__
#'1.24.0'
