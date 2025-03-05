#!/usr/bin/env bash
.  path_for_speaker_diarization_hltsz.sh
torchrun --nproc_per_node=2 tests/test_ddp.py 
