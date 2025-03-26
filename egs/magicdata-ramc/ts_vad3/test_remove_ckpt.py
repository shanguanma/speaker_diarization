#!/usr/bin/env python3
# Author: Duo MA
# Email: maduo@cuhk.edu.cn

from checkpoint import find_checkpoints
from checkpoint import remove_checkpoints
if __name__ == "__main__":
    ckpt_dir="/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_nofreeze_with_musan_rirs_wavlm_epoch40/backup"
    ans = find_checkpoints(ckpt_dir)
    print(ans)
    remove_checkpoints(ckpt_dir,topk=10)


