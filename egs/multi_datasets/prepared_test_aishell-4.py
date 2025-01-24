#!/usr/bin/env python3

import sys
from pathlib import Path
if __name__ == "__main__":
    inp_dir=Path(sys.argv[1])
    out_dir=Path(sys.argv[2])
    inp_rttm_dir=inp_dir/"TextGrid"
    inp_wav_dir=inp_dir/"wav"
    out_rttm=out_dir/"test.rttm"
    out_wavscp=out_dir/"wav.scp"
    with open(out_rttm,'w')as fw_rttm:
        for file in inp_rttm_dir.rglob("*.rttm"):
            with open(file,'r')as fr:
                for line in fr:
                    line = line.strip()
                    fw_rttm.write(f"{line}\n")


    with open(out_wavscp,'w')as fw_wavscp:
        for file in inp_wav_dir.rglob("*.flac"):
            key = str(file).split("/")[-1].split(".")[0]
            fw_wavscp.write(f"{key} {str(file)}\n")
