#!/usr/bin/env python3
# Author: Duo MA
# Email: maduo@cuhk.edu.cn

import sys
from pathlib import Path
if __name__ == "__main__":
    indir=sys.argv[1]
    indir = Path(indir)
    for wav_path in indir.rglob("**/*.wav"):
        #print(wav_path)
        idx = wav_path.stem
        spkid = idx
        #print(spkid)
        content = str(wav_path).split("/")
        #print(content[-2])
        uttid = spkid + "-" + content[-2]
        print(f"{uttid} {wav_path}")
