#!/usr/bin/env python3
# Author: Duo MA
# Email: maduo@cuhk.edu.cn

import subprocess
import sys
from pathlib import Path
from tqdm import tqdm
import os
if __name__ == "__main__":
    input_dir=sys.argv[1]
    out_mono_dir= sys.argv[2]
    input_dir = Path(input_dir)
    out_mono_dir = Path(out_mono_dir)
    os.makedirs(out_mono_dir, exist_ok=True)
    for path in tqdm(list(input_dir.rglob("*.wav")), desc=f"Preparing mono audio"):
        print(path)
        print(f"os.path.basename(path): {os.path.basename(path)}")
        wav_path_mono = out_mono_dir/f"{os.path.basename(path)}"
        cmd = f"sox {path} -c 1 {wav_path_mono}"
        subprocess.run(cmd, shell=True, check=True)
