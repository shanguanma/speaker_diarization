#!/usr/bin/env python3
# Author: Duo MA
# Email: maduo@cuhk.edu.cn


import sys
from pathlib import Path
if __name__ == "__main__":
    inp_dir= sys.argv[1]
    inp_dir = Path(inp_dir)
    for audio_path in inp_dir.rglob("**/*.wav"):
        idx = audio_path.stem
        print(f"{idx} {str(audio_path)}")
