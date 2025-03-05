#!/usr/bin/env python

import sys
import librosa

if __name__ == "__main__":
    inp_wavscp= sys.argv[1]
    with open(inp_wavscp, 'r') as f:
        for line in f:
            line = line.strip().split()
            audio_path = line[-1]
            y, sr = librosa.load(audio_path,sr=16000)
            dur_second = librosa.get_duration(y=y, sr=sr)
            print(f"{line[0]} 1 0.000 {dur_second:.3f}")
