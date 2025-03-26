#!/usr/bin/env python3
import os
import re
import sys
import shutil
import subprocess
import numpy as np
import tempfile

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
#MDEVAL_BIN = os.path.join(SCRIPT_DIR, 'md-eval-22.pl')
MDEVAL_BIN = os.path.join(SCRIPT_DIR, "md-eval.pl")
FILE_REO = re.compile(r'(?<=Speaker Diarization for).+(?=\*\*\*)')
SCORED_SPEAKER_REO = re.compile(r'(?<=SCORED SPEAKER TIME =)[\d.]+')
MISS_SPEAKER_REO = re.compile(r'(?<=MISSED SPEAKER TIME =)[\d.]+')
FA_SPEAKER_REO = re.compile(r'(?<=FALARM SPEAKER TIME =)[\d.]+')
ERROR_SPEAKER_REO = re.compile(r'(?<=SPEAKER ERROR TIME =)[\d.]+')

def der(ref_rttm: str, sys_rttm: str, collar: float=0.25, ignore_overlaps=False):
    tmp_dir = tempfile.mkdtemp()
    # Actually score.
    try:
        cmd = [MDEVAL_BIN,
               '-af',
               #'-af'
               '-r', ref_rttm,
               '-s', sys_rttm,
               '-c', str(collar),
               #'-u', uemf,
              ]
        if ignore_overlaps:
            cmd.append('-1')
        stdout = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        stdout = e.output
    finally:
        shutil.rmtree(tmp_dir)

    # Parse md-eval output to extract by-file and total scores.
    stdout = stdout.decode('utf-8')
    print(f"stdout: {stdout}")
    file_ids = [m.strip() for m in FILE_REO.findall(stdout)]
    file_ids = [file_id[2:] if file_id.startswith('f=') else file_id
                for file_id in file_ids]
    scored_speaker_times = np.array(
        [float(m) for m in SCORED_SPEAKER_REO.findall(stdout)])
    miss_speaker_times = np.array(
        [float(m) for m in MISS_SPEAKER_REO.findall(stdout)])
    fa_speaker_times = np.array(
        [float(m) for m in FA_SPEAKER_REO.findall(stdout)])
    error_speaker_times = np.array(
        [float(m) for m in ERROR_SPEAKER_REO.findall(stdout)])
    print(f"miss_speaker_times: {miss_speaker_times}")

    with np.errstate(invalid='ignore', divide='ignore'):
        error_times = miss_speaker_times + fa_speaker_times + error_speaker_times
        ders = error_times / scored_speaker_times
        print(f"ders: {ders}")
if __name__ == "__main__":
    ref_rttm=sys.argv[1]
    sys_rttm=sys.argv[2]
    der(ref_rttm, sys_rttm)
