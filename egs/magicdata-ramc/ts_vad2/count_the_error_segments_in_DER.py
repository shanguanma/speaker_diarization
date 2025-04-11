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
#FILE_REO = re.compile(r'(file=).+(?=\*\*\*)')
FILE_REO = re.compile(r"file=\*")
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
    print(f"md-eval stdout: {stdout}")

    speaker_errors_list=[]
    miss_speaker_list=[]
    falarm_speaker_list=[]
    scored_speech_list=[]
    for line in stdout.split("\n"):
    #for line in FILE_REO.findall(stdout):
        line = line.strip()
        #print(f"line sb: {line}")
        if "file=" in line:
            #print(f"line: {line}")
            line = line.split(",")
            #print(line)
            if line[1]=="ref_type=falarm_speaker_segs":
                #line[2].split("=")[-1]
                falarm_speaker_list.append(line[2].split("=")[-1])
            elif line[1]=="ref_type=miss_speaker_segs":
                miss_speaker_list.append(line[2].split("=")[-1])
            elif line[1]=="ref_type=speaker_error_segs":
                speaker_errors_list.append(line[2].split("=")[-1])
            elif line[1]=="ref_type=scored_speech_segs":
                scored_speech_list.append(line[2].split("=")[-1])
    #lines = [ line for line in stdout.split("\n") if "file=" in line.strip()]
    #print(lines)
    #falarm_speaker_list=[line for line in lines if line.split()[1] == "ref_type=falarm_speaker_segs" ]
    #print(falarm_speaker_list)

    falarm_speaker_segs = [float(i) for m in falarm_speaker_list for i in m.split()] 
    miss_speaker_segs = [float(i) for m in miss_speaker_list for i in m.split()]
    speaker_errors_segs = [float(i) for m in speaker_errors_list for i in m.split()]
    scored_speech_segs = [float(i) for m in scored_speech_list for i in m.split()] 
    #print(falarm_speaker_segs)
    print("""================""")
    print(f"falarm_speaker <=0.5 seconds segs numbers: {len([i for i in falarm_speaker_segs if i<=0.5])}")
    print(f"0.5<falarm_speaker <=1 seconds segs numbers: {len([i for i in falarm_speaker_segs if 0.5<i<=1])}")
    print(f"1<falarm_speaker <=2 seconds segs numbers: {len([i for i in falarm_speaker_segs if 1<i<=2])}")
    print(f"2<falarm_speaker seconds segs numbers: {len([i for i in falarm_speaker_segs if 2<i])}")
    print("""================""")    
    print(f"miss_speaker <=0.5 seconds segs numbers: {len([i for i in miss_speaker_segs if i<=0.5])}")
    print(f"0.5<miss_speaker <=1 seconds segs numbers: {len([i for i in miss_speaker_segs if 0.5<i<=1])}")
    print(f"1<miss_speaker <=2 seconds segs numbers: {len([i for i in miss_speaker_segs if 1<i<=2])}")
    print(f"2<miss_speaker seconds segs numbers: {len([i for i in miss_speaker_segs if 2<i])}")
    print("""================""")
    print(f"speaker_errors <=0.5 seconds segs numbers: {len([i for i in speaker_errors_segs if i<=0.5])}")
    print(f"0.5<speaker_errors <=1 seconds segs numbers: {len([i for i in speaker_errors_segs if 0.5<i<=1])}")
    print(f"1<speaker_errors <=2 seconds segs numbers: {len([i for i in speaker_errors_segs if 1<i<=2])}")
    print(f"2<speaker_errors seconds segs numbers: {len([i for i in speaker_errors_segs if 2<i])}")

    print("""================""")
    print(f"scored_speech_segs numbers: {len(scored_speech_segs)}")

    #for m in falarm_speaker_list:
    #    for i in m.split():
    #        print(f"seg: {i}")
    #lines = [ line for line in stdout.split("\n") if "file=" in line.strip()]                
    #print(lines)
    
    #file_ids = [m.strip() for m in FILE_REO.findall(stdout)]
    #print(f"file_ids: {file_ids}")
    #file_ids = [file_id[2:] if file_id.startswith('f=') else file_id
    #            for file_id in file_ids]
    #scored_speaker_times = np.array(
    #    [float(m) for m in SCORED_SPEAKER_REO.findall(stdout)])
    #miss_speaker_times = np.array(
    #    [float(m) for m in MISS_SPEAKER_REO.findall(stdout)])
    #fa_speaker_times = np.array(
    #    [float(m) for m in FA_SPEAKER_REO.findall(stdout)])
    #error_speaker_times = np.array(
    #    [float(m) for m in ERROR_SPEAKER_REO.findall(stdout)])
    #print(f"miss_speaker_times: {miss_speaker_times}")

    #with np.errstate(invalid='ignore', divide='ignore'):
    #    error_times = miss_speaker_times + fa_speaker_times + error_speaker_times
    #    ders = error_times / scored_speaker_times
    #    print(f"ders: {ders}")
if __name__ == "__main__":
    ref_rttm=sys.argv[1]
    sys_rttm=sys.argv[2]
    der(ref_rttm, sys_rttm)
