#!/usr/bin/env python3

import sys
import os
from typing import Dict, List
from collections import defaultdict

class Segment(object):
    def __init__(self, uttid, stime, etime):
        self.uttid = uttid
        self.stime = round(stime, 3)
        self.etime = round(etime, 3)

    def change_stime(self, time):
        self.stime = time

    def change_etime(self, time):
        self.etime = time



def load_oracle_rttm(file: str,seg_dur: float):
    file_name = os.path.basename(file).split(".")[0]
    print(f"file_name: {file_name}")
    dir_name = os.path.dirname(file)
    out_rttm = f"{dir_name}/{file_name}_{seg_dur}dur.rttm"
    segments = []
    with open(file,'r')as f,open(out_rttm,'w')as fw:
        for line in f:
            line_list = line.strip().split()
            if float(line_list[4])<= float(seg_dur):
                fw.write(line)
                key=line_list[1]
                start=float(line_list[3])
                end = float(line_list[3])+float(line_list[4])
                segments.append(Segment(key,start,end))

    intervals = defaultdict(list)
    for i in range(len(segments)):
        interval = [segments[i].stime, segments[i].etime]
        intervals[segments[i].uttid].append(interval)
    #print(intervals)
    return intervals

def filter_sys_rttm(oracle_dict: Dict, seg_dur: float, file: str):
    file_name = os.path.basename(file).split(".")[0]
    dir_name = os.path.dirname(file)
    out_rttm = f"{dir_name}/{file_name}_strict_{seg_dur}dur.rttm"
    with open(file,'r')as f,open(out_rttm,'w')as fw:
        for line in f:
            line_list = line.strip().split()
            start=float(line_list[3])
            end = float(line_list[3])+float(line_list[4])
            if line_list[1] in oracle_dict.keys() and float(line_list[4])<= float(seg_dur):
                for interval in oracle_dict[line_list[1]]:
                    s,e = interval
                    if start>=s and end <=e:
                        fw.write(line)
            #for key in oracle_dict:
            #    if key==line_list[1] and float(line_list[4])<= float(seg_dur):
            #        for interval in oracle_dict[key]:
            #            s,e = interval
            #            if start<=s and end <=e:
            #                fw.write(line)






if __name__ == "__main__":
    oracle_rttm = sys.argv[1]
    seg_dur = sys.argv[2]
    sys_rttm = sys.argv[3]
    oracle_dict = load_oracle_rttm(oracle_rttm,seg_dur)
    filter_sys_rttm(oracle_dict, seg_dur, sys_rttm)



    oracle_file_name = os.path.basename(oracle_rttm).split(".")[0]
    oracle_dir_name = os.path.dirname(oracle_rttm)
    oracle_out_rttm = f"{oracle_dir_name}/{oracle_file_name}_{seg_dur}dur.rttm"
    sys_file_name = os.path.basename(sys_rttm).split(".")[0]
    sys_dir_name = os.path.dirname(sys_rttm)
    sys_out_rttm = f"{sys_dir_name}/{sys_file_name}_strict_{seg_dur}dur.rttm"
    os.system(f"perl SCTK-2.4.12/src/md-eval/md-eval.pl -v -x  -c 0.25 -r {oracle_out_rttm} -s {sys_out_rttm}")

