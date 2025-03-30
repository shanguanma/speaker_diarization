#/usr/bin/env python3
import sys
import os

def load_specify_seg_rttm(file: str, seg_dur: float):
    file_name = os.path.basename(file).split(".")[0]
    dir_name = os.path.dirname(file)
    out_rttm = f"{dir_name}/{file_name}_{seg_dur}dur.rttm"
    with open(file,'r')as f,open(out_rttm,'w')as fw:
        for line in f:
            line_list = line.strip().split()
            if float(line_list[4])<= float(seg_dur):
                fw.write(line)


if __name__ == "__main__":
    oracle_rttm = sys.argv[1]
    seg_dur = sys.argv[2]
    sys_rttm = sys.argv[3]
    load_specify_seg_rttm(oracle_rttm,seg_dur)
    load_specify_seg_rttm(sys_rttm,seg_dur)
    oracle_file_name = os.path.basename(oracle_rttm).split(".")[0]
    oracle_dir_name = os.path.dirname(oracle_rttm)
    oracle_out_rttm = f"{oracle_dir_name}/{oracle_file_name}_{seg_dur}dur.rttm"
    sys_file_name = os.path.basename(sys_rttm).split(".")[0]
    sys_dir_name = os.path.dirname(sys_rttm)
    sys_out_rttm = f"{sys_dir_name}/{sys_file_name}_{seg_dur}dur.rttm"
    os.system(f"perl SCTK-2.4.12/src/md-eval/md-eval.pl -v -x  -c 0.25 -r {oracle_out_rttm} -s {sys_out_rttm}")

