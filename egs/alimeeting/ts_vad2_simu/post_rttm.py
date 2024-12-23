#!/usr/bin/env python3
# Author: Duo MA
# Email: maduo@cuhk.edu.cn
import sys
if __name__ == "__main__":
    #  # origin rttm
    # SPEAKER /mntcephfs/lab_data/maduo/datasets/simu_alimeeting/200h_fixed_4spks/eval/conv_val_dir/SPK8022-R8009_M8018_MS809_191_SPK8024-R8009_M8019_MS810_335_SPK8067-R8007_M8011_MS806_116_SPK8015-R8001_M8004_MS801_70.conv 1 0.310 2.590 <NA> <NA> SPK8024-R8009_M8019_MS810_335 <NA> <NA>
    input = sys.argv[1]

    #SPEAKER SPK8022-R8009_M8018_MS809_191_SPK8024-R8009_M8019_MS810_335_SPK8067-R8007_M8011_MS806_116_SPK8015-R8001_M8004_MS801_70 1 0.310 2.590 <NA> <NA> SPK8024 <NA> <NA>
    #output = sys.argv[2]
    with open(input,'r')as f:
        for line in f:
            line = line.strip().split()
            uttid = line[1].split("/")[-1].split(".")[0]
            spkid = line[7].split("-")[0]
            print(f"SPEAKER {uttid} {line[2]} {line[3]} {line[4]} <NA> <NA> {spkid} <NA> <NA>")


