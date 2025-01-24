#!/usr/bin/env python3
# Author: Duo MA
# Email: maduo@cuhk.edu.cn
#result=""
#json_files=["/data/maduo/datasets/alimeeting/Eval_Ali/Eval_Ali_far/Eval.json","/data/maduo/exp/speaker_diarization/ts_vad2/data/magicdata-ramc/dev/dev.json"]
#for f in json_files:
#    with open(f, "r") as infile:
#        result += infile.read()
#with open("dev.json", "w") as outfile:
#    outfile.writelines(result)
import random
import sys
import os
out_put_dir = sys.argv[1]
name=sys.argv[2]
json_files = sys.argv[3:]
result=[]
#json_files=["./dev_10.json"]
for f in json_files:
    with open(f, 'r')as infile:
        for line in infile:
            line = line.strip()
            result.append(line)

random.shuffle(result)
#with open("shuf_dev_10.json",'w')as fw:
os.makedirs(f"{out_put_dir}/{name}",exist_ok=True)
with open(f"{out_put_dir}/{name}/{name}.json",'w')as fw:
    for i in result:
        fw.write(f"{i}\n")
