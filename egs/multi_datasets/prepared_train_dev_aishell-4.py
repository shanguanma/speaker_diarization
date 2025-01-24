#!/usr/bin/env python3
import sys
import random
def file2dict(inp: str):
    out={}
    keys=[]
    with open(inp, 'r') as f:
        for line in f:
            line = line.strip()
            line_ =line .split('/')
            key = line_[-1].split(".")[0]
            keys.append(key)
            if key not in out:
                out[key] = line
    return out,keys


if __name__ == "__main__":
    inp_rttms=sys.argv[1]
    inp_wavs=sys.argv[2]
    out_train_rttms=sys.argv[3]
    out_train_wavs=sys.argv[4]
    out_dev_rttms=sys.argv[5]
    out_dev_wavs=sys.argv[6]

    out,keys = file2dict(inp_rttms)
    out_wavs,_ = file2dict(inp_wavs)

    dev_keys = random.sample(keys,10)
    print(f"dev_keys: {dev_keys}")
    train_keys = [i for i in keys if i not in dev_keys]
    print(f"train_keys: {train_keys}")
    # output train rttm
    with open(out_train_rttms,'w') as fw_train_rttms:
        for key in out.keys():
            if key in train_keys:
                with open(out[key], 'r')as f:
                    for line in f:
                        line = line.strip()
                        fw_train_rttms.write(f"{line}\n")
    # output dev rttm
    with open(out_dev_rttms,'w') as fw_dev_rttms:
        for key in out.keys(): 
            if key in dev_keys:
                with open(out[key], 'r')as f:
                    for line in f:
                        line = line.strip()
                        fw_dev_rttms.write(f"{line}\n")

    # output train wav.scp
    with open(out_train_wavs,'w') as fw_train_wavs:
        for key in out_wavs.keys():
            if key in train_keys:
                fw_train_wavs.write(f"{key} {out_wavs[key]}\n")

    # output dev wav.scp
    with open(out_dev_wavs,'w') as fw_dev_wavs:
        for key in out_wavs.keys():
            if key in dev_keys:
                fw_dev_wavs.write(f"{key} {out_wavs[key]}\n")

