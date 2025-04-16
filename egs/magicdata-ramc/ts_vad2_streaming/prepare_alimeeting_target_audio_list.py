#!/usr/bin/env python

import os

import sys
import glob

def main():
    input_dir=sys.argv[1]
    output = sys.argv[2]
    #input_dir=
    with open(output,'w')as fw:
        #for x in os.listdir(input_dir):
        for x in glob.glob(f"{input_dir}/*/*.wav"):
            if 'all' not in x:
                print(f"x: {x}!")
                fw.write(f"{x}\n")

if __name__=="__main__":
    main()
