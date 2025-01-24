#!/usr/bin/env python3
# Author: Duo MA
# Email: maduo@cuhk.edu.cn

import glob
import sys
import os
if __name__ == "__main__":
     out_dir=sys.argv[1]
     #name=sys.argv[2]
     fdirs=sys.argv[2:]
     for fdir in fdirs:
         for path in glob.glob(f"{fdir}/*"):
             print(f"path: {path}!")
             if not path.endswith("wavs.txt"):
                 print(path)
                 target = path.split("/")[-1]
                 print(target)
                 cmd = f"ln -svf {path} {out_dir}/{target}"
                 os.system(cmd)
