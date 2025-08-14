import sys
import glob
import os
from pathlib import Path
def gen_stm():
    in_dir = sys.argv[1]
    output_dir=sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)
    out_file = f"{output_dir}/ref.stm"
    #inputs = glob.glob(f"{in_dir}/*.txt")
    fw = open(out_file,'w')
    for file in glob.glob(f"{in_dir}/*.txt"):
        file_name = Path(file).stem
        print(f"file_name: {file_name}")
        with open(file,'r') as f:
            for line in f:
                line = line.strip().split("\t")
                start_str="".join(str(round(float(line[0]),3)).split("."))
                end_str ="".join(str(round(float(line[1]),3)).split(".")) 
                start = round(float(line[0]),3))
                end = round(float(line[1]),3))
                spkid=line[2]
                text = line[-1]
                fw.write(f"{file_name}_{start_str}-{end_str}_")

if __name__ == "__main__":
   gen_stm()


