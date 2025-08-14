import sys
import re
import argparse
from pathlib import Path
import pysrt

def convert_to_second(inp:str):
    """
    "00:00:01,432" -> 1.4320
    """
    a = inp
    tot = int(a.split(":")[0])*3600 + int(a.split(":")[1])*60 + int(a.split(":")[2].split(',')[0])+ int(a.split(":")[2].split(',')[1])/1000
    return round(tot,4)

def text_normalization(input_line_text):
    custom_punctuations = r'!"#$%&()*+,./:;<=>?@[\\]^_`{|}~。、？！・¿¡，'
    punctuation_pattern = re.compile(f'[{re.escape(custom_punctuations)}]')
    input_line_text = input_line_text.strip()
    ori_text = input_line_text.lower()
    text_tn = punctuation_pattern.sub('', ori_text)
    text_tn = re.sub(r' +', ' ', text_tn)
    return text_tn

def gen_hyp_stm(inp_srt_file_list: str, output: str):
    with open(inp_srt_file_list, "r", encoding='utf-8') as fin, open(output, 'w', encoding='utf-8')as fw:
        for line in fin:
            line = line.strip()
            uttid = Path(line).stem
            fsrt = pysrt.open(line)
            for lin in fsrt:
                if ":" in str(lin.text):
                    # normal case:
                    #230
                    #00:18:48,533 --> 00:18:51,417
                    #[SPEAKER_01]: But I hope I will go soon with my parents.
                    spkid = str(lin.text).split(':')[0]
                    spkid = re.sub(r"\[",'',spkid)
                    spkid =  re.sub(r"\]",'',spkid)
                    fw.write(f"{uttid} 1 {spkid} {convert_to_second(str(lin.start))} {convert_to_second(str(lin.end))} {text_normalization(str(lin.text).split(':')[-1])}\n")
                else:
                    # for case:
                    #231
                    #00:18:51,958 --> 00:18:52,178
                    #Yeah, yeah.
                    spkid = "SPEAKER_empty"
                    fw.write(f"{uttid} 1 {spkid} {convert_to_second(str(lin.start))} {convert_to_second(str(lin.end))} {text_normalization(str(lin.text))}\n")
                     
                    
    



if __name__ == "__main__":
    inp_srt_file_list = sys.argv[1]
    output = sys.argv[2]
    gen_hyp_stm(inp_srt_file_list, output)
    

