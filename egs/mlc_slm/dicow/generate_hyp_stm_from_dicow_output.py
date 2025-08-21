#!/usr/bin/env python3

"""
dicow output the below format txt:
🗣️ Speaker 0:
<|3.26|>hi carrie how are you today<|4.92|>
<|8.30|>i am very well thank you for asking and how was your day today<|12.06|>
<|28.86|>it was great i did a lot of stuff i cleaned the house i went to the grocery store<|36.48|>
<|36.66|>for garbage<|38.10|>
<|43.74|>so today we are going to talk about traveling<|48.00|>
<|49.18|>do you like to travel<|50.84|>
<|53.08|>how often do you travel<|54.74|>
<|88.38|>not recently but i just went to the to croatia ten days to roving and that was very very nice<|98.82|>
<|107.84|>yes that is correct yes i have been to croatia many many years ago to kvar with my best friend and we stayed for like two weeks and it was it was beautiful the the smell of the sea the people the food<|123.48|>
<|124.50|>and just the temperature of the sea i like to swim a lot<|127.90|>
<|160.70|>yeah because we d we do have the cold ocean<|163.68|>
<|164.10|>so you can not really swim in the ocean and it is always windy in the beaches<|168.90|>
<|183.14|>have you ever been abroad<|184.86|>
<|198.94|>yeah to traveling abroad i like europe i like the cities over there<|203.40|>
<|203.58|>germany<|205.72|>
<|206.10|>england<|207.92|>
<|208.10|>italy<|209.08|>
<|212.10|>i have been to france<|213.46|>
....

🗣️ Speaker 1:
<|1.14|>hi ivy<|2.96|>
<|5.32|>very well thank you and you<|8.28|>
<|12.42|>it was very good most of my day i spent<|17.40|>
<|18.32|>studying but afterwards i saw a friend that i have not seen for a while so that was very good and yours<|28.04|>
<|39.86|>well fun fun fun yeah<|42.44|>
<|51.40|>i love traveling<|52.84|>
<|55.14|>well i i like to go every year i like to go on a trip to another country and to visit like the all the cities and villages that are the the country is famous for<|72.86|>
<|73.20|>and also i like to go skiing and to go on a vacation like to the beach or something like that to the sea<|85.02|>
<|85.30|>how about you<|86.56|>
<|99.34|>have you ever been before to a croatian sea i think that is a adriatic sea is that right<|107.02|>
<|127.98|>yeah<|128.74|>
<|128.86|>i think since we are from america but n usually people do not travel to europe as much<|137.96|>
<|138.12|>but i prefer going to europe because it is much different than america i think the cities are the cities have different atmosphere they are<|147.48|>
<|148.68|>especially when when there is sea in the city and the beaches i think the the atmosphere is really nice and people are more relaxed and completely different than us<|160.56|>
<|169.12|>yeah and and i mean i am pretty scared of sharks so i do not just like<|175.22|>
<|175.58|>go swimming you know far away in the sea i just you know dip my toes<|180.22|>
<|185.88|>yeah as i said so i went to europe a lot and of course to a lot of places in america but europe i i prefer europe<|195.40|>
<|209.12|>yeah france have you been to france<|212.02|>
<|232.16|>yeah<|233.00|>
....


it is coverted to the below format txt:
American_0517_007 1 O1 1.8205 2.7620 hi ivy
American_0517_007 1 O2 3.1600 4.9839 hi carrie how are you today
American_0517_007 1 O1 5.2228 8.2700 i'm very well thank you and you
American_0517_007 1 O2 8.2700 12.0483 i'm very well thank you for asking and how was your day today
American_0517_007 1 O1 12.4297 19.5736 it was very good uh most of my day i uh spent hmm studying
American_0517_007 1 O1 19.8883 27.8135 but afterwards i saw a friend that i haven't seen for a while so that was very good and yours
American_0517_007 1 O2 28.9536 34.1489 it was great i i did a lot of stuff i cleaned the house um
American_0517_007 1 O2 34.1596 39.3061 i went to the grocery store for garbage yeah
American_0517_007 1 O2 43.6529 47.9869 so today we are going to talk about traveling
American_0517_007 1 O2 49.2654 50.7212 do you like to travel
American_0517_007 1 O1 51.1473 52.7312 i love traveling
American_0517_007 1 O2 53.0821 54.7164 how often do you travel
American_0517_007 1 O1 58.0821 65.8983 i like to go e- every year i like to go on um a trip uh to another country
....

"""
from pathlib import Path
import re
import sys
import glob


def convert_to_second(inp: str):
    return round(float(inp), 4)


def process(inp_dir: str, output: str):
    text_lists = glob.glob(f"{inp_dir}/*.txt", recursive=False)
    pattern = r"<\|([\d.]+)\|>"
    target = ""
    with open(output, "w") as fw:
        for line in text_lists:
            line = line.strip()
            uttid = Path(line).stem
            single_txt_list = open(line, "r").readlines()
            res = split_list(single_txt_list, "\n")  # use "\n" to split list
            for spk_content in res:
                spkid = "spk" + spk_content[0].strip().split(" ")[-1].split(":")[0]
                for li in spk_content[1:]:
                    li = li.strip()
                    matches = re.finditer(pattern, li)
                    timestamps = [(float(match.group(1))) for match in matches]
                    text = re.sub(pattern, target, li)
                    fw.write(
                        f"{uttid} 1 {spkid} {convert_to_second(timestamps[0])} {convert_to_second(timestamps[1])} {text}\n"
                    )


def split_list(lst, delimiter):
    """
    使用指定元素作为分隔符切分列表

    参数:
    lst -- 待切分的列表
    delimiter -- 用作分隔符的元素

    返回:
    一个包含子列表的列表，每个子列表是原列表中被分隔符切分的部分
    """
    result = []
    current_sublist = []

    for item in lst:
        if item == delimiter:
            if current_sublist:  # 避免添加空的子列表
                result.append(current_sublist)
                current_sublist = []
        else:
            current_sublist.append(item)

    # 添加最后一个子列表（如果有内容）
    if current_sublist:
        result.append(current_sublist)

    return result


def test_single_text_process():
    inp_file = (
        "/maduo/exp/asr_sd/dicow_offical_model_inference/American/American_0517_007.txt"
    )
    uttid = Path(inp_file).stem
    output_file = "./output.stm"
    target = ""
    pattern = r"<\|([\d.]+)\|>"
    with open(inp_file, "r") as f, open(output_file, "w") as fw:
        lines = f.readlines()
        res = split_list(lines, "\n")
        for spk_content in res:
            spkid = "spk" + spk_content[0].strip().split(" ")[-1].split(":")[0]
            for line in spk_content[1:]:
                line = line.strip()
                matches = re.finditer(pattern, line)
                timestamps = [(float(match.group(1))) for match in matches]
                text = re.sub(pattern, target, line)
                fw.write(
                    f"{uttid} 1 {spkid} {convert_to_second(timestamps[0])} {convert_to_second(timestamps[1])} {text}\n"
                )


if __name__ == "__main__":
    inp_dir = sys.argv[1]
    output = sys.argv[2]
    process(inp_dir, output)
