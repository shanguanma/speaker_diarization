# this code is modified from https://github.com/MagicHub-io/MagicData-RAMC/blob/main/sd/scripts/prepare_magicdata_180h.py
import os
import re
import sys
def main():
    #txt_dir = "/mntcephfs/lee_dataset/asr/MagicData-RAMC/MDT2021S003/TXT"
    #wav_dir = "/mntcephfs/lee_dataset/asr/MagicData-RAMC/MDT2021S003/WAV"
    #output_dir= "/mntnfs/lee_data1/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/data/magicdata-RAMC_debug"
    txt_dir = sys.argv[1]
    wav_dir = sys.argv[2]
    output_dir = sys.argv[3]
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/train", exist_ok=True)
    os.makedirs(f"{output_dir}/dev", exist_ok=True)
    os.makedirs(f"{output_dir}/test", exist_ok=True)
    txt_files = os.listdir(txt_dir)

    if len(txt_files) != 351:
        raise Exception("Dataset is incomplete. Please check dataset.")

    if len(os.listdir(wav_dir)) != 351:
        raise Exception("Dataset is incomplete. Please check dataset.")

    patt = '(\d+)-(\d+)-(\d+)-(\d+)'
    for i in range(len(txt_files)-1):
        for x in range(i+1, len(txt_files)):
            j = 1
            while j<5:
                # print(re.match(patt, txt_files[i]))
                lower = re.search(patt, txt_files[i]).group(j)
                # print(lower)
                upper = re.search(patt, txt_files[x]).group(j)
                if int(lower) < int(upper):
                    j = 5
                elif int(lower) == int(upper):
                    j += 1
                else:
                    txt_files[i],txt_files[x] = txt_files[x],txt_files[i]
                    j = 5

    # print(txt_files)

    seg_writter_train = open(f"{output_dir}/train/segments", "w+")
    scp_writter_train = open(f"{output_dir}/train/wav.scp", "w+")
    text_writter_train = open(f"{output_dir}/train/text", "w+")
    rttm_writter_train = open(f"{output_dir}/train/rttm", "w+")
    #rttm_writter_train = open(f"{output_dir}/train/rttm_debug", "w+")
    utt2spk_writter_train = open(f"{output_dir}/train/utt2spk", "w+")

    seg_writter_dev = open(f"{output_dir}/dev/segments", "w+")
    scp_writter_dev = open(f"{output_dir}/dev/wav.scp", "w+")
    text_writter_dev = open(f"{output_dir}/dev/text", "w+")
    rttm_writter_dev = open(f"{output_dir}/dev/rttm", "w+")
    #rttm_writter_dev = open(f"{output_dir}/dev/rttm_debug", "w+")
    utt2spk_writter_dev = open(f"{output_dir}/dev/utt2spk", "w+")

    seg_writter_test = open(f"{output_dir}/test/segments", "w+")
    scp_writter_test = open(f"{output_dir}/test/wav.scp", "w+")
    text_writter_test = open(f"{output_dir}/test/text", "w+")
    rttm_writter_test = open(f"{output_dir}/test/rttm", "w+")
    #rttm_writter_test = open(f"{output_dir}/test/rttm_debug", "w+")
    utt2spk_writter_test = open(f"{output_dir}/test/utt2spk", "w+")

    for index, txt_file in enumerate(txt_files):
        name = txt_file[:-4]
        wav_name = name + '.wav'
        wav_path = os.path.join(wav_dir, wav_name)
        txt_reader = open(os.path.join(txt_dir, txt_file), )
        txt_all = txt_reader.readlines()

        scp_line = name + ' ' + wav_path + '\n'
        if index < 43:
            scp_writter_test.writelines(scp_line)
        elif index < 332:
            scp_writter_train.writelines(scp_line)
        else:
            scp_writter_dev.writelines(scp_line)

        for line in txt_all:
            splited_str = line.split("\t", 4)
            time = splited_str[0][1:-1]
            time_str = time.split(",")
            start_time = time_str[0]
            end_time = time_str[1]
            during_time = str(round(float(end_time) - float(start_time), 2))
            person_id = splited_str[1]
            text = splited_str[3][:-1]
            if person_id =="G00000000":
                continue
            rttm_line = 'SPEAKER ' + name + ' 1 ' + start_time + ' ' + during_time + ' <NA> <NA> ' + person_id + ' <NA> <NA>\n'
            start_time_int = start_time.replace('.', '')
            end_time_int = end_time.replace('.', '')
            seg_line = person_id + '_' + name + '_' + start_time_int + '_' + end_time_int + ' ' + name + ' ' + start_time + ' ' + end_time + '\n'

            text_line =  person_id + '_' + name + '_' + start_time_int + '_' + end_time_int + ' ' + text + '\n'
            utt2spk_line = person_id + '_' + name + '_' + start_time_int + '_' + end_time_int + ' ' + person_id + '\n'

            if index < 43:
                rttm_writter_test.writelines(rttm_line)
            elif index < 332:
                rttm_writter_train.writelines(rttm_line)
            else:
                rttm_writter_dev.writelines(rttm_line)

            if index < 43:
                seg_writter_test.writelines(seg_line)
                text_writter_test.writelines(text_line)
                utt2spk_writter_test.writelines(utt2spk_line)
            elif index < 332:
                seg_writter_train.writelines(seg_line)
                text_writter_train.writelines(text_line)
                utt2spk_writter_train.writelines(utt2spk_line)
            else:
                seg_writter_dev.writelines(seg_line)
                text_writter_dev.writelines(text_line)
                utt2spk_writter_dev.writelines(utt2spk_line)


    return 0

if __name__ == "__main__":
    main()
