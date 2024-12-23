#!/usr/bin/env python3
# Author: Duo MA
# Email: maduo@cuhk.edu.cn

import glob, tqdm, os, textgrid, soundfile, copy, json, argparse, numpy, torch, wave
from collections import defaultdict
from typing import List
import logging
import time
"""
因为我们使用的单说话人语音句子进行模拟对话式的，且由overlap的diarization 训练数据
因此，我们使用模拟后的rttm 获得每一个句子的label(i.e.*.json)
然后利用rttm 的句子id 和rttm 的spkid 来获得对应的target speaker speech , 而不再利用rttm 的移除overlap 后的获得target speaker speech
"""
class Segment(object):
    def __init__(self, uttid, spkr, stime, etime, name,tgt_spkr_speech):
        self.uttid = uttid # filename
        self.spkr = spkr # speaker_key
        self.stime = round(stime, 2)
        self.etime = round(etime, 2)
        self.name = name # speaker_id
        self.tgt_spkr_speech = tgt_spkr_speech

    def change_stime(self, time):
        self.stime = time

    def change_etime(self, time):
        self.etime = time



def wavscp2dict(wavscp:str):
    scp2dict={}
    with open(wavscp, 'r')as f:
        for line in f:
            line = line.strip().split()
            scp2dict[line[0]] = line[1]
    return scp2dict
def load_wavs(args):
    audio_wavforms = glob.glob(args.audio_dir + "/*") # mix audio
    return audio_wavforms

def process(args,audio_wavforms):
    #audio_wavforms = glob.glob(args.audio_dir + "/*") # mix audio
    #print(f"audio_wavforms: {audio_wavforms}")
    outs = open(args.out_text, "w")
    scp2dict=wavscp2dict(args.target_speaker_speech_wavscp) # load target speaker speech into dictionary.

    for audio_wavform in tqdm.tqdm(audio_wavforms):
        # audio_wavform is str
        uttid = os.path.splitext(os.path.basename(audio_wavform))[0]
        #print(f"uttid:{uttid}")
        segments = []
        spk = {}
        num_spk = 1
        with open(args.simu_rttm, 'r')as f:
            for line in f:
                # i.e.:SPEAKER SPK0176-R0014_M0086_MS006_SPK2678-R2001_M2205_MS204 1 0.000 243.980 <NA> <NA> SPK0176-R0014_M0086_MS006 <NA> <NA>
                line = line.strip().split()
                if line[1] == uttid:
                    #spk_name_id = uttid + "_" + "spk" + str(line[7]) # The actual speaker name is bound to the audio name.
                    spk_name_id = line[7].split("-")[0] # SPK0177
                    target_speaker_speech_id = line[7] #
                    start = float(line[3])
                    end = start + float(line[4])
                    tgt_spkr_speech = scp2dict[target_speaker_speech_id]
                    if spk_name_id not in spk:
                        spk[spk_name_id] = spk_name_id
                        #num_spk +=1
                    #if uttid == "R8009_M8020_MS810":
                    #    print(line)
                    #    print(spk_name_id)
                    segments.append(
                        Segment(
                           uttid,
                           spk[spk_name_id], # speaker index number in this long audio
                           start,
                           end,
                           spk_name_id, # actual speaker name in this long audio
                           #tgt_speech,
                           tgt_spkr_speech,
                           )
                       )
        print(f"spk: {spk}")
        segments = sorted(segments, key=lambda x: x.spkr)
        intervals = defaultdict(list)
        new_intervals = defaultdict(list)
        dic = defaultdict()
        # Summary the intervals for all speakers
        for i in range(len(segments)):
            interval = [segments[i].stime, segments[i].etime, segments[i].tgt_spkr_speech]
            intervals[segments[i].spkr].append(interval)
            dic[str(segments[i].uttid) + "_" + str(segments[i].spkr)] = segments[
                i
            ].name
        #print(dic,len(dic))
        #print(intervals,len(intervals))
        print(f"dic: {dic}, its len: {len(dic)}")
        print(f"intervals.keys(): {intervals.keys()}")
        #print(f"intervals; {intervals}")
        # read mixture audio
        wav_file = audio_wavform
        #print(f"wav_file: {wav_file}")
        orig_audio, _ = soundfile.read(wav_file) # wav_file is mono audio
        #orig_audio = orig_audio[:, 0]
        length = len(orig_audio)

        # # Cut and save the clean speech part
        id_full = uttid
        # Save the labels and target speech
        for key in intervals:
            output_dir = os.path.join(args.target_wav, id_full)
            os.makedirs(output_dir, exist_ok=True)
            output_wav = os.path.join(output_dir, str(key) + ".wav")
            labels = [0] * int(length / 16000 * 25)  # 40ms, one label
            for interval in intervals[key]:
                s, e,tgt_speech = interval
                for i in range(int(s * 25), min(int(e * 25) + 1, len(labels))):
                    labels[i] = 1
                print(f"id_full: {id_full}, tgt_speech: {tgt_speech}")
                #tgt_speech_samples, _ = soundfile.read(tgt_speech)
                #print(f"tgt_speech: {tgt_speech}, sample: {tgt_speech_samples}")
            #soundfile.write(output_wav, tgt_speech_samples, 16000) ## target speech
            cmd=f"ln -svf {tgt_speech} {output_wav}"
            os.system(cmd)
            print(f"output_wav: {output_wav}")
            room_speaker_id = id_full + "_" + str(key) # uttid_keys
            speaker_id = dic[room_speaker_id]

            res = {
                "filename": id_full, # wavform name
                "speaker_key": key, # speaker index number
                "speaker_id": speaker_id, #
                "labels": labels,
            }
            json.dump(res, outs)
            outs.write("\n")
        ## mix audio
        output_wav = os.path.join(output_dir, "all.wav")
        soundfile.write(output_wav, orig_audio, 16000)
    outs.close()

def filter_bad_utterance(output_wav)-> str:
    # output_wav = "/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2_simu/data/200h_fixed_4spks6s/Train_Ali_far/target_audio/SPK0177-R0014_M0087_MS002_49_SPK0145-R0008_M0072_MS002_108_SPK0094-R0003_M0047_MS006_98_SPK4052-R1019_M4052_MS103_166/3.wav"
    # >>> b = output_wav.split(".")[0].split("/")[-2:]
    # >>> b
    # ['SPK0177-R0014_M0087_MS002_49_SPK0145-R0008_M0072_MS002_108_SPK0094-R0003_M0047_MS006_98_SPK4052-R1019_M4052_MS103_166', '3']
    # >>> c = "_".join(b)
    # >>> c
    # 'SPK0177-R0014_M0087_MS002_49_SPK0145-R0008_M0072_MS002_108_SPK0094-R0003_M0047_MS006_98_SPK4052-R1019_M4052_MS103_166_3'
    #bad_uttids=[]
    try:
        a,_ = soundfile.read(output_wav)
    except:
        bad_uttid="_".join(output_wav.split(".")[0].split("/")[-2:])
        print(f"It cann't read and this is bad file, bad_uttid: {bad_uttid}")

        print(f"It cann't read and this is bad file, will remove {output_wav}")
        os.system(f'rm {output_wav}')
        return bad_uttid
    else:
        if a.size==0: # it is also bad utterance
            bad_uttid="_".join(output_wav.split(".")[0].split("/")[-2:])
            print(f"The bad utt will be removed, bad_uttid: {bad_uttid}")
            print(f"The bad utt will be removed: {output_wav}")
            os.system(f'rm {output_wav}')
            return bad_uttid
        return None



def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--simu_rttm", help="the path for the alimeeting")
    parser.add_argument("--audio_dir", help="the path for the mix audio_dir")
    parser.add_argument("--target_speaker_speech_wavscp", help="before mixture per speaker target speech wav.scp path")
    parser.add_argument("--dest_dir", help="the director for output files")
    parser.add_argument("--type", help="Eval or Train")
    args = parser.parse_args()
    if args.type=="Eval" or args.type=="Test":
        args.path = os.path.join(
        args.dest_dir, "%s_Ali" % (args.type), "%s_Ali_far" % (args.type)

        )
    else:
        args.path = os.path.join(args.dest_dir,"%s_Ali_far" % (args.type))
    os.makedirs(args.path, exist_ok=True)
    args.target_wav = os.path.join(args.path, "target_audio")
    args.out_text = os.path.join(args.path, "%s.json" % (args.type))
    return args
if __name__ == "__main__":

    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    args = get_args()
    start_time = time.time()

    rank = int(os.environ['LOCAL_RANK'])        ## processing id
    threads_num = int(os.environ['WORLD_SIZE']) ## cpu numbers, is setted by --nproc_per_node
    logging.info("rank {}/{}.".format(
        rank, threads_num,
    ))
    paths = load_wavs(args)
    paths.sort(key=lambda x: x[0])
    local_paths = paths[rank::threads_num]
    logging.info(f"local_paths: {local_paths}")
    process(args,local_paths)
    end_time = time.time()
    print(f"total time: {end_time - start_time:.2f} seconds")
    # for debug
    #main()

