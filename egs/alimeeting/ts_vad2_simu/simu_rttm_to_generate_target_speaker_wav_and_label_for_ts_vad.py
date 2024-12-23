#!/usr/bin/env python3
# Author: Duo MA
# Email: maduo@cuhk.edu.cn

import glob, tqdm, os, textgrid, soundfile, copy, json, argparse, numpy, torch, wave
from collections import defaultdict
from typing import List
class Segment(object):
    def __init__(self, uttid, spkr, stime, etime, name):
        self.uttid = uttid
        self.spkr = spkr
        self.stime = round(stime, 2)
        self.etime = round(etime, 2)
        self.name = name

    def change_stime(self, time):
        self.stime = time

    def change_etime(self, time):
        self.etime = time


def remove_overlap(aa, bb):
    # Sort the intervals in both lists based on their start time
    a = aa.copy()
    b = bb.copy()
    a.sort()
    b.sort()

    # Initialize the new list of intervals
    result = []

    # Initialize variables to keep track of the current interval in list a and the remaining intervals in list b
    i = 0
    j = 0

    # Iterate through the intervals in list a
    while i < len(a):
        # If there are no more intervals in list b or the current interval in list a does not overlap with the current interval in list b, add it to the result and move on to the next interval in list a
        if j == len(b) or a[i][1] <= b[j][0]:
            result.append(a[i])
            i += 1
        # If the current interval in list a completely overlaps with the current interval in list b, skip it and move on to the next interval in list a
        elif a[i][0] >= b[j][0] and a[i][1] <= b[j][1]:
            i += 1
        # If the current interval in list a partially overlaps with the current interval in list b, add the non-overlapping part to the result and move on to the next interval in list a
        elif a[i][0] < b[j][1] and a[i][1] > b[j][0]:
            if a[i][0] < b[j][0]:
                result.append([a[i][0], b[j][0]])
            a[i][0] = b[j][1]
        # If the current interval in list a starts after the current interval in list b, move on to the next interval in list b
        elif a[i][0] >= b[j][1]:
            j += 1

    # Return the new list of intervals
    return result


def main(args):
#def main():
    # for debug
    #system_rttm = "/mntcephfs/lab_data/maduo/exp/speaker_diarization/spectral_cluster/exp/spectral_cluster/alimeeting_eval_system_sad_rttm_cam++_advanced"
    #audio_dir= "/mntcephfs/lab_data/maduo/datasets/alimeeting/Eval_Ali/Eval_Ali_far/audio_dir/"
    #target_wav="tests/R8009_M8020_MS810_target_audio_all"

    #system_rttm="tests/R8009_M8020_MS810_system.rttm"
    #audio_dir="tests/audio_dir"
    #out_text="tests/R8009_M8020_MS810.json"
    #target_wav="tests/R8009_M8020_MS810_target_audio"
    audio_wavforms = glob.glob(args.audio_dir + "/*")
    print(f"audio_wavforms: {audio_wavforms}")
    outs = open(args.out_text, "w")
    #bad_uttids=[]
    for audio_wavform in tqdm.tqdm(audio_wavforms):
        # audio_wavform is str
        uttid = os.path.splitext(os.path.basename(audio_wavform))[0]
        print(f"uttid:{uttid}")
        segments = []
        spk = {}
        num_spk = 1
        with open(args.system_rttm, 'r')as f:
            for line in f:
                # i.e. SPEAKER R8001_M8004_MS801 1 7.202 5.940 <NA> <NA> 2 <NA> <NA>
                line = line.strip().split()
                if line[1] == uttid:
                    spk_name_id = uttid + "_" + "spk" + str(line[7]) # The actual speaker name is bound to the audio name.
                    start = float(line[3])
                    end = start + float(line[4])
                    if spk_name_id not in spk:
                        spk[spk_name_id] = num_spk
                        num_spk +=1
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
                           )
                       )
        print(f"spk: {spk}")
        segments = sorted(segments, key=lambda x: x.spkr)
        intervals = defaultdict(list)
        new_intervals = defaultdict(list)

        dic = defaultdict()
        # Summary the intervals for all speakers
        for i in range(len(segments)):
            interval = [segments[i].stime, segments[i].etime]
            intervals[segments[i].spkr].append(interval)
            dic[str(segments[i].uttid) + "_" + str(segments[i].spkr)] = segments[
                i
            ].name
        #print(dic,len(dic))
        #print(intervals,len(intervals))
        print(f"dic: {dic}, its len: {len(dic)}")
        print(f"intervals.keys(): {intervals.keys()}")
        # Remove the overlapped speeech
        for key in intervals:
            new_interval = intervals[key]
            for o_key in intervals:
                if o_key != key:
                    new_interval = remove_overlap(
                        copy.deepcopy(new_interval), copy.deepcopy(intervals[o_key])
                    )
            new_intervals[key] = new_interval

        # read mixture audio
        wav_file = audio_wavform
        #print(f"wav_file: {wav_file}")
        orig_audio, _ = soundfile.read(wav_file)
        #orig_audio = orig_audio[:, 0]
        length = len(orig_audio)

        # # Cut and save the clean speech part
        id_full = uttid
        #output_dir=None
        bad_uttids=[]
        for key in new_intervals:
            output_dir = os.path.join(args.target_wav, id_full)
            os.makedirs(output_dir, exist_ok=True)
            output_wav = os.path.join(output_dir, str(key) + ".wav")
            new_audio = []
            labels = [0] * int(length / 16000 * 25)  # 40ms, one label
            for interval in new_intervals[key]:
                s, e = interval
                for i in range(int(s * 25), min(int(e * 25) + 1, len(labels))):
                    labels[i] = 1
                s *= 16000
                e *= 16000
                new_audio.extend(orig_audio[int(s) : int(e)])
            soundfile.write(output_wav, new_audio, 16000)
            # remove bad utterance
            #print(f"output_wav, {output_wav}")
            bad_uttid = filter_bad_utterance(output_wav)
            #print(f"bad_uttid;{bad_uttids} !!")
            bad_uttids.append(bad_uttid)
        print(f" bad_uttids: {bad_uttids}")
        output_wav = os.path.join(output_dir, "all.wav")
        soundfile.write(output_wav, orig_audio, 16000)

        # Save the labels
        for key in intervals:
            labels = [0] * int(length / 16000 * 25)  # 40ms, one label
            for interval in intervals[key]:
                s, e = interval
                for i in range(int(s * 25), min(int(e * 25) + 1, len(labels))):
                    labels[i] = 1

            room_speaker_id = id_full + "_" + str(key) # uttid_keys
            print(f"room_speaker_id: {room_speaker_id}, bad_uttids: {bad_uttids}")
            if room_speaker_id not in bad_uttids:

                speaker_id = dic[room_speaker_id]

                res = {
                    "filename": id_full, # wavform name
                    "speaker_key": key, # speaker index number
                    "speaker_id": speaker_id, #
                    "labels": labels,
                }
                json.dump(res, outs)
                outs.write("\n")
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
    parser.add_argument("--system_rttm", help="the path for the alimeeting")
    parser.add_argument("--audio_dir", help="the path for the alimeeting audio_dir")
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
    args = get_args()
    main(args)
    # for debug
    #main()

