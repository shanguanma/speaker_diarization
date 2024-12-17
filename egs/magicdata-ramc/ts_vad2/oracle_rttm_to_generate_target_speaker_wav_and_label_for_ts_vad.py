#!/usr/bin/env python3
# Author: Duo MA
# Email: maduo@cuhk.edu.cn

import glob, tqdm, os, textgrid, soundfile, copy, json, argparse, numpy, torch, wave
from collections import defaultdict

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

def load_wav_scp(wav_scp: str):
    wav2scp={}
    with open(wav_scp,'r')as f:
        for line in f:
            line = line.strip().split()
            if line[0] not in wav2scp:
                wav2scp[line[0]] = line[-1]
    return wav2scp

def main(args):
    #audio_wavforms = glob.glob(args.audio_dir + "/*")
    wav2scp = load_wav_scp(args.wavscp)
    outs = open(args.out_text, "w")
    #for audio_wavform in tqdm.tqdm(audio_wavforms):
    for uttid in wav2scp.keys():
        #uttid = os.path.splitext(os.path.basename(audio_wavform))[0]
        print(uttid)
        segments = []
        spk = {}
        num_spk = 1
        with open(args.oracle_rttm, 'r')as f:
            for line in f:
                # i.e. 
                line = line.strip().split()
                if line[1] == uttid:
                    # i.e. oracle rttm:
                    # SPEAKER CTS-CN-F2F-2019-11-15-1421 1 1.386 3.8 <NA> <NA> G00000224 <NA> <NA>
                    # I know the correct speaker name  in the session (CTS-CN-F2F-2019-11-15-1421)
                    # the speaker name id is  globally unique.
                    # however at system rttm
                    # SPEAKER CTS-CN-F2F-2019-11-15-1421 1 1.386 3.8 <NA> <NA> 1 <NA> <NA>
                    # I don't know the correct spekaer name in the session (CTS-CN-F2F-2019-11-15-1421)
                    # so I must speaker_name_id bound to the session( audio name).
                    spk_name_id = str(line[7])
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
        print(spk)
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
        wav_file = wav2scp[uttid]
        orig_audio, _ = soundfile.read(wav_file)
        #orig_audio = orig_audio[:,0] # magicdata-ramc is mono channel data, so it doesn't choice channel number
        length = len(orig_audio)

        # # Cut and save the clean speech part
        id_full = uttid
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

def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--oracle_rttm", help="the path for the magicdata-ramc oracle rttm file")
    parser.add_argument("--wavscp", help="the path for the magicdata-ramc wavscp file path")
    parser.add_argument("--dest_dir", help="the director for output files")

    parser.add_argument("--type", help=" dev or test or Train")
    args = parser.parse_args()
    #if args.type=="Eval" or args.type=="Test":
    #args.path = os.path.join(
    #    args.dest_dir, "%s_Ali" % (args.type), "%s_Ali_far" % (args.type)

    #    )
    #else:
    args.path = os.path.join(args.dest_dir,f"{args.type}")
    os.makedirs(args.path, exist_ok=True)
    args.target_wav = os.path.join(args.path, "target_audio")
    args.out_text = os.path.join(args.path, "%s.json" % (args.type))
    return args
if __name__ == "__main__":
    args = get_args()
    main(args)
    # for debug
    #main()

