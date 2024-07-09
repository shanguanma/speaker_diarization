#!/usr/bin/env python3
import glob, tqdm, os, textgrid, soundfile, copy, json, argparse, numpy, torch, wave
from collections import defaultdict

class Segment(object):
    def __init__(self, uttid, spkr, stime, etime, text, name):
        self.uttid = uttid
        self.spkr = spkr
        self.stime = round(stime, 2)
        self.etime = round(etime, 2)
        self.text = text
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

def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--data_path", help="the path for the alimeeting")
    parser.add_argument("--type", help="Eval or Train")

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    
    if args.type=="Train":
        args.path=os.path.join(args.data_path,f"{args.type}_Ali_far")
    else:
        args.path=os.path.join(args.data_path,f"{args.type}_Ali",f"{args.type}_Ali_far")
    args.path_wav = os.path.join(args.path, "audio_dir")
    args.path_grid = os.path.join(args.path, "textgrid_dir")
    args.target_wav = os.path.join(args.path, "non_overlap")


    text_grids = glob.glob(args.path_grid + "/*")
    #outs = open(args.out_text, "w")
    for text_grid in tqdm.tqdm(text_grids):
        tg = textgrid.TextGrid.fromFile(text_grid)
        segments = []
        spk = {}
        num_spk = 1
        uttid = text_grid.split("/")[-1][:-9]
        for i in range(tg.__len__()):
            for j in range(tg[i].__len__()):
                if tg[i][j].mark:
                    if tg[i].name not in spk:
                        #spk[tg[i].name] = num_spk
                        spk[tg[i].name] = tg[i].name.split("_")[-1] 
                        num_spk += 1
                    segments.append(
                        Segment(
                            uttid,
                            spk[tg[i].name],
                            tg[i][j].minTime,
                            tg[i][j].maxTime,
                            tg[i][j].mark.strip(),
                            tg[i].name,
                        )
                    )
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
            ].name.split("_")[-1]
        # Remove the overlapped speeech
        for key in intervals:
            new_interval = intervals[key]
            for o_key in intervals:
                if o_key != key:
                    new_interval = remove_overlap(
                        copy.deepcopy(new_interval), copy.deepcopy(intervals[o_key])
                    )
            new_intervals[key] = new_interval

        wav_file = glob.glob(os.path.join(args.path_wav, uttid) + "*.wav")[0]
        orig_audio, _ = soundfile.read(wav_file)
        orig_audio = orig_audio[:, 0]
        length = len(orig_audio)

        # # Cut and save the clean speech part
        id_full = wav_file.split("/")[-1][:-4]
        room_id = id_full[:11]
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
        #output_wav = os.path.join(output_dir, "all.wav")
        #soundfile.write(output_wav, orig_audio, 16000)
if __name__ == "__main__":
    main()
