import glob, os, soundfile, json, argparse
from collections import defaultdict
from tqdm import tqdm
import librosa


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


def get_args():
    parser = argparse.ArgumentParser(description="prepare target label json data for speaker diarization")
    parser.add_argument("--path_rttm", help="the path for the rttm of mixer wavform")
    parser.add_argument("--path_wav", help="the path for the mixer wavform")
    parser.add_argument("--out_dir", help="the path for output directory")
    parser.add_argument(
        "--label_rate", type=int, default=100, help="target speaker label rate"
    )
    parser.add_argument("--type", help="dev, test or train")

    args = parser.parse_args()

    #args.out_text = os.path.join(args.out_dir, f"{args.type}_{args.label_rate}.json")
    return args


def main():
    args = get_args()
    out_text = os.path.join(args.out_dir, f"{args.type}_{args.label_rate}.json")
    outs = open(out_text, "w")
    segments = {}
    spk_num = {}
    with open(args.path_rttm) as f:
        for line in f:
            _, uttid, _, stime, offset, _, _, spk_id, _, _ = line.strip("\n").split(" ")
            if uttid not in segments:
                num_spk = 1
                spk = {}
                spk_num[uttid] = []
                segments[uttid] = []
            if spk_id not in spk:
                spk[spk_id] = num_spk
                num_spk += 1
                spk_num[uttid].append(spk_id)
            segments[uttid].append(
                Segment(
                    uttid,
                    spk[spk_id],
                    float(stime),
                    float(stime) + float(offset),
                    spk_id,
                )
            )

    for uttid in tqdm(segments):
        segments[uttid] = sorted(segments[uttid], key=lambda x: x.spkr)
        intervals = defaultdict(list)

        dic = {}
        # Summary the intervals for all speakers
        for i in range(len(segments[uttid])):
            interval = [segments[uttid][i].stime, segments[uttid][i].etime]
            intervals[segments[uttid][i].spkr].append(interval)
            if (
                str(segments[uttid][i].uttid) + "_" + str(segments[uttid][i].spkr)
                not in dic
            ):
                dic[
                    str(segments[uttid][i].uttid) + "_" + str(segments[uttid][i].spkr)
                ] = segments[uttid][i].name

        # Save the labels
        wav_file = os.path.join(args.path_wav, uttid) + ".wav"
        duration = librosa.get_duration(path=wav_file)
        id_full = wav_file.split("/")[-1][:-4]
        for key in intervals:
            labels = [0] * int(duration * args.label_rate)  # 40ms, one label
            for interval in intervals[key]:
                s, e = interval
                for i in range(
                    int(s * args.label_rate),
                    min(int(e * args.label_rate) + 1, len(labels)),
                ):
                    labels[i] = 1

            room_speaker_id = id_full + "_" + str(key)
            speaker_id = dic[room_speaker_id]

            res = {
                "filename": id_full,
                "speaker_key": key,
                "speaker_id": speaker_id,
                "labels": labels,
                "spk_num": len(spk_num[id_full]),
            }
            json.dump(res, outs)
            outs.write("\n")


if __name__ == "__main__":
    main()
