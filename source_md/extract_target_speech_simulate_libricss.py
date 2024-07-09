import glob, tqdm, os, soundfile, copy, argparse
from collections import defaultdict
from joblib import Parallel, delayed


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
    parser.add_argument(
        "--rttm_path", required=True, help="the path for the rttm_files"
    )
    parser.add_argument(
        "--orig_audio_path", required=True, help="the path for the orig audio"
    )
    parser.add_argument(
        "--target_audio_path", required=True, help="the part for the output audio"
    )
    parser.add_argument("--sample_rate", default=16000, type=int, help="sample rate")
    args = parser.parse_args()

    return args


def get_target_audio(lines, room_id, args):
    intervals = defaultdict(list)
    new_intervals = defaultdict(list)
    for line in lines:
        data = line.split()
        if data[1] == room_id:
            stime = float(data[3])
            etime = float(data[3]) + float(data[4])
            spkr = data[-3]
            intervals[spkr].append([stime, etime])

    # Remove the overlapped speeech
    for key in intervals:
        new_interval = intervals[key]
        for o_key in intervals:
            if o_key != key:
                new_interval = remove_overlap(
                    copy.deepcopy(new_interval), copy.deepcopy(intervals[o_key])
                )
        new_intervals[key] = new_interval

    wav_file = glob.glob(os.path.join(args.orig_audio_path, room_id) + ".wav")[0]
    orig_audio, fs = soundfile.read(wav_file)

    if len(orig_audio.shape) > 1:
        orig_audio = orig_audio[:, 0]

    # # Cut and save the clean speech part
    id_full = wav_file.split("/")[-1][:-4]
    for key in new_intervals:
        output_dir = os.path.join(args.target_audio_path, id_full)
        os.makedirs(output_dir, exist_ok=True)
        output_wav = os.path.join(output_dir, str(key) + ".wav")
        new_audio = []
        for interval in new_intervals[key]:
            s, e = interval
            s *= args.sample_rate
            e *= args.sample_rate
            new_audio.extend(orig_audio[int(s) : int(e)])

        if len(new_audio) < 100:
            print(output_wav)

        soundfile.write(output_wav, new_audio, args.sample_rate)


def main():
    args = get_args()
    lines = open(args.rttm_path).read().splitlines()
    room_set = set()
    for line in lines:
        data = line.split()
        room_set.add(data[1])

    Parallel(n_jobs=120)(
        delayed(get_target_audio)(lines, room_id, args) for room_id in room_set
    )

    # for room_id in tqdm.tqdm(room_set):
    #     get_target_audio(lines, room_id, args)
    # output_wav = os.path.join(output_dir, 'all.wav')
    # soundfile.write(output_wav, orig_audio, 16000)


if __name__ == "__main__":
    main()
