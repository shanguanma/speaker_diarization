#!/usr/bin/env python3
import glob, tqdm, os, textgrid, soundfile, copy, argparse
from collections import defaultdict
import time
import multiprocessing
import logging

def str2bool(v):
    """Used in argparse.ArgumentParser.add_argument to indicate
    that a type is a bool type and user can enter

        - yes, true, t, y, 1, to represent True
        - no, false, f, n, 0, to represent False

    See https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse  # noqa
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

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
    parser.add_argument("--duration",type=int,default=2,help="less than --duration, will not stored it as non overlap target speaker speech")
    #parser.add_argument("--cat_cut",type=str2bool,default=True, help="if true, will splice the continuation to the specified duration,else will stitch all")
    parser.add_argument("--dest_name_dir", type=str,default="non_overlap_segment",help="stored target speech directory name")
    parser.add_argument("--nj",type=int,default=8,help="")
    args = parser.parse_args()
    return args

def text_grids_list(args)->list:
    if args.type=="Train":
        path=os.path.join(args.data_path,f"{args.type}_Ali_far")
    else:
        path=os.path.join(args.data_path,f"{args.type}_Ali",f"{args.type}_Ali_far")
    path_grid = os.path.join(path, "textgrid_dir")
    text_grids = glob.glob(path_grid + "/*")
    return text_grids


def load_data(args):
    if args.type=="Train":
        path=os.path.join(args.data_path,f"{args.type}_Ali_far")
    else:
        path=os.path.join(args.data_path,f"{args.type}_Ali",f"{args.type}_Ali_far")
    path_wavs = os.path.join(path, "audio_dir")
    path_grid = os.path.join(path, "textgrid_dir")
    target_wavs = os.path.join(path, f"{args.dest_name_dir}")
    return path_wavs,path_grid, target_wavs

def process(args,text_grids,path_wavs,path_grid,target_wavs):
    #args = get_args()

    #if args.type=="Train":
    #    path=os.path.join(args.data_path,f"{args.type}_Ali_far")
    #else:
    #    path=os.path.join(args.data_path,f"{args.type}_Ali",f"{args.type}_Ali_far")
    #path_wavs = os.path.join(path, "audio_dir")
    #path_grid = os.path.join(path, "textgrid_dir")
    #target_wavs = os.path.join(path, f"{args.dest_name_dir}")


    #text_grids = glob.glob(path_grid + "/*")
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

        wav_file = glob.glob(os.path.join(path_wavs, uttid) + "*.wav")[0]
        orig_audio, _ = soundfile.read(wav_file)
        orig_audio = orig_audio[:, 0]
        length = len(orig_audio)

        # # Cut and save the clean speech part
        id_full = wav_file.split("/")[-1][:-4]
        room_id = id_full[:11]
        for key in new_intervals:
            print(f"key: {key}!!")
            output_dir = os.path.join(target_wavs, id_full)
            os.makedirs(output_dir, exist_ok=True)
            output_wav = os.path.join(output_dir, str(key) + ".wav")
            new_audio = []
            #labels = [0] * int(length / 16000 * 25)  # 40ms, one label
            for i, interval in enumerate(new_intervals[key]):
                s, e = interval
                #dur = e - s
                #if dur >=args.duration: # remove <2s speech
                ss = s* 16000
                ee = e* 16000
                new_audio.extend(orig_audio[int(ss) : int(ee)])
            soundfile.write(output_wav, new_audio, 16000)

def filter_bad_utterance(output_wav):
    try:
        a,_ = soundfile.read(output_wav)
    #except Exception as e:
    except:
        print(f"It cann't read and this is bad file, will remove {output_wav}")
        os.system(f'rm {output_wav}')
    else:
        if a.size==0: # it is also bad utterance
            print(f"The bad utt will be removed: {output_wav}")
            os.system(f'rm {output_wav}')

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
    text_grids = text_grids_list(args)
    text_grids.sort(key=lambda x: x[0])
    local_paths = text_grids[rank::threads_num]
    logging.info(f"local_paths: {local_paths}")
    path_wavs, path_grid, target_wavs = load_data(args)
    #with multiprocessing.Pool(args.nj) as p:
    #    print(f"args.nj: {args.nj}")
    #    p.map(process,args)
    process(args,local_paths,path_wavs,path_grid,target_wavs)
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")
