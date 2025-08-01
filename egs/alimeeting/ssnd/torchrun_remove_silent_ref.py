#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
get intervals from .vads file, specify output data, and this script removes silences and saves the audio data in out path folder
paths=tests/wav20.scp
vads=tests/train_pa_20.vads
out=tests/wenetspeech_wo_silence
 torchrun --nproc_per_node=5 --master_port=12345  codebase/fairseq_speechtext/examples/wav2vec/unsupervised/scripts/remove_silence_for_wavscp_torchrun_parallel.py --wavscp tests/wav20.scp  --vads tests/train_pa_20.vads --out tests/wenetspeech_wo_silence
"""

import os
import argparse
import torch
import torchaudio
#import tqdm
from tqdm import tqdm
import soundfile

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wavscp", default="", type=str)
    parser.add_argument("--vads", default="", type=str)
    parser.add_argument("--out", type=str)
    return parser


def load_wav(wavscp):
    # load paths
    #paths = dict()
    paths = []
    with open(wavscp) as f:
        #root = next(f).rstrip()
        #for line in f:
        #    paths.append(os.path.join(root, line.rstrip().split("\t")[0]))
        for line in f:
            key, path = line.rstrip().split(maxsplit=1)
            #paths[key] = path
            paths.append((key,path))
            #paths.append(line.rstrip().split()[1])
    return paths
def load_vads(vads):


    # load vads
    intervals = dict()
    with open(vads) as f:
        for line in f:
            key, interval_ = line.rstrip().split(" ",maxsplit=1)

            interval = [
                [int(w.split(":")[0]), int(w.split(":")[1])] for w in interval_.split()
            ]
            intervals[key] = interval
            #list_intervals.append(interval)
    return intervals
    #assert len(paths) == len(intervals),f"audio utterances: {len(paths)}, vad utterances: {len(intervals)}"
    #return paths, intervals


def write_audio_wo_silence(args,paths, intervals):
    i=0
    for i, (uttid, path) in tqdm(enumerate(paths), total=len(paths),ascii=True):
        """
        >>> import torchaudio
        >>> import soundfile
        >>> b,d = soundfile.read("/mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000000_--5llN02F84.opus")
        >>> a,s = torchaudio.load("/mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000000_--5llN02F84.opus")
        b>>> b
        array([ 6.23337076e-11,  5.64897400e-11,  3.90823103e-11, ...,
            -3.00671127e-05, -2.87229668e-05, -2.66385159e-05])
        >>> d
        16000
        >>> a
        tensor([[ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  3.0518e-05,
                -3.0518e-05, -6.1035e-05]])
        >>> s
        48000
        >>> import torch
        >>> b1 = torch.from_numpy(b)
        >>> b1
        tensor([ 6.2334e-11,  5.6490e-11,  3.9082e-11,  ..., -3.0067e-05,
                -2.8723e-05, -2.6639e-05], dtype=torch.float64)
        >>> type(b)
        <class 'numpy.ndarray'>
        >>> a[0]
        tensor([ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  3.0518e-05,
                -3.0518e-05, -6.1035e-05])
        >>> a
        tensor([[ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  3.0518e-05,
                -3.0518e-05, -6.1035e-05]])
        >>> a.shape
        torch.Size([1, 119739385])
        >>> b.shape
        (39913128,)
        >>> b1.shape
        torch.Size([39913128])
        """
        data, sr = soundfile.read(path) # data is np.float64, it is same as  stage of computing vads.
        data = torch.from_numpy(data) # data is torch.float64
        assert sr==16000,f"expected sample rate is 16000, however uttid is {uttid}, sample rate: {sr}"
        data_filtered = None
        if uttid in intervals.keys():
            if len(intervals[uttid])>0:
                #data_filtered = torch.cat([data[int(it[0]) : int(it[1])] for it in intervals[uttid]]).unsqueeze(0)
                data_filtered = torch.cat([data[int(it[0]) : int(it[1])] for it in intervals[uttid]])
            else:
                data_filtered = data

        ## assume the wavform path is as follows:
        ## '/mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000000_--5llN02F84.opus'
        # >>> os.path.dirname(path).split("/",maxsplit=6)
        # ['', 'mntcephfs', 'lee_dataset', 'asr', 'WenetSpeech', 'untar', 'audio/train/youtube/B00000']
        # >>> os.path.splitext(path)
        # ('/mntcephfs/lee_dataset/asr/WenetSpeech/untar/audio/train/youtube/B00000/Y0000000000_--5llN02F84', '.opus')
        # >>> os.path.splitext(path)[0].split("/",maxsplit=6)
        # ['', 'mntcephfs', 'lee_dataset', 'asr', 'WenetSpeech', 'untar', 'audio/train/youtube/B00000/Y0000000000_--5llN02F84']
        os.makedirs(args.out,exist_ok=True)
        out = os.path.splitext(path)[0].split("/",maxsplit=6)[-1]
        #outpath = args.out + '/'+ out + ".wav" ## torchaudio.save don't support save as opus.
        outpath = args.out + '/'+ out + ".opus" # soundfile support save as opus, reference:https://github.com/bastibe/python-soundfile/issues/252
        output_dir = args.out + '/' + os.path.dirname(path).split("/",maxsplit=6)[-1]
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        #if not os.path.exists(outpath):
        #data_filtered = data_filtered.to(torch.float32)
        data_filtered = data_filtered.numpy()
        soundfile.write(outpath,data_filtered,samplerate=16000, format='OGG', subtype='OPUS')
        #torchaudio.save(outpath, data_filtered,sample_rate=16000)
        i = i+1
        logging.info(f"output wavform:{outpath}!")
    logging.info(f"write all {i} utterances , finish!!!")
def main():
    parser = get_parser()
    args = parser.parse_args()

    rank = int(os.environ['LOCAL_RANK'])        ## processing id
    threads_num = int(os.environ['WORLD_SIZE']) ## cpu numbers, is setted by --nproc_per_node
    logging.info("rank {}/{}.".format(
        rank, threads_num,
    ))
    paths = load_wav(args.wavscp)
    paths.sort(key=lambda x: x[0])
    local_all_paths = paths[rank::threads_num]
    logging.info(f"local_all_paths: {local_all_paths}")
    intervals= load_vads(args.vads)
    write_audio_wo_silence(args,local_all_paths, intervals)
    """
    i=0
    for i, (uttid, path) in tqdm(enumerate(local_all_paths), total=len(local_all_paths), ascii=True):
        data, sr = soundfile.read(path) # data is np.float64, it is same as  stage of computing vads.
        data = torch.from_numpy(data) # data is torch.float64
        assert sr==16000,f"expected sample rate is 16000, however uttid is {uttid}, sample rate: {sr}"
        data_filtered = None
        if uttid in intervals.keys():
            if len(intervals[uttid])>0:
                data_filtered = torch.cat([data[int(it[0]) : int(it[1])] for it in intervals[uttid]]).unsqueeze(0)
            else:
                data_filtered = data
        out = os.path.splitext(path)[0].split("/",maxsplit=6)
        outpath = args.out + '/'+ out + ".wav" ## torchaudio.save don't support save as opus.
        output_dir = args.out + '/' + os.path.dirname(path).split("/",maxsplit=6)[-1]
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        #if not os.path.exists(outpath):
        data_filtered = data_filtered.to(torch.float32)

        torchaudio.save(outpath, data_filtered,sample_rate=16000)
        i = i+1
        logging.info(f"output wavform:{outpath}!")
    logging.info(f"write all {i} utterances , finish!!!")
    """




if __name__ == "__main__":
    import logging
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
