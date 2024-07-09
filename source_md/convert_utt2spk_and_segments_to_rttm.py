#!/usr/bin/env python3
# Author: Duo MA
# Email: maduo@cuhk.edu.cn

"""This script converts kaldi-style utt2spk and segments to a NIST RTTM
file.

The RTTM format is
<type> <file-id> <channel-id> <begin-time> \
        <duration> <ortho> <stype> <name> <conf>

<type> = SPEAKER for each segment.
<file-id> - the File-ID of the recording
<channel-id> - the Channel-ID, usually 1
<begin-time> - start time of segment
<duration> - duration of segment
<ortho> - <NA> (this is ignored)
<stype> - <NA> (this is ignored)
<name> - speaker name or id
<conf> - <NA> (this is ignored)

"""
## this script is modified from https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/segmentation/convert_utt2spk_and_segments_to_rttm.py
from __future__ import print_function
import argparse
import sys

def get_args():
    parser = argparse.ArgumentParser(
        description="""This script converts kaldi-style utt2spk and
        segments to a NIST RTTM file""")

    parser.add_argument("--reco2file-and-channel", type=str,
                        action=common_lib.NullstrToNoneAction,
                        help="""Input reco2file_and_channel.
                        The format is <recording-id> <file-id> <channel-id>.
                        If not provided, then <recording-id> is taken as the
                        <file-id> with <channel-id> = 1.""")
    parser.add_argument("utt2spk", type=str,
                        help="Input utt2spk file")
    parser.add_argument("segments", type=str,
                        help="Input segments file")
    parser.add_argument("rttm_file", type=str,
                        help="Output RTTM file")

    args = parser.parse_args()
    return args

class smart_open(object):
    """
    This class is designed to be used with the "with" construct in python
    to open files. It is similar to the python open() function, but
    treats the input "-" specially to return either sys.stdout or sys.stdin
    depending on whether the mode is "w" or "r".

    e.g.: with smart_open(filename, 'w') as fh:
            print ("foo", file=fh)
    """
    def __init__(self, filename, mode="r"):
        self.filename = filename
        self.mode = mode
        assert self.mode == "w" or self.mode == "r"

    def __enter__(self):
        if self.filename == "-" and self.mode == "w":
            self.file_handle = sys.stdout
        elif self.filename == "-" and self.mode == "r":
            self.file_handle = sys.stdin
        else:
            self.file_handle = open(self.filename, self.mode)
        return self.file_handle

    def __exit__(self, *args):
        if self.filename != "-":
            self.file_handle.close()



def main():
    args = get_args()

    if args.reco2file_and_channel is not None:
        reco2file_and_channel = {}
        with smart_open(args.reco2file_and_channel) as fh:
            for line in fh:
                parts = line.strip().split()
                reco2file_and_channel[parts[0]] = (parts[1], parts[2])

    utt2spk = {}
    with smart_open(args.utt2spk) as fh:
        for line in fh:
            parts = line.strip().split()
            utt2spk[parts[0]] = parts[1]

    with smart_open(args.segments) as segments_reader, \
            smart_open(args.rttm_file, 'w') as rttm_writer:
        for line in segments_reader:
            parts = line.strip().split()

            utt = parts[0]
            spkr = utt2spk[utt]

            reco = parts[1]
            file_id = reco
            channel = 1

            if args.reco2file_and_channel is not None:
                try:
                    file_id, channel = reco2file_and_channel[reco]
                except KeyError:
                    raise RuntimeError(
                        "Could not find recording {0} in {1}".format(
                            reco, args.reco2file_and_channel))

            start_time = float(parts[2])
            duration = float(parts[3]) - start_time

            print("SPEAKER {0} {1} {2:7.2f} {3:7.2f} "
                  "<NA> <NA> {4} <NA>".format(
                      file_id, channel, start_time,
                      duration, spkr), file=rttm_writer)


if __name__ == '__main__':
    main()
