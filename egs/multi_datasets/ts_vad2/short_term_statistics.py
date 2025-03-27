#!/usr/bin/env python3
# Author: Duo MA
# Email: maduo@cuhk.edu.cn

"""
In order to obtain the DER of the non-overlap short-term segment,
we obtain the non-overlap short-term segment by obtaining the reference RTTM, and then we crop the RTTM output by the model prediction.

step1: We get the overlap segment from the reference rttm.
step2: We use the pyannote gap function to get the non-overlap segment.
step3: We get the short-term segment from the reference rttm.
step4: We crop the short-term segment extracted in the third step from the non-overlap segment obtained in the second step to get the non-overlap short-term segment.

step5: We crop the non-overlap segment from hyp rttm via non-overlap segment of the second step
step6: We crop the short-term segment from non-overlap segment of step 5 via short-term segment of step 3.


how to run it ?
computer der of less than 1s segment
python3 ts_vad2/short_term_statistics.py 1 hyp_rttm ref_rttm

computer der of less than 2s segment
python3 ts_vad2/short_term_statistics.py 2 hyp_rttm ref_rttm
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from pyannote.core import Segment
from pyannote.core import Timeline
from pyannote.core import Annotation
from pyannote.core import SlidingWindow
from pyannote.core import SlidingWindowFeature
from pyannote.metrics.diarization import DiarizationErrorRate

def get_uris_from_txt(txt_path: str):
    uris = []
    with open(txt_path,'r')as f:
        for line in f:
            content = line.strip().split(" ")[0]
            uris.append(content)

    return uris

# get utt id
def get_uris_from_rttm(rttm_path: str):
    uris = set()
    with open(rttm_path,'r')as f:
        for line in f:
            uri = line.strip().split()[1]
            uris.add(uri)
    return uris

def annotation2timeline(reference: Annotation):
    timeline = Timeline(uri=reference.uri)
    for sl, tl in reference.itertracks():
        timeline.add(sl)
        endtime =sl.end
    return timeline, endtime


def get_non_overlap_timeline(reference: Annotation):
    ori_timeline,ori_endtime = annotation2timeline(reference)
    overlap_timeline = Timeline(uri=reference.uri)
    # Compare two segments
    for (s1,t1),(s2,t2) in reference.co_iter(reference):
        l1 = reference[s1, t1]
        l2 = reference[s2, t2]
        if l1 == l2: # two segments are same.
            continue
        overlap_timeline.add(s1&s2)

    # build two fake segments, in order to crop all non-overlap via gap function.
    segment_start = Segment(start=-1, end=0)
    segment_end = Segment(ori_endtime,ori_endtime + 1)

    overlap_timeline.add(segment_start)
    overlap_timeline.add(segment_end)
    gap_timeline = overlap_timeline.gaps()
    non_overlap_reference = reference.crop(gap_timeline)
    non_overlap_timeline = ori_timeline.crop(gap_timeline)

    return non_overlap_reference,  non_overlap_timeline



def get_shortsegment_timeline(reference: Annotation, short_duration: int=1):
    ShortSegmentTimeline = Timeline()
    for s1, t1 in reference.itertracks():
        if s1.duration<=int(short_duration):
            ShortSegmentTimeline.add(s1)
    return ShortSegmentTimeline.support()



def crop_segment(ShortSegmentTimeline, reference: Annotation):
    shortsegment = reference.crop(ShortSegmentTimeline)

    return shortsegment.support()



def rttm_parser(path: str, wav_name: str, uri):
    with open(path, 'r')as f:
        ann = Annotation(uri)
        for line in f:
            line = line.strip().split()
            utt_id = line[1]
            if utt_id == wav_name:
                start = float(line[3])
                duration = float(line[4])
                label = line[7]
                end = start + duration
                ann[Segment(start,end)] = label.strip()
        return ann

if __name__ == "__main__":
    short_duration = sys.argv[1]
    res_path = sys.argv[2] # ts_vad model predict rttm
    ref_path = sys.argv[3] # ref rttm
    collar = sys.argv[4] # default 0.25
    uri_list = get_uris_from_rttm(res_path)
    print(f"total utts: {len(uri_list)}")
    der = DiarizationErrorRate(skip_overlap=False,collar=float(collar))
    for uri in uri_list:
        print(f"uri: {uri}")
        hyp = rttm_parser(res_path, uri,"")
        ref = rttm_parser(ref_path,uri, "")

        shortsegment_tl = get_shortsegment_timeline(ref, short_duration=short_duration)

        non_overlap_reference, non_overlap_timeline = get_non_overlap_timeline(ref)
        if shortsegment_tl == Timeline():
            continue
        shortsegment_ref = crop_segment(shortsegment_tl, non_overlap_reference)
        shortsegment_hyp = crop_segment(non_overlap_timeline, hyp)
        shortsegment_hyp = crop_segment(shortsegment_tl, shortsegment_hyp)
        der(shortsegment_ref, shortsegment_hyp)
    print(der.report().to_string())
    confusion_ = der.report()["confusion"].iloc[-1,-1]
    der_ = der.report()["diarization error rate"].iloc[-1,-1]
    false_ = der.report()["false alarm"].iloc[-1,-1]
    miss_ = der.report()["missed detection"].iloc[-1,-1]
    print(f"der={round(der_,2)},miss={round(miss_,2)}, false={round(false_,2)},confusion={round(confusion_,2)}")
