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

def get_uris_from_txt(txt_path: str):
    uris = []
    with open(txt_path,'r')as f:
        for line in f:
            content = line.strip().split(" ")[0]
            uris.append(content)

    return uris

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
        if s1.duration<=short_duration:
            ShortSegmentTimeline.add(s1)
    return ShortSegmentTimeline.support()



def crop_segment(ShortSegmentTimeline, reference: Annotation):
    shortsegment = reference.crop(ShortSegmentTimeline)

    return shortsegment.support()


def ref_file_parser(rttm_path: str, uri):
    with open(rttm_path, 'r')as f:
        ann = Annotation(uri)
        for line in f.readlines():
            line = line.strip().split()
            start = float(line[3])
            duration = float(line[4])
            label = line[7]
            end  = start + duration
            ann[Segment(start,end)] = label.strip()
        return ann

def ref_tsvad_rttm_parser(path: str, wav_name: str, uri):
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
     uri_list =
