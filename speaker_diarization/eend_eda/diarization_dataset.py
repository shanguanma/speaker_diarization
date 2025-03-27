#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Modified by: Yexin Yang
# Licensed under the MIT license.
#
import torch
import numpy as np
from speaker_diarization import kaldi_data
from speaker_diarization import feature
import logging


def _count_frames(data_len, size, step):
    # no padding at edges, last remaining samples are ignored
    return int((data_len - size + step) / step)


def _gen_frame_indices(
        data_length, size=2000, step=2000,
        use_last_samples=False,
        label_delay=0,
        subsampling=1):
    i = -1
    for i in range(_count_frames(data_length, size, step)):
        yield i * step, i * step + size
    if use_last_samples and i * step + size < data_length:
        if data_length - (i + 1) * step - subsampling * label_delay > 0:
            yield (i + 1) * step, data_length


def my_collate(batch):
    data, target = list(zip(*batch))
    return [data, target]


class KaldiDiarizationDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_dir,
            chunk_size=2000, # frames nums
            context_size=0,
            frame_size=1024, # sample point nums
            frame_shift=256, # sample point nums
            subsampling=1,
            rate=16000,
            input_transform=None,
            use_last_samples=False,
            label_delay=0,
            n_speakers=None,
            ):
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.context_size = context_size
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.subsampling = subsampling
        self.input_transform = input_transform
        self.n_speakers = n_speakers
        self.chunk_indices = []
        self.label_delay = label_delay

        self.data = kaldi_data.KaldiData(self.data_dir)
        self.rate = rate
        # make chunk indices: filepath, start_frame, end_frame
        for rec in self.data.wavs:
            data_len = int(self.data.reco2dur[rec] * rate / frame_shift)
            data_len = int(data_len / self.subsampling)
            for st, ed in _gen_frame_indices(
                    data_len, chunk_size, chunk_size, use_last_samples,
                    label_delay=self.label_delay,
                    subsampling=self.subsampling):
                self.chunk_indices.append(
                        (rec, st * self.subsampling, ed * self.subsampling))
        #print(len(self.chunk_indices), " chunks")
        logging.info(f"data set has {len(self.chunk_indices)} chunks!")

    def __len__(self):
        return len(self.chunk_indices)

    def __getitem__(self, i):
        rec, st, ed = self.chunk_indices[i]
        """
        Y, T, S = feature.get_labeledSTFT(
            self.data,
            rec,
            st,
            ed,
            self.frame_size,
            self.frame_shift,
            self.n_speakers)
        """
        Y, T, _ = feature.get_labeledSTFT(
            self.data,
            rec,
            st,
            ed,
            self.frame_size,
            self.frame_shift,
            self.n_speakers)

        # Y: (frame, num_ceps)
        #logging.info(f"in the __getitem__: feat Y shape: {Y.shape}")
        Y = feature.transform(Y, transform_type=self.input_transform, sample_rate=self.rate)


        # i.e. self.input_transform=logmel23_mn,it means that num_mels=23, Y shape:(frame,23)
        # Y_spliced: (frame, num_mels * (context_size * 2 + 1))
        #logging.info(f"in the __getitem__: after feature.transform Y shape: {Y.shape}")
        Y_spliced = feature.splice(Y, self.context_size)

        #logging.info(f"in the __getitem__: after splice Y shape: {Y_spliced.shape}")
        # Y_ss: (frame / subsampling, num_mels * (context_size * 2 + 1))
        # T_ss:(frame / subsampling, num_speakers)
        Y_ss, T_ss = feature.subsample(Y_spliced, T, self.subsampling)
        #logging.info(f"in the __getitem__: after subsample Y shape: {Y_ss.shape}, T_ss shape: {T_ss.shape}")

        # why add .copy ?  it can solve the below warning:
        # UserWarning: The given NumPy array is not writable,
        # and PyTorch does not support non-writable tensors.
        # This means writing to this tensor will result in undefined behavior.
        # You may want to copy the array to protect its data or make it writable before converting it to a tensor.
        # This type of warning will be suppressed for the rest of this program.
        # (Triggered internally at ../torch/csrc/utils/tensor_numpy.cpp:206.)
        # Y_ss = torch.from_numpy(Y_ss).float()

        Y_ss = torch.from_numpy(Y_ss.copy()).float()
        T_ss = torch.from_numpy(T_ss.copy()).float()
        return Y_ss, T_ss
