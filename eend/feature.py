# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Licensed under the MIT license.
#
# This module is for computing audio features
import logging
import numpy as np
import librosa
import scipy.signal

def get_input_dim(
        frame_size,
        context_size,
        transform_type,
        ):
    if transform_type.startswith('logmel23'):
        frame_size = 23
    else:
        fft_size = 1 << (frame_size-1).bit_length()
        frame_size = int(fft_size / 2) + 1
    input_dim = (2 * context_size + 1) * frame_size
    return input_dim


def transform(
        Y,
        transform_type=None,
        dtype=np.float32,
        sample_rate: int=16000,
        n_fft: int=1024, # it is same as stft setting, it is window size,the unit is sampling point
        ):
    """ Transform STFT feature

    Args:
        Y: STFT
            (n_frames, n_bins)-shaped np.complex array
        transform_type:
            None, "log"
        dtype: output data type
            np.float32 is expected
    Returns:
        Y (numpy.array): transformed feature
    """
    Y = np.abs(Y)
    if not transform_type:
        pass
    elif transform_type == 'log':
        Y = np.log(np.maximum(Y, 1e-10))
    elif transform_type == 'logmel':
        n_fft = 2 * (Y.shape[1] - 1)
        #sr = 16000
        sr=sample_rate
        n_mels = 40
        mel_basis = librosa.filters.mel(sr, n_fft, n_mels)
        Y = np.dot(Y ** 2, mel_basis.T)
        Y = np.log10(np.maximum(Y, 1e-10))
    elif transform_type == 'logmel23':
        n_fft = 2 * (Y.shape[1] - 1)
        #sr = 8000
        sr=sample_rate
        n_mels = 23
        mel_basis = librosa.filters.mel(sr, n_fft, n_mels)
        Y = np.dot(Y ** 2, mel_basis.T)
        Y = np.log10(np.maximum(Y, 1e-10))
    elif transform_type == 'logmel23_mn':
        n_fft = 2 * (Y.shape[1] - 1)
        #sr = 8000
        sr=sample_rate
        n_mels = 23
        mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
        Y = np.dot(Y ** 2, mel_basis.T)
        Y = np.log10(np.maximum(Y, 1e-10))
        mean = np.mean(Y, axis=0)
        Y = Y - mean
    elif transform_type == 'logmel23_swn':
        n_fft = 2 * (Y.shape[1] - 1)
        #sr = 8000
        sr=sample_rate
        n_mels = 23
        mel_basis = librosa.filters.mel(sr, n_fft, n_mels)
        Y = np.dot(Y ** 2, mel_basis.T)
        Y = np.log10(np.maximum(Y, 1e-10))
        #b = np.ones(300)/300
        #mean = scipy.signal.convolve2d(Y, b[:, None], mode='same')
        #
        # simple 2-means based threshoding for mean calculation
        powers = np.sum(Y, axis=1)
        th = (np.max(powers) + np.min(powers))/2.0
        for i in range(10):
            th = (np.mean(powers[powers >= th]) + np.mean(powers[powers < th])) / 2
        mean = np.mean(Y[powers > th,:], axis=0)
        Y = Y - mean
    elif transform_type == 'logmel23_mvn':
        n_fft = 2 * (Y.shape[1] - 1)
        #sr = 8000
        sr=sample_rate
        n_mels = 23
        mel_basis = librosa.filters.mel(sr, n_fft, n_mels)
        Y = np.dot(Y ** 2, mel_basis.T)
        Y = np.log10(np.maximum(Y, 1e-10))
        mean = np.mean(Y, axis=0)
        Y = Y - mean
        std = np.maximum(np.std(Y, axis=0), 1e-10)
        Y = Y / std
    elif transform_type == 'logmel23_espnet':
        # it is from https://github.com/espnet/espnet/blob/master/espnet/transform/spectrogram.py#L71
        fmin = 0
        fmax = sample_rate / 2
        n_mels=23
        eps=1e-10
        n_fft = 2 * (Y.shape[1] - 1) # because it is based on https://librosa.org/doc/main/generated/librosa.stft.html
        #if sample_rate==16000:
        #    assert n_fft == 400 # 25ms windows size
        #elif sample_rate==8000:
        #    assert n_fft == 200 # 25ms windows size
        mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
        Y = np.log10(np.maximum(eps, np.dot(Y, mel_basis.T)))
    else:
        raise ValueError('Unknown transform_type: %s' % transform_type)
    return Y.astype(dtype)


def subsample(Y, T, subsampling=1):
    """ Frame subsampling
    """
    Y_ss = Y[::subsampling]
    T_ss = T[::subsampling]
    return Y_ss, T_ss


def splice(Y, context_size=0):
    """ Frame splicing

    Args:
        Y: feature
            (n_frames, n_featdim)-shaped numpy array
        context_size:
            number of frames concatenated on left-side
            if context_size = 5, 11 frames are concatenated.

    Returns:
        Y_spliced: spliced feature
            (n_frames, n_featdim * (2 * context_size + 1))-shaped
    """
    Y_pad = np.pad(
            Y,
            [(context_size, context_size), (0, 0)],
            'constant')
    Y_spliced = np.lib.stride_tricks.as_strided(
            Y_pad,
            (Y.shape[0], Y.shape[1] * (2 * context_size + 1)),
            (Y.itemsize * Y.shape[1], Y.itemsize), writeable=False)
    return Y_spliced


def stft(
        data,
        frame_size=400,#25ms window size for 16000Hz
        frame_shift=160,# 10ms frame shift for 16000Hz
        #frame_size=200,#25ms window size for 8000Hz
        #frame_shift=80,# 10ms frame shift for 8000Hz
    ):
    """ Compute STFT features

    Args:
        data: audio signal
            (n_samples,)-shaped np.float32 array
        frame_size: number of samples in a frame (must be a power of two)
        frame_shift: number of samples between frames

    Returns:
        stft: STFT frames
            (n_frames, n_bins)-shaped np.complex64 array
            n_bins = 1+n_fft/2
    """
    # round up to nearest power of 2
    fft_size = 1 << (frame_size-1).bit_length()
    # HACK: The last frame is ommited
    #       as librosa.stft produces such an excessive frame
    if len(data) % frame_shift == 0:
        return librosa.stft(data, n_fft=fft_size, win_length=frame_size,
                            hop_length=frame_shift).T[:-1]
    else:
        return librosa.stft(data, n_fft=fft_size, win_length=frame_size,
                            hop_length=frame_shift).T


def _count_frames(data_len, size, shift):
    # HACK: Assuming librosa.stft(..., center=True)
    n_frames = 1 + int(data_len / shift)
    if data_len % shift == 0:
        n_frames = n_frames - 1
    return n_frames


def get_frame_labels(
        kaldi_obj,
        rec,
        start=0,
        end=None,
        frame_size=400,
        frame_shift=160,
        n_speakers=None):
    """ Get frame-aligned labels of given recording
    Args:
        kaldi_obj (KaldiData)
        rec (str): recording id
        start (int): start frame index
        end (int): end frame index
            None means the last frame of recording
        frame_size (int): number of frames in a frame
        frame_shift (int): number of shift samples
        n_speakers (int): number of speakers
            if None, the value is given from data
    Returns:
        T: label
            (n_frames, n_speakers)-shaped np.int32 array
    """
    filtered_segments = kaldi_obj.segments[kaldi_obj.segments['rec'] == rec]
    speakers = np.unique(
            [kaldi_obj.utt2spk[seg['utt']] for seg
                in filtered_segments]).tolist()
    if n_speakers is None:
        n_speakers = len(speakers)
    es = end * frame_shift if end is not None else None
    data, rate = kaldi_obj.load_wav(
            rec, start * frame_shift, es)
    n_frames = _count_frames(len(data), frame_size, frame_shift)
    T = np.zeros((n_frames, n_speakers), dtype=np.int32)
    if end is None:
        end = n_frames

    for seg in filtered_segments:
        speaker_index = speakers.index(kaldi_obj.utt2spk[seg['utt']])
        start_frame = np.rint(
                seg['st'] * rate / frame_shift).astype(int)
        end_frame = np.rint(
                seg['et'] * rate / frame_shift).astype(int)
        rel_start = rel_end = None
        if start <= start_frame and start_frame < end:
            rel_start = start_frame - start
        if start < end_frame and end_frame <= end:
            rel_end = end_frame - start
        if rel_start is not None or rel_end is not None:
            T[rel_start:rel_end, speaker_index] = 1
    return T


def get_labeledSTFT(
        kaldi_obj,
        rec, start, end, frame_size, frame_shift,
        n_speakers=None,
        use_speaker_id=False,
    ):
    """ Extracts STFT and corresponding labels

    Extracts STFT and corresponding diarization labels for
    given recording id and start/end times

    this function is used for stardand EEND or EEND-EDA diarization data format.
    it will ouput three types of data,
    1): speech feature i.e.: stft feature, shape: (frames, 1 + n_fft/2),
        Then, in the subsequent diarization dataset, we generate mel spectrograms, splice, subsample, and other operations.
    2): target feature, target feature, it is integer types, shape: (frames, num_speakers)
        The frame containing the speaker is set to 1, and the frame without the speaker is set to 0
    3): target speakers,(optional), it is not used at here.

    Args:
        kaldi_obj (KaldiData)
        rec (str): recording id
        start (int): start frame index
        end (int): end frame index
        frame_size (int): number of samples in a frame
        frame_shift (int): number of shift samples
        n_speakers (int): number of speakers
            if None, the value is given from data
    Returns:
        Y: STFT
            (n_frames, n_bins)-shaped np.complex64 array, n_bins = 1+n_fft/2
        T: label
            (n_frmaes, n_speakers)-shaped np.int32 array.
    """
    data, rate = kaldi_obj.load_wav(
            rec, start * frame_shift, end * frame_shift)
    Y = stft(data, frame_size, frame_shift)
    filtered_segments = kaldi_obj.segments[rec]
    #print(f"filtered_segments: {filtered_segments}")

    # filtered_segments = kaldi_obj.segments[kaldi_obj.segments['rec'] == rec]
    speakers = np.unique(
            [kaldi_obj.utt2spk[seg['utt']] for seg
                in filtered_segments]).tolist() # i.e. ['32-21631', '7312-92432']
    #print(f"speakers: {speakers}")
    if n_speakers is None:
        n_speakers = len(speakers)
    T = np.zeros((Y.shape[0], n_speakers), dtype=np.int32)

    if use_speaker_id:
        all_speakers = sorted(kaldi_obj.spk2utt.keys())
        S = np.zeros((Y.shape[0], len(all_speakers)), dtype=np.int32)
    
    for seg in filtered_segments:
        """
        >>> speakers = ['32-21631', '7312-92432']
        >>> speakers.index('32-21631')
        0
        >>> speakers.index('7312-92432') 
        1
        """
        speaker_index = speakers.index(kaldi_obj.utt2spk[seg['utt']])
        if use_speaker_id:
            all_speaker_index = all_speakers.index(kaldi_obj.utt2spk[seg['utt']])
        start_frame = np.rint(
                seg['st'] * rate / frame_shift).astype(int)
        end_frame = np.rint(
                seg['et'] * rate / frame_shift).astype(int)
        rel_start = rel_end = None
        if start <= start_frame and start_frame < end:
            rel_start = start_frame - start
        if start < end_frame and end_frame <= end:
            rel_end = end_frame - start
        if rel_start is not None or rel_end is not None:
            T[rel_start:rel_end, speaker_index] = 1
            print(f"rel_start: {rel_start}, rel_end: {rel_end}, T shape: {T.shape}, T: {T}")
            print(f"speaker_index: {speaker_index}") # 0, or 1  for two speaker
            print(f"Y shape: {Y.shape},") # 
            if use_speaker_id:
                S[rel_start:rel_end, all_speaker_index] = 1
                print(f"use_speaker_id, S shape: {S.shape}")
    
    if use_speaker_id:
        return Y, T, S
    else:
        return Y, T, None


def get_labeledSTFT_and_mask(
        kaldi_obj,
        rec, start, end, frame_size, frame_shift,
        n_speakers=None,
        use_speaker_id=False,
    ):
    """ Extracts STFT and corresponding labels

    Extracts STFT and corresponding diarization labels for
    given recording id and start/end times

    Compared with get_labeledSTFT(), this function is used for mask2former style diarization data format.

    Args:
        kaldi_obj (KaldiData)
        rec (str): recording id
        start (int): start frame index
        end (int): end frame index
        frame_size (int): number of samples in a frame
        frame_shift (int): number of shift samples
        n_speakers (int): number of speakers
            if None, the value is given from data
    Return:
        1): speech feature i.e.: stft feature, shape: (frames, 1 + n_fft/2),
        Then, in the subsequent diarization dataset, we generate mel spectrograms, splice, subsample, and other operations.
        2): target feature, it is bool style, shape (frames, num_segments_of_speakers),
            note: The frame containing the speaker is set to True, and the frame without the speaker is set to False
            At stardand EEND diarization setting, target feature, it is integer types.
            The frame containing the speaker is set to 1, and the frame without the speaker is set to 0
            it is used to compute dice and diarization(mask) loss
        3): target speaker label feature, it is one-dimension np.int32 array, shape (num_segments_of_speakers)
            Its length is The number of speech segments of the speaker included in this mixed speech.
            It is used to compute class loss.
    """
    logger = logging.getLogger(__name__)
    data, rate = kaldi_obj.load_wav(
            rec, start * frame_shift, end * frame_shift)
    Y = stft(data, frame_size, frame_shift)
    filtered_segments = kaldi_obj.segments[rec]
    #logger.warn(f"filtered_segments: {filtered_segments}, its length is : {len(filtered_segments)}")

    # filtered_segments = kaldi_obj.segments[kaldi_obj.segments['rec'] == rec]
    speakers = np.unique(
            [kaldi_obj.utt2spk[seg['utt']] for seg
                in filtered_segments]).tolist() # i.e. ['32-21631', '7312-92432']
    #print(f"speakers: {speakers}")
    if n_speakers is None:
        n_speakers = len(speakers)
    #T = np.zeros((Y.shape[0], n_speakers), dtype=np.int32)

    if use_speaker_id:
        all_speakers = sorted(kaldi_obj.spk2utt.keys())
        S = np.zeros((Y.shape[0], len(all_speakers)), dtype=np.int32)

    labels = [] # it will store speaker_index every speaker segments in the mix speech segments
    mask=[]
    for seg in filtered_segments:
        """
        >>> speakers = ['32-21631', '7312-92432']
        >>> speakers.index('32-21631')
        0
        >>> speakers.index('7312-92432')
        1
        """
        speaker_index = speakers.index(kaldi_obj.utt2spk[seg['utt']])
        if use_speaker_id:
            all_speaker_index = all_speakers.index(kaldi_obj.utt2spk[seg['utt']])
        start_frame = np.rint(
                seg['st'] * rate / frame_shift).astype(int)
        end_frame = np.rint(
                seg['et'] * rate / frame_shift).astype(int)
        rel_start = rel_end = None
        if start <= start_frame and start_frame < end:
            rel_start = start_frame - start
        if start < end_frame and end_frame <= end:
            rel_end = end_frame - start
        if rel_start is not None or rel_end is not None:
            #T[rel_start:rel_end, speaker_index] = 1
            #print(f"rel_start: {rel_start}, rel_end: {rel_end}, T shape: {T.shape}, T: {T}")
            #print(f"speaker_index: {speaker_index}") # 0, or 1  for two speaker
            #print(f"Y shape: {Y.shape},") #
           
            sub_mask = np.zeros(Y.shape[0],dtype=np.int32)
            sub_mask[rel_start:rel_end] = 1
            mask.append(sub_mask)
    
            labels.append(speaker_index)
            if use_speaker_id:
                S[rel_start:rel_end, all_speaker_index] = 1
                #print(f"use_speaker_id, S shape: {S.shape}")
    labels = np.array(labels) # list to np.array
    #logger.warn(f"labels: {labels}")
    #logger.warn(f"mask: {mask}")
    # mask=[], bool(mask)=False
    if bool(mask):
        #logger.warn(f"mask: {mask}")
        #return Y, None,None, None
        T = np.stack(mask, axis=0).T #(n_frmaes, num_segments_of_speakers)
        #print(f"labels: {labels}")
        T = T.astype(bool)
    else:
        T = np.zeros((Y.shape[0],labels.size),dtype=np.int32) 
    #logger.warn(f"T: {T}")
    #if mask:
    #    mask = np.stack(mask,axis=0).T
    #    mask = mask.astype(bool)
    if use_speaker_id:
        return Y, T, S, labels
    else:
        return Y, T, None,labels
