#!/usr/bin/env python3
# copy and modified from https://github.com/BUTSpeechFIT/EEND_dataprep/blob/main/v1/conv2wav.py
import numpy as np
import os
import kaldi_data
import scipy.fftpack
import scipy.io.wavfile
import random
import math
from types import SimpleNamespace
import argparse

seed = 3
np.random.seed(seed)
random.seed(seed)  # Python random module.

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

def parse_arguments() -> SimpleNamespace:
    parser = argparse.ArgumentParser(description='Conversations generator')
    parser.add_argument('--conversations-list-filename', type=str,
                        required=True, help='file with list of conversations \
                        to generate.')
    parser.add_argument('--input-wav-scp', type=str, required=True,
                        help='scp list of waveforms to use to create \
                        the conversations.')
    parser.add_argument('--in-conv-dir', type=str, required=True,
                        help='directory with the defined conversation files.')
    parser.add_argument('--out-wav-dir', type=str, required=True,
                        help='directory where waveforms corresponding to \
                        conversations will be generated.')
    parser.add_argument('--use-rirs', type=str2bool, default=False)
    #parser.add_argument('--no-use-rirs', dest='use_rirs', action='store_false')
    #parser.set_defaults(use_rirs=False)
    parser.add_argument('--rirs-wav-scp', type=str, required=False,
                        default=None, help='scp list of RIRs to apply.')
    parser.add_argument('--use-noises', type=str2bool, default=False)
    #parser.add_argument('--no-use-noises', dest='use_noises',
    #                    action='store_false')
    #parser.set_defaults(use_noises=False)
    parser.add_argument('--noises-wav-scp', type=str, required=False,
                        default=None, help='scp list of noises to apply.')
    parser.add_argument('--noises-snrs', type=str, required=False,
                        default=None, help='''SNR values used to add noises \
                        to the signal. Values must be separated by colons (:),i.e.SNRS="5:10:15:20"''')
    parser.add_argument('--sampling-frequency', type=int, required=False,
                        default=8000, help='data sampling frequency.')
    args = parser.parse_args()
    return args

# Note about reproducibility: the lists of noises and RIRs should be the
# same in between runs for the results to match.

if __name__ == '__main__':
    args = parse_arguments()

    convs_list = np.loadtxt(args.conversations_list_filename, dtype=object)
    wav_dict = kaldi_data.load_wav_scp(args.input_wav_scp)

    if args.rirs_wav_scp is None:
        rirs = {}
    else:
        rirs = kaldi_data.load_wav_scp(args.rirs_wav_scp)
        all_rirs = list(rirs.keys())
    if args.use_rirs:
        speech_rvb_probability = 0.5
    else:
        speech_rvb_probability = 0

    if args.noises_wav_scp is None:
        noises = {}
    else:
        noises = kaldi_data.load_wav_scp(args.noises_wav_scp)
        all_noises = list(noises.values())
    if args.noises_snrs:
        noise_snrs = [float(x) for x in args.noises_snrs.split(':')]
    #else:
    #    noise_snrs=[]

    # "-R" forces the default random seed for reproducibility
    resample_cmd = "sox -R -t wav - -t wav - rate {}".format(
        args.sampling_frequency)

    for conv_file in convs_list:
        print(conv_file)
        conversation = np.loadtxt(os.path.join(
            args.in_conv_dir, conv_file+'.conv'), dtype=object)
        if (conversation.ndim==1):
            conversation = np.expand_dims(conversation, axis=0)
        conversation[:, 2:] = conversation[:, 2:].astype(int)

        last_seg_end = max(conversation[:, 3] -
                           conversation[:, 2] + conversation[:, 4])

        all_session_files = np.unique(np.array([conversation[:, 1]]))

        all_session_data = {}
        rir=None
        for fname in all_session_files:
            signal = wav_dict[fname]

            # Randomly select a room impulse response
            if args.rirs_wav_scp is not None:
                choice_rir = random.choice(all_rirs)
                if random.random() < speech_rvb_probability:
                    rir = rirs[choice_rir]
                    # per speaker utterance is reverberated using room impulse response
                    preprocess = "wav-reverberate --print-args=false " \
                         " --impulse-response={} - -".format(rir)
                else:
                    rir = None

            # per speaker utterance is reverberated using room impulse response
            #preprocess = "wav-reverberate --print-args=false " \
            #             " --impulse-response={} - -".format(rir)
            if rir is not None:
                wav_rxfilename = kaldi_data.process_wav(
                    wav_dict[fname], preprocess)
                wav_rxfilename = kaldi_data.process_wav(
                    wav_rxfilename, resample_cmd)
            else:
                wav_rxfilename = wav_dict[fname]
            all_session_data[fname] = kaldi_data.load_wav(wav_rxfilename)[0]

        nspksall_session_vad = {fname: np.zeros_like(data, dtype=bool)
                                for fname, data in all_session_data.items()}

        out = np.zeros(last_seg_end)
        for seg in conversation:
            fname = seg[1]
            strt, end, pos = seg[2:]
            data = all_session_data[fname]
            if end > data.shape[0]:
                print(f"Trimming end of segment from {end} to {data.shape[0]} \
                    (error due to rounding in VAD segments)")
                end = data.shape[0]
            out[pos:pos+end-strt] += data[strt:end]

        #noise = random.choice(all_noises)
        #noise_snr = random.choice(noise_snrs)
        if args.use_noises:
            noise = random.choice(all_noises)
            noise_snr = random.choice(noise_snrs)
            # noise is repeated or cut for fitting to the mixture data length
            noise_resampled = kaldi_data.process_wav(noise, resample_cmd)
            noise_data, _ = kaldi_data.load_wav(noise_resampled)
            if last_seg_end > len(noise_data):
                noise_data = np.pad(noise_data, (
                    0, last_seg_end - len(noise_data)), 'wrap')
            else:
                noise_data = noise_data[:last_seg_end]
            # noise power is scaled according to selected SNR, then mixed
            signal_power = np.sum(out**2) / len(out)
            noise_power = np.sum(noise_data**2) / len(noise_data)
            scale = math.sqrt(
                math.pow(10, - noise_snr / 10) * signal_power / noise_power)
            out += noise_data * scale

        # Extra normalization to avoid clipping, in cases where it would,
        # and it maximizes the signal to cover the full range
        out = 2.*(out - np.min(out))/np.ptp(out)-1

        scipy.io.wavfile.write(os.path.join(
            args.out_wav_dir, os.path.splitext(conv_file)[0]+'.wav'
            ), args.sampling_frequency, out)
