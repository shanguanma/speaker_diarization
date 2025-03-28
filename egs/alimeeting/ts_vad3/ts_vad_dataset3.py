from collections import defaultdict
import logging
import os
from scipy import signal
import glob, json, random, wave
import re
from tqdm import tqdm

import numpy as np
import soundfile as sf

from torch.nn.utils.rnn import pad_sequence
import torch
import librosa
import torchaudio.compliance.kaldi as kaldi
from typing import Any, Dict

logger = logging.getLogger(__name__)


class FBank(object):
    def __init__(
        self,
        n_mels,
        sample_rate,
        mean_nor: bool = False,
    ):
        self.n_mels = n_mels
        self.mean_nor = mean_nor
        self.sample_rate = sample_rate

    def __call__(self, wav, dither=1.0):
        sr = 16000
        assert sr == self.sample_rate, f"sample_rate"
        if len(wav.shape) == 1:
            wav = wav.unsqueeze(0)
        # select single channel
        if wav.shape[0] > 1:
            wav = wav[0, :]
        assert len(wav.shape) == 2 and wav.shape[0] == 1
        wav = wav * (1 << 15)
        logger.debug(f"in the ts_vad_dataset.py , here wav is from ref_speech, wav: {wav}, its shape: {wav.shape}")
        feat = kaldi.fbank(
            wav,
            num_mel_bins=self.n_mels,
            sample_frequency=sr,
            dither=dither,
            window_type="hamming",
            use_energy=False,
        )
        logger.debug(f"in the ts_vad_dataset.py , here feat is from ref_speech,feat: {feat},its shape: {feat.shape}")
        # feat: [T, N]
        if self.mean_nor:
            feat = feat - feat.mean(0, keepdim=True)
        return feat


def _collate_data(
    frames,
    is_audio_input: bool = False,
    is_label_input: bool = False,
    is_embed_input: bool = False,
) -> torch.Tensor:
    if is_audio_input:
        max_len = max(frame.size(0) for frame in frames)
        out = frames[0].new_zeros((len(frames), max_len))
        for i, v in enumerate(frames):
            out[i, : v.size(0)] = v

    if is_label_input:
        max_len = max(frame.size(1) for frame in frames)
        out = frames[0].new_zeros((len(frames), frames[0].size(0), max_len))
        for i, v in enumerate(frames):
            out[i, :, : v.size(1)] = v

    if is_embed_input:
        if len(frames[0].size()) == 2:
            max_len = max(frame.size(0) for frame in frames)
            max_len2 = max(frame.size(1) for frame in frames)
            out = frames[0].new_zeros((len(frames), max_len, max_len2))
            for i, v in enumerate(frames):
                out[i, : v.size(0), : v.size(1)] = v
        elif len(frames[0].size()) == 3:
            max_len = max(frame.size(1) for frame in frames)
            out = frames[0].new_zeros(
                (len(frames), frames[0].size(0), max_len, frames[0].size(2))
            )
            for i, v in enumerate(frames):
                out[i, :, : v.size(1), :] = v

    return out


class TSVADDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        json_path: str,
        audio_path: str,
        ts_len: int,
        rs_len: int,
        is_train: bool,
        musan_path: str = None,
        rir_path: str = None,
        noise_ratio: float = 0.5,
        spk_path: str = None,
        fbank_spk_path: str = None,
        frame_spk_path: str = None,
        multi_level_fuse_spk: bool = False,
        rs_segment_shift: int = 2,
        zero_ratio: float = 0.5,
        max_num_speaker: int = 4,
        dataset_name: str = "alimeeting",
        sample_rate: int = 16000,
        embed_len: float = 1,
        embed_shift: float = 0.4,
        embed_input: bool = False,
        fbank_input: bool = False,
        label_rate: int = 25,
        random_channel: bool = False,
        support_mc: bool = False,
        random_mask_speaker_prob: float = 0.0,
        random_mask_speaker_step: int = 0,
        speaker_embed_dim: int = 192,  # same as speaker_embed_dim of model
    ):
        self.audio_path = audio_path
        self.spk_path = spk_path
        self.fbank_spk_path = fbank_spk_path
        self.frame_spk_path = frame_spk_path
        self.multi_level_fuse_spk = multi_level_fuse_spk
        self.ts_len = ts_len  # Number of second for target speech

        self.dataset_name = dataset_name

        self.random_mask_speaker_prob = random_mask_speaker_prob
        self.random_mask_speaker_step = random_mask_speaker_step
        self.speaker_embed_dim = speaker_embed_dim

        ## load data and label,
        ## it will prepare chunk segment information for mixture audio
        self.label_rate = label_rate
        self.sample_rate = sample_rate
        self.rs_segment_shift = rs_segment_shift
        self.rs_len = rs_len  # Number of second for reference speech
        self.is_train = is_train
        self.data_list, self.label_dic, self.sizes, self.spk2data, self.data2spk = (
            self.load_data_and_label(json_path)
        )

        # add noise and rir augment
        self.musan_path = musan_path
        self.rir_path = rir_path
        if musan_path is not None or rir_path is not None:
            self.noisesnr, self.numnoise, self.noiselist, self.rir_files = (
                self.load_musan_or_rirs(musan_path, rir_path)
            )

        self.noise_ratio = noise_ratio
        self.zero_ratio = zero_ratio
        self.max_num_speaker = max_num_speaker

        self.embed_len = int(self.sample_rate * embed_len)
        self.embed_shift = int(self.sample_rate * embed_shift)
        self.embed_input = embed_input
        self.label_rate = label_rate
        self.fbank_input = fbank_input
        self.random_channel = random_channel
        self.support_mc = support_mc
        self.update_num = 0
        if fbank_input:
            logger.info(
                f"model expect fbank as input , fbank_input should be {fbank_input} !!!"
            )
            self.feature_extractor = FBank(
                80, sample_rate=self.sample_rate, mean_nor=True
            )
        else:
            logger.info(
                f"speech encoder may be self supervise pretrain model, its input is wavform."
            )
            self.feature_extractor = None

        logger.info(
            f"loaded sentence={len(self.sizes)}, "
            f"shortest sent={min(self.sizes)}, longest sent={max(self.sizes)}, "
            f"rs_len={rs_len}, rs_segment_shift={rs_segment_shift},  rir={rir_path is not None}, "
            f"musan={musan_path is not None}, noise_ratio={noise_ratio}, zero_ratio={self.zero_ratio} "
        )

    def load_data_and_label(self, json_path):
        ###
        lines = open(json_path).read().splitlines()
        filename_set = set()
        label_dic = defaultdict(list)
        data_list = []
        sizes = []
        spk2data = {}
        data2spk = {}
        # Load the data and labels
        for line in tqdm(lines):
            dict = json.loads(line)
            length = len(dict["labels"])  # Number of frames (1s = 25 frames)
            filename = dict["filename"]
            labels = dict["labels"]
            speaker_id = str(dict["speaker_key"])
            speaker_id_full = str(dict["speaker_id"])

            if speaker_id_full not in spk2data:
                spk2data[speaker_id_full] = []
            if self.dataset_name == "alimeeting":
                spk2data[speaker_id_full].append(
                    filename + "/" + speaker_id
                )  # A speaker corresponds to all its speech segment in the long mixture audio.
                # in order to random choice target embedding base on the above list.
                data2spk[filename + "/" + speaker_id] = speaker_id_full
            else:
                raise Exception(
                    f"The given dataset {self.dataset_name} is not supported."
                )
            full_id = filename + "_" + speaker_id
            label_dic[full_id] = labels
            if filename in filename_set:
                pass
            else:
                filename_set.add(filename)
                dis = self.label_rate * self.rs_segment_shift
                chunk_size = self.label_rate * self.rs_len
                folder = (
                    self.audio_path + "/" + filename + "/*.wav"
                )  # target speaker wavform

                audios = glob.glob(folder)
                num_speaker = (
                    len(audios) - 1  # -1 means -1 all.wav(it is mixer wav)
                )  # The total number of speakers, 2 or 3 or 4

                ## get chunk segment information for mixture audio.
                ## We use fixed window length(i.e. 4s) data to train TSVAD
                ## length is from labels for long mixture auido.
                # we fix label rate is 25, mean that has 25 target labels in 1s mixture audio.
                for start in range(0, length, dis):
                    end = (
                        (start + chunk_size) if start + chunk_size < length else length
                    )
                    if self.is_train:
                        short_ratio = 3
                    else:
                        short_ratio = 0
                    if end - start > self.label_rate * short_ratio:
                        data_intro = [filename, num_speaker, start, end]
                        data_list.append(data_intro)
                        sizes.append((end - start) * self.sample_rate / self.label_rate)

        return data_list, label_dic, sizes, spk2data, data2spk

    def load_musan_or_rirs(self, musan_path, rir_path):
        # add noise and rir augment
        if musan_path is not None:
            noiselist = {}
            noisetypes = ["noise", "speech", "music"]
            noisesnr = {"noise": [0, 15], "speech": [13, 20], "music": [5, 15]}
            numnoise = {"noise": [1, 1], "speech": [3, 8], "music": [1, 1]}

            augment_files = glob.glob(
                os.path.join(musan_path, "*/*/*.wav")
            )  # musan/*/*/*.wav
            for file in augment_files:
                if file.split("/")[-3] not in noiselist:
                    noiselist[file.split("/")[-3]] = []
                noiselist[file.split("/")[-3]].append(file)

        if rir_path is not None:
            rir_files = glob.glob(
                os.path.join(rir_path, "*/*.wav")
            )  # RIRS_NOISES/*/*.wav
        return noisesnr, numnoise, noiselist, rir_files

    def repeat_to_fill(self, x, window_fs):
        length = x.size(0)
        num = (window_fs + length - 1) // length

        return x.repeat(num)[:window_fs]

    def segment_rs(self, ref_speech):
        subsegs = []
        for begin in range(0, ref_speech.size(0), self.embed_shift):
            end = min(begin + self.embed_len, ref_speech.size(0))
            subsegs.append(self.repeat_to_fill(ref_speech[begin:end], self.embed_len))

        return torch.stack(subsegs, dim=0)

    def get_ids(self, num_speaker, add_ext=False):
        speaker_ids = []
        for i in range(self.max_num_speaker - (1 if add_ext else 0)):
            if i < num_speaker:
                speaker_ids.append(i + 1)
            else:
                speaker_ids.append(-1)

        if self.is_train:
            random.shuffle(speaker_ids)

        if add_ext:
            speaker_ids.append(-2)
        return speaker_ids

    def load_rs(self, file, speaker_ids, start, stop):
        audio_start = self.sample_rate // self.label_rate * start
        audio_stop = self.sample_rate // self.label_rate * stop
        if self.dataset_name == "alimeeting":
            audio_path = os.path.join(
                self.audio_path, file + "/all.wav"
            )  ## This audio_path is single channel mixer audio,
            ## now it is used in alimeeting dataset,and is stored at target_audio directory.
            ref_speech, rc = self.read_audio_with_resample(
                audio_path,
                start=audio_start,
                length=(audio_stop - audio_start),
                support_mc=self.support_mc,
            )
            if len(ref_speech.shape) == 1:
                ref_speech = np.expand_dims(np.array(ref_speech), axis=0)

        frame_len = audio_stop - audio_start
        assert (
            frame_len - ref_speech.shape[1] <= 100
        ), f"frame_len {frame_len} ref_speech.shape[1] {ref_speech.shape[1]}"
        if frame_len - ref_speech.shape[1] > 0:
            new_ref_speech = np.zeros((ref_speech.shape[0], frame_len))
            new_ref_speech[:, : ref_speech.shape[1]] = ref_speech
            ref_speech = new_ref_speech

        ## add noise and rirs augment
        if self.rir_path is not None or self.musan_path is not None:
            add_noise = np.random.choice(2, p=[1 - self.noise_ratio, self.noise_ratio])
            if add_noise == 1:
                if self.rir_path is not None and self.musan_path is not None:
                    noise_type = random.randint(0, 1)
                    if noise_type == 0:
                        ref_speech = self.add_rev(ref_speech, length=frame_len)
                    elif noise_type == 1:
                        ref_speech = self.choose_and_add_noise(
                            random.randint(0, 2), ref_speech, frame_len
                        )
                elif self.rir_path is not None:
                    ref_speech = self.add_rev(ref_speech, length=frame_len)
                elif self.musan_path is not None:
                    ref_speech = self.choose_and_add_noise(
                        random.randint(0, 2), ref_speech, frame_len
                    )

        ref_speech = torch.FloatTensor(np.array(ref_speech))

        labels = []
        new_speaker_ids = []
        residual_label = np.zeros(stop - start)
        for speaker_id in speaker_ids:
            if speaker_id == -1:
                labels.append(np.zeros(stop - start))  # Obatin the labels for silence
            elif speaker_id == -2:  # Obatin the labels for extral
                residual_label[residual_label > 1] = 1
                labels.append(residual_label)
                new_speaker_ids.append(-2)
                continue
            else:  ##
                full_label_id = file + "_" + str(speaker_id)
                label = self.label_dic[full_label_id]
                labels.append(
                    label[start:stop]
                )  # Obatin the labels for the reference speech

            mask_prob = 0
            if self.random_mask_speaker_prob != 0:
                mask_prob = self.random_mask_speaker_prob * min(
                    self.update_num / self.random_mask_speaker_step, 1.0
                )
            if sum(labels[-1]) == 0 and self.is_train:
                new_speaker_ids.append(-1)
            elif (
                sum(new_speaker_ids) != -1 * len(new_speaker_ids)
                and np.random.choice(2, p=[1 - mask_prob, mask_prob])
                and self.is_train
            ):
                new_speaker_ids.append(-1)
                residual_label = residual_label + labels[-1]
                labels[-1] = np.zeros(stop - start)
            else:
                new_speaker_ids.append(speaker_id)
        labels = torch.from_numpy(np.array(labels)).float()  # 4, T
        return ref_speech, labels, new_speaker_ids, rc

    def load_alimeeting_multi_level_ts_embed(self,file,speaker_ids):
        """
        For target speaker embedding processing.
        silence case:
           inference: fill zero vector
           train: random other target speaker embedding and mean it.
        target speaker case:
           inference: random one segement (i.e. 6s ) from target speaker embedding
           train : mean on total segement target speaker embeddings.
        """
        target_speeches = [] # utterance level speaker embedding
        fbank_feats = [] # fbank level speaker embedding
        frame_feats = [] # frame level speaker embedding
        exist_spk = []
        # print(f"file:{file}, speaker_ids: {speaker_ids}")
        for speaker_id in speaker_ids:
            if speaker_id != -1 and speaker_id != -2:
                audio_filename = speaker_id
                exist_spk.append(self.data2spk[f"{file}/{audio_filename}"])

        for speaker_id in speaker_ids:
            if speaker_id == -1:  # Obatin the labels for silence
                ## prepared silence embedding at inference stage and dev set and testset
                if (
                    np.random.choice(2, p=[1 - self.zero_ratio, self.zero_ratio]) == 1
                    or not self.is_train
                ):

                    # (TODO) maduo add speaker embedding dimension parameter to replace hard code.
                    feature = torch.zeros(
                        self.speaker_embed_dim
                    )  # speaker embedding dimension of speaker model
                    if self.fbank_spk_path is not None:
                        fbank_feat = torch.zeros(self.ts_len*100,80) # (T,D) # T: 1s have 100frames, i.e.ts_len=6s, T=600,80 means fbank dimension
                    if self.frame_spk_path is not None:
                        frame_feat = torch.zeros(512,self.ts_len*100//2) # i.e. cam++ model, subsample is 2. frame dimension is 512.
                else:
                    # ## prepared silence embedding at train stage and trainset
                    random_spk = random.choice(list(self.spk2data))

                    while random_spk in exist_spk:
                        random_spk = random.choice(list(self.spk2data))
                    exist_spk.append(random_spk)

                    random_spk_embed=random.choice(self.spk2data[random_spk])
                    logger.debug(f"random_spk_embed: {random_spk_embed}")
                    path = os.path.join(
                        self.spk_path,## utterance level speaker embedding
                        f"{random_spk_embed}.pt",
                    )
                    feature = torch.load(path, map_location="cpu")
                    if self.fbank_spk_path is not None:
                        fbank_path = os.path.join(
                            self.fbank_spk_path,
                            f"{random_spk_embed}.pt",
                        )
                        fbank_feat = torch.load(fbank_path,map_location="cpu")
                    if self.frame_spk_path is not None:
                        frame_path = os.path.join(
                            self.frame_spk_path,
                            f"{random_spk_embed}.pt",
                        )
                        frame_feat = torch.load(frame_path,map_location="cpu")

            elif speaker_id == -2:  # # Obatin the labels for extral
                feature = torch.zeros(
                    self.speaker_embed_dim
                )  # speaker embedding dimension of speaker model
                if self.fbank_spk_path is not None:
                    fbank_feat = torch.zeros(self.ts_len*100,80) # (T,D) # T: 1s have 100frames, i.e.ts_len=6s, T=600,80 means fbank dimension
                if self.frame_spk_path is not None:
                    fbank_feat = torch.zeros(512,self.ts_len*100//2) #(F,T//2) i.e. cam++ model, subsample is 2. frame dimension is 512.
            else:  # # Obatin the labels for speaker
                audio_filename = speaker_id
                path = os.path.join(self.spk_path, file, str(audio_filename) + ".pt")
                logger.debug(f"file, str(audio_filename): {file}/{str(audio_filename)}")
                logger.debug(f"feature path: {path}")
                feature = torch.load(path, map_location="cpu")
                if self.fbank_spk_path is not None:
                    fbank_path = os.path.join(self.fbank_spk_path,file, str(audio_filename) + ".pt")
                    fbank_feat = torch.load(fbank_path,map_location="cpu") # (num_segments, T,80)

                if self.frame_spk_path is not None:
                    frame_path = os.path.join(self.frame_spk_path,file, str(audio_filename) + ".pt")
                    logger.debug(f"frame_path: {frame_path}")
                    frame_feat = torch.load(frame_path,map_location="cpu") # (num_segments, 512,T//2)


            if len(feature.size()) == 2 :
                if self.is_train:
                    segments_index = random.randint(0, feature.shape[0] - 1) #
                    logger.debug(f"feature shape: {feature.shape}, segments_index: {segments_index}")
                    feature = feature[segments_index, :]
                    if self.fbank_spk_path is not None and len(fbank_feat.size()) == 3:
                        fbank_feat = fbank_feat[segments_index,:,:]
                    if self.frame_spk_path is not None and len(frame_feat.size()) == 3:
                        logger.debug(f"frame_feat shape: {frame_feat.shape}, segments_index: {segments_index}")
                        frame_feat = frame_feat[segments_index,:,:]
                else:
                    # feature = torch.mean(feature, dim = 0)
                    feature = torch.mean(feature, dim=0)
                    if self.fbank_spk_path is not None and len(fbank_feat.size()) == 3:
                        fbank_feat = torch.mean(fbank_feat,dim=0)
                    if self.frame_spk_path is not None and len(frame_feat.size()) == 3:
                        frame_feat = torch.mean(frame_feat,dim=0)
            target_speeches.append(feature)
            if self.fbank_spk_path is not None:
                fbank_feats.append(fbank_feat)
            if self.frame_spk_path is not None:
                frame_feats.append(frame_feat)
        target_speeches = torch.stack(target_speeches)
        fbank_featss = torch.zeros(4,self.ts_len*100,80)
        frame_featss = torch.zeros(4,512,self.ts_len*100//2)
        if self.fbank_spk_path is not None:
            #if fbank_feats.size(1)!=self.ts_len*100:
            #    pad_len=self.ts_len*100-fbank_feats.size(1)
                #fbank_feats = torch.nn.functional.pad(fbank_feats,(0,0,0,pad_len),mode='constant', value=0)
            fbank_featss = pad_sequence(fbank_feats,batch_first=True)
            logger.debug(f"fbank_featss: {fbank_featss},its shape: {fbank_featss.shape}")
            #fbank_featss = torch.stack(fbank_feats)
        if self.frame_spk_path is not None:
            # frame_feats: [(D,T1),(D,T2)...] ->[(T1,D),(T2,D),...]
            frame_feats = [frame_feat.permute(1,0) for frame_feat in frame_feats]
            frame_featss = pad_sequence(frame_feats,batch_first=True) #(4,T,D)
            #frame_featss = frame_featss.permute(0,2,1) #(4,D,T)

            logger.debug(f"frame_featss: {frame_featss},its shape: {frame_featss.shape}")
        return target_speeches,fbank_featss, frame_featss

    def load_alimeeting_ts_embed(self, file, speaker_ids):
        """
        For target speaker embedding processing.
        silence case:
           inference: fill zero vector
           train: random other target speaker embedding and mean it.
        target speaker case:
           inference: random one segement (i.e. 6s ) from target speaker embedding
           train : mean on total segement target speaker embeddings.
        """
        target_speeches = []
        exist_spk = []
        # print(f"file:{file}, speaker_ids: {speaker_ids}")
        for speaker_id in speaker_ids:
            if speaker_id != -1 and speaker_id != -2:
                audio_filename = speaker_id
                exist_spk.append(self.data2spk[f"{file}/{audio_filename}"])

        for speaker_id in speaker_ids:
            if speaker_id == -1:  # Obatin the labels for silence
                ## prepared silence embedding at inference stage and dev set and testset
                if (
                    np.random.choice(2, p=[1 - self.zero_ratio, self.zero_ratio]) == 1
                    or not self.is_train
                ):

                    # (TODO) maduo add speaker embedding dimension parameter to replace hard code.
                    feature = torch.zeros(
                        self.speaker_embed_dim
                    )  # speaker embedding dimension of speaker model
                else:
                    # ## prepared silence embedding at train stage and trainset
                    random_spk = random.choice(list(self.spk2data))

                    while random_spk in exist_spk:
                        random_spk = random.choice(list(self.spk2data))
                    exist_spk.append(random_spk)

                    path = os.path.join(
                        self.spk_path,
                        f"{random.choice(self.spk2data[random_spk])}.pt",
                    )
                    feature = torch.load(path, map_location="cpu")

            elif speaker_id == -2:  # # Obatin the labels for extral
                feature = torch.zeros(
                    self.speaker_embed_dim
                )  # speaker embedding dimension of speaker model
            else:  # # Obatin the labels for speaker
                audio_filename = speaker_id
                path = os.path.join(self.spk_path, file, str(audio_filename) + ".pt")
                feature = torch.load(path, map_location="cpu")

            if len(feature.size()) == 2:
                if self.is_train:
                    feature = feature[random.randint(0, feature.shape[0] - 1), :]
                else:
                    # feature = torch.mean(feature, dim = 0)
                    feature = torch.mean(feature, dim=0)
            target_speeches.append(feature)
        target_speeches = torch.stack(target_speeches)
        return target_speeches

    def load_ts_embed(self, file, speaker_ids):
        ## prepared target speaker embedding data
        if self.dataset_name == "alimeeting":
            target_speeches, fbank_feats, frame_feats = self.load_alimeeting_multi_level_ts_embed(file, speaker_ids)
                #target_speeches = self.load_alimeeting_ts_embed(file, speaker_ids)
            return target_speeches,fbank_feats, frame_feats

    def load_ts(self, file, speaker_ids):
        ## prepared target speaker wavform data
        if self.dataset_name == "alimeeting":
            # rc=0 means that we only select first channel data if multi channel wavfrom data.
            target_speeches = self.load_alimeeting_ts(file, speaker_ids, rc=0)
        return target_speeches

    def load_alimeeting_ts(self, file, speaker_ids, rc=0):
        """
        For target speaker wavform processing.
        silence case:
            inference: fill zero wavfrom
            train:  random target speaker wavform as silence case.
             > ts_len: random ts_len wavfrom from other target speaker wavform utterance.
             < ts_len: read total other target speaker wavfrom utterance.
        target speaker:
             inference:
              > ts_len: read frist ts_len wavfrom from target speaker wavform utterance
              < ts_len: read total target speaker wavfrom utterance.
             train :
              > ts_len: read random ts_len wavfrom from target speaker wavform utterance
              < ts_len: read total target speaker wavfrom utterance.

        """
        target_speeches = []
        exist_spk = []
        speaker_id_full_list = []
        ts_mask = []
        for speaker_id in speaker_ids:
            if speaker_id != -1 and speaker_id != -2:
                audio_filename = speaker_id
                exist_spk.append(self.data2spk[f"{file}/{audio_filename}"])
        logger.debug(f"exist_spk: {exist_spk}")
        for speaker_id in speaker_ids:
            if speaker_id == -1:  # Obatin the labels for silence
                if (
                    np.random.choice(2, p=[1 - self.zero_ratio, self.zero_ratio]) == 0
                    and self.is_train
                ):
                    # prepared silence target path at train stage
                    random_spk = random.choice(list(self.spk2data))
                    while random_spk in exist_spk:
                        random_spk = random.choice(list(self.spk2data))
                    exist_spk.append(random_spk)
                    path = os.path.join(self.audio_path, f"{random.choice(self.spk2data[random_spk])}.wav")
                    logger.debug(
                         f"speaker_id==-1,random target speech :{path} random_spk: {random_spk}")
                    target_speech = self.cut_target_speech(path,rc)
                    #target_speech = self.cut_target_speech_v2(path,rc)
                else:
                    # for inference silence case or use zeros vector to instead random target speech.
                    target_speech = torch.zeros(self.ts_len * self.sample_rate)  # fake one

            elif speaker_id == -2:
                target_speech = torch.zeros(self.ts_len * self.sample_rate)  # fake one
            else:  # # Obatin the labels for speaker
                audio_filename = speaker_id
                path = os.path.join(self.audio_path, file, str(audio_filename) + ".wav")
                target_speech = self.cut_target_speech(path,rc)
                #target_speech = self.cut_target_speech_v2(path,rc)
                logger.debug(f"target speaker wavform: {path},self.audio_path: {self.audio_path}, file: {file},str(audio_filename): {str(audio_filename)}!!!")
            target_speeches.append(target_speech)
        logger.debug(f"target_speeches: {target_speeches}")
        target_speeches = _collate_data(target_speeches, is_audio_input=True) #(4,self.ts_len * self.sample_rate)
        logger.debug(f"after _collate_data fn, target_speeches: {target_speeches},its shape: {target_speeches.shape}")
        return target_speeches

    def cut_target_speech_v3(self,path,rc):
        """
        if trainset and devset align to offline mode,
        I need to add codes switch trainset or devset.
        the below code will output the shape target speech tensor(num_segs,args.ts_len * 16000) or (target_speech)
        ## for logger target_speech wavform, it will many num_segs. online speaker model input shape (B,4,num_segs,args.ts_len * 16000),
        ## its limitation will OOM on big speaker model.
        """
        import wave
        wav_length = wave.open(path, "rb").getnframes()  # entire length for target speech
        target_speechs=[]
        if wav_length <= self.ts_len*16000:
            target_speechss, _ = sf.read(path)
            target_speechss = torch.from_numpy(target_speechss)
        else:
            for start in range(0,wav_length - int(self.ts_len * 16000), int(1* 16000)):
                stop = random_start + int(self.ts_len * 16000)
                target_speech, _ = sf.read(path, start=random_start, stop=stop)
                target_speechs.append(torch.from_numpy(target_speech))
            target_speechss = torch.stack(target_speechs) #(num_segs,args.ts_len * 16000)
        return target_speechss


    def cut_target_speech_v2(self,path,rc):
        """
        trainset align to offline mode
        devset doesn't align to offline mode
        """
        import wave
        wav_length = wave.open(path, "rb").getnframes()  # entire length for target speech
        if wav_length <= self.ts_len*16000:
            target_speech, _ = sf.read(path,dtype="float32")
        else:
            if self.is_train:
                start = list(range(0,wav_length - int(self.ts_len * 16000), int(1* 16000)))
                random_start = random.randint(0,len(start)-1)

            else:
                random_start=0
            stop = random_start + int(self.ts_len * 16000)
            target_speech, _ = sf.read(path, start=random_start, stop=stop,dtype="float32")
        target_speech = torch.from_numpy(target_speech)
        return target_speech

    def cut_target_speech(self,path,rc):
        assert librosa.get_duration(path=path) > 0.01
        aux_len = librosa.get_duration(path=path) # aux_len's unit is second.
        if aux_len <= self.ts_len:
            target_speech, _ = sf.read(path)
        else:
            sr_cur = self.sample_rate
            if self.is_train:
                start_frame = np.int64(
                    random.random() * (aux_len - self.ts_len) * sr_cur
                )
            else:
                start_frame = 0
            target_speech, _ = self.read_audio_with_resample(
                path,
                start=start_frame,
                length=int(self.ts_len * sr_cur),
                sr_cur=sr_cur,
                rc=rc,
            )
        target_speech = torch.from_numpy(target_speech)
        return target_speech


    def add_rev(self, audio, length):
        rir_file = random.choice(self.rir_files)
        rir, _ = self.read_audio_with_resample(rir_file)
        rir = np.expand_dims(rir.astype(float), 0)
        rir = rir / np.sqrt(np.sum(rir**2))
        return signal.convolve(audio, rir, mode="full")[:, :length]

    def add_noise(self, audio, noisecat, length):
        clean_db = 10 * np.log10(max(1e-4, np.mean(audio**2)))
        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(
            self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1])
        )
        noises = []
        for noise in noiselist:
            noiselength = wave.open(noise, "rb").getnframes()
            noise_sample_rate = wave.open(noise, "rb").getframerate()
            if noise_sample_rate != self.sample_rate:
                noiselength = int(noiselength * self.sample_rate / noise_sample_rate)
            if noiselength <= length:
                noiseaudio, _ = self.read_audio_with_resample(noise)
                noiseaudio = np.pad(noiseaudio, (0, length - noiselength), "wrap")
            else:
                start_frame = np.int64(random.random() * (noiselength - length))
                noiseaudio, _ = self.read_audio_with_resample(
                    noise, start=start_frame, length=length
                )
            noiseaudio = np.stack([noiseaudio], axis=0)
            noise_db = 10 * np.log10(max(1e-4, np.mean(noiseaudio**2)))
            noisesnr = random.uniform(
                self.noisesnr[noisecat][0], self.noisesnr[noisecat][1]
            )
            noises.append(
                np.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio
            )
        noise = np.sum(np.concatenate(noises, axis=0), axis=0, keepdims=True)
        if noise.shape[1] < length:
            assert length - noise.shape[1] < 10
            audio[:, : noise.shape[1]] = noise + audio[:, : noise.shape[1]]
            return audio
        else:
            return noise[:, :length] + audio

    def choose_and_add_noise(self, noise_type, ref_speech, frame_len):
        assert self.musan_path is not None
        if noise_type == 0:
            return self.add_noise(ref_speech, "speech", length=frame_len)
        elif noise_type == 1:
            return self.add_noise(ref_speech, "music", length=frame_len)
        elif noise_type == 2:
            return self.add_noise(ref_speech, "noise", length=frame_len)

    def __len__(self):
        return len(self.sizes)

    def size(self, index):
        return self.sizes[index]

    def num_tokens(self, index):
        return self.size(index)
    # for batch
    def collater(self, samples):
        if len([s["labels"] for s in samples]) == 0:
            return {}

        ref_speech_len = [s["ref_speech"].size(1) for s in samples]
        if sum(ref_speech_len) == len(ref_speech_len) * self.rs_len * self.sample_rate:
            labels = torch.stack([s["labels"] for s in samples], dim=0)
            ref_speech = torch.stack([s["ref_speech"] for s in samples], dim=0)
        else:
            labels = _collate_data([s["labels"] for s in samples], is_label_input=True)
            ref_speech = _collate_data(
                [s["ref_speech"] for s in samples], is_embed_input=True
            )

        target_speech = torch.stack([s["target_speech"] for s in samples], dim=0)
        #fbank_feat = torch.stack([s["fbank_feat"] for s in samples], dim=0)
        fbank_feat = _collate_data(
            [s["fbank_feat"] for s in samples], is_embed_input=True
        )# (B,4,T,F)
        #frame_feat = torch.stack([s["frame_feat"] for s in samples], dim=0)
        frame_feat = _collate_data(
                [s["frame_feat"] for s in samples], is_embed_input=True
        ) #(B,4,T,D)
        frame_feat = frame_feat.permute(0,1,3,2) #(B,4,D,T)
        labels_len = torch.tensor(
            [s["labels"].size(1) for s in samples], dtype=torch.long
        )

        if not self.support_mc:
            assert ref_speech.size(1) == 1
            ref_speech = ref_speech[:, 0, :]
        net_input = {
            "ref_speech": ref_speech,
            "target_speech": target_speech,
            "fbank_feat": fbank_feat,
            "frame_feat": frame_feat,
            "labels": labels,
            "labels_len": labels_len,
            "file_path": [s["file_path"] for s in samples],
            "speaker_ids": [s["speaker_ids"] for s in samples],
            "start": [s["start"] for s in samples],
        }

        batch = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": net_input,
        }

        return batch

    def __getitem__(self, index):
        # T: number of frames (1 frame = 0.04s)
        # ref_speech : 16000 * (T / 25)
        # labels : 4, T
        # target_speech: 4, 16000 * (T / 25)
        # file, num_speaker, start, stop, _ = self.data_list[index]
        file, num_speaker, start, stop = self.data_list[index]
        speaker_ids = self.get_ids(
            num_speaker
        )  # num_speaker means that it contains number of speaker in current mixture utterance.

        ref_speech, labels, new_speaker_ids, _ = self.load_rs(
            file, speaker_ids, start, stop
        )

        if self.embed_input:
            ref_speech = self.segment_rs(ref_speech)

        if self.fbank_input:
            ref_speech = [self.feature_extractor(rs) for rs in ref_speech]
            ref_speech = torch.stack(ref_speech)

        if self.spk_path is None:
            # target_speech, _, _, _ = self.load_ts(file, new_speaker_ids)
            target_speech = self.load_ts(file, new_speaker_ids)
            # logger.info(f"in the __getitem__ fn: target_speech shape: {target_speech.shape}")
        else:
            target_speech,fbank_feat,frame_feat = self.load_ts_embed(file, new_speaker_ids)

        samples = {
            "id": index,
            "ref_speech": ref_speech,
            "target_speech": target_speech,
            "fbank_feat": fbank_feat,
            "frame_feat": frame_feat,
            "labels": labels,
            "file_path": file,
            "speaker_ids": np.array(speaker_ids),
            "start": np.array(start),
        }
        return samples

    def ordered_indices(self):
        order = [np.random.permutation(len(self))]
        order.append(self.sizes)
        return np.lexsort(order)[::-1]

    def read_audio_with_resample(
        self, audio_path, start=None, length=None, sr_cur=None, support_mc=False, rc=-1
    ):
        if sr_cur is None:
            sr_cur = self.sample_rate
        audio_sr = librosa.get_samplerate(audio_path)

        if audio_sr != self.sample_rate:
            try:
                if start is not None:
                    audio, _ = librosa.load(
                        audio_path,
                        offset=start / sr_cur,
                        duration=length / sr_cur,
                        sr=sr_cur,
                        mono=False,
                    )
                else:
                    audio, _ = librosa.load(audio_path, sr=sr_cur, mono=False)
            except Exception as e:
                logger.info(e)
                audio, _ = librosa.load(audio_path, sr=sr_cur, mono=False)
                audio = audio[start : start + length]
            if len(audio.shape) > 1 and not support_mc:
                if self.random_channel and self.is_train:
                    if rc == -1:
                        rc = np.random.randint(0, audio.shape[1])
                    audio = audio[rc, :]
                else:
                    # use reference channel
                    audio = audio[0, :]
        else:
            try:
                if start is not None:
                    audio, _ = sf.read(
                        audio_path, start=start, stop=start + length, dtype="float32"
                    )
                else:
                    audio, _ = sf.read(audio_path, dtype="float32")
            except Exception as e:
                logger.info(e)
                audio, _ = sf.read(audio_path, dtype="float32")
                audio = audio[start : start + length]
            if len(audio.shape) > 1 and not support_mc:
                if self.random_channel and self.is_train:
                    if rc == -1:
                        rc = np.random.randint(0, audio.shape[1])
                    audio = audio[:, rc]
                else:
                    # use reference channel
                    audio = audio[:, 0]
            else:
                audio = np.transpose(audio)

        return audio, rc
