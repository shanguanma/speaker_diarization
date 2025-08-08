import os
import random
import numpy as np
import soundfile as sf
import librosa
from typing import List, Tuple, Dict, Callable
from scipy import signal
import random
import glob
import time
class SimuDiarMixer:
    def __init__(self,
                 spk2chunks: Dict[str, List[np.ndarray]],
                 sample_rate: int = 16000,
                 frame_length: float = 0.025, # 25ms
                 frame_shift: float = 0.04, # 40ms only for preparing label,0.04s means 25 frames in 1s, means 1s has 25 labels, means downsample rate is 4.
                 num_mel_bins: int = 80,
                 max_mix_len: float = 30.0,
                 min_silence: float = 0.0,
                 max_silence: float = 4.0,
                 min_speakers: int = 2,
                 max_speakers: int = 4,
                 target_overlap: float = 0.2,
                 musan_path: str = None,
                 rir_path: str = None,
                 noise_ratio: float = 0.8,
                 ):
        """
        spk2chunks: {spk_id: [vad后片段, ...]}
        sample_rate: 采样率
        max_mix_len: 混合音频总时长（秒）
        min_silence, max_silence: 静音片段长度范围（秒）
        target_overlap: 目标重叠率（帧级别多于1人说话的帧占比）
        """
        
        self.spk2chunks = spk2chunks
        self.sr = sample_rate
        self.frame_length = frame_length
        self.num_mel_bins = num_mel_bins
        self.frame_shift = frame_shift
        self.num_mel_bins = num_mel_bins
        self.max_mix_len = max_mix_len
        self.min_silence = min_silence
        self.max_silence = max_silence
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.target_overlap = target_overlap
        self.musan_path = musan_path
        self.rir_path = rir_path
        self.noise_ratio = noise_ratio
        
        # 计算数据集大小，用于PyTorch DataLoader
        self.dataset_size = max(1000, len(spk2chunks) * 10)  # 每个epoch随机生成样本
        
    def __len__(self):
        """返回数据集大小"""
        return self.dataset_size
    
    def __getitem__(self, idx):
        """获取数据样本"""
        # 忽略idx，每次都随机生成新样本
        return self.sample()

    def load_musan_or_rirs(self, musan_path, rir_path):
        # add noise and rir augment
        if musan_path is not None:
            noiselist = {}
            noisetypes = ["noise", "speech", "music"]
            noisesnr = {"noise": [0, 15], "speech": [13, 20], "music": [5, 15]}
            # setting is from wespeaker
            noisesnr = {"noise": [0, 15], "speech": [10, 30], "music": [5, 15]}
            numnoise = {"noise": [1, 1], "speech": [3, 8], "music": [1, 1]}

            augment_files = glob.glob(
                os.path.join(musan_path, "*/*/*.wav")
            )  # musan/*/*/*.wav
            for file in augment_files:
                if file.split("/")[-3] not in noiselist:
                    noiselist[file.split("/")[-3]] = []
                noiselist[file.split("/")[-3]].append(file)
                #print(f'file.split("/")[-3]: {file.split("/")[-3]}')
        else:
            noisesnr=None
            numnoise=None
            noisetypes=None
            noiselist=None
        if rir_path is not None:
            rir_files = glob.glob(
                os.path.join(rir_path, "*/*.wav")
            )  # RIRS_NOISES/*/*.wav
        else:
            rir_files =None
        return noisesnr, numnoise, noiselist, rir_files, noisetypes

    def add_reverb(self, audio, rir_files):
        rir_file = random.choice(rir_files)
        print(f"rir_file: {rir_file}")
        rir, _ = sf.read(rir_file)
        print(f"rir shape: {rir.shape}")
        if len(rir.shape)>1:
             # it is multi channel, (samples, num_channels)
            rir = rir[:,0]

        rir = np.expand_dims(rir.astype(float), 0)
        rir = rir / np.sqrt(np.sum(rir**2))
        print(f"rir shape: {rir.shape}, audio shape: {audio.shape}")
        return signal.convolve(audio, rir, mode="full")[:, :audio.shape[1]]

    def add_noise(self, audio, noiselist, noisesnr, noisetype, numnoise):
        #clean_db = 10 * np.log10(max(1e-4, np.mean(audio**2)))
        clean_db = 10 * np.log10(1e-4 + np.mean(audio**2))
        noiselist_cat = random.sample(noiselist[noisetype], random.randint(numnoise[noisetype][0], numnoise[noisetype][1]))
        print(f"noiselist_cat: {noiselist_cat}, its len: {len(noiselist_cat)}")
        noises = []
        length = audio.shape[1]
        for noise in noiselist_cat:
            noiseaudio, sr = sf.read(noise)
            if sr != 16000:
                noiseaudio = librosa.resample(noiseaudio, orig_sr=sr, target_sr=16000)

            if noiseaudio.shape[0] < length:
                noiseaudio = np.pad(noiseaudio, (0, length - noiseaudio.shape[0]), "wrap")
            else:
                start_frame = int(random.random() * (noiseaudio.shape[0] - length))
                noiseaudio = noiseaudio[start_frame:start_frame+length]
            noiseaudio = np.expand_dims(noiseaudio, 0)
            #noise_db = 10 * np.log10(max(1e-4, np.mean(noiseaudio**2)))
            noise_db = 10 * np.log10(1e-4 + np.mean(noiseaudio**2))
            snr = random.uniform(noisesnr[noisetype][0], noisesnr[noisetype][1])
            noises.append(np.sqrt(10 ** ((clean_db - noise_db - snr) / 10)) * noiseaudio)
        noise = np.sum(np.concatenate(noises, axis=0), axis=0, keepdims=True)
        return noise[:, :length] + audio

    def sample_post(self):
        """
        对sample()输出的mix进行加噪和/或加混响，保证长度不变。
        - mix: np.ndarray [T]
        - label: np.ndarray [N, T]
        - spk_ids: list
        - add_noise: bool，是否加噪
        - add_reverb: bool，是否加混响
        返回: (mix_new, label, spk_ids)
        """
        mix, label,spk_ids = self.sample()
        
        noisesnr, numnoise, noiselist, rir_files, noisetypes = self.load_musan_or_rirs(self.musan_path, self.rir_path)
        if self.rir_path is not None or self.musan_path is not None:
            add_noise = np.random.choice(2, p=[1 - self.noise_ratio, self.noise_ratio])
            if add_noise == 1:
                if self.rir_path is not None and self.musan_path is not None:
                    mix =  np.expand_dims(np.array(mix), axis=0) #(1,T)
                    # add_noise and add_reverb required 2d data
                    ntypes = random.choice(noisetypes) # choice one from  ["noise", "speech", "music"]
                    mix = self.add_noise(mix, noiselist, noisesnr, ntypes, numnoise)
                    mix = self.add_reverb(mix,rir_files)
                elif self.rir_path is not None:
                    mix =  np.expand_dims(np.array(mix), axis=0) #(1,T)
                    mix = self.add_reverb(mix, rir_files)
                elif self.musan_path is not None:
                    mix =  np.expand_dims(np.array(mix), axis=0) #(1,T)
                    ntypes = random.choice(noisetypes) # choice one from  ["noise", "speech", "music"]
                    mix = self.add_noise(mix, noiselist, noisesnr, ntypes, numnoise)
                mix = np.squeeze(mix,axis=0) #(1,T) ->(T)
        return mix, label, spk_ids
            
    def _find_silence_regions(self, active, min_len=1, value=0):
        # active: 0/1 array, value: 0(静音) or 1(有声/重叠)
        regions = []
        cur = 0
        start = None
        for i, v in enumerate(active):
            if v == value:
                if start is None:
                    start = i
                cur += 1
            else:
                if cur >= min_len:
                    regions.append((start, i))
                cur = 0
                start = None
        if cur >= min_len:
            regions.append((start, len(active)))
        return regions

    @staticmethod
    def extract_fbank(wav, sample_rate=16000, num_mel_bins=80, frame_length=0.025, frame_shift=0.01):
        """
        提取 kaldi 风格的 fbank 特征。
        wav: numpy array, 1D
        sample_rate: 采样率
        num_mel_bins: 梅尔滤波器组数
        frame_length: 帧长(s)
        frame_shift: 帧移(s)
        返回: [num_frames, num_mel_bins] 的 torch.Tensor
        """
        import torch
        import torchaudio
        if isinstance(wav, np.ndarray):
            wav_tensor = torch.tensor(wav, dtype=torch.float32)
        else:
            wav_tensor = wav.float()
        if wav_tensor.dim() == 2:
            wav_tensor = wav_tensor[:, 0]  # 只取第一通道
        fbank = torchaudio.compliance.kaldi.fbank(
            wav_tensor.unsqueeze(0),
            num_mel_bins=num_mel_bins,
            frame_length=25,
            frame_shift=10,
            sample_frequency=sample_rate,
            use_log_fbank=True,
            dither=1.0,
            window_type='hamming'
        ).squeeze(0)  # [num_frames, num_mel_bins]
        fbank = (fbank - fbank.mean(dim=0)) / fbank.std(dim=0).clamp(min=1e-8)
        return fbank

    def collate_fn(self, batch, vad_out_len=200):
        # batch: list of (mix, label, spk_ids)
        max_len = max([len(x[0]) for x in batch])
        # 先提取所有fbank帧数，确定max_frames
        fbanks = []
        fbank_lens = []
        for mix, _, _ in batch:
            fbank = self.extract_fbank(mix)
            fbanks.append(fbank)
            fbank_lens.append(fbank.shape[0])
        max_frames = max(fbank_lens)
        max_spks = max([x[1].shape[0] for x in batch])
        wavs = []
        labels = []
        spk_ids_list = []
        fbanks_pad = []
        labels_len = []
        for (mix, label, spk_ids), fbank in zip(batch, fbanks):
            pad_wav = np.pad(mix, (0, max_len - len(mix)), 'constant')
            wavs.append(pad_wav)
            pad_spk_ids = list(spk_ids) + [None] * (max_spks - len(spk_ids))
            spk_ids_list.append(pad_spk_ids)
            # fbank pad
            pad_fbank = np.pad(fbank, ((0, max_frames - fbank.shape[0]), (0, 0)), 'constant')
            fbanks_pad.append(pad_fbank)
            # label对齐到fbank帧(这里的fbank 帧的帧移是40ms 也就是考虑了fbank 特征送入ResNetExtractor 后输出是下采样4倍的 )
            N, T_sample = label.shape
            T_fbank = fbank.shape[0]
            aligned_label = np.zeros((N, T_fbank), dtype=np.float32)
            win_length = int(self.frame_length * self.sr)
            hop_length = int(self.frame_shift * self.sr)
            for n in range(N):
                for t in range(int(T_fbank/4)):
                    start = t * hop_length
                    end = min(start + win_length, T_sample)
                    aligned_label[n, t] = label[n, start:end].max()
            pad_label = np.zeros((max_spks, max_frames), dtype=np.float32)
            pad_label[:aligned_label.shape[0], :aligned_label.shape[1]] = aligned_label
            labels.append(pad_label)

            labels_len.append(min(label.shape[1], vad_out_len) if label.ndim > 1 else 0)
        import torch
        wavs = torch.tensor(np.stack(wavs), dtype=torch.float32)
        labels = torch.tensor(np.stack(labels), dtype=torch.float32)
        fbanks = torch.tensor(np.stack(fbanks_pad), dtype=torch.float32)
        labels_len = torch.tensor(labels_len, dtype=torch.int32)

        return wavs, labels, spk_ids_list, fbanks, labels_len
    

    def collate_fn_wo_feat(self, batch):
        # batch: list of (mix, label, spk_ids)
        # 只做padding，不提取特征
        max_len = max([len(x[0]) for x in batch])
        max_frames = max([x[1].shape[1] for x in batch])
        max_spks = max([x[1].shape[0] for x in batch])
        wavs = []
        labels = []
        spk_ids_list = []
        for mix, label, spk_ids in batch:
            pad_wav = np.pad(mix, (0, max_len - len(mix)), 'constant')
            pad_label = np.zeros((max_spks, max_frames), dtype=np.float32)
            pad_label[:label.shape[0], :label.shape[1]] = label
            wavs.append(pad_wav)
            labels.append(pad_label)
            pad_spk_ids = list(spk_ids) + [None] * (max_spks - len(spk_ids))
            spk_ids_list.append(pad_spk_ids)
        import torch
        wavs = torch.tensor(np.stack(wavs), dtype=torch.float32)
        labels = torch.tensor(np.stack(labels), dtype=torch.float32)
        return wavs, labels, spk_ids_list

    def sample(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        随机采样生成一个混合样本
        返回: (混合音频, 活动标签, spk_ids)
        标签 shape: [num_speakers, T]，每帧0/1
        性能优化：
        - 尽量减少np.where/np.sum等全量操作
        - 插入成功后及时break
        - 只shuffle一次
        """
        sr = self.sr
        total_len = int(self.max_mix_len * sr)
        all_spk_ids = list(self.spk2chunks.keys())
        n_spk = np.random.randint(self.min_speakers, min(self.max_speakers, len(all_spk_ids)) + 1)
        spk_ids = list(np.random.choice(all_spk_ids, n_spk, replace=False))
        num_spk = len(spk_ids)
        main_spk = np.random.choice(spk_ids)
        # 优化：提前shuffle
        main_chunks = self.spk2chunks[main_spk].copy()
        np.random.shuffle(main_chunks)
        timeline = np.zeros(total_len, dtype=np.float32)
        label_mat = np.zeros((num_spk, total_len), dtype=np.float32)
        t = 0
        # 1. 主说话人顺序排布片段，静音区间严格切割
        for chunk in main_chunks:
            chunk_len = min(len(chunk), total_len - t)
            if chunk_len <= 0:
                break
            timeline[t:t+chunk_len] += chunk[:chunk_len]
            label_mat[spk_ids.index(main_spk), t:t+chunk_len] = 1
            t += chunk_len
            if t >= total_len:
                break
            remain = int(sr * np.random.uniform(self.min_silence, self.max_silence))
            t = min(t + remain, total_len)
        # 2. 其他说话人片段插入，优先重叠区，后静音区
        target_overlap_frames = int(self.target_overlap * total_len)
        # 优化：缓存active和overlap_mask，减少sum操作
        for i, spk in enumerate(spk_ids):
            if spk == main_spk:
                continue
            chunks = self.spk2chunks[spk].copy()
            np.random.shuffle(chunks)
            for chunk in chunks:
                # 优先重叠区
                active = (label_mat.sum(axis=0) > 0).astype(int)
                overlap_regions = self._find_silence_regions(active, min_len=1, value=1)
                candidate_starts = []
                for start, end in overlap_regions:
                    region_len = end - start
                    insert_len = min(len(chunk), region_len)
                    if insert_len > 0:
                        candidate_starts.append((start, insert_len))
                np.random.shuffle(candidate_starts)
                inserted = False
                for start, insert_len in candidate_starts:
                    end = start + insert_len
                    if label_mat[i, start:end].sum() > 0:
                        continue
                    # 计算插入后重叠帧数
                    overlap_mask = (label_mat.sum(axis=0) > 0).astype(int)
                    chunk_mask = np.zeros(total_len, dtype=int)
                    chunk_mask[start:end] = 1
                    overlap_incr = ((overlap_mask + chunk_mask) > 1).sum() - (overlap_mask > 1).sum()
                    current_overlap = (label_mat.sum(axis=0) > 1).sum()
                    if current_overlap + overlap_incr > target_overlap_frames:
                        remain = target_overlap_frames - current_overlap
                        if remain > 0:
                            for l in range(remain, 0, -1):
                                if start + l > total_len:
                                    continue
                                test_mask = np.zeros(total_len, dtype=int)
                                test_mask[start:start+l] = 1
                                test_incr = ((overlap_mask + test_mask) > 1).sum() - (overlap_mask > 1).sum()
                                if current_overlap + test_incr <= target_overlap_frames:
                                    label_mat[i, start:start+l] = 1
                                    timeline[start:start+l] += chunk[:l]
                                    inserted = True
                                    break
                        break
                    else:
                        label_mat[i, start:end] = 1
                        timeline[start:end] += chunk[:end-start]
                        inserted = True
                        break
                if inserted and (label_mat.sum(axis=0) > 1).sum() >= target_overlap_frames:
                    break
            # 如果重叠率已达标，尝试填补静音区
            if (label_mat.sum(axis=0) > 1).sum() >= target_overlap_frames:
                for chunk in chunks:
                    active = (label_mat.sum(axis=0) > 0).astype(int)
                    silence_regions = self._find_silence_regions(active, min_len=1, value=0)
                    silence_regions = [r for r in silence_regions if (r[1]-r[0]) > 0]
                    if not silence_regions:
                        break
                    region = silence_regions[np.random.randint(len(silence_regions))]
                    region_len = region[1] - region[0]
                    insert_len = min(len(chunk), region_len, int(self.max_silence * sr))
                    if insert_len <= 0:
                        continue
                    start = region[0]
                    end = start + insert_len
                    if label_mat[i, start:end].sum() > 0:
                        continue
                    label_mat[i, start:end] = 1
                    timeline[start:end] += chunk[:insert_len]
        # 3. 最后强制切割所有静音区间为max_silence
        active = (label_mat.sum(axis=0) > 0).astype(int)
        silence_regions = self._find_silence_regions(active, min_len=1)
        for start, end in silence_regions:
            region_len = end - start
            if region_len > int(self.max_silence * sr):
                for s in range(start, end, int(self.max_silence * sr)):
                    e = min(s + int(self.max_silence * sr), end)
                    if e - s > int(self.max_silence * sr):
                        timeline[s+int(self.max_silence*sr):e] += np.random.normal(0, 1e-6, e-s-int(self.max_silence*sr))
        if np.max(np.abs(timeline)) > 0:
            timeline = timeline / (np.max(np.abs(timeline)) + 1e-8)
        active_spk_mask = (label_mat.sum(axis=1) > 0)
        label_mat = label_mat[active_spk_mask]
        used_spk_ids = [spk for spk, m in zip(spk_ids, active_spk_mask) if m]
        return timeline, label_mat, used_spk_ids
        
# 用法示例：
# vad_func = ... # 你的VAD函数
# spk2wav = {'spk1': [...], ...}
# spk2chunks = {spk: [vad_func(wav, sr) for wav in wavs] for spk, wavs in spk2wav.items()}
# mixer = SimuDiarMixer(spk2chunks, sample_rate=16000, max_mix_len=30.0, min_silence=0.0, max_silence=4.0, target_overlap=0.2)
# mix, label, spk_ids = mixer.sample()

# 用法示例：
# 假设你有spk2wav = {'spk1': ['a.wav', ...], ...}
# 以及vad_func(wav, sr) -> [(start, end), ...]
# 你可以用pyannote.audio/pipeline, webrtcvad, funasr等实现vad_func
#
# def vad_func(wav, sr):
#     ... # 返回[(start, end), ...]，单位秒
#
# dataset = SimuDiarDataset(spk2wav, vad_func)
# mix, labels = dataset.sample() 
