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
import json
import gzip
import logging
import math
# 设置日志
logger = logging.getLogger(__name__)

class SimuDiarMixer:
    def __init__(self,
                 spk2chunks: Dict[str, List[np.ndarray]] = None,
                 voxceleb2_spk2chunks_json: str = None,
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
                 downsample_factor: int = 4, # downsample rate is 4 , only for preparing label, because we hope that 1s has 25 labels, fbank is 1s 100 labels(frames), 
                 ):
        """
        spk2chunks: {spk_id: [vad后片段, ...]} (可选，用于传统模式)
        voxceleb2_spk2chunks_json: VAD文件路径 (可选，用于懒加载模式)
        sample_rate: 采样率
        max_mix_len: 混合音频总时长（秒）
        min_silence, max_silence: 静音片段长度范围（秒）
        target_overlap: 目标重叠率（帧级别多于1人说话的帧占比）
        """
        
        self.spk2chunks = spk2chunks
        self.voxceleb2_spk2chunks_json = voxceleb2_spk2chunks_json
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
        self.downsample_factor = downsample_factor  
        # 懒加载模式相关
        self.lazy_mode = voxceleb2_spk2chunks_json is not None
        if self.lazy_mode:
            self._speaker_cache = {}  # 缓存已加载的说话人数据
            self._speaker_list = None  # 说话人列表缓存
            logger.info(f"启用懒加载模式，VAD文件: {voxceleb2_spk2chunks_json}")
        else:
            logger.info("使用传统模式，需要预先加载spk2chunks")
        
        # 音频增强相关
        if self.musan_path is not None:
            if os.path.exists(self.musan_path):
                logger.info(f"启用噪声增强，MUSAN路径: {self.musan_path}")
            else:
                logger.warning(f"MUSAN路径不存在: {self.musan_path}")
                self.musan_path = None
        
        if self.rir_path is not None:
            if os.path.exists(self.rir_path):
                logger.info(f"启用混响增强，RIR路径: {self.rir_path}")
            else:
                logger.warning(f"RIR路径不存在: {self.rir_path}")
                self.rir_path = None
        
        if self.musan_path is not None or self.rir_path is not None:
            logger.info(f"音频增强概率: {self.noise_ratio:.2f}")
        
        # 计算数据集大小，用于PyTorch DataLoader
        if self.lazy_mode:
            # 懒加载模式下，数据集大小基于VAD文件中的说话人数量
            self.dataset_size = max(1000, self._get_speaker_count() * 10)
        else:
            self.dataset_size = max(1000, len(spk2chunks) * 10)  # 每个epoch随机生成样本
        
    def __len__(self):
        """返回数据集大小"""
        return self.dataset_size
    
    def __getitem__(self, idx):
        """获取数据样本"""
        # 忽略idx，每次都随机生成新样本
        if self.lazy_mode:
            # i.e.mix shape: (128000,), label shape: (3, 128000), spk_ids: ['5038', '3787', '5660']
            mix, label, spk_ids = self.sample_lazy()
            #print(f"mix shape: {mix.shape}, label shape: {label.shape}, spk_ids: {spk_ids}")
        else:
            mix, label, spk_ids = self.sample()
        
        # 应用音频增强（加噪和/或加混响）
        mix = self.apply_audio_augmentation(mix)
        
        # 添加数据来源标识：1表示模拟数据
        data_source = 1  # 0: real data, 1: simulated data
        return mix, label, spk_ids, data_source

    def apply_audio_augmentation(self, mix, force_augment=False):
        """
        对音频进行加噪和/或加混响增强
        Args:
            mix: np.ndarray [T] - 输入的混合音频
            force_augment: bool - 是否强制应用增强（忽略noise_ratio）
        Returns:
            np.ndarray [T] - 增强后的音频
        """
        if self.musan_path is None and self.rir_path is None:
            return mix
        
        # 根据noise_ratio决定是否进行增强，除非强制增强
        if not force_augment and np.random.random() > self.noise_ratio:
            return mix
        
        try:
            # 加载增强资源
            noisesnr, numnoise, noiselist, rir_files, noisetypes = self.load_musan_or_rirs(self.musan_path, self.rir_path)
            
            # 检查资源是否可用
            if (self.musan_path is not None and (noiselist is None or len(noiselist) == 0)) or \
               (self.rir_path is not None and (rir_files is None or len(rir_files) == 0)):
                logger.warning("增强资源不可用，跳过音频增强")
                return mix
            
            # 转换为2D格式，因为add_noise和add_reverb需要2D数据
            mix_2d = np.expand_dims(np.array(mix), axis=0)  # (1, T)
            
            # 加噪
            if self.musan_path is not None and noiselist is not None and len(noiselist) > 0:
                ntypes = random.choice(noisetypes)  # 随机选择噪声类型
                mix_2d = self.add_noise(mix_2d, noiselist, noisesnr, ntypes, numnoise)
            
            # 加混响
            if self.rir_path is not None and rir_files is not None and len(rir_files) > 0:
                mix_2d = self.add_reverb(mix_2d, rir_files)
            
            # 转换回1D格式
            mix_augmented = np.squeeze(mix_2d, axis=0)  # (T)
            
            return mix_augmented
            
        except Exception as e:
            logger.error(f"音频增强过程中发生错误: {e}")
            logger.warning("跳过音频增强，返回原始音频")
            return mix

    def is_augmentation_available(self):
        """
        检查音频增强是否可用
        Returns:
            bool: 如果增强资源可用返回True，否则返回False
        """
        if self.musan_path is None and self.rir_path is None:
            return False
        
        try:
            noisesnr, numnoise, noiselist, rir_files, noisetypes = self.load_musan_or_rirs(self.musan_path, self.rir_path)
            
            musan_available = (self.musan_path is not None and 
                              noiselist is not None and 
                              len(noiselist) > 0)
            
            rir_available = (self.rir_path is not None and 
                            rir_files is not None and 
                            len(rir_files) > 0)
            
            return musan_available or rir_available
        except Exception:
            return False
    
    def get_augmentation_info(self):
        """
        获取音频增强的详细信息
        Returns:
            dict: 包含增强配置信息的字典
        """
        info = {
            'musan_path': self.musan_path,
            'rir_path': self.rir_path,
            'noise_ratio': self.noise_ratio,
            'augmentation_available': self.is_augmentation_available()
        }
        
        if self.is_augmentation_available():
            try:
                noisesnr, numnoise, noiselist, rir_files, noisetypes = self.load_musan_or_rirs(self.musan_path, self.rir_path)
                
                if self.musan_path is not None and noiselist is not None:
                    info['musan_categories'] = list(noiselist.keys())
                    info['musan_file_counts'] = {k: len(v) for k, v in noiselist.items()}
                
                if self.rir_path is not None and rir_files is not None:
                    info['rir_file_count'] = len(rir_files)
                    
            except Exception as e:
                info['error'] = str(e)
        
        return info

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
        rir_file = random.choice(rir_files) # rir wav path str
        #print(f"rir_file: {rir_file}")  
        rir, _ = sf.read(rir_file) # (223104,) 
        #print(f"rir shape: {rir.shape}")
        if len(rir.shape)>1:
             # it is multi channel, (samples, num_channels)
            rir = rir[:,0]

        rir = np.expand_dims(rir.astype(float), 0)
        rir = rir / np.sqrt(np.sum(rir**2)) #
        #i.e. rir shape: (1, 369980), audio shape: (1, 128000)
        #print(f"rir shape: {rir.shape}, audio shape: {audio.shape}")
        return signal.convolve(audio, rir, mode="full")[:, :audio.shape[1]]

    def add_noise(self, audio, noiselist, noisesnr, noisetype, numnoise):
        #clean_db = 10 * np.log10(max(1e-4, np.mean(audio**2)))
        clean_db = 10 * np.log10(1e-4 + np.mean(audio**2))
        # i.e. noiselist_cat: ['/maduo/datasets/musan/noise/free-sound/noise-free-sound-0493.wav'], its len: 1
        noiselist_cat = random.sample(noiselist[noisetype], random.randint(numnoise[noisetype][0], numnoise[noisetype][1]))
        #print(f"noiselist_cat: {noiselist_cat}, its len: {len(noiselist_cat)}")
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

    def sample_post(self, mix=None, label=None, spk_ids=None, force_augment=False):
        """
        对音频进行加噪和/或加混响，保证长度不变。
        Args:
            mix: np.ndarray [T] - 输入的混合音频，如果为None则调用self.sample()
            label: np.ndarray [N, T] - 活动标签，如果为None则调用self.sample()
            spk_ids: list - 说话人ID列表，如果为None则调用self.sample()
            force_augment: bool - 是否强制应用增强（忽略noise_ratio）
        返回: (mix_new, label, spk_ids)
        """
        if mix is None or label is None or spk_ids is None:
            mix, label, spk_ids = self.sample()
        
        # 应用音频增强
        mix_augmented = self.apply_audio_augmentation(mix, force_augment)
        
        return mix_augmented, label, spk_ids
            
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

#    def collate_fn(self, batch, vad_out_len=200):
#        # batch: list of (mix, label, spk_ids, data_source)
#        max_len = max([len(x[0]) for x in batch])
#        # 先提取所有fbank帧数，确定max_frames
#        fbanks = []
#        fbank_lens = []
#        for mix, _, _, _ in batch:
#            fbank = self.extract_fbank(mix)
#            fbanks.append(fbank)
#            fbank_lens.append(fbank.shape[0])
#        max_frames = max(fbank_lens)
#        max_spks = max([x[1].shape[0] for x in batch])
#        wavs = []
#        labels = []
#        spk_ids_list = []
#        fbanks_pad = []
#        labels_len = []
#        data_sources = []
#        for (mix, label, spk_ids, data_source), fbank in zip(batch, fbanks):
#            pad_wav = np.pad(mix, (0, max_len - len(mix)), 'constant')
#            wavs.append(pad_wav)
#            pad_spk_ids = list(spk_ids) + [None] * (max_spks - len(spk_ids))
#            spk_ids_list.append(pad_spk_ids)
#            # fbank pad
#            pad_fbank = np.pad(fbank, ((0, max_frames - fbank.shape[0]), (0, 0)), 'constant')
#            fbanks_pad.append(pad_fbank)
#            # label对齐到fbank帧(这里的fbank 帧的帧移是40ms 也就是考虑了fbank 特征送入ResNetExtractor 后输出是下采样4倍的 )
#            N, T_sample = label.shape
#            T_fbank = fbank.shape[0]
#            aligned_label = np.zeros((N, T_fbank), dtype=np.float32)
#            win_length = int(self.frame_length * self.sr)
#            hop_length = int(self.frame_shift * self.sr)
#            for n in range(N):
#                for t in range(int(T_fbank/4)):
#                    start = t * hop_length
#                    end = min(start + win_length, T_sample)
#                    aligned_label[n, t] = label[n, start:end].max()
#            pad_label = np.zeros((max_spks, max_frames), dtype=np.float32)
#            pad_label[:aligned_label.shape[0], :aligned_label.shape[1]] = aligned_label
#            labels.append(pad_label)
#            data_sources.append(data_source)
#
#            labels_len.append(min(label.shape[1], vad_out_len) if label.ndim > 1 else 0)
#        import torch
#        wavs = torch.tensor(np.stack(wavs), dtype=torch.float32)
#        labels = torch.tensor(np.stack(labels), dtype=torch.float32)
#        fbanks = torch.tensor(np.stack(fbanks_pad), dtype=torch.float32)
#        labels_len = torch.tensor(labels_len, dtype=torch.int32)
#        data_sources = torch.tensor(data_sources, dtype=torch.int32)
#        
#        return wavs, labels, spk_ids_list, fbanks, labels_len, data_sources
   

    def collate_fn(self, batch, vad_out_len=200):
        # batch: list of (mix, label, spk_ids, data_source)
        max_len = max([len(x[0]) for x in batch])
        # 先提取所有fbank帧数，确定max_frames
        fbanks = []
        fbank_lens = []
        for mix, _, _, _ in batch:
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
        data_sources = []
        max_labels_frame= math.ceil(max_frames/self.downsample_factor)  # math.ceil(798/4)=200
        for (mix, label, spk_ids, data_source), fbank in zip(batch, fbanks):
            pad_wav = np.pad(mix, (0, max_len - len(mix)), 'constant')
            wavs.append(pad_wav)
            pad_spk_ids = list(spk_ids) + [None] * (max_spks - len(spk_ids))
            spk_ids_list.append(pad_spk_ids)
            # fbank pad
            pad_fbank = np.pad(fbank, ((0, max_frames - fbank.shape[0]), (0, 0)), 'constant')
            fbanks_pad.append(pad_fbank)
            # label对齐到fbank帧(这里的fbank 帧的帧移是40ms 也就是考虑了fbank 特征送入ResNetExtractor 后输出是下采样4倍的 )
            N, T_sample = label.shape
            #T_fbank = fbank.shape[0]
            #aligned_label = np.zeros((N, T_fbank), dtype=np.float32)
            win_length = int(self.frame_length * self.sr)
            hop_length = int(self.frame_shift * self.sr)
            # 计算总帧数 (与FBANK特征提取逻辑一致)
            num_frames = (T_sample - win_length) // hop_length + 1
            # 初始化帧级标签
            frame_label = np.zeros((N, num_frames), dtype=np.int32)
            for n in range(N):
                for t in range(int(num_frames)):
                    start = t * hop_length
                    end = min(start + win_length, T_sample)
                    frame_label[n, t] = 1 if np.any(label[n][start:end]) else 0
           
            # 标签下采样 (取每4帧的第1帧)
            downsampled_label = frame_label[:,::self.downsample_factor]
            pad_label = np.zeros((max_spks, max_labels_frame), dtype=np.float32)
            pad_label[:downsampled_label.shape[0], :downsampled_label.shape[1]] = downsampled_label
            labels.append(pad_label)
            data_sources.append(data_source)

            labels_len.append(min(downsampled_label.shape[1], vad_out_len) if label.ndim > 1 else 0)
        import torch
        wavs = torch.tensor(np.stack(wavs), dtype=torch.float32)
        labels = torch.tensor(np.stack(labels), dtype=torch.float32)
        
        fbanks = torch.tensor(np.stack(fbanks_pad), dtype=torch.float32)
        labels_len = torch.tensor(labels_len, dtype=torch.int32)
        data_sources = torch.tensor(data_sources, dtype=torch.int32)
         
        return wavs, labels, spk_ids_list, fbanks, labels_len, data_sources

    # 标签处理函数
    def process_labels(self,T_sample, labels, sample_rate, window_size, window_shift, downsample_factor):
        # 计算帧参数
        frame_length = int(self.frame_length * sample_rate)  # 400个采样点
        frame_shift = int(self.frame_shift * sample_rate)  # 160个采样点
        
        total_samples= T_sample
        # 计算总帧数 (与FBANK特征提取逻辑一致)
        num_frames = (total_samples - frame_length) // frame_shift + 1
        
        # 初始化帧级标签
        frame_labels = np.zeros(num_frames, dtype=np.int32)
        
        # 为每帧分配标签 (只要帧内有1就标记为1)
        for i in range(num_frames):
            start = i * frame_shift
            end = start + frame_length
            frame_labels[i] = 1 if np.any(labels[start:end]) else 0
        
        # 标签下采样 (取每4帧的第1帧)
        downsampled_labels = frame_labels[::downsample_factor]
        
        return frame_labels, downsampled_labels

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
        
    def sample_lazy(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        懒加载版本的采样函数，实时处理VAD文件和原始音频
        返回: (混合音频, 活动标签, spk_ids)
        """
        if not self.lazy_mode:
            raise ValueError("此函数仅在懒加载模式下可用")
        
        sr = self.sr
        total_len = int(self.max_mix_len * sr)
        
        # 随机选择说话人数量
        n_spk = np.random.randint(self.min_speakers, min(self.max_speakers, self._get_speaker_count()) + 1)
        
        # 随机选择说话人
        available_speakers = list(range(self._get_speaker_count()))
        spk_ids = list(np.random.choice(available_speakers, n_spk, replace=False))
        
        # 加载主说话人数据
        main_spk = spk_ids[0]
        main_chunks = self._load_speaker_data(main_spk)
        if not main_chunks:
            # 如果主说话人没有数据，随机选择另一个
            for spk in spk_ids[1:]:
                chunks = self._load_speaker_data(spk)
                if chunks:
                    main_spk = spk
                    main_spk_idx = spk_ids.index(spk)
                    spk_ids[0], spk_ids[main_spk_idx] = spk_ids[main_spk_idx], spk_ids[0]
                    main_chunks = chunks
                    break
        
        if not main_chunks:
            # 如果所有说话人都没有数据，返回静音
            logger.warning("所有选择的说话人都没有有效数据，返回静音")
            timeline = np.zeros(total_len, dtype=np.float32)
            label_mat = np.zeros((n_spk, total_len), dtype=np.float32)
            return timeline, label_mat, [str(spk) for spk in spk_ids]
        
        # 随机打乱主说话人的片段
        np.random.shuffle(main_chunks)
        
        # 初始化时间线和标签矩阵
        timeline = np.zeros(total_len, dtype=np.float32)
        label_mat = np.zeros((n_spk, total_len), dtype=np.float32)
        t = 0
        
        # 1. 主说话人顺序排布片段，静音区间严格切割
        for chunk in main_chunks:
            chunk_len = min(len(chunk), total_len - t)
            if chunk_len <= 0:
                break
            timeline[t:t+chunk_len] += chunk[:chunk_len]
            label_mat[0, t:t+chunk_len] = 1  # 主说话人在索引0
            t += chunk_len
            if t >= total_len:
                break
            remain = int(sr * np.random.uniform(self.min_silence, self.max_silence))
            t = min(t + remain, total_len)
        
        # 2. 其他说话人片段插入，优先重叠区，后静音区
        target_overlap_frames = int(self.target_overlap * total_len)
        
        for i, spk in enumerate(spk_ids[1:], 1):  # 从索引1开始，跳过主说话人
            chunks = self._load_speaker_data(spk)
            if not chunks:
                continue
                
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
        
        # 归一化音频
        if np.max(np.abs(timeline)) > 0:
            timeline = timeline / (np.max(np.abs(timeline)) + 1e-8)
        
        # 只保留有活动的说话人
        active_spk_mask = (label_mat.sum(axis=1) > 0)
        label_mat = label_mat[active_spk_mask]
        used_spk_ids = [str(spk) for spk, m in zip(spk_ids, active_spk_mask) if m]
        
        return timeline, label_mat, used_spk_ids
        
    def _get_speaker_count(self) -> int:
        """
        从VAD文件中获取说话人总数
        返回: 说话人数量
        """
        if not self.lazy_mode:
            raise ValueError("此函数仅在懒加载模式下可用")
        
        if self._speaker_list is None:
            self._speaker_list = self._load_speaker_list()
        
        return len(self._speaker_list)
    
    def _load_speaker_list(self) -> List[str]:
        """
        从VAD文件中加载说话人列表
        返回: 说话人ID列表
        """
        if not self.lazy_mode:
            raise ValueError("此函数仅在懒加载模式下可用")
        
        speaker_list = []
        
        try:
            if self.voxceleb2_spk2chunks_json.endswith('.gz') or self.voxceleb2_spk2chunks_json.endswith('.jsonl_gzip'):
                # 处理gzip压缩文件（包括.jsonl_gzip格式）
                import gzip
                with gzip.open(self.voxceleb2_spk2chunks_json, "rt", encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                data = json.loads(line)
                                spk_id = data.get("spk_id")
                                if spk_id:
                                    speaker_list.append(spk_id)
                            except json.JSONDecodeError:
                                continue
            else:
                # 处理普通文本文件
                with open(self.voxceleb2_spk2chunks_json, "r", encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                data = json.loads(line)
                                spk_id = data.get("spk_id")
                                if spk_id:
                                    speaker_list.append(spk_id)
                            except json.JSONDecodeError:
                                continue
            
            logger.info(f"从VAD文件加载了 {len(speaker_list)} 个说话人")
            return speaker_list
            
        except Exception as e:
            logger.error(f"加载说话人列表失败: {e}")
            return []
    
    def _load_speaker_data(self, spk_idx: int) -> List[np.ndarray]:
        """
        加载指定索引的说话人数据
        Args:
            spk_idx: 说话人在列表中的索引
        返回: 音频片段列表
        """
        if not self.lazy_mode:
            raise ValueError("此函数仅在懒加载模式下可用")
        
        if self._speaker_list is None:
            self._speaker_list = self._load_speaker_list()
        
        if spk_idx >= len(self._speaker_list):
            logger.warning(f"说话人索引 {spk_idx} 超出范围，最大索引: {len(self._speaker_list) - 1}")
            return []
        
        spk_id = self._speaker_list[spk_idx]
        
        # 检查缓存
        if spk_id in self._speaker_cache:
            return self._speaker_cache[spk_id]
        
        # 从VAD文件加载数据
        chunks = []
        try:
            if self.voxceleb2_spk2chunks_json.endswith('.gz') or self.voxceleb2_spk2chunks_json.endswith('.jsonl_gzip'):
                import gzip
                with gzip.open(self.voxceleb2_spk2chunks_json, "rt", encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                data = json.loads(line)
                                if data.get("spk_id") == spk_id:
                                    wav_paths = data.get("wav_paths", [])
                                    time_stamps = data.get("results", [])
                                    
                                    if len(wav_paths) == len(time_stamps):
                                        for wav_path, time_stamp_list in zip(wav_paths, time_stamps):
                                            try:
                                                if not os.path.exists(wav_path):
                                                    continue
                                                
                                                wav, sr = sf.read(wav_path)
                                                if sr != self.sr:
                                                    wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sr)
                                                
                                                # 提取VAD后的音频片段
                                                for start, end in time_stamp_list: # time_stamp_list is ms unit, start/1000 * 16000 = start *16
                                                    start_frame = int(start * 16)
                                                    end_frame = int(end * 16)
                                                    if start_frame < end_frame and start_frame < len(wav):
                                                        chunk = wav[start_frame:min(end_frame, len(wav))]
                                                        if len(chunk) > 0:
                                                            chunks.append(chunk)
                                                
                                                # 释放内存
                                                del wav
                                                
                                            except Exception as e:
                                                logger.warning(f"处理音频文件失败 {wav_path}: {e}")
                                                continue
                                    break  # 找到目标说话人后退出循环
                            except json.JSONDecodeError:
                                continue
            
            # 缓存结果
            self._speaker_cache[spk_id] = chunks
            logger.debug(f"加载说话人 {spk_id} 数据完成，共 {len(chunks)} 个音频片段")
            return chunks
            
        except Exception as e:
            logger.error(f"加载说话人 {spk_id} 数据失败: {e}")
            return []

    def clear_cache(self):
        """
        清理说话人数据缓存，释放内存
        """
        if hasattr(self, '_speaker_cache'):
            self._speaker_cache.clear()
        if hasattr(self, '_speaker_list'):
            self._speaker_list = None
        logger.info("已清理说话人数据缓存")
    
    def get_cache_info(self) -> Dict[str, int]:
        """
        获取缓存信息
        返回: 包含缓存状态的字典
        """
        cache_info = {
            'cached_speakers': len(self._speaker_cache) if hasattr(self, '_speaker_cache') else 0,
            'total_speakers': len(self._speaker_list) if hasattr(self, '_speaker_list') and self._speaker_list else 0,
            'lazy_mode': self.lazy_mode
        }
        return cache_info

# 用法示例：
# vad_func = ... # 你的VAD函数
# spk2wav = {'spk1': [...], ...}
# spk2chunks = {spk: [vad_func(wav, sr) for wav in wavs] for spk, wavs in spk2wav.items()}

# 1. 基本用法（无音频增强）
# mixer = SimuDiarMixer(spk2chunks, sample_rate=16000, max_mix_len=30.0, min_silence=0.0, max_silence=4.0, target_overlap=0.2)
# mix, label, spk_ids = mixer.sample()

# 2. 启用音频增强（加噪和混响）
# mixer = SimuDiarMixer(
#     spk2chunks, 
#     sample_rate=16000, 
#     max_mix_len=30.0, 
#     min_silence=0.0, 
#     max_silence=4.0, 
#     target_overlap=0.2,
#     musan_path="/path/to/musan",  # MUSAN数据集路径
#     rir_path="/path/to/rirs",     # RIR数据集路径
#     noise_ratio=0.8               # 80%的概率应用增强
# )

# 3. 检查增强资源状态
# info = mixer.get_augmentation_info()
# print(f"增强可用: {info['augmentation_available']}")
# if info['augmentation_available']:
#     print(f"MUSAN类别: {info.get('musan_categories', [])}")
#     print(f"RIR文件数: {info.get('rir_file_count', 0)}")

# 4. 强制应用增强（忽略noise_ratio）
# mix, label, spk_ids = mixer.sample_post(force_augment=True)

# 5. 在PyTorch DataLoader中使用
# from torch.utils.data import DataLoader
# dataset = SimuDiarMixer(...)
# dataloader = DataLoader(dataset, batch_size=8, collate_fn=dataset.collate_fn)
# for batch in dataloader:
#     wavs, labels, spk_ids_list, fbanks, labels_len, data_sources = batch
#     # 处理批次数据...

# 6. 懒加载模式（从VAD文件加载）
# mixer = SimuDiarMixer(
#     voxceleb2_spk2chunks_json="/path/to/vad_data.json",
#     musan_path="/path/to/musan",
#     rir_path="/path/to/rirs",
#     noise_ratio=0.8
# ) 
