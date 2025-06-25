import os
import numpy as np
import soundfile as sf
import textgrid
from torch.utils.data import Dataset
import torch
import glob
import random

class AlimeetingDiarDataset(Dataset):
    def __init__(self, wav_dir, textgrid_dir, sample_rate=16000, frame_length=0.025, frame_shift=0.04,
                 musan_path=None, rir_path=None, noise_ratio=0.8, window_sec=8, window_shift_sec=0.4):
        self.wav_dir = wav_dir
        self.textgrid_dir = textgrid_dir
        self.sample_rate = sample_rate
        self.frame_shift = frame_shift # 0.04s means 25 frames in 1s, means 1s has 25 labels, means downsample rate is 4. 
        self.frame_length = frame_length
        self.musan_path = musan_path
        self.rir_path = rir_path
        self.noise_ratio = noise_ratio
        self.window_sec = window_sec
        self.window_shift_sec = window_shift_sec
        self.data, self.windows = self._collect_data_and_windows()

    def _collect_data_and_windows(self):
        data = []
        windows = []
        print(f"Scanning {self.textgrid_dir} for TextGrid files...")
        for tg_file in os.listdir(self.textgrid_dir):
            if not tg_file.endswith('.TextGrid'):
                continue
            uttid = tg_file.replace('.TextGrid', '')
            wav_path = glob.glob(os.path.join(self.wav_dir, uttid + '*.wav'))[0]
            tg_path = os.path.join(self.textgrid_dir, tg_file)
            if os.path.exists(wav_path):
                data.append({'uttid': uttid, 'wav': wav_path, 'tg': tg_path})
                wav_len = sf.info(wav_path).frames
                win_size = int(self.window_sec * self.sample_rate)
                win_shift = int(self.window_shift_sec * self.sample_rate)
                for start in range(0, wav_len, win_shift):
                    end = min(start + win_size, wav_len)
                    if end - start < win_size // 2:
                        break
                    windows.append((len(data)-1, start, end))
            else:
                print(f"Warning: {wav_path} not found for {tg_file}")
        print(f"Collected {len(data)} samples.")
        return data, windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        import soundfile as sf
        import textgrid
        import numpy as np
        data_idx, start, end = self.windows[idx]
        item = self.data[data_idx]
        wav, sr = sf.read(item['wav'], start=start, stop=end)
        if wav.ndim > 1:
            wav = wav[:, 0]
        tg = textgrid.TextGrid.fromFile(item['tg'])
        spk2intervals = {}
        for tier in tg:
            spk = tier.name
            intervals = []
            for interval in tier:
                if interval.mark.strip():
                    intervals.append([interval.minTime, interval.maxTime])
            if intervals:
                spk2intervals[spk] = intervals
        win_start_sec = start / self.sample_rate
        win_end_sec = end / self.sample_rate
        num_frames = int(len(wav) / self.sample_rate / self.frame_shift)
        label = np.zeros((len(spk2intervals), num_frames), dtype=np.float32)
        spk_ids = list(spk2intervals.keys())
        for i, spk in enumerate(spk_ids):
            for s, e in spk2intervals[spk]:
                s_clip = max(s, win_start_sec) - win_start_sec
                e_clip = min(e, win_end_sec) - win_start_sec
                if e_clip <= 0 or s_clip >= (win_end_sec - win_start_sec):
                    continue
                start_frame = int(s_clip / self.frame_shift)
                end_frame = int(e_clip / self.frame_shift)
                label[i, start_frame:end_frame] = 1
        # 只保留窗口内有活动的说话人
        active_mask = label.sum(axis=1) > 0
        label = label[active_mask]
        spk_ids = [spk for spk, active in zip(spk_ids, active_mask) if active]
        if self.musan_path or self.rir_path:
            wav, label, spk_ids = self.sample_post_window(wav, label, spk_ids)
        return wav, label, spk_ids

    def sample_post_window(self, wav, label, spk_ids):
        import numpy as np
        noisesnr, numnoise, noiselist, rir_files, noisetypes = self.load_musan_or_rirs(self.musan_path, self.rir_path)
        if self.rir_path is not None or self.musan_path is not None:
            add_noise = np.random.choice(2, p=[1 - self.noise_ratio, self.noise_ratio])
            if add_noise == 1:
                import random
                if self.rir_path is not None and self.musan_path is not None:
                    wav_aug = np.expand_dims(np.array(wav), axis=0)
                    ntypes = random.choice(noisetypes)
                    wav_aug = self.add_noise(wav_aug, noiselist, noisesnr, ntypes, numnoise)
                    wav_aug = self.add_reverb(wav_aug, rir_files)
                elif self.rir_path is not None:
                    wav_aug = np.expand_dims(np.array(wav), axis=0)
                    wav_aug = self.add_reverb(wav_aug, rir_files)
                elif self.musan_path is not None:
                    wav_aug = np.expand_dims(np.array(wav), axis=0)
                    ntypes = random.choice(noisetypes)
                    wav_aug = self.add_noise(wav_aug, noiselist, noisesnr, ntypes, numnoise)
                wav = np.squeeze(wav_aug, axis=0)
        return wav, label, spk_ids

    def collate_fn(self, batch, vad_out_len=200):
        if not isinstance(batch, (list, tuple)) or not hasattr(batch[0], '__getitem__'):
            raise ValueError("collate_fn expects a list of samples, not a Dataset object! Please pass [dataset[i] for i in ...]")
        # batch: list of (wav, label, spk_ids)
        batch = [(x[0], np.array(x[1]) if not isinstance(x[1], np.ndarray) else x[1], x[2]) for x in batch]
        fbanks_unpadded = [self.extract_fbank(wav) for wav, _, _ in batch]
        max_fbank_frames = max(f.shape[0] for f in fbanks_unpadded) if fbanks_unpadded else 0
        max_spks = max([x[1].shape[0] for x in batch if x[1].ndim > 1], default=0)
        max_wav_len = max([len(x[0]) for x in batch])
        wavs, labels, spk_ids_list, fbanks, labels_len = [], [], [], [], []
        for i, (wav, label, spk_ids) in enumerate(batch):
            pad_wav = np.pad(wav, (0, max_wav_len - len(wav)), 'constant')
            wavs.append(pad_wav)
            # Pad/crop label to vad_out_len
            pad_label = np.zeros((max_spks, vad_out_len), dtype=np.float32)
            if label.shape[0] > 0:
                length = min(label.shape[1], vad_out_len)
                pad_label[:label.shape[0], :length] = label[:, :length]
            labels.append(pad_label)
            fbank = fbanks_unpadded[i]
            pad_fbank = np.pad(fbank, ((0, max_fbank_frames - fbank.shape[0]), (0, 0)), 'constant')
            fbanks.append(pad_fbank)
            pad_spk_ids = spk_ids + [None] * (max_spks - len(spk_ids))
            spk_ids_list.append(pad_spk_ids)
            labels_len.append(min(label.shape[1], vad_out_len) if label.ndim > 1 else 0)
        wavs = torch.tensor(np.stack(wavs), dtype=torch.float32)
        labels = torch.tensor(np.stack(labels), dtype=torch.float32)
        fbanks = torch.tensor(np.stack(fbanks), dtype=torch.float32)
        labels_len = torch.tensor(labels_len, dtype=torch.int32)
        return wavs, labels, spk_ids_list, fbanks, labels_len

    @staticmethod
    def compute_overlap(label):
        # label: [num_spk, num_frames]
        overlap_frames = (label.sum(axis=0) > 1).sum()
        total_frames = label.shape[1]
        overlap_ratio = overlap_frames / total_frames
        return overlap_ratio

    def extract_fbank(self, wav, sample_rate=16000):
        """
        提取 kaldi 风格的 fbank 特征。
        wav: numpy array, 1D
        sample_rate: 采样率
        返回: [num_frames, 80] 的 torch.Tensor
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
            num_mel_bins=80,
            frame_length=25,
            frame_shift=10,
            sample_frequency=sample_rate,
            use_log_fbank=True,
            dither=1.0,
            window_type='hamming'
        ).squeeze(0)  # [num_frames, num_mel_bins]
        return fbank

    def load_musan_or_rirs(self, musan_path, rir_path):
        import glob
        if musan_path is not None:
            noiselist = {}
            noisetypes = ["noise", "speech", "music"]
            noisesnr = {"noise": [0, 15], "speech": [10, 30], "music": [5, 15]}
            numnoise = {"noise": [1, 1], "speech": [3, 8], "music": [1, 1]}
            augment_files = glob.glob(
                os.path.join(musan_path, "*/*/*.wav")
            )
            for file in augment_files:
                if file.split("/")[-3] not in noiselist:
                    noiselist[file.split("/")[-3]] = []
                noiselist[file.split("/")[-3]].append(file)
        else:
            noisesnr = None
            numnoise = None
            noisetypes = None
            noiselist = None
        if rir_path is not None:
            rir_files = glob.glob(
                os.path.join(rir_path, "*/*.wav")
            )
        else:
            rir_files = None
        return noisesnr, numnoise, noiselist, rir_files, noisetypes

    def add_reverb(self, audio, rir_files):
        import soundfile as sf
        import numpy as np
        from scipy import signal
        import random
        rir_file = random.choice(rir_files)
        rir, _ = sf.read(rir_file)
        if len(rir.shape) > 1:
            rir = rir[:, 0]
        rir = np.expand_dims(rir.astype(float), 0)
        rir = rir / np.sqrt(np.sum(rir ** 2))
        return signal.convolve(audio, rir, mode="full")[:, :audio.shape[1]]

    def add_noise(self, audio, noiselist, noisesnr, noisetype, numnoise):
        import soundfile as sf
        import numpy as np
        import librosa
        import random
        clean_db = 10 * np.log10(1e-4 + np.mean(audio ** 2))
        noiselist_cat = random.sample(noiselist[noisetype], random.randint(numnoise[noisetype][0], numnoise[noisetype][1]))
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
                noiseaudio = noiseaudio[start_frame:start_frame + length]
            noiseaudio = np.expand_dims(noiseaudio, 0)
            noise_db = 10 * np.log10(1e-4 + np.mean(noiseaudio ** 2))
            snr = random.uniform(noisesnr[noisetype][0], noisesnr[noisetype][1])
            noises.append(np.sqrt(10 ** ((clean_db - noise_db - snr) / 10)) * noiseaudio)
        noise = np.sum(np.concatenate(noises, axis=0), axis=0, keepdims=True)
        return noise[:, :length] + audio

def test_sample(wav_dir,textgrid_dir):
    dataset = AlimeetingDiarDataset(wav_dir, textgrid_dir)
    wav, label, spk_ids = dataset[0]
    print(f"wav: {wav}, its shape: {wav.shape}, label shape: {label.shape}, spk_ids: {spk_ids}")
    batch = [dataset[i] for i in range(2)]
    mix_pad, label_pad, spk_ids_list,fbanks, labels_len = dataset.collate_fn(batch)
    print(f"mix_pad: {mix_pad.shape}, label_pad shape: {label_pad.shape}, spk_ids_list: {spk_ids_list}, fbanks shape: {fbanks.shape}, labels_len: {labels_len}")

#def test_sample2(wav_dir, textgrid_dir):
# dataset = AlimeetingDiarDataset(wav_dir, textgrid_dir)
#    wav, label, spk_ids = dataset.sample_post_window(wav, label, spk_ids)
#   print(f"wav: {wav}, its shape: {wav.shape}, label shape: {label.shape}, spk_ids: {spk_ids}")

def test_dataloader_augmentation(wav_dir, textgrid_dir, musan_path=None, rir_path=None, batch_size=2):
    from torch.utils.data import DataLoader
    import numpy as np

    # 分别构造不增强和增强的dataset
    dataset_noaug = AlimeetingDiarDataset(wav_dir, textgrid_dir)
    dataset_aug = AlimeetingDiarDataset(wav_dir, textgrid_dir, musan_path=musan_path, rir_path=rir_path)

    loader_noaug = DataLoader(dataset_noaug, batch_size=batch_size, collate_fn=dataset_noaug.collate_fn)
    loader_aug = DataLoader(dataset_aug, batch_size=batch_size, collate_fn=dataset_aug.collate_fn)

    # 取一批数据
    batch_noaug = next(iter(loader_noaug))
    batch_aug = next(iter(loader_aug))

    mix_pad_noaug = batch_noaug[0].numpy()
    mix_pad_aug = batch_aug[0].numpy()

    # 对比同一条音频增强前后的差异
    print("NoAug batch mix_pad[0][:10]:", mix_pad_noaug[0][:10])
    print("Aug batch mix_pad[0][:10]:", mix_pad_aug[0][:10])
    print("Difference (L2 norm):", np.linalg.norm(mix_pad_noaug[0] - mix_pad_aug[0]))

    # 判断是否有明显差异
    if not np.allclose(mix_pad_noaug[0], mix_pad_aug[0]):
        print("增强生效：增强后的音频与原音频不同。")
    else:
        print("增强未生效：增强后的音频与原音频相同。")

if __name__ == "__main__":
    wav_dir="/data/maduo/datasets/alimeeting/Eval_Ali/Eval_Ali_far/audio_dir"
    textgrid_dir="/data/maduo/datasets/alimeeting/Eval_Ali/Eval_Ali_far/textgrid_dir"
    test_sample(wav_dir, textgrid_dir)
    #test_sample2(wav_dir, textgrid_dir)
    musan_path = "/data/maduo/datasets/musan"
    rir_path = "/data/maduo/datasets/RIRS_NOISES"
    musan_path=""
    rir_path = ""
    test_dataloader_augmentation(wav_dir, textgrid_dir, musan_path, rir_path)
