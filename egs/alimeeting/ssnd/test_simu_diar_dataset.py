import numpy as np
from simu_diar_dataset import SimuDiarMixer
from funasr import AutoModel # pip install funasr
import soundfile as sf
import random
import torchaudio
def fix_random_seed2(seed: int):
    #torch.manual_seed(seed)
    #if torch.cuda.is_available():
    #    torch.cuda.manual_seed(seed)
    #    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups                                                                                                        
    random.seed(seed)
    np.random.seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False

def test():
    fsmn_vad_model = AutoModel(model="fsmn-vad", model_revision="v2.0.4")

    def vad_func(wav, sr):
        if wav.dtype != np.int16:
            wav = (wav * 32767).astype(np.int16)
        result = fsmn_vad_model.generate(wav, fs=sr)
        print(result)# [{'key': 'rand_key_9GFz6Y1t9EwL5', 'value': [[0, 4530], [4810, 7970]]}]
        time_stamp = result[0]['value']
        print("time_stamp:",time_stamp)
        return time_stamp

    # 构造假数据和VAD函数
    #def fake_vad_func(wav, sr):
        # 假设全是语音段，返回[(0, len(wav)/sr)]
    #    return [(0, len(wav)/sr)]

    #def make_fake_wav(length_sec, sr):
        # 生成正弦波，模拟音频
    #    t = np.linspace(0, length_sec, int(length_sec*sr), endpoint=False)
    #    return 0.1 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    # 假设有3个说话人，每人2段音频
    sr = 16000
    #spk2wav = {
    #    'spk1': ['spk1_1.wav', 'spk1_2.wav'],
    #    'spk2': ['spk2_1.wav', 'spk2_2.wav'],
    #    'spk3': ['spk3_1.wav', 'spk3_2.wav'],
    #}
    spk2wav={"spk1":["/data/maduo/datasets/test_wavs/3-sichuan.wav"],
            'spk2':["/data/maduo/datasets/test_wavs/5-henan.wav"],
            'spk3':["/data/maduo/datasets/test_wavs/zh.wav"],
            }





    # 生成假音频文件

    #for spk, wavs in spk2wav.items():
    #    for fname in wavs:
    #        wav = make_fake_wav(5, sr)
    #        sf.write(fname, wav, sr)

    # 创建数据集
    simu_dataset = SimuDiarDataset(
        spk2wav=spk2wav,
        #vad_func=fake_vad_func,
        vad_func=vad_func,
        min_seg=1.0,
        max_seg=3.0,
        max_silence_seg=4.0,
        sample_rate=sr,
        max_speakers=3,
        min_speakers=1,
        max_mix_len=30.0,
        overlap_ratio=0.2,
        allow_repeat=False,
    )

    # 单样本测试
    mix, label,spk_ids= simu_dataset.sample()
    sf.write("mix.wav",mix,sr)
    print("mix:",mix,"label:",label,"spk_ids: ", spk_ids)
    print('混合音频 shape:', mix.shape)
    print('标签 shape:', label.shape)
    print('活动帧比例:', label.sum() / np.prod(label.shape))
    print('每个说话人活动帧:', label.sum(axis=1))
    # 假设 mix, label, spk_ids = simu_dataset.sample()
    # 统计全局重叠率
    overlap_frames = (label.sum(axis=0) > 1).sum()
    total_frames = label.shape[1]
    overlap_ratio_actual = overlap_frames / total_frames
    print(f'实际全局重叠率: {overlap_ratio_actual:.3f}')

    # 统计静音片段分布
    active = (label.sum(axis=0) > 0).astype(int)
    silence_lengths = []
    cur = 0
    for v in active:
        if v == 0:
            cur += 1
        elif cur > 0:
            silence_lengths.append(cur)
            cur = 0
    if cur > 0:
        silence_lengths.append(cur)
    if silence_lengths:
        silence_lengths_sec = np.array(silence_lengths) / sr
        print(f'静音片段长度分布（秒）: min={silence_lengths_sec.min():.2f}, max={silence_lengths_sec.max():.2f}, mean={silence_lengths_sec.mean():.2f}')
    else:
        print('无静音片段')

    import matplotlib.pyplot as plt

    def plot_mix_and_labels(mix, label, sr):
        t = np.arange(len(mix)) / sr
        plt.figure(figsize=(12, 6))
        plt.plot(t, mix, label='Mix')
        for i, l in enumerate(label):
            plt.plot(t, l * 0.2 + 0.2 * i, label=f'Speaker {i+1} label')  # label上移避免重叠
        plt.legend()
        plt.xlabel('Time (s)')
        plt.title('Mix waveform and speaker activity labels')
        plt.savefig("./test_image.jpg") # save a figure and use image previewer extention to see.
        plt.show()

    # 用法
    plot_mix_and_labels(mix, label, sr)


    # 批量测试
    batch = [simu_dataset.sample() for _ in range(4)]
    print("batch:",batch)
    mix_pad, label_pad, spk_ids_list = SimuDiarDataset.batch_collate_fn(batch)
    print('batch mix_pad shape:', mix_pad.shape)
    print('batch label_pad shape:', label_pad.shape)
    print('batch spk_ids_list:', spk_ids_list)



    # 清理假音频文件
    #import os
    #for wavs in spk2wav.values():
    #    for fname in wavs:
    #        os.remove(fname) 


def test2():
    fsmn_vad_model = AutoModel(model="fsmn-vad", model_revision="v2.0.4")

    def vad_func(wav, sr):
        if wav.dtype != np.int16:
            wav = (wav * 32767).astype(np.int16)
        result = fsmn_vad_model.generate(wav, fs=sr)
        print(result)# [{'key': 'rand_key_9GFz6Y1t9EwL5', 'value': [[0, 4530], [4810, 7970]]}] # 
        time_stamp = result[0]['value']
        print("time_stamp:",time_stamp) 
        return time_stamp # in ms
    def spktochunks():
        from collections import defaultdict
        import librosa
        spk2wav={"spk1":["/data/maduo/datasets/test_wavs/3-sichuan.wav"],
            'spk2':["/data/maduo/datasets/test_wavs/5-henan.wav"],
            'spk3':["/data/maduo/datasets/test_wavs/zh.wav"],
            'spk4':["/data/maduo/datasets/test_wavs/yue.wav"],
            }
        spk2chunks=defaultdict(list)
        for spk_id in spk2wav.keys():
            for wav_path in spk2wav[spk_id]:
                wav, sr = sf.read(wav_path)
                if sr != 16000:
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
                time_stamp_list = vad_func(wav,sr=16000)
                # in ms ->(/1000) in second ->(*16000) in sample points
                speech_chunks = [wav[int(s*16):int(e*16)] for s, e in time_stamp_list]
                if spk_id in spk2chunks:
                    spk2chunks[spk_id].append(speech_chunks)
                else:
                    spk2chunks[spk_id] = speech_chunks
        return spk2chunks

            
    spk2chunks= spktochunks()
    print("spk2chunks: ", spk2chunks) 
    # 假设你已经有 spk2chunks = {spk: [vad后片段, ...], ...}
    import time
    start = time.time()
    mixer = SimuDiarMixer(
        spk2chunks,
        sample_rate=16000,
        max_mix_len=15.0,
        min_silence=0.0,
        max_silence=4.0,
        target_overlap=0.2
    )
    print(f"finish run the SimuDiarMixer time consume: {time.time()-start}second")
    mix, label, spk_ids = mixer.sample()
    sf.write("mix.wav",mix,16000)
    print("mix:",mix,"label:",label,"spk_ids: ", spk_ids)
    print('混合音频 shape:', mix.shape)
    print('标签 shape:', label.shape)
    print('活动帧比例:', label.sum() / np.prod(label.shape))
    print('每个说话人活动帧:', label.sum(axis=1))

    overlap_frames = (label.sum(axis=0) > 1).sum()
    total_frames = label.shape[1]
    overlap_ratio_actual = overlap_frames / total_frames
    print(f'实际全局重叠率: {overlap_ratio_actual:.3f}')

    # 统计静音片段分布
    active = (label.sum(axis=0) > 0).astype(int)
    silence_lengths = []
    cur = 0
    for v in active:
        if v == 0:
            cur += 1
        elif cur > 0:
            silence_lengths.append(cur)
            cur = 0
    if cur > 0:
        silence_lengths.append(cur)
    if silence_lengths:
        silence_lengths_sec = np.array(silence_lengths) / 16000
        print(f'静音片段长度分布（秒）: min={silence_lengths_sec.min():.2f}, max={silence_lengths_sec.max():.2f}, mean={silence_lengths_sec.mean():.2f}')
    else:
        print('无静音片段')

    import matplotlib.pyplot as plt

    def plot_mix_and_labels(mix, label, spk_ids,sr):
        t = np.arange(len(mix)) / sr
        plt.figure(figsize=(12, 6))
        plt.plot(t, mix, label='Mix')
        for i, l in enumerate(label):
            plt.plot(t, l * 0.2 + 0.2 * i, label=f'Speaker {spk_ids[i]} label')  # label上移避免重叠
        plt.legend()
        plt.xlabel('Time (s)')
        plt.title('Mix waveform and speaker activity labels')
        plt.savefig("./test_image.jpg")
        plt.show()

    # 用法
    plot_mix_and_labels(mix, label,spk_ids, sr=16000)
    # 批量测试
    batch = [mixer.sample() for _ in range(4)]
    print("batch:",batch)
    mix_pad, label_pad, spk_ids_list = SimuDiarMixer.collate_fn(batch)
    print('batch mix_pad shape:', mix_pad.shape)
    print('batch label_pad shape:', label_pad.shape)
    print('batch spk_ids_list:', spk_ids_list)

def plot_wav_and_check(mix, label, spk_ids,name="clean"):
    sf.write(f"mix_{name}.wav",mix,16000)
    print("mix:",mix,"label:",label,"spk_ids: ", spk_ids)
    print(f'{name}混合音频 shape:{mix.shape}')
    print(f'{name}标签 shape: {label.shape}')
    print(f'{name}活动帧比例: {label.sum() / np.prod(label.shape)}')
    print(f'{name}每个说话人活动帧: {label.sum(axis=1)}')

    overlap_frames = (label.sum(axis=0) > 1).sum()
    total_frames = label.shape[1]
    overlap_ratio_actual = overlap_frames / total_frames
    print(f'{name}实际全局重叠率: {overlap_ratio_actual:.3f}')

    # 统计静音片段分布
    active = (label.sum(axis=0) > 0).astype(int)
    silence_lengths = []
    cur = 0
    for v in active:
        if v == 0:
            cur += 1
        elif cur > 0:
            silence_lengths.append(cur)
            cur = 0
    if cur > 0:
        silence_lengths.append(cur)
    if silence_lengths:
        silence_lengths_sec = np.array(silence_lengths) / 16000
        print(f'{name}静音片段长度分布（秒）: min={silence_lengths_sec.min():.2f}, max={silence_lengths_sec.max():.2f}, mean={silence_lengths_sec.mean():.2f}')
    else:
        print('无静音片段')

    import matplotlib.pyplot as plt

    def plot_mix_and_labels(mix, label, spk_ids,sr):
        t = np.arange(len(mix)) / sr
        plt.figure(figsize=(12, 6))
        plt.plot(t, mix, label='Mix')
        for i, l in enumerate(label):
            plt.plot(t, l * 0.2 + 0.2 * i, label=f'Speaker {spk_ids[i]} label')  # label上移避免重叠
        plt.legend()
        plt.xlabel('Time (s)')
        plt.title(f'{name} Mix waveform and speaker activity labels')
        plt.savefig(f"./{name}_test_image.jpg")
        plt.show()

    # 用法
    plot_mix_and_labels(mix, label,spk_ids, sr=16000)
    
def test3():
    from simu_diar_dataset import SimuDiarMixer
    import torch
    from ssnd_model import SSNDModel
    import soundfile as sf
    from collections import defaultdict
    import librosa
    import torchaudio
    # 1. 构造真实 spk2chunks
    def vad_func(wav, sr):
        from funasr import AutoModel
        fsmn_vad_model = AutoModel(model="fsmn-vad", model_revision="v2.0.4",disable_update=True)
        if wav.dtype != np.int16:
            wav = (wav * 32767).astype(np.int16)
        result = fsmn_vad_model.generate(wav, fs=sr)
        time_stamp = result[0]['value']
        return time_stamp
    spk2wav={
        "spk1":["/data/maduo/datasets/test_wavs/3-sichuan.wav"],
        'spk2':["/data/maduo/datasets/test_wavs/5-henan.wav"],
        'spk3':["/data/maduo/datasets/test_wavs/zh.wav"],
        'spk4':["/data/maduo/datasets/test_wavs/yue.wav"],
    }
    spk2chunks=defaultdict(list)
    for spk_id in spk2wav.keys():
        for wav_path in spk2wav[spk_id]:
            wav, sr = sf.read(wav_path)
            if sr != 16000:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            time_stamp_list = vad_func(wav,sr=16000)
            speech_chunks = [wav[int(s*16):int(e*16)] for s, e in time_stamp_list]
            spk2chunks[spk_id].extend(speech_chunks)
    # 2. 采样batch
    
    mixer = SimuDiarMixer(
        spk2chunks,
        sample_rate=16000,
        max_mix_len=15.0,
        min_silence=0.0,
        max_silence=4.0,
        target_overlap=0.2,
        #musan_path="/data/maduo/datasets/musan",
        #rir_path="/data/maduo/datasets/RIRS_NOISES",
    )
    
   
    # plot and check
    mix, label,spk_ids = mixer.sample_post()
    plot_wav_and_check(mix,label,spk_ids, name="clean")

    #batch = [mixer.sample() for _ in range(4)]
    batch = [mixer.sample_post() for _ in range(4)]
    mix_pad, label_pad, spk_ids_list = SimuDiarMixer.collate_fn(batch)
    # 3. 构造spk_label_idx
    all_spk = sorted({spk for spk_ids in spk_ids_list for spk in spk_ids if spk is not None})
    spk2idx = {spk: i for i, spk in enumerate(all_spk)}
    B, N, T = label_pad.shape
    spk_label_idx = torch.full((B, N), fill_value=-1, dtype=torch.long)
    for b, spk_ids in enumerate(spk_ids_list):
        for n, spk in enumerate(spk_ids):
            if spk is not None:
                spk_label_idx[b, n] = spk2idx[spk]
    spk_labels = spk_label_idx.clone()
    # 4. 特征提取：kaldi风格fbank+label帧对齐
    feats_list = []
    label_frame_list = []
    win_length = int(0.025 * 16000)
    hop_length = int(0.01 * 16000)
    for i in range(B):
        wav_tensor = torch.tensor(mix_pad[i].cpu().numpy(), dtype=torch.float32)
        fbank = torchaudio.compliance.kaldi.fbank(
            wav_tensor.unsqueeze(0),
            num_mel_bins=80,
            frame_length=25,
            frame_shift=10,
            sample_frequency=16000,
            use_log_fbank=True,
            dither=1.0,
            window_type='hamming'
        ).squeeze(0)  # [num_frames, 80]
        feats_list.append(fbank)
        # label对齐
        label_tensor = torch.tensor(label_pad[i], dtype=torch.float32)  # [N, T]
        num_frames = fbank.shape[0]
        frame_labels = []
        for n in range(N):
            l = label_tensor[n]
            frames = []
            for j in range(num_frames):
                start = j * hop_length
                end = min(start + win_length, l.shape[0])
                frames.append(l[start:end].max())
            frame_labels.append(torch.stack(frames))
        label_frame = torch.stack(frame_labels)  # [N, num_frames]
        label_frame_list.append(label_frame)
    # pad feats和label到最大帧
    max_frames = max([f.shape[0] for f in feats_list])
    feats_pad = torch.zeros(B, max_frames, 80)
    label_pad_frame = torch.zeros(B, N, max_frames)
    for i, (fbank, label_frame) in enumerate(zip(feats_list, label_frame_list)):
        feats_pad[i, :fbank.shape[0], :] = fbank
        label_pad_frame[i, :, :label_frame.shape[1]] = label_frame
    # 5. 实例化模型
    model = SSNDModel(
        feat_dim=80,
        emb_dim=32,
        d_model=32,
        nhead=4,
        d_ff=64,
        num_layers=2,
        max_speakers=N,
        vad_out_len=max_frames,
        arcface_num_classes=len(all_spk),
        pos_emb_dim=32,
        max_seq_len=max_frames,
        n_all_speakers=len(all_spk),
        mask_prob=0.5,
        training=True,
    )
    # 6. 前向传播
    vad_labels = label_pad_frame  # [B, N, max_frames]
    feats = feats_pad  # [B, max_frames, 80]
    vad_pred, spk_emb_pred, bce_loss, arcface_loss, mask_info = model(
        feats, spk_label_idx, vad_labels, spk_labels
    )
    print('vad_pred shape:', vad_pred.shape)
    print('spk_emb_pred shape:', spk_emb_pred.shape)
    print('bce_loss:', bce_loss.item())
    print('arcface_loss:', arcface_loss.item() if arcface_loss is not None else None)
    print('mask_info:', mask_info)

if __name__== "__main__":
    fix_random_seed2(43)
    #test() 
    #test2()
    test3()
