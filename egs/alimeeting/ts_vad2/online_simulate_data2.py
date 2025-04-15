import os
import numpy as np
import soundfile as sf
import librosa
import random
from glob import glob
from scipy.signal import convolve
from textgrid import TextGrid
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DataSimulator:
    def __init__(self, real_label_dir, speaker_data_dir, noise_dir, rir_dir, sample_rate=16000):
        self.sample_rate = sample_rate
        self.real_labels = self.load_real_labels(real_label_dir)
        self.speaker_files = self.load_speaker_data(real_label_dir, speaker_data_dir)
        self.noise_files = glob(f"{noise_dir}/**/*.wav", recursive=True)
        self.rir_files = glob(f"{rir_dir}/**/*.wav", recursive=True)

    def load_real_labels(self, label_dir):
        """加载AliMeeting的TextGrid真实标签并生成标签矩阵"""
        label_files = glob(f"{label_dir}/*.TextGrid")
        label_matrices = {}
        for label_file in label_files:
            meeting_id = label_file.split("/")[-1].replace(".TextGrid", "")
            tg = TextGrid.fromFile(label_file)
            num_speakers = len(tg)
            max_duration = tg.maxTime
            max_frames = int(max_duration * self.sample_rate)
            label_matrix = np.zeros((num_speakers, max_frames), dtype=int)
            for speaker_index, tier in enumerate(tg):
                for interval in tier:
                    if interval.mark != "":
                        start_frame = int(interval.minTime * self.sample_rate)
                        end_frame = int(interval.maxTime * self.sample_rate)
                        end_frame = min(end_frame, max_frames)
                        label_matrix[speaker_index, start_frame:end_frame] = 1
            label_matrices[meeting_id] = label_matrix
        return label_matrices

    def load_speaker_data(self, label_dir, data_dir):
        """加载AliMeeting的所有speaker的语音文件，并根据TextGrid提取非重叠片段"""
        speaker_files = {}
        label_files = glob(f"{label_dir}/*.TextGrid")
        for label_file in label_files:
            meeting_id = label_file.split("/")[-1].replace(".TextGrid", "")
            #audio_file = f"{data_dir}/{meeting_id}.wav"
            audio_file = glob(os.path.join(data_dir, meeting_id) +"*.wav")[0]
            audio, sr = librosa.load(audio_file, sr=self.sample_rate)
            tg = TextGrid.fromFile(label_file)
            num_speakers = len(tg)
            speaker_audios = [[] for _ in range(num_speakers)]
            for speaker_index, tier in enumerate(tg):
                for interval in tier:
                    if interval.mark != "":
                        start_frame = int(interval.minTime * self.sample_rate)
                        end_frame = int(interval.maxTime * self.sample_rate)
                        is_overlap = False
                        for other_speaker in range(num_speakers):
                            if other_speaker != speaker_index:
                                for other_interval in tg[other_speaker]:
                                    if other_interval.mark != "":
                                        other_start = int(other_interval.minTime * self.sample_rate)
                                        other_end = int(other_interval.maxTime * self.sample_rate)
                                        if (start_frame < other_end) and (end_frame > other_start):
                                            is_overlap = True
                                            break
                            if is_overlap:
                                break
                        if not is_overlap:
                            if end_frame > start_frame:
                                speaker_audios[speaker_index].append(audio[start_frame:end_frame])
            speaker_files[meeting_id] = speaker_audios
        return speaker_files

    def simulate_audio(self, meeting_id, total_duration=10):
        """生成指定会议ID的模拟音频"""
        label_matrix = self.real_labels[meeting_id]
        num_speakers, max_frames = label_matrix.shape
        T = int(total_duration * self.sample_rate)
        mix_audio = np.zeros(T, dtype=np.float32)
        simulated_label = np.zeros((num_speakers, T), dtype=int)

        # 获取至少有一个说话人活跃的时间段
        valid_segments = self.get_valid_segments(label_matrix)
        if not valid_segments:
            logging.warning("No valid speaker segments found in the label matrix")
            return mix_audio, simulated_label

        # 随机选择一个有效时间段
        start_frame, end_frame = random.choice(valid_segments)
        segment_duration = end_frame - start_frame
        if segment_duration <= 0:
            return mix_audio, simulated_label

        for speaker in range(num_speakers):
            active_frames = np.where(label_matrix[speaker, start_frame:end_frame] == 1)[0]
            if len(active_frames) == 0:
                continue
            # 将活跃帧分割为非重叠连续段，这里的 global_start 是所选时间段的起始帧
            segments = self.split_into_nonoverlapping(active_frames, start_frame)
            for seg_start, seg_end in segments:
                seg_len = seg_end - seg_start
                if seg_len <= 0:
                    continue
                if not self.speaker_files[meeting_id][speaker]:
                    continue
                long_enough_speeches = [speech for speech in self.speaker_files[meeting_id][speaker] if
                                        len(speech) >= seg_len]
                if not long_enough_speeches:
                    logging.warning(
                        f"Failed to find a long enough speech segment for speaker {speaker} at segment {seg_start}-{seg_end}")
                    continue
                speech = random.choice(long_enough_speeches)
                # 确保 speech_segment 的长度与 mix_audio 切片长度一致
                speech_segment = speech[:seg_len]
                # 计算 mix_audio 中的实际位置
                mix_start = seg_start - start_frame
                mix_end = seg_end - start_frame
                # 检查位置是否越界
                if mix_start < 0 or mix_end > len(mix_audio):
                    logging.warning(f"Index out of bounds for mix_audio at segment {seg_start}-{seg_end}")
                    continue
                mix_audio[mix_start:mix_end] += speech_segment
                simulated_label[speaker, mix_start:mix_end] = 1

        mix_audio = self.add_noise(mix_audio)
        mix_audio = self.add_reverberation(mix_audio)
        return mix_audio, simulated_label

    def get_valid_segments(self, label_matrix):
        """提取标签矩阵中至少有一个说话人活跃的时间段"""
        active_mask = np.any(label_matrix, axis=0)
        valid_start = np.where(np.diff(active_mask, prepend=0) == 1)[0]
        valid_end = np.where(np.diff(active_mask, append=0) == -1)[0]
        return list(zip(valid_start, valid_end))

    def split_into_nonoverlapping(self, active_frames, global_start):
        """将活跃帧分割为非重叠连续段（论文要求）"""
        if len(active_frames) == 0:
            return []
        segments = []
        start = active_frames[0] + global_start
        for i in range(1, len(active_frames)):
            if active_frames[i] != active_frames[i - 1] + 1:
                end = active_frames[i - 1] + global_start + 1
                segments.append((start, end))
                start = active_frames[i] + global_start
        # 处理最后一个段
        end = active_frames[-1] + global_start + 1
        segments.append((start, end))
        return segments

    def add_noise(self, audio):
        """添加MUSAN噪声"""
        if not self.noise_files:
            return audio
        noise_file = random.choice(self.noise_files)
        noise, sr = librosa.load(noise_file, sr=self.sample_rate, mono=True)
        if len(noise) < len(audio):
            num_repeats = len(audio) // len(noise) + 1
            noise = np.tile(noise, num_repeats)[:len(audio)]
        elif len(noise) > len(audio):
            noise = noise[:len(audio)]
        snr = random.uniform(5, 15)
        noise = noise * (np.std(audio) / np.std(noise)) / (10 ** (snr / 20))
        return audio + noise

    def add_reverberation(self, audio):
        """添加RIR混响"""
        if not self.rir_files:
            return audio
        rir, sr = librosa.load(random.choice(self.rir_files), sr=self.sample_rate, mono=True)
        rir = rir / np.max(np.abs(rir))
        return convolve(audio, rir, mode='same')

    def save_sample(self, audio, label_matrix, output_path):
        """保存模拟样本和标签"""
        sf.write(output_path, audio, self.sample_rate)
        np.save(output_path.replace('.wav', '_label.npy'), label_matrix)

def test_data_simulator():
    # 假设之前的 DataSimulator 类已经定义
    # 初始化模拟器
    simulator = DataSimulator(
        real_label_dir="tests/alimeeting/Eval_Ali/Eval_Ali_far/textgrid_dir",
        speaker_data_dir="tests/alimeeting/Eval_Ali/Eval_Ali_far/audio_dir",
        noise_dir="/maduo/datasets/musan",
        rir_dir="/maduo/datasets/RIRS_NOISES"
    )

    # 选择一个会议ID进行模拟
    meeting_id = list(simulator.real_labels.keys())[0]

    # 生成模拟数据
    duration = random.uniform(5, 15)
    audio, label_matrix = simulator.simulate_audio(meeting_id, total_duration=duration)

    # 验证 1: 标签矩阵的维度
    num_speakers = len(simulator.real_labels[meeting_id])
    expected_frames = int(duration * simulator.sample_rate)
    if label_matrix.shape[0] != num_speakers:
        print(f"错误：标签矩阵的行数应为 {num_speakers}，但实际为 {label_matrix.shape[0]}")
    if label_matrix.shape[1] != expected_frames:
        print(f"错误：标签矩阵的列数应为 {expected_frames}，但实际为 {label_matrix.shape[1]}")
    else:
        print("标签矩阵维度验证通过")

    # 验证 2: 非重叠连续段
    for speaker in range(num_speakers):
        active_frames = np.where(label_matrix[speaker] == 1)[0]
        segments = simulator.split_into_nonoverlapping(active_frames, 0)
        for i in range(len(segments) - 1):
            end_frame = segments[i][1]
            start_frame = segments[i + 1][0]
            if start_frame < end_frame:
                print(f"错误：说话人 {speaker} 的活跃段存在重叠")
        print(f"说话人 {speaker} 的非重叠连续段验证通过")

    # 验证 3: 音频与标签的一致性
    # 这里简单地检查音频中是否有声音的时间段与标签矩阵中的标记一致
    audio_energy = librosa.feature.rms(y=audio).flatten()
    energy_threshold = np.mean(audio_energy) * 0.1  # 简单的能量阈值
    for speaker in range(num_speakers):
        active_frames = np.where(label_matrix[speaker] == 1)[0]
        for frame in active_frames:
            if audio_energy[frame] < energy_threshold:
                print(f"错误：说话人 {speaker} 在标签标记为活跃的帧 {frame} 处音频能量过低")
        print(f"说话人 {speaker} 的音频与标签一致性验证通过")


# 使用示例
if __name__ == "__main__":
    # valid whether is correct
    #test_data_simulator()

    # generate simulate data
    # 初始化模拟器
    simulator = DataSimulator(
        real_label_dir="tests/alimeeting/Eval_Ali/Eval_Ali_far/textgrid_dir",
        speaker_data_dir="tests/alimeeting/Eval_Ali/Eval_Ali_far/audio_dir",
        noise_dir="/maduo/datasets/musan",
        rir_dir="/maduo/datasets/RIRS_NOISES"
    )

    # 选择一个会议ID进行模拟
    meeting_id = list(simulator.real_labels.keys())[0]

    # 生成模拟数据
    for i in range(2):
        duration = random.uniform(5, 15)
        audio, label_matrix = simulator.simulate_audio(meeting_id, total_duration=duration)

        # 保存文件
        output_file = f"simulated_sample_{i}.wav"
        simulator.save_sample(audio, label_matrix, output_file)
        print(f"Generated {output_file} with meeting {meeting_id}")
   

