import numpy as np
import soundfile as sf
import librosa
import random
from glob import glob
from scipy.signal import convolve
from textgrid import TextGrid


class DataSimulator:
    def __init__(self, real_label_dir, speaker_data_dir, noise_dir, rir_dir, sample_rate=16000):
        """
        初始化数据模拟器

        参数:
            real_label_dir: 真实标签文件所在目录 (TextGrid格式)
            speaker_data_dir:  speaker语音数据目录 (AliMeeting)
            noise_dir: MUSAN噪声目录
            rir_dir: RIR混响脉冲响应目录
            sample_rate: 音频采样率
        """
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
            label_matrix = np.zeros((num_speakers, int(max_duration * self.sample_rate)), dtype=int)
            for speaker_index, tier in enumerate(tg):
                for interval in tier:
                    if interval.mark != "":
                        start_frame = int(interval.minTime * self.sample_rate)
                        end_frame = int(interval.maxTime * self.sample_rate)
                        label_matrix[speaker_index, start_frame:end_frame] = 1
            label_matrices[meeting_id] = label_matrix
        return label_matrices

    def load_speaker_data(self, label_dir, data_dir):
        """加载AliMeeting的所有speaker的语音文件，并根据TextGrid提取非重叠片段"""
        speaker_files = {}
        label_files = glob(f"{label_dir}/*.TextGrid")
        for label_file in label_files:
            meeting_id = label_file.split("/")[-1].replace(".TextGrid", "")
            audio_file = f"{data_dir}/{meeting_id}.wav"
            audio, sr = librosa.load(audio_file, sr=self.sample_rate)
            tg = TextGrid.fromFile(label_file)
            num_speakers = len(tg)
            speaker_audios = [[] for _ in range(num_speakers)]
            for speaker_index, tier in enumerate(tg):
                for interval in tier:
                    if interval.mark != "":
                        start_frame = int(interval.minTime * self.sample_rate)
                        end_frame = int(interval.maxTime * self.sample_rate)
                        # 检查是否有重叠
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
                            speaker_audios[speaker_index].append(audio[start_frame:end_frame])
            speaker_files[meeting_id] = speaker_audios
        return speaker_files

    def simulate_audio(self, meeting_id, total_duration=10):
        """生成指定会议ID的模拟音频"""
        T = int(total_duration * self.sample_rate)  # 总帧数
        label_matrix = self.real_labels[meeting_id]
        num_speakers = label_matrix.shape[0]
        mix_audio = np.zeros(T, dtype=np.float32)
        speaker_audios = []

        for speaker in range(num_speakers):
            speaker_audio = np.zeros(T, dtype=np.float32)
            active_frames = np.where(label_matrix[speaker] == 1)[0]
            if len(active_frames) == 0:
                continue

            # 分割活跃段为连续区间
            segments = self.split_into_segments(active_frames)
            for seg_start, seg_end in segments:
                seg_duration = seg_end - seg_start
                if self.speaker_files[meeting_id][speaker]:
                    speech = random.choice(self.speaker_files[meeting_id][speaker])
                    if len(speech) >= seg_duration:
                        start = random.randint(0, len(speech) - seg_duration)
                        speaker_audio[seg_start:seg_end] = speech[start:start + seg_duration]

            speaker_audios.append(speaker_audio)

        # 混合所有speaker音频
        mix_audio = np.sum(speaker_audios, axis=0)
        # 添加噪声和混响
        mix_audio = self.add_noise(mix_audio)
        mix_audio = self.add_reverberation(mix_audio)

        return mix_audio, label_matrix

    def select_speech_segment(self, meeting_id, speaker, duration):
        """从指定会议和说话人数据中选择指定时长的片段"""
        if self.speaker_files[meeting_id][speaker]:
            speech = random.choice(self.speaker_files[meeting_id][speaker])
            if len(speech) >= duration:
                start = random.randint(0, len(speech) - duration)
                return speech[start:start + duration]
        return np.zeros(duration)

    def split_into_segments(self, active_frames):
        """将连续活跃帧分割为不重叠的段"""
        segments = []
        current_start = active_frames[0]
        for frame in active_frames[1:]:
            if frame > current_start + 1:
                segments.append((current_start, frame))
                current_start = frame
        segments.append((current_start, active_frames[-1]))
        return segments

    def add_noise(self, audio):
        """添加MUSAN噪声"""
        if not self.noise_files:
            return audio
        noise_file = random.choice(self.noise_files)
        noise, sr = librosa.load(noise_file, sr=self.sample_rate, mono=True)
        noise = noise[:len(audio)]
        snr = random.uniform(5, 15)  # 随机信噪比
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


# 使用示例
if __name__ == "__main__":
    # 初始化模拟器
    simulator = DataSimulator(
        real_label_dir="path/to/alimeeting_textgrid_labels",
        speaker_data_dir="path/to/alimeeting_wav",
        noise_dir="path/to/musan_noise",
        rir_dir="path/to/rir_corpus"
    )

    # 选择一个会议ID进行模拟
    meeting_id = list(simulator.real_labels.keys())[0]

    # 生成模拟数据
    for i in range(10):  # 生成10个模拟样本
        duration = random.uniform(5, 15)  # 5 - 15秒时长
        audio, label_matrix = simulator.simulate_audio(meeting_id, duration)

        # 保存文件
        output_file = f"simulated_sample_{i}.wav"
        simulator.save_sample(audio, label_matrix, output_file)
        print(f"Generated {output_file} with meeting {meeting_id}")
    
