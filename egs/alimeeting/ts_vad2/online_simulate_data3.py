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
        label_files = glob(f"{label_dir}/*.TextGrid")
        label_matrices = {}
        for label_file in label_files:
            meeting_id = label_file.split("/")[-1].replace(".TextGrid", "")
            tg = TextGrid.fromFile(label_file)
            num_speakers = len(tg)
            max_duration = tg.maxTime
            max_frames = int(max_duration * self.sample_rate)  # 总帧数
            label_matrix = np.zeros((num_speakers, max_frames), dtype=int)
            for speaker_index, tier in enumerate(tg):
                for interval in tier:
                    if interval.mark != "":
                        print(f"interval.minTime: {interval.minTime}, interval.maxTime: {interval.maxTime}")
                        start_frame = int(interval.minTime * self.sample_rate)
                        end_frame = int(interval.maxTime * self.sample_rate)
                        # 确保结束帧不超过总帧数
                        end_frame = min(end_frame, max_frames)
                        label_matrix[speaker_index, start_frame:end_frame] = 1
            label_matrices[meeting_id] = label_matrix
        return label_matrices


    def load_speaker_data(self, label_dir, data_dir):
        speaker_files = {}
        label_files = glob(f"{label_dir}/*.TextGrid")
        for label_file in label_files:
            meeting_id = label_file.split("/")[-1].replace(".TextGrid", "")
            #audio_file = f"{data_dir}/{meeting_id}.wav"
            audio_file = glob(os.path.join(data_dir, meeting_id) + "*.wav")[0]
            audio, sr = librosa.load(audio_file, sr=self.sample_rate)
            tg = TextGrid.fromFile(label_file)
            num_speakers = len(tg)
            speaker_audios = [[] for _ in range(num_speakers)]
            for speaker_index, tier in enumerate(tg):
                for interval in tier:
                    if interval.mark != "":
                        start_frame = int(interval.minTime * self.sample_rate)
                        end_frame = int(interval.maxTime * self.sample_rate)
                        # 提取非重叠片段（原逻辑）
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
                            # 确保片段长度合法
                            if end_frame > start_frame:
                                speaker_audios[speaker_index].append(audio[start_frame:end_frame])
            speaker_files[meeting_id] = speaker_audios
        return speaker_files


    def simulate_audio(self, meeting_id, total_duration=10):
        label_matrix = self.real_labels[meeting_id]
        num_speakers, max_frames = label_matrix.shape
        print(f"meeting_id: {meeting_id},label_matrix num_speakers: {num_speakers}, max_frames: {max_frames}")
        T = max_frames  # 使用标签矩阵的总帧数，避免越界
        mix_audio = np.zeros(T, dtype=np.float32)
        speaker_audios = []
        simulated_label = np.zeros((num_speakers, T), dtype=int)

        for speaker in range(num_speakers):
            logging.info(f"Processing speaker {speaker}")
            speaker_audio = np.zeros(T, dtype=np.float32)
            active_frames = np.where(label_matrix[speaker] == 1)[0]
            if len(active_frames) == 0:
                continue

            segments = self.split_into_segments(active_frames, T)  # 传入总帧数校验边界
            for seg_start, seg_end in segments:
                seg_duration = seg_end - seg_start
                if seg_duration <= 0:
                    continue
                if not self.speaker_files[meeting_id][speaker]:
                    continue

                long_enough_speeches = [speech for speech in self.speaker_files[meeting_id][speaker] if len(speech) >= seg_duration]
                if not long_enough_speeches:
                    logging.warning(f"Failed to find a long enough speech segment for speaker {speaker} at segment {seg_start}-{seg_end}")
                    continue
                speech = random.choice(long_enough_speeches)
                start = random.randint(0, len(speech) - seg_duration)
                speech_segment = speech[start:start + seg_duration]  # 严格截取目标长度
                print(f"speech_segment len: {len(speech_segment)}, seg_duration: {seg_duration},seg_start: {seg_start}, seg_end: {seg_end}")
                # 确保切片长度匹配（处理潜在的边界问题）
                target_slice = speaker_audio[seg_start:seg_end]
                print(f"target_slice len: {len(target_slice)}")
                if len(target_slice) != seg_duration:
                    continue  # 跳过异常切片

                speaker_audio[seg_start:seg_end] = speech_segment
                simulated_label[speaker, seg_start:seg_end] = 1
            speaker_audios.append(speaker_audio)
        print(f"start sum all speaker_audios")
        mix_audio = np.sum(speaker_audios, axis=0)
        print(f"start add noise")
        mix_audio = self.add_noise(mix_audio)
        print(f"start add reverberation")
        mix_audio = self.add_reverberation(mix_audio)
        return mix_audio, simulated_label

    def split_into_segments(self, active_frames, max_frames):
        """新增max_frames参数，确保段不越界"""
        if len(active_frames) == 0:
            return []
        segments = []
        current_start = active_frames[0]
        for i in range(1, len(active_frames)):
            if active_frames[i] > active_frames[i-1] + 1:
                # 结束帧不超过总帧数
                end = min(active_frames[i-1] + 1, max_frames)
                segments.append((current_start, end))
                current_start = active_frames[i]
        # 处理最后一个段
        end = min(active_frames[-1] + 1, max_frames)
        segments.append((current_start, end))
        return segments

    def add_noise(self, audio):
        """添加MUSAN噪声"""
        if not self.noise_files:
            return audio
        noise_file = random.choice(self.noise_files)
        noise, sr = librosa.load(noise_file, sr=self.sample_rate, mono=True)
        # 确保噪声长度和音频长度一致
        if len(noise) < len(audio):
            # 循环拼接噪声
            num_repeats = len(audio) // len(noise) + 1
            noise = np.tile(noise, num_repeats)[:len(audio)]
        elif len(noise) > len(audio):
            # 截断噪声
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

def verify_correspondence(simulator, meeting_id):
    audio, label_matrix = simulator.simulate_audio(meeting_id)
    num_speakers, num_frames = label_matrix.shape

    # 遍历每个说话者的标签
    for speaker in range(num_speakers):
        speaker_active_frames = np.where(label_matrix[speaker] == 1)[0]
        if len(speaker_active_frames) == 0:
            continue

        # 找到连续的活动片段
        segments = simulator.split_into_segments(speaker_active_frames, num_frames)
        for seg_start, seg_end in segments:
            # 检查该片段内音频是否不为零
            audio_segment = audio[seg_start:seg_end]
            if np.all(audio_segment == 0):
                raise ValueError(f"Speaker {speaker} is labeled as active from frame {seg_start} to {seg_end}, "
                                 f"but the corresponding audio segment is all zeros.")

    print("Verification passed: Audio and labels are in correspondence.")




# 使用示例
if __name__ == "__main__":
    # valid whether is correct
    #test_data_simulator()

    #verify_correspondence(simulator, meeting_id) 
    # generate simulate data


    # 初始化模拟器
    simulator = DataSimulator(
        real_label_dir="tests/alimeeting/Eval_Ali/Eval_Ali_far/textgrid_dir",
        speaker_data_dir="tests/alimeeting/Eval_Ali/Eval_Ali_far/audio_dir",
        noise_dir="/data/maduo/datasets/musan",
        rir_dir="/data/maduo/datasets/RIRS_NOISES"
    )

    # 选择一个会议ID进行模拟
    meeting_id = list(simulator.real_labels.keys())[0]

    verify_correspondence(simulator, meeting_id)
    # 生成模拟数据
#    for i in range(2):  # 生成10个模拟样本
#        print(f"generated {i} sample")
#        duration = random.uniform(5, 15)  # 5 - 15秒时长
#        audio, label_matrix = simulator.simulate_audio(meeting_id, total_duration=duration)
#
#        # 保存文件
#        output_file = f"simulated_codes3_sample_{i}.wav"
#        print(f"before save sample")
#        simulator.save_sample(audio, label_matrix, output_file)
#        print(f"sb")
#        print(f"Generated {output_file} with meeting {meeting_id}")
