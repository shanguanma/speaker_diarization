import random
import numpy as np

class AlimeetingDataset(Dataset):
    def __init__(self, audio_dir, textgrid_dir, num_speakers=4):
        self.num_speakers = num_speakers
        self.audio_files = [...]  # 加载单通道音频路径
        self.textgrid_files = [...]  # 对应textgrid路径
        # 其他初始化代码与之前相同...
    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        # 加载原始音频和textgrid
        audio = load_audio(self.audio_files[idx])  # (T,)
        sr = 16000
        tg = TextGrid.fromFile(self.textgrid_files[idx])
        
        # 生成原始标签
        duration = len(audio) / sr
        frame_labels = torch.zeros(self.num_speakers, int(duration*100))
        
        # 概率决定是否模拟左半部分
        if random.random() < 0.5:
            # 生成模拟左半部分
            chunk_sec = 4  # 4秒块
            left_feat, left_label = self._generate_simulated_chunk(tg, audio, sr)
        else:
            # 随机选择原始左半部分
            start = random.randint(0, int(duration) - 8)
            left_audio = audio[start*sr : (start+4)*sr]
            left_feat = extract_fbank(left_audio).unsqueeze(0)
            left_label = self._extract_labels(tg, start, 4)

        # 随机选择右半部分（与左半部分无关）
        start_r = random.randint(0, int(duration) - 4)
        right_audio = audio[start_r*sr : (start_r+4)*sr]
        right_feat = extract_fbank(right_audio).unsqueeze(0)
        right_label = self._extract_labels(tg, start_r, 4)

        return {
            'left_feat': left_feat,
            'right_feat': right_feat,
            'left_label': left_label,
            'right_label': right_label
        }

    def _generate_simulated_chunk(self, tg, full_audio, sr):
        """生成模拟信号的核心方法"""
        chunk_sec = 4
        segments = []
        labels = torch.zeros(self.num_speakers, chunk_sec*100)
        
        # 为每个说话人分配等长时间段
        seg_per_spk = chunk_sec / self.num_speakers
        for spk in range(self.num_speakers):
            # 在textgrid中找到该说话人所有有效区间
            intervals = [i for i in tg.tiers[spk].intervals if i.mark != ""]
            
            # 随机选择一个有效区间
            if len(intervals) == 0:
                selected_interval = (0, seg_per_spk)  # 容错处理
            else:
                interval = random.choice(intervals)
                selected_interval = (interval.minTime, interval.maxTime)
            
            # 计算截取区间
            max_start = max(selected_interval[0], selected_interval[1] - seg_per_spk)
            start = random.uniform(max(0, max_start - 0.1), max_start)
            end = start + seg_per_spk
            
            # 截取音频
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            segment = full_audio[start_sample:end_sample]
            
            # 处理长度不足
            if len(segment) < seg_per_spk * sr:
                pad_len = int(seg_per_spk * sr) - len(segment)
                segment = np.pad(segment, (0, pad_len), mode='constant')
                
            segments.append(segment)
            
            # 生成标签
            label_start = int(spk * seg_per_spk * 100)
            label_end = int((spk+1) * seg_per_spk * 100)
            labels[spk, label_start:label_end] = 1
        
        # 混合所有说话人音频
        mixed_audio = np.sum(segments, axis=0) if len(segments) > 1 else segments[0]
        feat = extract_fbank(mixed_audio).unsqueeze(0)
        return feat, labels

    def _extract_labels(self, tg, start_sec, duration_sec):
        """从textgrid中提取指定区间的标签"""
        labels = torch.zeros(self.num_speakers, duration_sec*100)
        end_sec = start_sec + duration_sec
        
        for spk in range(self.num_speakers):
            for interval in tg.tiers[spk].intervals:
                if interval.mark == "":
                    continue
                overlap_start = max(start_sec, interval.minTime)
                overlap_end = min(end_sec, interval.maxTime)
                if overlap_start < overlap_end:
                    start = int((overlap_start - start_sec)*100)
                    end = int((overlap_end - start_sec)*100)
                    labels[spk, start:end] = 1
        return labels

class OnlineTSVAD(nn.Module):
    def forward(self, x_left, x_right, y_left):
        # 修改后的目标说话人嵌入计算
        batch_size, num_speakers, _ = y_left.shape
        
        # 计算每个说话人的平均嵌入
        target_emb = []
        for b in range(batch_size):
            speaker_embs = []
            for n in range(num_speakers):
                # 获取有效帧掩码
                valid_mask = y_left[b, n].unsqueeze(-1)  # (T, 1)
                num_valid = valid_mask.sum() + 1e-8
                
                # 加权平均
                sum_emb = (emb_left[b] * valid_mask).sum(dim=0)
                speaker_emb = sum_emb / num_valid
                speaker_embs.append(speaker_emb)
            target_emb.append(torch.stack(speaker_embs))
        
        target_emb = torch.stack(target_emb)  # (B, N, 256)
        return self.backend(emb_right, target_emb)
