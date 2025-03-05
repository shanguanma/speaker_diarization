import torch
import torch.nn as nn
import torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import random
import os

# 1. TS-VAD 模型定义（保持不变）
class TSVAD(nn.Module):
    def __init__(self, audio_feature_dim=40, speaker_embedding_dim=128, hidden_dim=256):
        super(TSVAD, self).__init__()
        self.audio_encoder = nn.LSTM(audio_feature_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim * 2 + speaker_embedding_dim, nhead=4),
            num_layers=2
        )
        self.fc = nn.Linear(hidden_dim * 2 + speaker_embedding_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, audio_features, speaker_embeddings):
        audio_enc, _ = self.audio_encoder(audio_features)
        batch_size, time_steps, _ = audio_enc.size()
        num_speakers = speaker_embeddings.size(1)
        speaker_embeddings = speaker_embeddings.unsqueeze(1).repeat(1, time_steps, 1, 1)
        audio_enc = audio_enc.unsqueeze(2).repeat(1, 1, num_speakers, 1)
        combined = torch.cat([audio_enc, speaker_embeddings], dim=-1)
        combined = combined.permute(1, 0, 2, 3).reshape(time_steps, batch_size * num_speakers, -1)
        transformer_out = self.transformer(combined)
        transformer_out = transformer_out.view(time_steps, batch_size, num_speakers, -1)
        transformer_out = transformer_out.permute(1, 0, 2, 3)
        logits = self.fc(transformer_out).squeeze(-1)
        probs = self.sigmoid(logits)
        return probs

# 2. MagicData-RAMC 数据集类
class MagicDataRAMCDataset(Dataset):
    def __init__(self, data_dir, subset="train", segment_length=200, sample_rate=16000):
        """
        :param data_dir: MagicData-RAMC 根目录
        :param subset: 'train', 'dev', 或 'test'
        :param segment_length: 每个样本的时间步长度
        :param sample_rate: 采样率（MagicData-RAMC 为 16kHz）
        """
        self.audio_dir = Path(data_dir) / subset / "audio"
        self.rttm_dir = Path(data_dir) / subset / "rttm"
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=40)
        
        # 加载所有音频和 RTTM 文件
        self.audio_files = sorted([f for f in self.audio_dir.glob("*.wav")])
        self.rttm_files = sorted([f for f in self.rttm_dir.glob("*.rttm")])
        assert len(self.audio_files) == len(self.rttm_files), "Audio and RTTM file counts must match!"

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        # 加载音频
        audio_path = self.audio_files[idx]
        waveform, sr = torchaudio.load(audio_path)
        assert sr == self.sample_rate, f"Sample rate mismatch: {sr} != {self.sample_rate}"
        
        # 提取 Mel 特征
        mel_features = self.mel_transform(waveform).squeeze(0).T  # [time_steps, n_mels]
        
        # 裁剪或填充到固定长度
        if mel_features.size(0) > self.segment_length:
            start = random.randint(0, mel_features.size(0) - self.segment_length)
            mel_features = mel_features[start:start + self.segment_length]
        elif mel_features.size(0) < self.segment_length:
            padding = torch.zeros(self.segment_length - mel_features.size(0), mel_features.size(1))
            mel_features = torch.cat([mel_features, padding], dim=0)
        
        # 加载 RTTM 文件并解析说话人活动
        rttm_path = self.rttm_files[idx]
        rttm_df = pd.read_csv(rttm_path, delim_whitespace=True, header=None,
                              names=["type", "file_id", "channel", "start", "duration", "ortho", "conf", "speaker", "x", "y"])
        speakers = rttm_df["speaker"].unique()
        num_speakers = len(speakers)
        
        # 生成标签（时间步级别的 0/1 矩阵）
        frame_duration = 0.01  # 每帧 10ms，与 Mel 特征帧移一致
        total_frames = self.segment_length
        labels = torch.zeros(total_frames, num_speakers)
        for _, row in rttm_df.iterrows():
            start_frame = int(row["start"] / frame_duration)
            end_frame = int((row["start"] + row["duration"]) / frame_duration)
            spk_idx = np.where(speakers == row["speaker"])[0][0]
            start_frame = max(0, start_frame)
            end_frame = min(total_frames, end_frame)
            labels[start_frame:end_frame, spk_idx] = 1
        
        # 模拟说话人嵌入（实际应用中需替换为 x-vector 等）
        speaker_embeddings = torch.randn(num_speakers, 128)
        
        return mel_features, speaker_embeddings, labels, num_speakers

# 3. 训练函数
def train_tsvad(model, train_loader, num_epochs=10, learning_rate=0.001, device="cuda"):
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (audio_features, speaker_embeddings, labels, num_speakers) in enumerate(train_loader):
            audio_features = audio_features.to(device)  # [batch, time, feature_dim]
            speaker_embeddings = speaker_embeddings.to(device)  # [batch, num_speakers, emb_dim]
            labels = labels.to(device)  # [batch, time, num_speakers]
            
            # 前向传播
            optimizer.zero_grad()
            probs = model(audio_features, speaker_embeddings)  # [batch, time, num_speakers]
            
            # 计算损失
            loss = criterion(probs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    torch.save(model.state_dict(), "tsvad_model.pth")
    print("Model saved to tsvad_model.pth")

# 4. 主函数：数据准备 + 训练
if __name__ == "__main__":
    # 数据路径
    data_dir = "/path/to/MagicData-RAMC"  # 替换为实际路径
    
    # 数据集和加载器
    dataset = MagicDataRAMCDataset(data_dir, subset="train", segment_length=200)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    
    # 初始化模型
    model = TSVAD(audio_feature_dim=40, speaker_embedding_dim=128, hidden_dim=256)
    
    # 训练模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tsvad(model, train_loader, num_epochs=10, learning_rate=0.001, device=device)
