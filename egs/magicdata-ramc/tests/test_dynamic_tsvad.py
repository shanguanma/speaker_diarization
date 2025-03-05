import torch
import torch.nn as nn
import torchaudio
import numpy as np

# 1. 定义 TS-VAD 模型
class TSVAD(nn.Module):
    def __init__(self, audio_feature_dim=40, speaker_embedding_dim=128, hidden_dim=256):
        super(TSVAD, self).__init__()
        # 音频特征编码器（基于 LSTM）
        self.audio_encoder = nn.LSTM(audio_feature_dim, hidden_dim, batch_first=True, bidirectional=True)
        # Transformer 用于融合音频和说话人嵌入
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim * 2 + speaker_embedding_dim, nhead=4),
            num_layers=2
        )
        # 输出层，动态适应说话人数量
        self.fc = nn.Linear(hidden_dim * 2 + speaker_embedding_dim, 1)  # 二分类：说话/不说话
        self.sigmoid = nn.Sigmoid()

    def forward(self, audio_features, speaker_embeddings):
        # audio_features: [batch_size, time_steps, audio_feature_dim]
        # speaker_embeddings: [batch_size, num_speakers, speaker_embedding_dim]
        
        # 编码音频特征
        audio_enc, _ = self.audio_encoder(audio_features)  # [batch, time, hidden_dim * 2]
        
        # 将说话人嵌入扩展到时间维度
        batch_size, time_steps, _ = audio_enc.size()
        num_speakers = speaker_embeddings.size(1)
        speaker_embeddings = speaker_embeddings.unsqueeze(1).repeat(1, time_steps, 1, 1)  # [batch, time, num_speakers, emb_dim]
        
        # 拼接音频特征和每个说话人的嵌入
        audio_enc = audio_enc.unsqueeze(2).repeat(1, 1, num_speakers, 1)  # [batch, time, num_speakers, hidden_dim * 2]
        combined = torch.cat([audio_enc, speaker_embeddings], dim=-1)  # [batch, time, num_speakers, total_dim]
        
        # Transformer 处理
        combined = combined.permute(1, 0, 2, 3)  # [time, batch, num_speakers, total_dim]
        combined = combined.reshape(time_steps, batch_size * num_speakers, -1)
        transformer_out = self.transformer(combined)  # [time, batch * num_speakers, total_dim]
        
        # 恢复维度并预测
        transformer_out = transformer_out.view(time_steps, batch_size, num_speakers, -1)
        transformer_out = transformer_out.permute(1, 0, 2, 3)  # [batch, time, num_speakers, total_dim]
        logits = self.fc(transformer_out).squeeze(-1)  # [batch, time, num_speakers]
        probs = self.sigmoid(logits)  # 每个说话人的语音活动概率
        
        return probs

# 2. 数据预处理函数
def preprocess_audio(audio_path, sample_rate=16000):
    waveform, sr = torchaudio.load(audio_path)
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
    # 提取 Mel 频谱特征
    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=40)
    mel_features = mel_transform(waveform).squeeze(0).T  # [time_steps, n_mels]
    return mel_features

# 3. 日志处理函数
def generate_speaker_log(audio_path, speaker_embeddings, model, threshold=0.5):
    # 加载和预处理音频
    audio_features = preprocess_audio(audio_path)
    audio_features = audio_features.unsqueeze(0)  # [1, time_steps, feature_dim]
    
    # 将数据移到 GPU（如果可用）
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    audio_features = audio_features.to(device)
    speaker_embeddings = speaker_embeddings.to(device)
    model = model.to(device)
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        probs = model(audio_features, speaker_embeddings)  # [1, time, num_speakers]
        preds = (probs > threshold).int().squeeze(0)  # [time, num_speakers]
    
    # 生成日志
    time_steps, num_speakers = preds.size()
    log = []
    for t in range(time_steps):
        active_speakers = [i for i in range(num_speakers) if preds[t, i] == 1]
        if active_speakers:
            log.append(f"Time {t * 0.01:.2f}s: Speakers {active_speakers}")
    
    return log

# 4. 示例用法
if __name__ == "__main__":
    # 初始化模型
    model = TSVAD(audio_feature_dim=40, speaker_embedding_dim=128, hidden_dim=256)
    
    # 模拟输入数据
    audio_path = "/data/maduo/datasets/alimeeting/Eval_Ali/Eval_Ali_far/target_audio/R8001_M8004_MS801/all.wav"  # 替换为实际音频路径
    num_speakers = 3  # 可变人数
    speaker_embeddings = torch.randn(1, num_speakers, 128)  # 随机生成的说话人嵌入
    
    # 生成日志
    speaker_log = generate_speaker_log(audio_path, speaker_embeddings, model)
    for entry in speaker_log:
        print(entry)
