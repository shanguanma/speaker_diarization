import torch
import torch.nn as nn
import torchaudio
import numpy as np

# 1. 定义 TS-VAD 模型（保持不变）
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

# 2. 数据预处理函数（保持不变）
def preprocess_audio(audio_path, sample_rate=16000):
    waveform, sr = torchaudio.load(audio_path)
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=40)
    mel_features = mel_transform(waveform).squeeze(0).T
    return mel_features

# 3. 分块处理音频的函数
def chunk_audio_features(audio_features, chunk_size=100, hop_size=50):
    """
    将音频特征分成固定长度的块，支持重叠（hop_size）。
    :param audio_features: [time_steps, feature_dim]
    :param chunk_size: 每个块的时间步数
    :param hop_size: 块之间的步幅（重叠部分 = chunk_size - hop_size）
    :return: 分块后的特征列表
    """
    time_steps, feature_dim = audio_features.size()
    chunks = []
    for start in range(0, time_steps - chunk_size + 1, hop_size):
        end = start + chunk_size
        chunk = audio_features[start:end]
        chunks.append((start, chunk))  # 保存起始时间步和块数据
    # 处理最后一个不完整的块（可选填充）
    if time_steps % hop_size != 0:
        last_chunk = audio_features[-chunk_size:]
        chunks.append((time_steps - chunk_size, last_chunk))
    return chunks

# 4. 流式日志处理函数
def generate_streaming_speaker_log(audio_path, speaker_embeddings, model, chunk_size=100, hop_size=50, threshold=0.5, frame_duration=0.01):
    """
    流式处理长音频并生成说话人日志。
    :param chunk_size: 每个块的时间步数
    :param hop_size: 块之间的步幅
    :param frame_duration: 每个时间步对应的真实时间（秒）
    """
    # 加载和预处理音频
    audio_features = preprocess_audio(audio_path)
    
    # 分块处理
    chunks = chunk_audio_features(audio_features, chunk_size, hop_size)
    
    # 将数据移到 GPU（如果可用）
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    speaker_embeddings = speaker_embeddings.to(device)
    model = model.to(device)
    
    # 流式处理每个块
    model.eval()
    log = []
    with torch.no_grad():
        for chunk_idx, (start_step, chunk) in enumerate(chunks):
            chunk = chunk.unsqueeze(0).to(device)  # [1, chunk_size, feature_dim]
            probs = model(chunk, speaker_embeddings)  # [1, chunk_size, num_speakers]
            preds = (probs > threshold).int().squeeze(0)  # [chunk_size, num_speakers]
            
            # 生成该块的日志
            for t in range(preds.size(0)):
                global_time_step = start_step + t
                active_speakers = [i for i in range(preds.size(1)) if preds[t, i] == 1]
                if active_speakers:
                    log.append(f"Time {global_time_step * frame_duration:.2f}s: Speakers {active_speakers}")
    
    return log

# 5. 示例用法
if __name__ == "__main__":
    # 初始化模型
    model = TSVAD(audio_feature_dim=40, speaker_embedding_dim=128, hidden_dim=256)
    
    # 模拟输入数据
    audio_path = "/data/maduo/datasets/alimeeting/Eval_Ali/Eval_Ali_far/target_audio/R8001_M8004_MS801/all.wav"  # 替换为实际音频路径
    num_speakers = 4  # 可变人数
    speaker_embeddings = torch.randn(1, num_speakers, 128)  # 随机生成的说话人嵌入
    
    # 流式生成日志
    speaker_log = generate_streaming_speaker_log(
        audio_path=audio_path,
        speaker_embeddings=speaker_embeddings,
        model=model,
        chunk_size=100,  # 每个块 100 个时间步
        hop_size=50,     # 步幅 50，50% 重叠
        threshold=0.5,
        frame_duration=0.01  # 每个时间步对应 10ms
    )
    
    # 输出日志
    for entry in speaker_log:
        print(entry)
