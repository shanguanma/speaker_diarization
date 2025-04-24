import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

class OTSVADInference:
    def __init__(self, front_end, back_end, max_n_speaker=8, upper_threshold=0.6, lower_threshold=0.3):
        self.front_end = front_end  # 前端特征提取模型
        self.back_end = back_end    # 后端预测模型
        self.max_n_speaker = max_n_speaker
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        
        # 初始化缓冲区
        self.Y_hat = None  # 输出缓冲区 (N, T)
        self.E_hat = None  # 嵌入缓冲区 (D, T)
        self.ct = None     # 帧计数 (T,)
        
    def chunking(self, X, l, m):
        """生成块迭代器，返回(start, end, chunk)"""
        T = X.shape[-1]
        for start in range(0, T, m):
            end = min(start + l, T)
            yield start, end, X[..., start:end]

    def binarize(self, Y, threshold=0.5):
        """二值化处理"""
        return (Y >= threshold).float()

    def forward(self, X, l, m):
        """算法1完整推理过程"""
        N, D = self.max_n_speaker, self.front_end.output_dim  # 假设前端输出维度为D
        device = next(self.front_end.parameters()).device
        
        # 初始化缓冲区
        self.Y_hat = torch.zeros(N, 0, device=device)  # (N, T)
        self.E_hat = torch.zeros(D, 0, device=device)  # (D, T)
        self.ct = torch.ones(0, device=device)          # (T,)
        
        for start, end, Xk in self.chunking(X, l, m):
            Tk = end - start
            Ek = self.front_end(Xk)  # 前端提取帧级嵌入 (D, Tk)
            
            # 初始化第一个块
            if self.Y_hat.shape[1] == 0:
                self.Y_hat = torch.zeros(N, Tk, device=device)
                self.Y_hat[0, :] = 1.0  # 假设第一个块只有1个说话人
                self.E_hat = Ek
                self.ct = torch.ones(Tk, device=device)
                continue
            
            # 步骤12: 二值化历史输出
            Y_bar = self.binarize(self.Y_hat, self.upper_threshold)  # (N, T_prev)
            
            # 步骤13: 计算目标说话人嵌入
            numerator = Y_bar @ self.E_hat.transpose(0, 1)  # (N, D)
            denominator = torch.sum(Y_bar, dim=1, keepdim=True)  # (N, 1)
            ek = numerator / denominator.clamp(min=1e-8)  # (N, D)
            
            # 步骤14: 后端预测当前块
            Yk = self.back_end(Ek, ek)  # (N, Tk)
            
            # 步骤17-19: 新说话人检测
            last_m_frames = self.Y_hat[:, -m:] if m <= self.Y_hat.shape[1] else self.Y_hat
            if (last_m_frames < self.lower_threshold).all() and (self.max_n_speaker > self.Y_hat.size(0)):
                new_speaker_idx = self.Y_hat.size(0)
                self.Y_hat = torch.cat([self.Y_hat, torch.zeros(1, last_m_frames.shape[1], device=device)], dim=0)
                self.Y_hat[new_speaker_idx, -m:] = 1.0  # 设置上限阈值
                self.E_hat = torch.cat([self.E_hat, torch.zeros(1, last_m_frames.shape[1], device=device)], dim=0)
            
            # 步骤21-29: 缓冲区更新
            if self.ct.size(0) < end:
                # 恢复历史累积值
                valid_len = min(end - start, self.Y_hat.shape[1])
                self.Y_hat[:, :valid_len] *= self.ct[:valid_len]
                self.E_hat[:, :valid_len] *= self.ct[:valid_len]
                
                # 填充新空间
                pad_len = end - self.ct.size(0)
                self.ct = torch.cat([self.ct, torch.zeros(pad_len, device=device)], dim=0)
                self.Y_hat = torch.cat([self.Y_hat, torch.zeros(N, pad_len, device=device)], dim=1)
                self.E_hat = torch.cat([self.E_hat, torch.zeros(D, pad_len, device=device)], dim=1)
            
            # 更新计数和平均值
            self.ct[start:end] += 1
            self.Y_hat[:, start:end] = (self.Y_hat[:, start:end] + Yk) / self.ct[start:end]
            self.E_hat[:, start:end] = (self.E_hat[:, start:end] + Ek) / self.ct[start:end]
        
        # 步骤32: 最终二值化输出
        return self.binarize(self.Y_hat, threshold=0.5)

# 示例前端和后端模型（需根据实际网络结构实现）
class DummyFrontEnd(nn.Module):
    def __init__(self, input_dim=80, output_dim=192):
        super().__init__()
        self.output_dim = output_dim
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.fc(x)  # 示例输出 (D, T) #(B,T,D)

class DummyBackEnd(nn.Module):
    def __init__(self, input_dim=192, hidden_dim=128, num_speakers=8):
        super().__init__()
        self.conformer = nn.Linear(input_dim*2, hidden_dim)
        self.bilstm = nn.LSTM(hidden_dim, hidden_dim*2, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_dim*2, num_speakers)
    
    def forward(self, Ek, ek):
        N, T = ek.shape[0], Ek.shape[1]
        ek_repeated = ek.unsqueeze(2).repeat(1, 1, T)  # (N, D, T)
        concat = torch.cat([Ek, ek_repeated], dim=0)  # (N+D, T)
        h = self.conformer(concat.transpose(0, 1))  # (T, N+D)
        h, _ = self.bilstm(h.unsqueeze(0))  # (1, T, 2*hidden_dim)
        return self.linear(h.squeeze(0)).transpose(0, 1).sigmoid()  # (N, T)

# 使用示例
if __name__ == "__main__":
    # 初始化模型
    front_end = DummyFrontEnd()
    back_end = DummyBackEnd()
    ots_vad = OTSVADInference(front_end, back_end)
    
    # 模拟输入音频特征 (假设为单通道MFCC特征, shape=(1, 40, 16000))
    x = torch.randn(1, 80, 1600)  # 示例输入特征 (B, F, T) # 16s audio
    x = x.permute(0,2,1) #(B,T,F)
    l, m = 800, 80  # 块长8秒,800帧，帧移(0.8s)100帧*0.8=80帧
    
    # 推理过程
    #output = ots_vad.forward(X.squeeze(0).transpose(0, 1), l, m)  # 转换为 (T, F) 后输入
    output = ots_vad.forward(X.squeeze(0), l, m)
    print("Inference output shape:", output.shape)  # 应输出 (max_n_speaker, T)
