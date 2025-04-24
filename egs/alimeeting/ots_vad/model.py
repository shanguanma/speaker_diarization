import torch
import torch.nn as nn
from conformer import ConformerBlock  # 需要自定义或使用现有实现

class FrontEnd(nn.Module):
    """前端网络（以ResNet34为例）"""
    def __init__(self, pretrained=True):
        super().__init__()
        # 基于ResNet34的修改版本（时间下采样8倍）
        self.resnet = torch.hub.load('pytorch/vision', 'resnet34', pretrained=False)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)  # 输入通道改为1
        self.resnet.fc = nn.Identity()  # 移除最后的全连接层
        
        # 全局统计池化（GSP）层
        self.gsp = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)),  # 保持时间维度
            nn.Flatten(start_dim=2)
        )
        
        # 帧嵌入投影层
        self.proj = nn.Linear(512 * 2, 256)  # 均值+方差 → 256维
        
    def forward(self, x):
        # x: (batch, 1, freq, time)
        features = self.resnet(x)  # (batch, 512, time/8, 1)
        
        # 全局统计池化
        mean = torch.mean(features, dim=2, keepdim=True)
        std = torch.std(features, dim=2, keepdim=True)
        stats = torch.cat([mean, std], dim=1)  # (batch, 1024, T, 1)
        
        # 投影到帧嵌入
        embeddings = self.proj(stats.squeeze(-1).transpose(1,2))  # (batch, T, 256)
        return embeddings

class BackEnd(nn.Module):
    """后端网络"""
    def __init__(self, num_speakers=4):
        super().__init__()
        self.num_speakers = num_speakers
        
        # Conformer编码器
        self.conformer = nn.Sequential(
            *[ConformerBlock(dim=256, dim_head=64, heads=8) for _ in range(6)]
        )
        
        # BiLSTM
        self.bilstm = nn.LSTM(256 * 2, 256, bidirectional=True, batch_first=True)
        
        # 分类层
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_speakers),
            nn.Sigmoid()
        )
        
    def forward(self, frame_emb, target_emb):
        """
        frame_emb: 当前块的帧嵌入 (batch, T, 256)
        target_emb: 目标说话人嵌入 (batch, N, 256)
        """
        batch, T, _ = frame_emb.shape
        N = self.num_speakers
        
        # 重复拼接
        frame_emb = frame_emb.unsqueeze(1).repeat(1,N,1,1)  # (B,N,T,D)
        target_emb = target_emb.unsqueeze(2).repeat(1,1,T,1) # (B,N,T,D)
        concat = torch.cat([frame_emb, target_emb], dim=-1)  # (B,N,T,512)
        
        # 处理每个说话者
        concat = concat.view(batch*N, T, 512)
        x = self.conformer(concat)  # (B*N, T, 256)
        
        # BiLSTM聚合
        x, _ = self.bilstm(x)  # (B*N, T, 512)
        
        # 分类
        logits = self.fc(x)  # (B*N, T, 1)
        return logits.view(batch, N, T)

class OnlineTSVAD(nn.Module):
    def __init__(self, frontend, num_speakers=4):
        super().__init__()
        self.frontend = frontend
        self.backend = BackEnd(num_speakers)
        
    def forward(self, x_left, x_right, y_left):
        """
        x_left: 左半部分音频特征 (B, 1, F, T)
        x_right: 右半部分音频特征 (B, 1, F, T)
        y_left: 左半部分标签 (B, N, T)
        """
        # 前端处理
        emb_left = self.frontend(x_left)  # (B, T/8, 256)
        emb_right = self.frontend(x_right)
        
        # 计算目标说话人嵌入（训练时使用真实标签）
        target_emb = []
        for b in range(emb_left.size(0)):
            speaker_embs = []
            for n in range(self.backend.num_speakers):
                # 提取该说话者的有效帧
                mask = y_left[b,n].unsqueeze(-1)  # (T, 1)
                valid_emb = emb_left[b] * mask  # (T, 256)
                sum_emb = valid_emb.sum(dim=0)
                count = mask.sum()
                speaker_emb = sum_emb / (count + 1e-8)
                speaker_embs.append(speaker_emb)
            target_emb.append(torch.stack(speaker_embs))
        target_emb = torch.stack(target_emb)  # (B, N, 256)
        
        # 后端处理
        pred = self.backend(emb_right, target_emb)
        return pred
