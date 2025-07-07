import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.models import Conformer  # 使用torchaudio的Conformer
from torch.utils.checkpoint import checkpoint
from typing import Any, Callable   

from resnet_wespeaker import ResNetWithGSP, BasicBlock
from cam_pplus_wespeaker import CAMPPlusWithGSP
class BatchNorm1D(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.bn = nn.BatchNorm1d(*args, **kwargs)

    def forward(self, input):
        if torch.sum(torch.isnan(input)) == 0:
            output = self.bn(input)
        else:
            output = input
        return output
    
class SpeechFeatUpsample2(nn.Module):
    def __init__(self, speaker_embed_dim: int, upsample: int, model_dim: int = 2560):
        super(SpeechFeatUpsample2, self).__init__()
        self.speaker_embed_dim = speaker_embed_dim
        # here model_dim means it is feature dimension  before pool layer of resnet34_wespeaker or samresnet model dimension
        self.up = nn.ConvTranspose1d(
            model_dim,
            speaker_embed_dim,
            5,
            stride=upsample,
            padding=2,
            output_padding=1,
        )
        self.batchnorm = BatchNorm1D(num_features=speaker_embed_dim)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)  # (B,F,T) -> (B,D,2T)
        x = self.batchnorm(x)
        x = self.act(x)
        return x  # (B,D,2T)
class BatchNorm1D(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.bn = nn.BatchNorm1d(*args, **kwargs)

    def forward(self, input):
        if torch.sum(torch.isnan(input)) == 0:
            output = self.bn(input)
        else:
            output = input
        return output  
# 1. ResNet-based Extractor
class ResNetExtractor(nn.Module):
    def __init__(self, device, speaker_pretrain_model_path, in_dim=80, out_dim=256, extractor_model_type='CAM++_wo_gsp'):
        super().__init__()
         # they are same as the version from `Sequence-to-Sequence Neural Diarization with Automatic Speaker Detection and Representation`
        if extractor_model_type == 'resnet34_32ch': # 5.454688 M parameters
            self.speech_encoder = ResNetWithGSP(BasicBlock,[3, 4, 6, 3],m_channels=32, feat_dim=in_dim,embed_dim=out_dim, out_dim=out_dim)
            # input of resnet model is fbank, means that 1s has 100 frames
            # we set target label rate is 25, means that 1s has 25 frames
            # resnet34_wespeaker model downsample scale is 8, so frame rate  is 12.5, so I should set stride equal to 2.
            upsample = 2
            ## the input shape of self.speech_up except is (B,F,T)
            self.speech_down_or_up = SpeechFeatUpsample2(
                speaker_embed_dim=out_dim,
                upsample=upsample,
                model_dim=out_dim,
            )   
        elif extractor_model_type == 'resnet34_64ch': # 21.53824 M parameters
            self.speech_encoder = ResNetWithGSP(BasicBlock,[3, 4, 6, 3],m_channels=64, feat_dim=in_dim,embed_dim=out_dim, out_dim=out_dim)
            # input of resnet model is fbank, means that 1s has 100 frames
            # we set target label rate is 25, means that 1s has 25 frames
            # resnet34_wespeaker model downsample scale is 8, so frame rate  is 12.5, so I should set stride equal to 2.
            upsample = 2
            ## the input shape of self.speech_up except is (B,F,T)
            self.speech_down_or_up = SpeechFeatUpsample2(
                speaker_embed_dim=out_dim,
                upsample=upsample,
                model_dim=out_dim,
            )   
        elif extractor_model_type == 'resnet152': # 58.140096 M parameters
            self.speech_encoder = ResNetWithGSP(BasicBlock,[3, 8, 36, 3],m_channels=64, feat_dim=in_dim,embed_dim=out_dim, out_dim=out_dim)
            # input of resnet model is fbank, means that 1s has 100 frames
            # we set target label rate is 25, means that 1s has 25 frames
            # resnet34_wespeaker model downsample scale is 8, so frame rate  is 12.5, so I should set stride equal to 2.
            upsample = 2
            ## the input shape of self.speech_up except is (B,F,T)
            self.speech_down_or_up= SpeechFeatUpsample2(
                speaker_embed_dim=out_dim,
                upsample=upsample,
                model_dim=out_dim,
            )   
        elif extractor_model_type == 'CAM++_wo_gsp':
            # input of resnet model is fbank, means that 1s has 100 frames
            # we set target label rate is 25, means that 1s has 25 frames
            # CAMPPlusWithGSP model downsample scale is 2, so frame rate  is 50, so I should set stride equal to 2.
            self.speech_encoder=CAMPPlusWithGSP(feat_dim=in_dim,use_gsp=False) # downsample is 2
            self.speech_encoder.train()
            self.load_speaker_encoder(speaker_pretrain_model_path, device=device, module_name="speech_encoder")
            pretrain_speech_encoder_dim = out_dim
            # downsample is 2
            self.speech_down_or_up = nn.Sequential(
                nn.Conv1d(
                    pretrain_speech_encoder_dim,
                    out_dim,
                    5,
                    stride=2,
                    padding=2,
                ),
                BatchNorm1D(num_features=out_dim),
                nn.ReLU(),
            )
        elif extractor_model_type == 'CAM++_gsp':
            # input of resnet model is fbank, means that 1s has 100 frames
            # we set target label rate is 25, means that 1s has 25 frames
            # CAMPPlusWithGSP model downsample scale is 2, so frame rate  is 50, so I should set stride equal to 2.
            self.speech_encoder=CAMPPlusWithGSP(feat_dim=in_dim,use_gsp=True) # downsample is 2
            self.speech_encoder.train()
            self.load_speaker_encoder(speaker_pretrain_model_path, device=device, module_name="speech_encoder")
            pretrain_speech_encoder_dim = out_dim
            # downsample is 2
            self.speech_down_or_up = nn.Sequential(
                nn.Conv1d(
                    pretrain_speech_encoder_dim,
                    out_dim,
                    5,
                    stride=2,
                    padding=2,
                ),
                BatchNorm1D(num_features=out_dim),
                nn.ReLU(),
            )
        else:
            raise ValueError(f"Unknown model_type: {extractor_model_type}")
        
    def load_speaker_encoder(self, model_path, device, module_name="speech_encoder"):
        loadedState = torch.load(model_path, map_location=device)
        selfState = self.state_dict()
        for name, param in loadedState.items():
            origname = name
            if (
                module_name == "speech_encoder"
                and hasattr(self.speech_encoder, "bn1")
                and isinstance(self.speech_encoder.bn1, BatchNorm1D)
                and ".".join(name.split(".")[:-1]) + ".running_mean" in loadedState
            ):
                name = ".".join(name.split(".")[:-1]) + ".bn." + name.split(".")[-1]

            name = f"{module_name}." + name

            if name not in selfState:
                logger.warn("%s is not in the model." % origname)
                continue
            if selfState[name].size() != loadedState[origname].size():
                sys.stderr.write(
                    "Wrong parameter length: %s, model: %s, loaded: %s"
                    % (origname, selfState[name].size(), loadedState[origname].size())
                )
                continue
            selfState[name].copy_(param)
    def forward(self, x):
        # x: [B, T, F]
        out = self.speech_encoder(x)  # [B, T/8, D], or [B, T/2, D] 
        out = out.permute(0,2,1) # [B, D, T/8] or [B, D, T/2] 
        out = self.speech_down_or_up(out) # [B,D,T/4]  
        out = out.permute(0,2,1) # [B,T/4, D]  
        return out

# 2. Conformer Encoder（使用torchaudio实现）
class SSNDConformerEncoder(nn.Module):
    def __init__(self, in_dim=256, d_model=256, num_layers=4, nhead=8, d_ff=512, cnn_kernel_size=15):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, d_model)
        self.encoder = Conformer(
            input_dim=d_model,
            num_heads=nhead,
            ffn_dim=d_ff,
            num_layers=num_layers,
            depthwise_conv_kernel_size=cnn_kernel_size,
            dropout=0.1,
        )
    # conformer has not downsample
    def forward(self, x):
        # x: [B, T, in_dim]
        #print(f"conformer input shape: {x.shape}")
        x = self.input_proj(x)
        # torchaudio.models.Conformer需要input, lengths
        lengths = torch.full((x.size(0),), x.size(1), dtype=torch.long, device=x.device)
        x, _ = self.encoder(x, lengths)  # 返回 (output, output_lengths)
        #print(f"conformer output shape: {x.shape}")
        return x

# ========== 新增：Fq/Fk融合、Detection/Representation Decoder ==========
class FqFusion(nn.Module):
    """
    Fq(X_dec, Q_aux): Q = X_dec + Linear(Q_aux)/sqrt(D)
    """
    def __init__(self, d_model, d_aux):
        super().__init__()
        self.linear = nn.Linear(d_aux, d_model)
        self.scale = d_model ** 0.5
    def forward(self, x_dec, q_aux):
        # x_dec: [B, N, D], q_aux: [B, N, D_aux]
        q_aux_proj = self.linear(q_aux) / self.scale  # [B, N, D]
        return x_dec + q_aux_proj

class FkFusion(nn.Module):
    """
    Fk(X_fea, K_pos): K = X_fea + Linear(K_pos)/sqrt(D)
    """
    def __init__(self, d_model, d_pos):
        super().__init__()
        self.linear = nn.Linear(d_pos, d_model)
        self.scale = d_model ** 0.5
    def forward(self, x_fea, k_pos):
        # x_fea: [B, T, D], k_pos: [B, T, D_pos]
        k_pos_proj = self.linear(k_pos) / self.scale  # [B, T, D]
        return x_fea + k_pos_proj

class SWDecoderBlockV2(nn.Module):
    """
    支持自定义Q/K融合的Decoder Block，结构如Fig.2
    """
    def __init__(self, d_model, nhead, d_ff, d_aux=None, d_pos=None):
        super().__init__()
        self.fq = FqFusion(d_model, d_aux) if d_aux is not None else None
        self.fk = FkFusion(d_model, d_pos) if d_pos is not None else None
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x_dec, x_fea, q_aux, k_pos):
        # x_dec: [B, N, D], x_fea: [B, T, D], q_aux: [B, N, D_aux], k_pos: [B, T, D_pos]
        Q = self.fq(x_dec, q_aux) if self.fq is not None else x_dec
        K = self.fk(x_fea, k_pos) if self.fk is not None else x_fea
        V = x_fea  # V 直接用 feature emb

        #print(f'Q shape: {Q.shape}')  # [B, N, D]
        #print(f'K shape: {K.shape}')  # [B, T, D]
        #print(f'V shape: {V.shape}')  # [B, T, D]

        # Cross-attention: Q attends to K, V
        x2, _ = self.cross_attn(Q, K, V)
        x = self.norm1(x_dec + x2)
        # Self-attention
        x2, _ = self.self_attn(x, x, x)
        x = self.norm2(x + x2)
        # FFN
        x2 = self.ffn(x)
        x = self.norm3(x + x2)
        #print(f'Output x shape: {x.shape}')  # [B, N, D]
        return x
    
class DetectionDecoder(nn.Module):
    """
    Detection Decoder: 输入decoder emb, feature emb, aux query（L2归一化），pos emb，输出VAD
    """
    def __init__(self, d_model, nhead, d_ff, num_layers, d_aux, d_pos, out_vad_len, out_bias=-0.5):
        super().__init__()
        #self.input_proj = nn.Linear(d_feat, d_model)
        self.layers = nn.ModuleList([
            SWDecoderBlockV2(d_model, nhead, d_ff, d_aux=d_aux, d_pos=d_pos)
            for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(d_model, out_vad_len)
        # 初始化bias为0.0，让模型初始时输出概率更中性
        torch.nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x_dec, x_fea, q_aux, k_pos):
        # x_dec:[B,N,D], it is setting to 0, it applys on query
        # x_fea:[B,T,D], it is ouput of encoder and it applys on value
        # q_aux:[B,N,D'], it is speaker embedding and it applys on query
        # pos_emb:[B,T,D'], it is postion embedding and it applys on key
        # L2 normalize auxiliary queries
        q_aux = F.normalize(q_aux, p=2, dim=-1)
        for layer in self.layers:
            x_dec = layer(x_dec, x_fea, q_aux, k_pos)
        out = self.out_proj(x_dec)  # [B,N,D]->[B, N, T]
        return out

    def focal_bce_loss(self, logits, targets, alpha=0.75, gamma=2.0):
        """
        Focal loss for binary classification to handle class imbalance.
        logits: [B, N, T]
        targets: [B, N, T]
        """
        # 计算sigmoid概率
        probs = torch.sigmoid(logits)
        # 计算focal loss
        pt = probs * targets + (1 - probs) * (1 - targets)  # p_t
        focal_weight = (1 - pt) ** gamma
        # 计算BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        # 应用focal weight和alpha weight
        alpha_weight = alpha * targets + (1 - alpha) * (1 - targets)
        focal_loss = alpha_weight * focal_weight * bce_loss
        return focal_loss

class RepresentationDecoder(nn.Module):
    """
    Representation Decoder: 输入decoder emb, feature emb, aux query（VAD），pos emb，输出speaker emb
    """
    def __init__(self, d_model, d_feat, nhead, d_ff, num_layers, d_aux, d_pos, speaker_embed_dim):
        super().__init__()
        self.input_proj = nn.Linear(d_feat, d_model)  # [B, T, F] -> [B, T, D]
        self.xdec_proj = nn.Linear(1, d_model)        # [B, N, 1] -> [B, N, D]
        self.qaux_proj = nn.Linear(1, d_aux)          # [B, N, 1] -> [B, N, d_aux]
        self.layers = nn.ModuleList([
            SWDecoderBlockV2(d_model, nhead, d_ff, d_aux=d_aux, d_pos=d_pos)
            for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(d_model, speaker_embed_dim)

    def forward(self, x_dec, x_fea, q_aux, k_pos):
        # x_dec: [B, N, T]
        # x_fea: [B, T, F]
        # q_aux: [B, N, T]
        # k_pos: [B, T, D_pos]
        x_fea = self.input_proj(x_fea)  # [B, T, D]
        x_dec_pooled = x_dec.mean(-1, keepdim=True)  # [B, N, 1]
        x_dec_proj = self.xdec_proj(x_dec_pooled)    # [B, N, D]
        q_aux_pooled = q_aux.mean(-1, keepdim=True)  # [B, N, 1]
        q_aux_proj = self.qaux_proj(q_aux_pooled)    # [B, N, d_aux]
        for layer in self.layers:
            x_dec_proj = layer(x_dec_proj, x_fea, q_aux_proj, k_pos)
        out = self.out_proj(x_dec_proj)  # [B, N, speaker_embed_dim]
        return out

# 5. SSND整体模型，集成Conformer
class SSNDModel(nn.Module):
    def __init__(
        self,
        speaker_pretrain_model_path,
        extractor_model_type='CAM++_wo_gsp',
        feat_dim=80,
        emb_dim=256, # speaker embedding dim and hidden_dim
        q_det_aux_dim=256, # query dim, in detection decoder, it is speaker embedding dim,
        q_rep_aux_dim=256,#  in representation decoder, it is vad lenght
        d_model=256,
        nhead=8,
        d_ff=512,
        num_layers=4,
        max_speakers=30,
        vad_out_len=100,  # 由block/chunk长度决定
        #arcface_num_classes=None,  # 说话人总数
        arcface_margin=0.2,
        arcface_scale=32.0,
        pos_emb_dim=256,   # 新增：位置编码维度
        max_seq_len=1000,  # 新增：最大帧数
        n_all_speakers=1000, # 训练集说话人总数
        mask_prob=0.5,      # mask概率
        training=True,      # 是否训练模式
        device=torch.device("cpu"),
        mask_prob_warmup=0.8,   # 新增：训练初期mask概率
        mask_prob_warmup_epochs=3,  # 新增：前多少个epoch用高mask概率
        out_bias=-0.5,        
    ):
        super().__init__()
        self.extractor = ResNetExtractor(device,speaker_pretrain_model_path, in_dim=feat_dim, out_dim=emb_dim, extractor_model_type=extractor_model_type)
        self.encoder = SSNDConformerEncoder(emb_dim, d_model, num_layers, nhead, d_ff)
        self.det_decoder = DetectionDecoder(d_model, nhead, d_ff, num_layers, q_det_aux_dim, pos_emb_dim, vad_out_len, out_bias=out_bias)
        self.rep_decoder = RepresentationDecoder(d_model,emb_dim, nhead, d_ff, num_layers, q_rep_aux_dim, pos_emb_dim,emb_dim)
        self.d_model = d_model
        self.max_speakers = max_speakers
        self.emb_dim = emb_dim
        self.pos_emb_dim = pos_emb_dim
        self.max_seq_len = max_seq_len
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, pos_emb_dim))
        self.n_all_speakers = n_all_speakers
        self.mask_prob = mask_prob
        self.mask_prob_warmup = mask_prob_warmup
        self.mask_prob_warmup_epochs = mask_prob_warmup_epochs
        self.cur_epoch = 0  # 训练脚本每个epoch前要设置
        self.training_mode = training
        # 可学习全体说话人embedding矩阵 E_all [N_all, S] - 使用更好的初始化
        self.E_all = nn.Parameter(torch.randn(n_all_speakers, emb_dim) )
        # 可学习伪说话人embedding e_pse [1, S]
        self.e_pse = nn.Parameter(torch.randn(1, emb_dim) )
        # 可学习non-speech embedding e_non [1, S]
        self.e_non = nn.Parameter(torch.randn(1, emb_dim) )
        # 可学习DetectionDecoder的query embedding [N, D]
        self.det_query_emb = nn.Parameter(torch.randn(max_speakers, d_model) )
        # 可学习RepresentationDecoder的query embedding [N, T_max]
        self.rep_query_emb = nn.Parameter(torch.randn(max_speakers, vad_out_len) )
        # 保存 arcface margin/scale
        self.arcface_margin = arcface_margin
        self.arcface_scale = arcface_scale
        self.gradient_checkpointing = False
        # 移除可学习的loss权重，使用固定权重
        # self.log_s_bce = nn.Parameter(torch.tensor(0.0))  # exp(-0.0) = 1.0, BCE weight = 1.0
        # self.log_s_arcface = nn.Parameter(torch.tensor(0.0))  


    def compute_arcface_loss(self, spk_emb_pred, spk_labels):
        """
        改进的 ArcFace loss 实现，添加数值稳定性和梯度裁剪。
        spk_emb_pred: [num_valid, emb_dim]
        spk_labels: [num_valid]
        """
        # 添加梯度裁剪防止梯度爆炸
        spk_emb_pred = torch.clamp(spk_emb_pred, -10, 10)
        
        spk_emb_norm = F.normalize(spk_emb_pred, p=2, dim=-1)  # [num_valid, emb_dim]
        E_all_norm = F.normalize(self.E_all, p=2, dim=-1)      # [n_all_speakers, emb_dim]
        
        # 更安全的clamp范围
        logits = torch.matmul(spk_emb_norm, E_all_norm.t()).clamp(-0.9999, 0.9999)  # [num_valid, n_all_speakers]
        labels = spk_labels  # [num_valid]
        margin = self.arcface_margin
        scale = self.arcface_scale
        
        # 先算 theta
        theta = torch.acos(logits)
        # 对正类加 margin
        theta_m = theta.clone()
        theta_m[torch.arange(theta.size(0)), labels] += margin
        # 再转回 cos
        logits_arc = torch.cos(theta_m)
        # 乘 scale
        logits_arc = logits_arc * scale
        
        # 添加label smoothing提高稳定性
        loss = F.cross_entropy(logits_arc, labels, label_smoothing=0.05)  # 从0.1降到0.05
        
        # 减少正则化项，让模型更容易学习说话人表示
        embedding_norm_penalty = 0.001 * torch.mean(torch.norm(spk_emb_pred, p=2, dim=1))  # 从0.01降到0.001
        loss = loss + embedding_norm_penalty
        
        return loss
    

    def focal_bce_loss(self, logits, targets, alpha=0.75, gamma=2.0):
        """
        Focal loss for binary classification to handle class imbalance.
        logits: [B, N, T]
        targets: [B, N, T]
        """
        # 计算sigmoid概率
        probs = torch.sigmoid(logits)
        # 计算focal loss
        pt = probs * targets + (1 - probs) * (1 - targets)  # p_t
        focal_weight = (1 - pt) ** gamma
        # 计算BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        # 应用focal weight和alpha weight
        alpha_weight = alpha * targets + (1 - alpha) * (1 - targets)
        focal_loss = alpha_weight * focal_weight * bce_loss
        return focal_loss
    
    """
    def focal_bce_loss(self, logits, targets, alpha=0.25, gamma=3.0):
        """
        Focal loss for binary classification to handle class imbalance.
        logits: [B, N, T]
        targets: [B, N, T]
        """
        # 计算sigmoid概率
        probs = torch.sigmoid(logits)
        
        # 计算focal loss
        pt = probs * targets + (1 - probs) * (1 - targets)  # p_t
        focal_weight = (1 - pt) ** gamma
        
        # 计算BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # 应用focal weight和alpha weight
        alpha_weight = alpha * targets + (1 - alpha) * (1 - targets)
        focal_loss = alpha_weight * focal_weight * bce_loss
        
        # 增强概率正则化，强烈鼓励模型输出更极端的概率
        prob_entropy = -probs * torch.log(probs + 1e-8) - (1 - probs) * torch.log(1 - probs + 1e-8)
        entropy_penalty = 0.5 * prob_entropy  # 增加熵惩罚权重
        
        # 添加额外的极端概率奖励
        extreme_reward = 0.2 * torch.exp(-10 * (probs - 0.5).abs())  # 奖励接近0或1的预测
        
        return focal_loss + entropy_penalty - extreme_reward
    """
    def print_loss_grad_norms(self, bce_loss, arcface_loss):
        """
        分析BCE loss和ArcFace loss对参数的梯度主导作用，并打印梯度范数。
        """
        # 1. BCE loss对参数的梯度范数
        self.zero_grad()
        bce_loss.backward(retain_graph=True)
        bce_grad_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                bce_grad_norm += p.grad.norm(2).item() ** 2
        bce_grad_norm = bce_grad_norm ** 0.5
        self.zero_grad()
        # 2. ArcFace loss对参数的梯度范数
        if arcface_loss.requires_grad and arcface_loss != 0.0:
            arcface_loss.backward(retain_graph=True)
            arcface_grad_norm = 0.0
            for p in self.parameters():
                if p.grad is not None:
                    arcface_grad_norm += p.grad.norm(2).item() ** 2
            arcface_grad_norm = arcface_grad_norm ** 0.5
            self.zero_grad()
        else:
            arcface_grad_norm = 0.0
        print(f"[GRAD DIAG] BCE grad norm: {bce_grad_norm:.4f}, ArcFace grad norm: {arcface_grad_norm:.4f}")

    def forward(self, feats, spk_label_idx, vad_labels, spk_labels=None):
        """
        feats: [B, T, F]
        spk_label_idx: [B, N_loc]，每个block的说话人index（int，指向E_all）
        vad_labels: [B, N_loc, T]
        spk_labels: [B, N_loc]，可选，ArcFace用
        """
        B, T, _ = feats.shape
        N_loc = spk_label_idx.shape[1]
        device = feats.device
        
        # 1. 查表获得输入speaker embedding (one-hot lookup as described in the paper)
        spk_label_idx_safe = spk_label_idx.clone()
        unknown_mask = spk_label_idx < 0
        spk_label_idx_safe[unknown_mask] = 0 # Temporarily set to 0 for one-hot
        
        # One-hot lookup
        one_hot_vectors = F.one_hot(spk_label_idx_safe, num_classes=self.n_all_speakers).float()
        speaker_embs = torch.matmul(one_hot_vectors, self.E_all)

        # Overwrite unknown speakers with pseudo-speaker embedding
        if unknown_mask.any():
            speaker_embs[unknown_mask] = self.e_pse.to(speaker_embs.dtype)

        # 2. Mask策略（训练时，动态mask概率）
        mask_info = None
        if self.training_mode:
            # 动态mask概率
            if hasattr(self, 'cur_epoch'):
                mask_prob = self.mask_prob_warmup if self.cur_epoch < self.mask_prob_warmup_epochs else self.mask_prob
            else:
                mask_prob = self.mask_prob  # 兼容性
            mask_info = []
            for b in range(B):
                if N_loc > 0 and torch.rand(1).item() < mask_prob:
                    mask_idx = torch.randint(0, N_loc, (1,)).item()
                    speaker_embs[b, mask_idx] = self.e_pse.to(speaker_embs.dtype)
                    # VAD标签不变
                    mask_info.append((b, mask_idx, spk_label_idx[b, mask_idx].item()))
        # 3. Padding策略（严格按论文50% e_non, 50%未出现说话人embedding）
        N = self.max_speakers
        if N_loc < N:
            pad_num = N - N_loc
            pad_embs = []
            pad_vad = []
            pad_idx = []
            for b in range(B):
                cur_spk = set(spk_label_idx[b].tolist())
                all_spk = set(range(self.n_all_speakers))
                unused_spk = list(all_spk - cur_spk)
                for i in range(pad_num):
                    if torch.rand(1).item() < 0.5 and len(unused_spk) > 0:
                        rand_spk = np.random.choice(unused_spk)
                        pad_embs.append(self.E_all[rand_spk].to(speaker_embs.dtype))  # shape [emb_dim]
                        pad_idx.append(rand_spk)
                    else:
                        pad_embs.append(self.e_non[0].to(speaker_embs.dtype))  # shape [emb_dim]
                        pad_idx.append(-1)
                    pad_vad.append(torch.zeros(vad_labels.shape[2], device=device))
            pad_embs = torch.stack(pad_embs).reshape(B, pad_num, self.emb_dim)
            pad_vad = torch.stack(pad_vad).reshape(B, pad_num, vad_labels.shape[2])
            pad_idx = torch.tensor(pad_idx, dtype=spk_label_idx.dtype, device=device).reshape(B, pad_num)
            speaker_embs = torch.cat([speaker_embs, pad_embs], dim=1)  # [B, N, S]
            vad_labels = torch.cat([vad_labels, pad_vad], dim=1)       # [B, N, T]
            spk_label_idx = torch.cat([spk_label_idx, pad_idx], dim=1)
            if spk_labels is not None:
                pad_labels = torch.full((B, pad_num), -1, dtype=spk_labels.dtype, device=device)
                spk_labels = torch.cat([spk_labels, pad_labels], dim=1)
        elif N_loc > N:
            speaker_embs = speaker_embs[:, :N, :]
            vad_labels = vad_labels[:, :N, :]
            if spk_labels is not None:
                spk_labels = spk_labels[:, :N]
        # 4. 特征提取
        x = self.extractor(feats)  # [B, T, emb_dim]
        assert vad_labels.shape[-1] == x.shape[1], f"Label/feature length mismatch: {vad_labels.shape[-1]} vs {x.shape[1]}"
        enc_out = self.encoder(x)  # [B, T, d_model]
        _, T_enc, _ = enc_out.shape
        pos_emb = self.pos_emb[:, :T_enc, :].expand(B, T_enc, self.pos_emb_dim)  # [B, T, D_pos]
        # 5. DetectionDecoder
        # x_dec:[B,N,D],use x_det_dec, it is setting to 0, it applys on query
        # x_fea:[B,T,D], it is ouput of encoder and it applys on value
        # q_aux:[B,N,D'], it is speaker embedding and it applys on query
        # pos_emb:[B,T,D'], it is postion embedding and it applys on key
        #print(f"speaker_embs shape: {speaker_embs.shape}")
        # 使用可学习的query embedding而不是全零向量
        x_det_dec = self.det_query_emb.unsqueeze(0).expand(B, N, self.d_model)
        #print(f"x_det_dec shape: {x_det_dec.shape}, enc_out shape: {enc_out.shape}")
        #print(f"speaker_embs shape: {speaker_embs.shape}, pos_emb shape: {pos_emb.shape}")
        # x_det_dec shape: torch.Size([16, 30, 256]), enc_out shape: torch.Size([16, 200, 256])
        # speaker_embs shape: torch.Size([16, 30, 256]), pos_emb shape: torch.Size([16, 200, 256])
        vad_pred = self.det_decoder(x_det_dec, enc_out, speaker_embs, pos_emb)  # [B, N, T]
        # 6. RepresentationDecoder (训练时q_aux用vad_labels)
        # x_dec:[B,N,T'] , use x_rep_dec, it is setting to 0, it applys on query
        # x_fea:[B,T,F], it is output of extrator and it applys on value
        # q_aux:[B,N,T'], it is vad ground-truth label and it applys on query
        # pos_emb:[B,T,F'] ,its length is equal to x_fea and it applys on key
        # 使用可学习的query embedding而不是全零向量
        x_rep_dec = self.rep_query_emb.unsqueeze(0).expand(B, N, vad_labels.shape[2])
        #
        #print(f"x_rep_dec shape: {x_rep_dec.shape}, x shape: {x.shape}")
        #print(f"vad_labels shape: {vad_labels.shape}, pos_emb shape: {pos_emb.shape}")
        # torch.Size([16, 30, 200]), pos_emb shape: torch.Size([16, 200, 256])
        spk_emb_pred = self.rep_decoder(x_rep_dec, x, vad_labels, pos_emb)      # [B, N, S]
        # 7. 损失
        # Clamp VAD predictions for numerical stability
        vad_pred = torch.clamp(vad_pred, -15, 15)
        # Focal loss with mask
        valid_mask = (spk_label_idx >= 0).unsqueeze(-1)  # [B, N, 1]
        focal_loss = self.focal_bce_loss(vad_pred, vad_labels, alpha=0.25, gamma=1.0)
        bce_loss = (focal_loss * valid_mask).sum() / valid_mask.sum()
        
        # ArcFace loss（只对有效说话人）- 进一步降低权重
        arcface_weight = 0.5  # 进一步降低ArcFace权重
        arcface_loss = torch.tensor(0.0, device=device)
        if spk_labels is not None and arcface_weight > 0.0:
            valid = (spk_labels >= 0)
            vad_label_active = vad_labels.sum(dim=2) > 0  # [B, N]
            valid = valid & vad_label_active
            if valid.sum() > 0:
                arcface_loss = self.compute_arcface_loss(spk_emb_pred[valid], spk_labels[valid])
                arcface_loss = arcface_loss * arcface_weight
        
        # 使用固定的loss权重，让BCE主导训练
        loss = 2.0 * bce_loss + arcface_loss  # 增加BCE权重
        l2_reg = 0.0005 * sum(p.norm(2) for p in self.parameters() if p.requires_grad)  # 减少L2正则化
        loss = loss + l2_reg

        # ========== 新增：分析各loss对参数的主导作用 ==========
        if self.training and loss.requires_grad:
            self.print_loss_grad_norms(bce_loss, arcface_loss)
        # ========== END ==========

        # 添加调试信息
        with torch.no_grad():
            vad_probs = torch.sigmoid(vad_pred)
            pred_positive_ratio = vad_probs.mean().item()
            true_positive_ratio = vad_labels.mean().item()
            print(f"DEBUG - True positive ratio: {true_positive_ratio:.4f}, Pred positive ratio: {pred_positive_ratio:.4f}")
            print(f"DEBUG - BCE loss: {bce_loss.item():.4f}, ArcFace loss: {arcface_loss.item():.4f}")
            print(f"DEBUG - Total loss: {loss.item():.4f}")
            # 添加VAD预测分布分析
            print(f"DEBUG - VAD probs stats: mean={vad_probs.mean().item():.4f}, std={vad_probs.std().item():.4f}")
            print(f"DEBUG - VAD probs range: [{vad_probs.min().item():.4f}, {vad_probs.max().item():.4f}]")
            # 分析预测的极端性
            extreme_preds = ((vad_probs > 0.8) | (vad_probs < 0.2)).float().mean().item()
            print(f"DEBUG - Extreme predictions ratio: {extreme_preds:.4f}")
            # 添加更详细的分布分析
            very_low = (vad_probs < 0.1).float().mean().item()
            low = ((vad_probs >= 0.1) & (vad_probs < 0.3)).float().mean().item()
            mid = ((vad_probs >= 0.3) & (vad_probs < 0.7)).float().mean().item()
            high = ((vad_probs >= 0.7) & (vad_probs < 0.9)).float().mean().item()
            very_high = (vad_probs >= 0.9).float().mean().item()
            print(f"DEBUG - VAD probs distribution: <0.1:{very_low:.4f}, 0.1-0.3:{low:.4f}, 0.3-0.7:{mid:.4f}, 0.7-0.9:{high:.4f}, >0.9:{very_high:.4f}")
            
        print("labels.sum() / labels.numel():", vad_labels.sum().item() / vad_labels.numel())
        for n in range(vad_labels.shape[1]):
            print(f"Channel {n} positive ratio:", vad_labels[0, n, :].sum().item() / vad_labels.shape[2])
        
        print("spk_ids_list[0]:", spk_label_idx[0])
        print("spk_label_idx[0]:", spk_label_idx[0])
        print("labels[0]:", vad_labels[0])
        return vad_pred, spk_emb_pred, loss, bce_loss, arcface_loss, mask_info, vad_labels
    
    def infer(self, feats, speaker_embs):
        """
        SSND block-wise推理接口。
        输入：
            feats: [1, T, F]，单个block的音频特征
            speaker_embs: [1, N, S]，当前buffer中的说话人embedding
        输出：
            vad_pred: [1, N, T']，每个说话人每帧的VAD概率
            emb_pred: [1, N, S]，每个说话人的新embedding
        """
        x = self.extractor(feats)              # [1, T, emb_dim]
        enc_out = self.encoder(x)              # [1, T, d_model]
        B, T, _ = enc_out.shape
        _,N,S = speaker_embs.shape
        device = feats.device
        pos_emb = self.pos_emb[:, :T, :].expand(B, T, self.pos_emb_dim)  # [1, T, D_pos]
        x_det_dec = self.det_query_emb.unsqueeze(0).expand(B, N, self.d_model)
        x_rep_dec = self.rep_query_emb.unsqueeze(0).expand(B, N, T)
        
        # 修正推理逻辑：DetectionDecoder也需要推理
        vad_pred = self.det_decoder(x_det_dec, enc_out, speaker_embs, pos_emb)
        # RepresentationDecoder的q_aux在推理时应该用vad_pred（sigmoid后）
        vad_prob_for_rep = torch.sigmoid(vad_pred)
        emb_pred = self.rep_decoder(x_rep_dec, x, vad_prob_for_rep, pos_emb)          # [1, N, S]
        return vad_pred, emb_pred 

    def offline_diarization(self, feats, threshold=0.5):
        """
        离线推理接口，输入整段音频特征，输出每帧说话人标签。
        feats: [T, F] or [1, T, F]
        threshold: VAD概率阈值
        返回:
            diarization_result: [N, T] 0/1矩阵，N为最大说话人数
            vad_prob: [N, T] 概率
        """
        self.eval()
        with torch.no_grad():
            if feats.ndim == 2:
                feats = feats.unsqueeze(0)  # [1, T, F]
            B, T, F = feats.shape
            N = self.max_speakers
            device = feats.device
            # 用E_all前N个作为初始speaker_embs
            speaker_embs = self.E_all[:N].unsqueeze(0).expand(B, N, self.emb_dim).to(device)
            # 推理
            vad_pred, spk_emb_pred = self.infer(feats, speaker_embs)
            vad_prob = torch.sigmoid(vad_pred)  # [1, N, T]
            diarization_result = (vad_prob > threshold).long().squeeze(0)  # [N, T]
            return diarization_result, vad_prob.squeeze(0) 
    @torch.no_grad()
    def online_infer(
        self,
        blocks,# 8s
        l_c,# 0.64s
        l_r,# 0.16s
        t1=0.5,
        t2=0.5,
        device=None
    ):
        """
        SSND在线推理接口，严格按照论文伪代码和流程图实现。
        blocks: List[Tensor] or List[np.ndarray], 每块 shape [chunk_size, feat_dim]
        l_c: 当前块输出帧数
        l_r: 右上下文帧数
        t1: 伪说话人embedding权重阈值
        t2: 注册说话人embedding权重阈值
        返回:
            dia_result: dict, {spk_id: [帧标签]}
        """
        dia_result = {}  # {spk_id: [帧标签]}
        emb_buffer = {}  # {spk_id: [(embedding, weight), ...]}
        num_frames = 0
        S = self.emb_dim
        N = self.max_speakers
        e_pse = self.e_pse.squeeze(0)  # [S]
        e_non = self.e_non.squeeze(0)  # [S]
        if device is None:
            device = e_pse.device
        # 初始化伪说话人id
        pse_id = 0
        for block in blocks:
            if not torch.is_tensor(block):
                block = torch.tensor(block, dtype=torch.float32, device=device)
            T, F = block.shape
            # 1. 构造 emb_list, spk_list
            emb_list = [e_pse]
            spk_list = [pse_id]
            # 2. 已注册说话人embedding加权平均
            for spk_id in emb_buffer.keys():
                e_sum = torch.zeros(S, device=device)
                w_sum = 0.0
                for e, w in emb_buffer[spk_id]:
                    e_sum += e * w
                    w_sum += w
                if w_sum > 0:
                    e_mean = e_sum / w_sum
                else:
                    e_mean = e_non
                emb_list.append(e_mean)
                spk_list.append(spk_id)
            # 3. pad 到 N
            while len(emb_list) < N:
                emb_list.append(e_non)
                spk_list.append(-1)
            emb_tensor = torch.stack(emb_list)  # [N, S]
            emb_tensor = emb_tensor.unsqueeze(0)  # [1, N, S]
            # 4. 前向推理
            block = block.unsqueeze(0)  # [1, T, F]
            vad_pred, spk_emb_pred = self.infer(block, emb_tensor)
            vad_prob = torch.sigmoid(vad_pred)[0]  # [N, T]
            # 5. 伪说话人处理
            y_pse = vad_prob[0]  # [T]
            e_pse_new = spk_emb_pred[0, 0]  # [S]
            v_pse = y_pse.mean().item()  # embedding weight
            # 只保留当前块的目标帧
            print(f"l_c: {l_c}, l_r: {l_r}")
            print(f"y_pse shape: {y_pse.shape}")
            print(f"y_pse[-(l_c+l_r):-l_r]: {y_pse[-(l_c+l_r):-l_r]}")
            current_y = y_pse[-(l_c + l_r):-l_r] if l_r > 0 else y_pse[-l_c:]
            # 拼接输出
            if pse_id not in dia_result:
                dia_result[pse_id] = torch.zeros(num_frames, device=device)
            dia_result[pse_id] = torch.cat([dia_result[pse_id], current_y.cpu()])
            # 更新buffer
            if v_pse > t1:
                emb_buffer[pse_id] = [(e_pse_new.detach(), v_pse)]
            # 6. 已注册说话人处理
            for n in range(1, N):
                if spk_list[n] == -1:
                    continue
                y_n = vad_prob[n]  # [T]
                e_n = spk_emb_pred[0, n]  # [S]
                v_n = y_n.mean().item()
                current_y_n = y_n[-(l_c + l_r):-l_r] if l_r > 0 else y_n[-l_c:]
                if spk_list[n] not in dia_result:
                    dia_result[spk_list[n]] = torch.zeros(num_frames, device=device)
                dia_result[spk_list[n]] = torch.cat([dia_result[spk_list[n]], current_y_n.cpu()])
                if v_n > t2:
                    if spk_list[n] not in emb_buffer:
                        emb_buffer[spk_list[n]] = []
                    emb_buffer[spk_list[n]].append((e_n.detach(), v_n))
            num_frames += l_c
        # 转为numpy输出
        for k in dia_result:
            dia_result[k] = dia_result[k].cpu().numpy()
        return dia_result
    
    def offline_rescore(self,
        blocks,
        l_c,
        l_r,
        t1=0.5,
        t2=0.5,
        threshold=0.5,
        device=None
    ):
        """
        论文式offline推理：先online收集全局emb_buffer，再re-decode。
        blocks: List[Tensor] or List[np.ndarray]
        l_c, l_r, t1, t2: 同online_infer
        threshold: VAD概率阈值
        返回:
            diarization_result: [N, T] 0/1矩阵
        """
        # 1. 先跑online_infer，收集全局emb_buffer
        dia_result = {}  # 不用
        emb_buffer = {}  # {spk_id: [(embedding, weight), ...]}
        num_frames = 0
        S = self.emb_dim
        N = self.max_speakers
        e_pse = self.e_pse.squeeze(0)
        e_non = self.e_non.squeeze(0)
        if device is None:
            device = e_pse.device
        pse_id = 0
        # 先收集全局buffer
        for block in blocks:
            if not torch.is_tensor(block):
                block = torch.tensor(block, dtype=torch.float32, device=device)
            T, F = block.shape
            emb_list = [e_pse]
            spk_list = [pse_id]
            for spk_id in emb_buffer.keys():
                e_sum = torch.zeros(S, device=device)
                w_sum = 0.0
                for e, w in emb_buffer[spk_id]:
                    e_sum += e * w
                    w_sum += w
                if w_sum > 0:
                    e_mean = e_sum / w_sum
                else:
                    e_mean = e_non
                emb_list.append(e_mean)
                spk_list.append(spk_id)
            while len(emb_list) < N:
                emb_list.append(e_non)
                spk_list.append(-1)
            emb_tensor = torch.stack(emb_list).unsqueeze(0)
            block = block.unsqueeze(0)
            vad_pred, spk_emb_pred = self.infer(block, emb_tensor)
            vad_prob = torch.sigmoid(vad_pred)[0]
            # 伪说话人
            y_pse = vad_prob[0]
            e_pse_new = spk_emb_pred[0, 0]
            v_pse = y_pse.mean().item()
            if v_pse > t1:
                emb_buffer[pse_id] = [(e_pse_new.detach(), v_pse)]
            # 已注册说话人
            for n in range(1, N):
                if spk_list[n] == -1:
                    continue
                y_n = vad_prob[n]
                e_n = spk_emb_pred[0, n]
                v_n = y_n.mean().item()
                if v_n > t2:
                    if spk_list[n] not in emb_buffer:
                        emb_buffer[spk_list[n]] = []
                    emb_buffer[spk_list[n]].append((e_n.detach(), v_n))
        # 2. 构造全局embedding列表
        global_emb_list = [e_pse]
        global_spk_list = [pse_id]
        for spk_id in emb_buffer.keys():
            e_sum = torch.zeros(S, device=device)
            w_sum = 0.0
            for e, w in emb_buffer[spk_id]:
                e_sum += e * w
                w_sum += w
            if w_sum > 0:
                e_mean = e_sum / w_sum
            else:
                e_mean = e_non
            global_emb_list.append(e_mean)
            global_spk_list.append(spk_id)
        while len(global_emb_list) < N:
            global_emb_list.append(e_non)
            global_spk_list.append(-1)
        global_emb_tensor = torch.stack(global_emb_list).unsqueeze(0)  # [1, N, S]
        # 3. 用全局embedding对所有块重新推理
        diarization_result = []  # [N, T]
        for block in blocks:
            if not torch.is_tensor(block):
                block = torch.tensor(block, dtype=torch.float32, device=device)
            block = block.unsqueeze(0)
            vad_pred, _ = self.infer(block, global_emb_tensor)
            vad_prob = torch.sigmoid(vad_pred)[0]  # [N, T]
            diarization_result.append((vad_prob > threshold).long())
        diarization_result = torch.cat(diarization_result, dim=1)  # [N, T_total]
        return diarization_result.cpu().numpy(), global_spk_list
    
    def extract_sentence_timestamps_and_embeddings(
        self,
        feats,
        vad_threshold=0.5,
        min_speech_duration_ms=200,
        min_silence_duration_ms=500,
        frame_shift_ms=10,
    ):
        """
        Offline extracts sentence-level timestamps and speaker embeddings.
        A sentence is a continuous speech segment from a single speaker.

        Args:
            feats (Tensor): Input features [T, F] or [1, T, F].
            vad_threshold (float): VAD probability threshold.
            min_speech_duration_ms (int): Minimum duration for a speech segment.
            min_silence_duration_ms (int): Minimum duration of silence to break sentences.
            frame_shift_ms (int): Frame shift in milliseconds.

        Returns:
            list: A list of dicts, each with 'start_ms', 'end_ms', 'speaker_id', 'embedding'.
        """
        self.eval()
        if not torch.is_tensor(feats):
            feats = torch.tensor(feats, dtype=torch.float32)
        device = self.E_all.device
        feats = feats.to(device)

        min_speech_frames = min_speech_duration_ms // frame_shift_ms
        min_silence_frames = min_silence_duration_ms // frame_shift_ms

        with torch.no_grad():
            if feats.ndim == 2:
                feats = feats.unsqueeze(0)
            
            B, T, F = feats.shape
            N = self.max_speakers
            
            speaker_embs = self.E_all[:N].unsqueeze(0).expand(B, N, self.emb_dim).to(device)
            vad_pred, spk_emb_pred = self.infer(feats, speaker_embs)
            
            vad_prob = torch.sigmoid(vad_pred).squeeze(0)
            spk_emb_pred = spk_emb_pred.squeeze(0)

            max_probs, speaker_indices = torch.max(vad_prob, dim=0)
            speech_frames = (max_probs > vad_threshold).long()

            sentences = []
            in_speech = False
            start_frame = 0
            
            for i in range(T):
                if not in_speech and speech_frames[i]:
                    in_speech = True
                    start_frame = i
                elif in_speech and not speech_frames[i]:
                    if i - start_frame >= min_speech_frames:
                        # Find dominant speaker in the segment
                        dom_spk = torch.mode(speaker_indices[start_frame:i])[0].item()
                        sentences.append({'start': start_frame, 'end': i, 'spk': dom_spk})
                    in_speech = False
                elif in_speech and i > start_frame:
                    # Check for speaker change
                    current_speaker = speaker_indices[i]
                    prev_speaker = speaker_indices[i-1]
                    if current_speaker != prev_speaker:
                         if i - start_frame >= min_speech_frames:
                            dom_spk = torch.mode(speaker_indices[start_frame:i])[0].item()
                            sentences.append({'start': start_frame, 'end': i, 'spk': dom_spk})
                         start_frame = i

            if in_speech:
                if T - start_frame >= min_speech_frames:
                    dom_spk = torch.mode(speaker_indices[start_frame:T])[0].item()
                    sentences.append({'start': start_frame, 'end': T, 'spk': dom_spk})

            # Merge consecutive segments from the same speaker
            if not sentences:
                return []

            merged_sentences = [sentences[0]]
            for i in range(1, len(sentences)):
                last_sent = merged_sentences[-1]
                curr_sent = sentences[i]
                # Check for silence gap
                silence_gap = curr_sent['start'] - last_sent['end']
                if curr_sent['spk'] == last_sent['spk'] and silence_gap < min_silence_frames:
                    merged_sentences[-1]['end'] = curr_sent['end']
                else:
                    merged_sentences.append(curr_sent)
            
            results = []
            for sent in merged_sentences:
                spk_id = sent['spk']
                results.append({
                    'start_ms': sent['start'] * frame_shift_ms,
                    'end_ms': sent['end'] * frame_shift_ms,
                    'speaker_id': spk_id,
                    'embedding': spk_emb_pred[spk_id].cpu().numpy()
                })

            return results
    
    def get_sentences_from_diarization_output(
        self,
        vad_prob,
        spk_embeddings,
        vad_threshold=0.5,
        min_speech_duration_ms=200,
        min_silence_duration_ms=500,
        frame_shift_ms=10,
    ):
        """
        Extracts sentence-level timestamps and speaker embeddings from diarization model output.
        This is a post-processing function that operates on model predictions.

        Args:
            vad_prob (Tensor): VAD probabilities for each speaker [N, T].
            spk_embeddings (Tensor): Speaker embeddings for each speaker [N, S].
            vad_threshold (float): VAD probability threshold.
            min_speech_duration_ms (int): Minimum duration for a speech segment.
            min_silence_duration_ms (int): Minimum duration of silence to break sentences.
            frame_shift_ms (int): Frame shift in milliseconds.

        Returns:
            list: A list of dicts, each with 'start_ms', 'end_ms', 'speaker_id', 'embedding'.
        """
        if not torch.is_tensor(vad_prob):
            vad_prob = torch.from_numpy(vad_prob)
        if not torch.is_tensor(spk_embeddings):
            spk_embeddings = torch.from_numpy(spk_embeddings)
        
        device = spk_embeddings.device
        vad_prob = vad_prob.to(device)

        min_speech_frames = min_speech_duration_ms // frame_shift_ms
        min_silence_frames = min_silence_duration_ms // frame_shift_ms
        
        N, T = vad_prob.shape

        max_probs, speaker_indices = torch.max(vad_prob, dim=0)
        speech_frames = (max_probs > vad_threshold).long()

        sentences = []
        in_speech = False
        start_frame = 0
        
        for t in range(T):
            is_speech_now = speech_frames[t] == 1
            if in_speech:
                speaker_changed = t > 0 and speaker_indices[t] != speaker_indices[t-1]
                if not is_speech_now or speaker_changed:
                    if t - start_frame >= min_speech_frames:
                        dom_spk = torch.mode(speaker_indices[start_frame:t])[0].item()
                        sentences.append({'start': start_frame, 'end': t, 'spk': dom_spk})
                    if is_speech_now and speaker_changed:
                        start_frame = t
                    else:
                        in_speech = False
            if not in_speech and is_speech_now:
                in_speech = True
                start_frame = t

        if in_speech and T - start_frame >= min_speech_frames:
            dom_spk = torch.mode(speaker_indices[start_frame:T])[0].item()
            sentences.append({'start': start_frame, 'end': T, 'spk': dom_spk})

        if not sentences:
            return []

        merged_sentences = [sentences[0]]
        for i in range(1, len(sentences)):
            last_sent = merged_sentences[-1]
            curr_sent = sentences[i]
            silence_gap = curr_sent['start'] - last_sent['end']
            if curr_sent['spk'] == last_sent['spk'] and silence_gap < min_silence_frames:
                merged_sentences[-1]['end'] = curr_sent['end']
            else:
                merged_sentences.append(curr_sent)
        
        results = []
        for sent in merged_sentences:
            spk_id = sent['spk']
            results.append({
                'start_ms': sent['start'] * frame_shift_ms,
                'end_ms': sent['end'] * frame_shift_ms,
                'speaker_id': spk_id,
                'embedding': spk_embeddings[spk_id].cpu().numpy()
            })

        return results
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """                                                                                                                                                                                               Activates gradient checkpointing for the current model.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".

        We pass the `__call__` method of the modules instead of `forward` because `__call__` attaches all the hooks of
        the module. https://discuss.pytorch.org/t/any-different-between-model-input-and-model-forward-input/3690/2

        Args:
            gradient_checkpointing_kwargs (dict, *optional*):
                Additional keyword arguments passed along to the `torch.utils.checkpoint.checkpoint` function.
        """
        # if not self.supports_gradient_checkpointing:
        #    raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")

        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {"use_reentrant": True}
        import functools

        gradient_checkpointing_func = functools.partial(
            checkpoint, **gradient_checkpointing_kwargs
        )

        # For old GC format (transformers < 4.35.0) for models that live on the Hub
        # we will fall back to the overwritten `_set_gradient_checkpointing` method
        # _is_using_old_format = "value" in inspect.signature(self._set_gradient_checkpointing).parameters

        # if not _is_using_old_format:
        self._set_gradient_checkpointing(
            enable=True, gradient_checkpointing_func=gradient_checkpointing_func
        )
    def _set_gradient_checkpointing(
        self, enable: bool = True, gradient_checkpointing_func: Callable = checkpoint
    ):
        is_gradient_checkpointing_set = False

        # Apply it on the top-level module in case the top-level modules supports it
        # for example, LongT5Stack inherits from `PreTrainedModel`.
        if hasattr(self, "gradient_checkpointing"):
            self._gradient_checkpointing_func = gradient_checkpointing_func
            self.gradient_checkpointing = enable
            is_gradient_checkpointing_set = True

        for module in self.modules():
            if hasattr(module, "gradient_checkpointing"):
                module._gradient_checkpointing_func = gradient_checkpointing_func
                module.gradient_checkpointing = enable
                is_gradient_checkpointing_set = True

        if not is_gradient_checkpointing_set:
            raise ValueError(
                f"{self.__class__.__name__} is not compatible with gradient checkpointing. Make sure all the architecture support it by setting a boolean attribute"
                " `gradient_checkpointing` to modules of the model that uses checkpointing."
            )
    
    def calc_diarization_error(self, outs_prob, labels, labels_len):
        """
        计算说话人分离的相关指标，包括DER、FA、MI、CF、ACC。
        outs_prob: [B, N, T]，概率输出
        labels: [B, N, T]，标签
        labels_len: [B]，每个batch的有效长度
        """
        import numpy as np
        batch_size, n_spk, max_len = labels.shape
        # mask padding部分
        mask = np.zeros((batch_size, n_spk, max_len))
        for i in range(batch_size):
            mask[i, :, :labels_len[i]] = 1
        label_np = labels.data.cpu().numpy().astype(int)
        pred_np = (outs_prob > 0.5).astype(int)
        label_np = label_np * mask
        pred_np = pred_np * mask
        length = labels_len.data.cpu().numpy()
        num_frames = np.sum(length)  # 总帧数
        ref_speech = np.sum(label_np, axis=1)  # [B, T]
        sys_speech = np.sum(pred_np, axis=1)   # [B, T]
        miss = np.sum((ref_speech > 0) & (sys_speech == 0))
        fa = np.sum((ref_speech == 0) & (sys_speech > 0))
        # old 
        n_map = np.sum(np.logical_and(label_np == 1, pred_np == 1), axis=1)
        conf = float(np.sum(np.minimum(ref_speech, sys_speech) - n_map))
        # new
        #conf = np.sum(np.abs(ref_speech - sys_speech) * ((ref_speech > 0) & (sys_speech > 0)))
        correct = np.sum((label_np == pred_np) * mask) / n_spk
        return correct, num_frames, miss, fa, conf

    def calc_diarization_result(self, outs_prob, labels, labels_len):
        """
        计算DER等相关指标，返回mi, fa, cf, acc, der。
        """
        correct, num_frames, miss, fa, conf = self.calc_diarization_error(outs_prob, labels, labels_len)
        acc = correct / num_frames if num_frames > 0 else 0.0
        der = (miss + fa + conf) / num_frames if num_frames > 0 else 0.0
        return (
            miss / num_frames if num_frames > 0 else 0.0,
            fa / num_frames if num_frames > 0 else 0.0,
            conf / num_frames if num_frames > 0 else 0.0,
            acc,
            der
        )
    
