import torch
import torch.nn as nn
import torch.nn.functional as F
from wenet.transformer.encoder import ConformerEncoder  # 集成Conformer
from pytorch_metric_learning.losses import ArcFaceLoss  # 集成ArcFace

# 1. ResNet-based Extractor（简化版，可替换为ResNet34等）
class ResNetExtractor(nn.Module):
    def __init__(self, in_dim=80, out_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((None, 1))
        )
        self.proj = nn.Linear(64, out_dim)

    def forward(self, x):
        # x: [B, T, F]
        x = x.unsqueeze(1)  # [B, 1, T, F]
        x = self.conv(x)    # [B, 64, T, 1]
        x = x.squeeze(-1).transpose(1, 2)  # [B, T, 64]
        x = self.proj(x)    # [B, T, out_dim]
        return x

# 2. Conformer Encoder（集成wenet实现）
class SSNDConformerEncoder(nn.Module):
    def __init__(self, in_dim=256, d_model=256, num_layers=4, nhead=8, d_ff=512):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, d_model)
        self.encoder = ConformerEncoder(
            input_size=d_model,
            output_size=d_model,
            attention_heads=nhead,
            linear_units=d_ff,
            num_blocks=num_layers,
            dropout_rate=0.1,
            positional_dropout_rate=0.1,
            attention_dropout_rate=0.0,
            input_layer="linear",
            pos_enc_layer_type="rel_pos",
            normalize_before=True,
            cnn_module_kernel=15,
        )
    def forward(self, x):
        # x: [B, T, in_dim]
        x = self.input_proj(x)
        # ConformerEncoder需要xs, xs_lens
        xs_lens = torch.full((x.size(0),), x.size(1), dtype=torch.long, device=x.device)
        x, _ = self.encoder(x, xs_lens)  # [B, T, d_model]
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
        # Cross-attention: Q attends to K
        x2, _ = self.cross_attn(Q, K, K)
        x = self.norm1(x_dec + x2)
        # Self-attention
        x2, _ = self.self_attn(x, x, x)
        x = self.norm2(x + x2)
        # FFN
        x2 = self.ffn(x)
        x = self.norm3(x + x2)
        return x

class DetectionDecoder(nn.Module):
    """
    Detection Decoder: 输入decoder emb, feature emb, aux query（L2归一化），pos emb，输出VAD
    """
    def __init__(self, d_model, nhead, d_ff, num_layers, d_aux, d_pos, out_dim):
        super().__init__()
        self.layers = nn.ModuleList([
            SWDecoderBlockV2(d_model, nhead, d_ff, d_aux=d_aux, d_pos=d_pos)
            for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(d_model, out_dim)

    def forward(self, x_dec, x_fea, q_aux, k_pos):
        # L2 normalize auxiliary queries
        q_aux = F.normalize(q_aux, p=2, dim=-1)
        for layer in self.layers:
            x_dec = layer(x_dec, x_fea, q_aux, k_pos)
        out = self.out_proj(x_dec)  # [B, N, T]
        return out

class RepresentationDecoder(nn.Module):
    """
    Representation Decoder: 输入decoder emb, feature emb, aux query（VAD），pos emb，输出speaker emb
    """
    def __init__(self, d_model, d_feat, nhead, d_ff, num_layers, d_aux, d_pos, out_dim):
        super().__init__()
        self.input_proj = nn.Linear(d_feat, d_model)
        self.qaux_proj = nn.Linear(1, d_aux)  # 新增：将池化后的q_aux投影到d_aux
        self.layers = nn.ModuleList([
            SWDecoderBlockV2(d_model, nhead, d_ff, d_aux=d_aux, d_pos=d_pos)
            for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(d_model, out_dim)

    def forward(self, x_dec, x_fea, q_aux, k_pos):
        x_fea = self.input_proj(x_fea) # feat dim: F-> D
        # q_aux: [B, N, T] -> [B, N, 1] -> [B, N, d_aux]
        q_aux_pooled = q_aux.mean(-1, keepdim=True)
        q_aux_vec = self.qaux_proj(q_aux_pooled)
        for layer in self.layers:
            x_dec = layer(x_dec, x_fea, q_aux_vec, k_pos)
        out = self.out_proj(x_dec)  # [B, N, S]
        return out

# 5. SSND整体模型，集成Conformer和ArcFace
class SSNDModel(nn.Module):
    def __init__(
        self,
        feat_dim=80,
        emb_dim=256,
        d_model=256,
        nhead=8,
        d_ff=512,
        num_layers=4,
        max_speakers=30,
        vad_out_len=100,  # 由block/chunk长度决定
        arcface_num_classes=None,  # 说话人总数
        arcface_margin=0.2,
        arcface_scale=32.0,
        pos_emb_dim=256,   # 新增：位置编码维度
        max_seq_len=1000,  # 新增：最大帧数
        n_all_speakers=1000, # 训练集说话人总数
        mask_prob=0.5,      # mask概率
        training=True,      # 是否训练模式
    ):
        super().__init__()
        self.extractor = ResNetExtractor(feat_dim, emb_dim)
        self.encoder = SSNDConformerEncoder(emb_dim, d_model, num_layers, nhead, d_ff)
        self.det_decoder = DetectionDecoder(d_model, nhead, d_ff, num_layers, emb_dim, pos_emb_dim, vad_out_len)
        self.rep_decoder = RepresentationDecoder(d_model, emb_dim, nhead, d_ff, num_layers, emb_dim, pos_emb_dim, emb_dim)
        self.max_speakers = max_speakers
        self.emb_dim = emb_dim
        self.pos_emb_dim = pos_emb_dim
        self.max_seq_len = max_seq_len
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, pos_emb_dim))
        self.n_all_speakers = n_all_speakers
        self.mask_prob = mask_prob
        self.training_mode = training
        # 可学习全体说话人embedding矩阵 E_all [N_all, S]
        self.E_all = nn.Parameter(torch.randn(n_all_speakers, emb_dim))
        # 可学习伪说话人embedding e_pse [1, S]
        self.e_pse = nn.Parameter(torch.randn(1, emb_dim))
        # 可学习non-speech embedding e_non [1, S]
        self.e_non = nn.Parameter(torch.randn(1, emb_dim))
        # ArcFace损失
        self.arcface_loss = ArcFaceLoss(
            num_classes=n_all_speakers,
            embedding_size=emb_dim,
            margin=arcface_margin,
            scale=arcface_scale,
        )

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
        # 1. 查表获得输入speaker embedding
        spk_label_idx_safe = spk_label_idx.clone()
        spk_label_idx_safe[spk_label_idx_safe < 0] = 0
        speaker_embs = torch.matmul(
            F.one_hot(spk_label_idx_safe, num_classes=self.n_all_speakers).float(),
            self.E_all
        )  # [B, N_loc, S]
        # 2. Mask策略（训练时）
        mask_info = None
        if self.training_mode:
            mask_info = []
            for b in range(B):
                if torch.rand(1).item() < self.mask_prob and N_loc > 0:
                    mask_idx = torch.randint(0, N_loc, (1,)).item()
                    # 用e_pse替换
                    speaker_embs[b, mask_idx] = self.e_pse
                    # VAD标签分配给伪说话人
                    # 记录mask位置，后续loss用
                    mask_info.append((b, mask_idx, spk_label_idx[b, mask_idx].item()))
        # 3. Padding策略
        N = self.max_speakers
        if N_loc < N:
            pad_num = N - N_loc
            # 用e_non填充，VAD全0
            pad_embs = self.e_non.expand(B, pad_num, self.emb_dim)
            pad_vad = torch.zeros(B, pad_num, T, device=device)
            speaker_embs = torch.cat([speaker_embs, pad_embs], dim=1)  # [B, N, S]
            vad_labels = torch.cat([vad_labels, pad_vad], dim=1)       # [B, N, T]
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
        enc_out = self.encoder(x)  # [B, T, d_model]
        _, T_enc, _ = enc_out.shape
        pos_emb = self.pos_emb[:, :T_enc, :].expand(B, T_enc, self.pos_emb_dim)  # [B, T, D_pos]
        # 5. DetectionDecoder
        vad_pred = self.det_decoder(speaker_embs, enc_out, speaker_embs, pos_emb)  # [B, N, T]
        # 6. RepresentationDecoder (修正：x_dec用speaker_embs，q_aux用vad_pred)
        spk_emb_pred = self.rep_decoder(speaker_embs, enc_out, vad_pred, pos_emb)      # [B, N, S]
        # 7. 损失
        # BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(vad_pred, vad_labels, reduction='mean')
        # ArcFace loss（只对有效说话人）
        arcface_loss = None
        if spk_labels is not None:
            # mask掉填充的-1
            valid = (spk_labels >= 0)
            if valid.sum() > 0:
                arcface_loss = self.arcface_loss(
                    spk_emb_pred[valid],
                    spk_labels[valid]
                )
        return vad_pred, spk_emb_pred, bce_loss, arcface_loss, mask_info
    
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
        pos_emb = self.pos_emb[:, :T, :].expand(B, T, self.pos_emb_dim)  # [1, T, D_pos]
        vad_pred = self.det_decoder(speaker_embs, enc_out, speaker_embs, pos_emb)  # [1, N, T']
        emb_pred = self.rep_decoder(vad_pred, enc_out, vad_pred, pos_emb)          # [1, N, S]
        return vad_pred, emb_pred 