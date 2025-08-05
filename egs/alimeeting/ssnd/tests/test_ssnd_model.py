import torch
from ssnd_model import SSNDModel

def test_ssnd_model():
    # 配置参数
    B = 2                # batch size
    T = 128              # 帧数
    F = 80               # 特征维度
    N_loc = 3            # 当前block说话人数
    N_all = 10           # 训练集说话人总数
    N_max = 5            # 最大说话人数
    emb_dim = 32         # embedding维度
    pos_emb_dim = 32     # 位置编码维度

    # 构造输入
    feats = torch.randn(B, T, F)
    spk_label_idx = torch.randint(0, N_all, (B, N_loc))  # [B, N_loc]
    vad_labels = torch.randint(0, 2, (B, N_loc, T)).float()  # [B, N_loc, T]
    spk_labels = spk_label_idx.clone()  # 直接用index做类别标签

    # 实例化模型
    model = SSNDModel(
        feat_dim=F,
        emb_dim=emb_dim,
        d_model=emb_dim,
        nhead=4,
        d_ff=64,
        num_layers=2,
        max_speakers=N_max,
        vad_out_len=T,
        arcface_num_classes=N_all,
        pos_emb_dim=pos_emb_dim,
        max_seq_len=T,
        n_all_speakers=N_all,
        mask_prob=1.0,  # 强制mask，便于测试
        training=True,
    )

    # 前向传播
    vad_pred, spk_emb_pred, bce_loss, arcface_loss, mask_info = model(
        feats, spk_label_idx, vad_labels, spk_labels
    )

    # 检查输出shape
    print('vad_pred shape:', vad_pred.shape)           # [B, N_max, T]
    print('spk_emb_pred shape:', spk_emb_pred.shape)   # [B, N_max, emb_dim]
    print('bce_loss:', bce_loss.item())
    print('arcface_loss:', arcface_loss.item() if arcface_loss is not None else None)
    print('mask_info:', mask_info)

    # 检查反向传播
    (bce_loss + (arcface_loss if arcface_loss is not None else 0)).backward()
    print('Backward pass successful!')

if __name__ == '__main__':
    test_ssnd_model() 