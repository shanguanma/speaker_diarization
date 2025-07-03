# test_toy_ssnd.py
import torch
from torch.utils.data import DataLoader
from ssnd_model import SSNDModel

# 1. Toy Dataset
class ToyDiarDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.fbanks = torch.ones(2, 10, 80)
        self.labels = torch.zeros(2, 2, 10)
        self.labels[0, 0, :] = 1
        self.labels[1, 1, :] = 1
        self.spk_label_idx = torch.tensor([[0, 1], [1, 0]])
        self.labels_len = torch.tensor([10, 10], dtype=torch.int32)
    def __len__(self): return 2
    def __getitem__(self, idx):
        return self.fbanks[idx], self.labels[idx], self.spk_label_idx[idx], self.labels_len[idx]

def toy_collate_fn(batch):
    fbanks = torch.stack([x[0] for x in batch])
    labels = torch.stack([x[1] for x in batch])
    spk_label_idx = torch.stack([x[2] for x in batch])
    labels_len = torch.stack([x[3] for x in batch])
    return fbanks, labels, spk_label_idx, labels_len

# 2. Model
model = SSNDModel(
    speaker_pretrain_model_path='/maduo/model_hub/speaker_pretrain_model/zh_cn/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin',
    extractor_model_type='CAM++_wo_gsp',
    feat_dim=80,
    emb_dim=256,
    q_det_aux_dim=256,
    q_rep_aux_dim=256,
    d_model=16,
    nhead=2,
    d_ff=32,
    num_layers=1,
    max_speakers=2,
    vad_out_len=10,
    arcface_margin=0.2,
    arcface_scale=10.0,
    pos_emb_dim=16,
    max_seq_len=10,
    n_all_speakers=2,
    mask_prob=0.0,
    training=True,
    out_bias=0.0,
)

# 3. DataLoader
dataset = ToyDiarDataset()
dataloader = DataLoader(dataset, batch_size=2, collate_fn=toy_collate_fn)

# 4. Forward & Print
for batch in dataloader:
    fbanks, labels, spk_label_idx, labels_len = batch
    vad_pred, spk_emb_pred, loss, bce_loss, arcface_loss, mask_info, padded_vad_labels = model(
        fbanks, spk_label_idx, labels, spk_labels=spk_label_idx
    )
    print("vad_pred:", vad_pred)
    print("labels:", labels)
    print("spk_label_idx:", spk_label_idx)
    print("loss:", loss)
    print("bce_loss:", bce_loss)
    print("arcface_loss:", arcface_loss)
    print("mask_info:", mask_info)
    print("padded_vad_labels:", padded_vad_labels)
    break

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
for epoch in range(10):
    for batch in dataloader:
        fbanks, labels, spk_label_idx, labels_len = batch
        vad_pred, spk_emb_pred, loss, bce_loss, arcface_loss, mask_info, padded_vad_labels = model(
            fbanks, spk_label_idx, labels, spk_labels=spk_label_idx
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, loss: {loss.item()}")
