import torch
import torch.optim as optim
from ssnd_model import SSNDModel
from alimeeting_diar_dataset import AlimeetingDiarDataset
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav-dir', type=str, required=True, help='Train wav dir')
    parser.add_argument('--textgrid-dir', type=str, required=True, help='Train textgrid dir')
    parser.add_argument('--speaker-pretrain-model-path', type=str, required=True)
    parser.add_argument('--extractor-model-type', type=str, default='CAM++_wo_gsp')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--max-speakers', type=int, default=4)
    parser.add_argument('--vad-out-len', type=int, default=100)
    parser.add_argument('--num-steps', type=int, default=1000)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 构造dataset和dataloader（无增强、无shuffle）
    dataset = AlimeetingDiarDataset(
        wav_dir=args.wav_dir,
        textgrid_dir=args.textgrid_dir,
        sample_rate=16000,
        frame_shift=0.04,
        musan_path=None,
        rir_path=None,
        noise_ratio=0.0,
        window_sec=8.0,
        window_shift_sec=0.4
    )
    def collate_fn_wrapper(batch):
        wavs, labels, spk_ids_list, fbanks, labels_len = dataset.collate_fn(batch, vad_out_len=args.vad_out_len)
        # 构造spk2int
        all_spk = set()
        for spk_ids in spk_ids_list:
            all_spk.update([s for s in spk_ids if s is not None])
        spk2int = {spk: i for i, spk in enumerate(sorted(list(all_spk)))}
        max_spks_in_batch = labels.shape[1]
        spk_label_indices = torch.full((len(spk_ids_list), max_spks_in_batch), -1, dtype=torch.long)
        for i, spk_id_sample in enumerate(spk_ids_list):
            for j, spk_id in enumerate(spk_id_sample):
                if spk_id and spk_id in spk2int:
                    spk_label_indices[i, j] = spk2int[spk_id]
        return fbanks, labels, spk_label_indices, labels_len
    from torch.utils.data import DataLoader
    train_dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_wrapper)

    # 取一个batch
    batch = next(iter(train_dl))
    fbanks, labels, spk_label_idx, labels_len = batch
    print(f"Batch shapes: fbanks {fbanks.shape}, labels {labels.shape}, spk_label_idx {spk_label_idx.shape}, labels_len {labels_len.shape}")
    fbanks = fbanks.to(device)
    labels = labels.to(device)
    spk_label_idx = spk_label_idx.to(device)
    labels_len = labels_len.to(device)

    # 构造模型
    n_all_speakers = int((spk_label_idx.max() + 1).item()) if (spk_label_idx >= 0).any() else args.max_speakers
    model = SSNDModel(
        speaker_pretrain_model_path=args.speaker_pretrain_model_path,
        extractor_model_type=args.extractor_model_type,
        feat_dim=80,
        emb_dim=256,
        q_det_aux_dim=256,
        q_rep_aux_dim=256,
        d_model=256,
        nhead=8,
        d_ff=512,
        num_layers=4,
        max_speakers=args.max_speakers,
        vad_out_len=args.vad_out_len,
        arcface_margin=0.2,
        arcface_scale=32.0,
        pos_emb_dim=256,
        max_seq_len=1000,
        n_all_speakers=n_all_speakers,
        mask_prob=0.0,
        training=True,
        device=device,
        out_bias=-0.5,
    ).to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for step in range(args.num_steps):
        optimizer.zero_grad()
        vad_pred, spk_emb_pred, loss, bce_loss, arcface_loss, mask_info, vad_labels = model(
            fbanks, spk_label_idx, labels, spk_labels=spk_label_idx
        )
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            vad_probs = torch.sigmoid(vad_pred)
            print(f"Step {step}: loss={loss.item():.4f}, bce_loss={bce_loss.item():.4f}, arcface_loss={arcface_loss.item():.4f}")
            print(f"vad_probs mean: {vad_probs.mean().item():.4f}, std: {vad_probs.std().item():.4f}")
            print(f"vad_labels mean: {vad_labels.mean().item():.4f}")
            for n in range(min(args.max_speakers, vad_probs.shape[1])):
                print(f"Speaker {n} pred>0.5 ratio: {(vad_probs[:, n, :] > 0.5).float().mean().item():.4f}")
                print(f"Speaker {n} label>0.5 ratio: {(vad_labels[:, n, :] > 0.5).float().mean().item():.4f}")
        if loss.item() < 1e-2:
            print("Loss very low, stop early.")
            break

if __name__ == "__main__":
    main() 