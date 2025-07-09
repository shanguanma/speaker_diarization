import torch
import torch.optim as optim
from ssnd_model import SSNDModel
from typing import Any, Dict, Optional, Tuple, Union, List
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from alimeeting_diar_dataset import AlimeetingDiarDataset
import argparse
#from train_accelerate_ddp import compute_loss, compute_validation_loss

def compute_loss(
    model: Union[nn.Module, DDP],
    batch: tuple,
    is_training: bool,
    device: torch.device,
):
    """
    batch: (fbanks, labels, spk_label_idx, labels_len)
    """
    with torch.set_grad_enabled(is_training):
        fbanks, labels, spk_label_idx, labels_len = batch  # [B, T, F], [B, N, T], [B, N], [B]
        fbanks = fbanks.to(device)
        labels = labels.to(device)
        spk_label_idx = spk_label_idx.to(device)
        labels_len = labels_len.to(device)
        B, N, T = labels.shape
        # 诊断：打印输入shape和部分内容
        if is_training and B > 0:
            print(f"[DIAG] fbanks.shape: {fbanks.shape}, labels.shape: {labels.shape}, spk_label_idx.shape: {spk_label_idx.shape}, labels_len: {labels_len}")
            print(f"[DIAG] labels[0, :, :10]: {labels[0, :, :10]}")
            print(f"[DIAG] spk_label_idx[0]: {spk_label_idx[0]}")
            # 添加更多诊断信息
            print(f"[DIAG] labels.sum(): {labels.sum()}, labels.mean(): {labels.mean()}")
            print(f"[DIAG] valid_speakers: {(spk_label_idx >= 0).sum()}")
        # forward
        (
            vad_pred,
            spk_emb_pred,
            loss,
            bce_loss,
            arcface_loss,
            mask_info,
            padded_vad_labels,
        ) = model(fbanks, spk_label_idx, labels, spk_labels=spk_label_idx)
        # 诊断：打印模型输出和标签
        #if is_training and B > 0:
        #    print(f"[LOSS DIAG] vad_pred.shape={vad_pred.shape}, padded_vad_labels.shape={padded_vad_labels.shape}")
        #    print(f"[LOSS DIAG] vad_pred[0, :, :10]={vad_pred[0, :, :10].detach().cpu().numpy()}")
        #    print(f"[LOSS DIAG] padded_vad_labels[0, :, :10]={padded_vad_labels[0, :, :10].detach().cpu().numpy()}")
        #    print(f"[DIAG] loss: {loss.item()}, bce_loss: {bce_loss.item()}, arcface_loss: {arcface_loss.item()}")
            # 添加预测概率的统计信息
        #    vad_probs = torch.sigmoid(vad_pred)
        #    print(f"[DIAG] vad_probs.mean(): {vad_probs.mean().item()}, vad_probs.std(): {vad_probs.std().item()}")
        #    print(f"[DIAG] vad_probs.max(): {vad_probs.max().item()}, vad_probs.min(): {vad_probs.min().item()}")
            
            # 添加VAD预测分布分析
        #    vad_probs_flat = vad_probs.flatten()
        #    positive_preds = vad_probs_flat > 0.5
        #    print(f"[VAD DIAG] 预测为正样本的比例: {positive_preds.float().mean().item():.4f}")
        #    print(f"[VAD DIAG] 真实正样本比例: {padded_vad_labels.float().mean().item():.4f}")
            
            # 分析每个说话人的预测
        #    for i in range(min(3, vad_probs.shape[1])):  # 只看前3个说话人
        #        spk_probs = vad_probs[0, i, :]
        #        spk_labels = padded_vad_labels[0, i, :]
        #        spk_positive_preds = spk_probs > 0.5
        #        print(f"[VAD DIAG] Speaker {i}: 预测正样本比例={spk_positive_preds.float().mean().item():.4f}, 真实正样本比例={spk_labels.float().mean().item():.4f}")
        # DER 计算
        outs_prob = torch.sigmoid(vad_pred).detach().cpu().numpy()
        #print(f"[DER DIAG] outs_prob.shape={outs_prob.shape}, padded_vad_labels.shape={padded_vad_labels.shape}, labels_len={labels_len}")
        #print(f"[DER DIAG] labels_len.sum()={labels_len.sum() if hasattr(labels_len, 'sum') else labels_len}")
        mi, fa, cf, acc, der = model.calc_diarization_result(
            outs_prob, padded_vad_labels, labels_len
        )

    # 始终返回loss（含两个loss）
    total_loss = loss
    info = {
        "loss": total_loss.detach().cpu().item(),
        "bce_loss": bce_loss.detach().cpu().item(),
        "arcface_loss": arcface_loss.detach().cpu().item(),
        "DER": der,
        "ACC": acc,
        "MI": mi,
        "FA": fa,
        "CF": cf,
        "vad_pred": torch.sigmoid(vad_pred),
        "vad_labels": padded_vad_labels,
    }
    # 移除可学习权重的日志，因为已经改为固定权重
    # if is_training:
    #     info["log_s_bce"] = model.module.log_s_bce.item()
    #     info["log_s_arcface"] = model.module.log_s_arcface.item()
    return total_loss, info

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
    from train_accelerate_ddp import build_spk2int
    def build_train_dl_with_global_spk2int(dataset, args):
        spk2int = build_spk2int(args.textgrid_dir)

        def collate_fn_wrapper(batch):
            wavs, labels, spk_ids_list, fbanks, labels_len = dataset.collate_fn(batch, vad_out_len=args.vad_out_len)
            print(f"wavs.shape: {wavs.shape}, labels.shape: {labels.shape}, spk_ids_list: {spk_ids_list}, fbanks.shape: {fbanks.shape}, labels_len.shape: {labels_len.shape}")
            # 构造spk2int
            #all_spk = set()
            #for spk_ids in spk_ids_list:
            #    all_spk.update([s for s in spk_ids if s is not None])
            #spk2int = {spk: i for i, spk in enumerate(sorted(list(all_spk)))}
            max_spks_in_batch = labels.shape[1]
            print(f"max_spks_in_batch: {max_spks_in_batch}")
            spk_label_indices = torch.full((len(spk_ids_list), max_spks_in_batch), -1, dtype=torch.long)
            for i, spk_id_sample in enumerate(spk_ids_list):
                for j, spk_id in enumerate(spk_id_sample):
                    if spk_id and spk_id in spk2int:
                        spk_label_indices[i, j] = spk2int[spk_id]
            return fbanks, labels, spk_label_indices, labels_len
        from torch.utils.data import DataLoader
        train_dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_wrapper)
        return train_dl
    
    def build_train_dl_with_local_spk2int(dataset, args):
        #spk2int = build_spk2int(args.textgrid_dir)

        def collate_fn_wrapper(batch):
            wavs, labels, spk_ids_list, fbanks, labels_len = dataset.collate_fn(batch, vad_out_len=args.vad_out_len)
            print(f"wavs.shape: {wavs.shape}, labels.shape: {labels.shape}, spk_ids_list: {spk_ids_list}, fbanks.shape: {fbanks.shape}, labels_len.shape: {labels_len.shape}")
            # 构造spk2int
            all_spk = set()
            for spk_ids in spk_ids_list:
                all_spk.update([s for s in spk_ids if s is not None])
            spk2int = {spk: i for i, spk in enumerate(sorted(list(all_spk)))}
            max_spks_in_batch = labels.shape[1]
            print(f"max_spks_in_batch: {max_spks_in_batch}")
            spk_label_indices = torch.full((len(spk_ids_list), max_spks_in_batch), -1, dtype=torch.long)
            for i, spk_id_sample in enumerate(spk_ids_list):
                for j, spk_id in enumerate(spk_id_sample):
                    if spk_id and spk_id in spk2int:
                        spk_label_indices[i, j] = spk2int[spk_id]
            return fbanks, labels, spk_label_indices, labels_len
        from torch.utils.data import DataLoader
        train_dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_wrapper)
        return train_dl
    #train_dl = build_train_dl_with_local_spk2int(dataset, args) # very fast coverage
    train_dl = build_train_dl_with_global_spk2int(dataset,args) # 模型的loss也可以下降，
                                                                #但是loss在运行到999步时，loss 是0.045 ，
                                                                # 而且大部分来自arcface loss ，
                                                                # bce loss 占比很小，DER 也变为0
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
        #vad_pred, spk_emb_pred, loss, bce_loss, arcface_loss, mask_info, vad_labels = model(
        #    fbanks, spk_label_idx, labels, spk_labels=spk_label_idx
        #)
        loss, info = compute_loss(model, batch, is_training=True, device=device)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            vad_probs = info["vad_pred"]
            vad_labels = info["vad_labels"]
            #print(f"Step {step}: loss={loss.item():.4f}, bce_loss={bce_loss.item():.4f}, arcface_loss={arcface_loss.item():.4f}")
            print(f"Step {step}: loss={loss.item():.4f}, info={info}")
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