import sys
import argparse
import copy
import logging
import warnings
import gzip
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, Optional, Tuple, Union, List
from collections import defaultdict
import librosa
import soundfile as sf
import numpy as np
#import random.Random as rng

# import optim
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed as dist
# from torch.nn.utils import clip_grad_norm_


from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from dataclasses import dataclass
from accelerate import Accelerator
from accelerate.utils import GradScalerKwargs

import logging
import textgrid
from tqdm import tqdm
import json
import pickle

from utils import fix_random_seed
from checkpoint import (
    load_checkpoint,
    remove_checkpoints,
    remove_epochs,
    update_averaged_model,
)
from checkpoint import save_checkpoint as save_checkpoint_impl
from checkpoint import (
    save_checkpoint_with_global_batch_idx,
)
from utils import (
    AttributeDict,
    MetricsTracker,
    str2bool,
    none_or_str,
)

from ssnd_model import SSNDModel
from alimeeting_diar_dataset import AlimeetingDiarDataset
from simu_diar_dataset import SimuDiarMixer
from funasr import AutoModel # pip install funasr # only for simu data

#logging.basicConfig(
#    level=logging.INFO,
#    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
#)

# 设置日志
def setup_logging():
    """设置日志配置"""
    # 强制重新配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s',
        force=True,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

# 初始化日志
logger = setup_logging()
def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--debug", type=str2bool, default=False, help="for debug" )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=30,
        help="Number of epochs to train.",
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        help="Resume training from this epoch.",
    )
    parser.add_argument(
        "--start-batch",
        type=int,
        default=0,
        help="""If positive, --start-epoch is ignored and
        it loads the checkpoint from exp-dir/checkpoint-{start_batch}.pt
        """,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for random generators intended for reproducibility",
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        default="exp/ssnd",
        help="""The experiment dir.""",
    )
    parser.add_argument(
        "--lr", type=float, default=5e-4, help="learning rate."
    )
    parser.add_argument("--lr-type", type=str, default="PolynomialDecayLR", help="scheduler type of adamw, choise from `PolynomialDecayLR`, `CosineAnnealingLR`, `ReduceLROnPlateau`.")
    parser.add_argument(
        "--average-period",
        type=int,
        default=200,
        help="""Update the averaged model, namely `model_avg`, after processing
        this number of batches. `model_avg` is a separate version of model,
        in which each floating-point parameter is the average of all the
        parameters from the start of training. Each time we take the average,
        we do: `model_avg = model * (average_period / batch_idx_train) +
            model_avg * ((batch_idx_train - average_period) / batch_idx_train)`.
        """,
    )
    parser.add_argument(
        "--train-on-average",
        type=str2bool,
        default=False,
        help="train on the average model, how to average model, you can see '--average-period'",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="batch size"
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of GPUs for DDP training.",
    )
    parser.add_argument(
        "--grad-clip",
        type=str2bool,
        default=False,
        help="whether grad clip norm at traing stage",
    )
    parser.add_argument(
        "--tensorboard",
        type=str2bool,
        default=True,
        help="Should various information be logged in tensorboard.",
    )
    parser.add_argument(
        "--max-updates",
        type=int,
        default=40000,
        help="number iters of training ",
    )
    parser.add_argument(
        "--warmup-updates",
        type=int,
        default=400,
        help="number warmup iters of training ",
    )
    parser.add_argument(
        "--save-every-n",
        type=int,
        default=1500,
        help="""Save checkpoint after processing this number of batches"
        periodically. We save checkpoint to exp-dir/ whenever
        params.batch_idx_train % save_every_n == 0. The checkpoint filename
        has the form: f'exp-dir/checkpoint-{params.batch_idx_train}.pt'
        Note: It also saves checkpoint to `exp-dir/epoch-xxx.pt` at the
        end of each epoch where `xxx` is the epoch number counting from 0.
        """,
    )
    parser.add_argument(
        "--keep-last-k",
        type=int,
        default=20,
        help="""Only keep this number of checkpoints on disk.
        For instance, if it is 3, there are only 3 checkpoints
        in the exp-dir with filenames `checkpoint-xxx.pt`.
        It does not affect checkpoints with name `epoch-xxx.pt`.
        """,
    )
    parser.add_argument(
        "--keep-last-epoch",
        type=int,
        default=10,
        help="""Only keep this number of epoch checkpoint on disk.
        For instance, if it is 10, there are only 10 checkpoints
        in the exp-dir with filenames `epoch-xxx.pt`.
        It does not affect checkpoints with name `checkpoint-xxx.pt`.
        """,
    )
    parser.add_argument("--extrator_frozen_steps", type=int, default=100000,help="frozen speaker pretrain model")

    # Data arguments
    parser.add_argument('--train_wav_dir', type=str, required=True, help='Directory containing train WAV files.')
    parser.add_argument('--train_textgrid_dir', type=str, required=True, help='Directory containing train TextGrid files.')
    parser.add_argument('--valid_wav_dir', type=str, required=True, help='Directory containing valid WAV files.')
    parser.add_argument('--valid_textgrid_dir', type=str, required=True, help='Directory containing valid TextGrid files.')
    #parser.add_argument('--test_wav_dir', type=str, help='Directory containing test WAV files.')
    #parser.add_argument('--test_textgrid_dir', type=str, help='Directory containing test TextGrid files.')
    parser.add_argument('--musan-path', type=str, default=None, help='Path to MUSAN dataset for data augmentation.')
    parser.add_argument('--rir-path', type=str, default=None, help='Path to RIR dataset for data augmentation.')
    parser.add_argument('--noise-ratio', type=float, default=0.8, help='Probability of adding noise.')
    parser.add_argument('--window-sec', type=float, default=8.0, help='Window size in seconds.')
    parser.add_argument('--window-shift-sec', type=float, default=0.4, help='Window shift in seconds.')

    # Model arguments
    parser.add_argument("--speaker_pretrain_model_path", type=str, required=True, help="speaker pretrained model ckpt")
    parser.add_argument("--extractor_model_type", type=str, default='CAM++_wo_gsp',help="speaker pretrained model type")
    parser.add_argument('--feat-dim', type=int, default=80)
    parser.add_argument('--emb-dim', type=int, default=256)
    parser.add_argument('--q-det-aux-dim', type=int, default=256)
    parser.add_argument('--q-rep-aux-dim', type=int, default=256)
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--d-ff', type=int, default=512)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--max-speakers', type=int, default=4)
    parser.add_argument('--vad-out-len', type=int, default=200) # 8s / (16000/160) / 2 * 2 = 200, it is 8/0.04
    parser.add_argument('--arcface-margin', type=float, default=0.2)
    parser.add_argument('--arcface-scale', type=float, default=32.0)
    parser.add_argument('--pos-emb-dim', type=int, default=256)
    parser.add_argument('--max-seq-len', type=int, default=200)
    #parser.add_argument('--n-all-speakers', type=int, default=2097) # Alimeeting has 2097 speakers
    parser.add_argument('--mask-prob', type=float, default=0.5)
    #parser.add_argument('--out-bias', type=float, default=-0.5, help="output bias of detection decoder, >0 means more confident positive samples, <0 means more confident negative samples")
    parser.add_argument("--arcface-weight",type=float, default=0.01, help="arcface loss weight")
    parser.add_argument("--bce-alpha", type=float, default=0.75, help="focal bce loss scale")
    parser.add_argument("--bce-gamma", type=float, default=2.0, help="focal bce loss scale")
    parser.add_argument("--use-standard-bce", type=str2bool, default=False, help="Use standard BCE loss instead of focal loss")
    parser.add_argument("--weight-decay",type=float, default=0.001, help= "AdamW optimizer weight_decay")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate for regularization")
    parser.add_argument("--label-smoothing", type=float, default=0.00, help="Label smoothing factor")
    parser.add_argument("--gradient-clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--early-stopping-patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.001, help="Early stopping minimum improvement")
    #parser.add_argument("--focal-alpha", type=float, default=0.75, help="Focal loss alpha parameter")
    #parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal loss gamma parameter")
    parser.add_argument("--decoder-dropout", type=float, default=0.0, help="two decoder dropout rate")
    parser.add_argument("--extractor-dropout", type=float, default=0.0, help="apply dropout into output of extractor")
    parser.add_argument("--voxceleb2-dataset-dir", type=str, default="/maduo/datasets/voxceleb2/vox2_dev/", help="voxceleb2 kaldi format data dir")
    parser.add_argument("--train-stage", type=int, default=1, help="use numbers to determine train stage")
    parser.add_argument("--voxceleb2-spk2chunks-json", type=str, default="/maduo/datasets/voxceleb2/vox2_dev/train.jsonl_gzip", help="vad timestamp and spk_id  and wav_path")
    
    parser.add_argument("--compression-type", type=str, default="gzip", help="compression method type for vad json files.")
    
    # 加速处理相关参数
    # parser.add_argument("--use-fast-spktochunks", type=str2bool, default=True, help="是否使用加速版本的spktochunks函数")
    # parser.add_argument("--use-lazy-loading", type=str2bool, default=False, help="是否使用懒加载模式（内存优化）")
    # parser.add_argument("--use-memory-safe", type=str2bool, default=False, help="是否使用超级内存安全模式（避免OOM）")
    # parser.add_argument("--fast-batch-size", type=int, default=2, help="加速版本的批处理大小（控制内存使用）")
    # parser.add_argument("--fast-max-memory-mb", type=int, default=0, help="加速版本的最大内存限制（MB），0表示自动检测并使用可用内存的指定百分比")
    # parser.add_argument("--memory-usage-ratio", type=float, default=0.5, help="自动内存检测时使用的内存比例（0.0-1.0），默认0.5表示50%")
    # parser.add_argument("--fast-sub-batch-size", type=int, default=20, help="子批次大小（每个子批次处理的音频文件数量）")
    # parser.add_argument("--strict-memory-check", type=str2bool, default=False, help="是否启用严格的内存检查（True时会跳过超限的批次）")
    # parser.add_argument("--max-speakers-test", type=int, default=None, help="测试时限制最大说话人数量")
    # parser.add_argument("--max-files-per-speaker-test", type=int, default=None, help="测试时限制每个说话人的最大文件数量")
    # parser.add_argument("--disable-cache", type=str2bool, default=False, help="禁用缓存功能")
    
    # 懒加载模拟数据相关参数
    parser.add_argument("--use-lazy-simu", type=str2bool, default=False, help="是否启用懒加载模拟数据模式（跳过spktochunks预处理）")

    add_finetune_arguments(parser)
    return parser

def add_finetune_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--do-finetune",
        type=str2bool,
        default=False,
        help="If true, finetune from a pre-trained checkpoint",
    )
    parser.add_argument(
        "--init-modules",
        type=str,
        default=None,
        help="""
        Modules to be initialized. It matches all parameters starting with
        a specific key. The keys are given with Comma seperated. If None,
        all modules will be initialised. For example, if you only want to
        initialise all parameters staring with "encoder", use "encoder";
        if you want to initialise parameters starting with encoder or decoder,
        use "encoder,joiner".
        """,
    )

    parser.add_argument(
        "--finetune-ckpt",
        type=str,
        default=None,
        help="Fine-tuning from which checkpoint (path to a .pt file)",
    )

def get_optimizer_scheduler(params, model):
    from torch.optim import AdamW

    # 增加weight_decay来减少过拟合
    optimizer = AdamW(
        model.parameters(),
        lr=params.lr,
        betas=(0.9, 0.98),
        eps=1e-08,
        weight_decay=params.weight_decay,
    )
    if params.lr_type=="PolynomialDecayLR":
        # optimizer = AdamW(model.parameters(),lr=5e-5,betas=(0.9, 0.98)) # same as fairseq2
        from polynomial import PolynomialDecayLR

        scheduler = PolynomialDecayLR(
            optimizer, params.max_updates, params.warmup_updates, power=0.5  # 降低power，使学习率增长更平缓
        )
    elif params.lr_type=="CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100, eta_min=1e-6
        )
    elif params.lr_type=="ReduceLROnPlateau":
        # 或在损失平台期重置学习率
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        )
    return optimizer, scheduler

def get_params(args) -> AttributeDict:
    params = AttributeDict(
        {
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_der": float("inf"),
            "best_valid_der": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "log_interval": 500 if not args.debug else 5,  # same as fairseq
            #"log_interval":5,
            "reset_interval": 200 if not args.debug else 2,
            # "valid_interval": 1500,  # same as fairseq
            "valid_interval": 500 if not args.debug else 5,
            "batch_size": 64,
            "use_standard_bce": args.use_standard_bce,
            # 新增正则化参数
            "dropout": args.dropout,
            "label_smoothing": args.label_smoothing,
            "gradient_clip": args.gradient_clip,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
            # 早停相关
            "patience_counter": 0,
            "best_valid_der_for_early_stop": float("inf"),
        }
    )
    return params
#def calculate_loss(outs, labels, labels_len):
#    total_loss = 0
#    for i in range(labels_len.size(0)):
#        total_loss += F.binary_cross_entropy_with_logits(
#            outs[i, :, : labels_len[i]], labels[i, :, : labels_len[i]]
#        )
#    return total_loss / labels_len.size(0)

def compute_loss(
    model: Union[nn.Module, DDP],
    batch: tuple,
    is_training: bool,
    use_standard_bce: bool = False,
    params: AttributeDict = None,
):
    """
    batch: (fbanks, labels, spk_label_idx, labels_len)
    """
    with torch.set_grad_enabled(is_training):
        
        #fbanks, labels, spk_label_idx, labels_len, _= batch  # [B, T, F], [B, N, T], [B, N], [B], N is num of speakers,
        fbanks, labels, spk_label_indices, labels_len, data_source = batch
        #wavs, labels, spk_label_idx, fbanks, labels_len, data_sources = batch
        print(f"labels shape: {labels.shape}, spk_label_indices shape: {spk_label_indices.shape}, fbanks shape: {fbanks.shape}, labels_len : {labels_len.shape}, data_sources: {data_sources.shape}")
        B, N, T = labels.shape
        
        # 应用label smoothing
        if params and params.label_smoothing > 0 and is_training:
            labels = labels * (1 - params.label_smoothing) + 0.5 * params.label_smoothing
        
        # 诊断：打印输入shape和部分内容
        # if is_training and B > 0:
        #     print(f"[DIAG] fbanks.shape: {fbanks.shape}, labels.shape: {labels.shape}, spk_label_idx.shape: {spk_label_idx.shape}, labels_len: {labels_len}")
        #     print(f"[DIAG] labels[0, :, :10]: {labels[0, :, :10]}")
        #     print(f"[DIAG] spk_label_idx[0]: {spk_label_idx[0]}")
        #     # 添加更多诊断信息
        #     print(f"[DIAG] labels.sum(): {labels.sum()}, labels.mean(): {labels.mean()}")
        #     print(f"[DIAG] valid_speakers: {(spk_label_idx >= 0).sum()}")
        
        # forward
        (
            vad_pred,
            spk_emb_pred,
            loss,
            bce_loss,
            arcface_loss,
            mask_info,
            padded_vad_labels,
        ) = model(fbanks, spk_label_idx, labels, spk_labels=spk_label_idx, use_standard_bce=use_standard_bce)
        
        # 诊断：打印模型输出和标签
        #if is_training and B > 0:
        #    print(f"[LOSS DIAG] vad_pred.shape={vad_pred.shape}, padded_vad_labels.shape={padded_vad_labels.shape}")
        #    print(f"[LOSS DIAG] vad_pred[0, :, :10]={vad_pred[0, :, :10].detach().cpu().numpy()}")
        #    print(f"[LOSS DIAG] padded_vad_labels[0, :, :10]={padded_vad_labels[0, :, :10].detach().cpu().numpy()}")
        #    print(f"[DIAG] loss: {loss.item()}, bce_loss: {bce_loss.item()}, arcface_loss: {arcface_loss.item()}")
        #    # 添加预测概率的统计信息
            # vad_probs = torch.sigmoid(vad_pred)
            # print(f"[DIAG] vad_probs.mean(): {vad_probs.mean().item()}, vad_probs.std(): {vad_probs.std().item()}")
            # print(f"[DIAG] vad_probs.max(): {vad_probs.max().item()}, vad_probs.min(): {vad_probs.min().item()}")
            
            # # 添加VAD预测分布分析
            # vad_probs_flat = vad_probs.flatten()
            # positive_preds = vad_probs_flat > 0.5
            # print(f"[VAD DIAG] 预测为正样本的比例: {positive_preds.float().mean().item():.4f}")
            # print(f"[VAD DIAG] 真实正样本比例: {padded_vad_labels.float().mean().item():.4f}")
            
            # # 分析每个说话人的预测
            # for i in range(min(3, vad_probs.shape[1])):  # 只看前3个说话人
            #     spk_probs = vad_probs[0, i, :]
            #     spk_labels = padded_vad_labels[0, i, :]
            #     spk_positive_preds = spk_probs > 0.5
            #     print(f"[VAD DIAG] Speaker {i}: 预测正样本比例={spk_positive_preds.float().mean().item():.4f}, 真实正样本比例={spk_labels.float().mean().item():.4f}")
        
        # DER 计算
        outs_prob = torch.sigmoid(vad_pred).detach().cpu().numpy()
        #print(f"[DER DIAG] outs_prob.shape={outs_prob.shape}, padded_vad_labels.shape={padded_vad_labels.shape}, labels_len={labels_len}")
        #print(f"[DER DIAG] labels_len.sum()={labels_len.sum() if hasattr(labels_len, 'sum') else labels_len}")
        mi, fa, cf, acc_all, acc_spks, der = model.module.calc_diarization_result(
            outs_prob, padded_vad_labels, labels_len
        )
    
    # 始终返回loss（含两个loss）
    total_loss = loss
    info = {
        "loss": total_loss.detach().cpu().item(),
        "bce_loss": bce_loss.detach().cpu().item(),
        "arcface_loss": arcface_loss.detach().cpu().item(),
        "DER": der,
        "ACC_ALL": acc_all,
        "ACC_SPKS": acc_spks,
        "MI": mi,
        "FA": fa,
        "CF": cf,
    }
    # 移除可学习权重的日志，因为已经改为固定权重
    # if is_training:
    #     info["log_s_bce"] = model.module.log_s_bce.item()
    #     info["log_s_arcface"] = model.module.log_s_arcface.item()
    return total_loss, info

def compute_validation_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    valid_dl: torch.utils.data.DataLoader,
    batch_idx_train: int = 0,
    writer: Optional[SummaryWriter] = None,
    #train_der: float = None,  # 新增：训练集DER
) -> MetricsTracker:
    """Compute validation loss."""
    model.eval()
    
    tot_loss = MetricsTracker()
    
    batch_nums = []
    tot_loss_valid = 0
    for batch_idx, batch in enumerate(valid_dl):
        loss, loss_info = compute_loss(model, batch, is_training=False, use_standard_bce=params.use_standard_bce, params=params)
        assert loss.requires_grad is False
        #tot_loss.update(info)
        batch_nums.append(batch_idx)
        assert loss.requires_grad is False
        tot_loss_valid = tot_loss_valid + loss_info["loss"]
        tot_loss = tot_loss + loss_info

    for item in tot_loss.keys():
        tot_loss[item] = tot_loss[item] / len(batch_nums)

    loss_value = tot_loss["loss"]
    if loss_value < params.best_valid_loss:
        params.best_valid_epoch = params.cur_epoch
        params.best_valid_loss = loss_value

    der_value = tot_loss["DER"]
    if der_value < params.best_valid_der:
        params.best_valid_epoch = params.cur_epoch
        params.best_valid_der = der_value
    # log to tensorboard
    if writer is not None:
        for key, value in tot_loss.items():
            writer.add_scalar(f"valid/{key}", value, batch_idx_train)
    
    return tot_loss

def save_checkpoint(
    params: AttributeDict,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    model_avg: Optional[nn.Module] = None,
    scaler: Optional[GradScaler] = None,
) -> None:
    """Save model, optimizer, scheduler and training stats to file.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The training model.
    """
    filename = params.exp_dir / f"epoch-{params.cur_epoch}.pt"
    save_checkpoint_impl(
        filename=filename,
        model=model,
        params=params,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        model_avg=model_avg,
    )
    logging.info(f" end of epoch {params.cur_epoch}, Saved checkpoint to {filename} ")
    if params.best_train_epoch == params.cur_epoch:
        best_train_filename = params.exp_dir / "best-train-loss.pt"
        best_train_der_filename = params.exp_dir / "best-train-der.pt"
        copyfile(src=filename, dst=best_train_filename)
        copyfile(src=filename, dst=best_train_der_filename)
    if params.best_valid_epoch == params.cur_epoch:
        best_valid_filename = params.exp_dir / "best-valid-loss.pt"
        best_valid_der_filename = params.exp_dir / "best-valid-der.pt"
        copyfile(src=filename, dst=best_valid_filename)
        copyfile(src=filename, dst=best_valid_der_filename)

def do_save_and_remove_once(
    params, model, model_avg, optimizer, scheduler, train_dl, scaler
):
    # save model
    save_checkpoint_with_global_batch_idx(
        out_dir=params.exp_dir,
        global_batch_idx=params.batch_idx_train,
        model=model,
        model_avg=model_avg,
        params=params,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=train_dl.sampler,
        scaler=scaler,
    )
    # remove unsed ckpt
    remove_checkpoints(
        out_dir=params.exp_dir,
        topk=params.keep_last_k,
    )

    remove_epochs(
        out_dir=params.exp_dir,
        topk=params.keep_last_epoch,
    )

def train_one_epoch(
    params: AttributeDict,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    accelerator,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    model_avg: Optional[nn.Module] = None,
    writer: Optional[SummaryWriter] = None,
) -> bool:
    """Train the model for one epoch.

    The training loss from the mean of all frames is saved in
    `params.train_loss`. It runs the validation process every
    `params.valid_interval` batches.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The model for training.
      optimizer:
        The optimizer we are using.
      train_dl_simu:
        Dataloader for the simu training dataset.
      train_dl:
        Dataloader for the training dataset.
      valid_dl:
        Dataloader for the validation dataset.
    """
    if hasattr(model, 'module'):
        model.module.cur_epoch = params.cur_epoch
    else:
        model.cur_epoch = params.cur_epoch
    model.train()

    tot_loss = MetricsTracker()
    train_batch_nums = []
 
    for batch_idx, batch in enumerate(train_dl):
        # 每个batch前判断是否需要冻结/解冻extrator
        if  params.batch_idx_train < params.extrator_frozen_steps:
            for p in model.module.extractor.speech_encoder.parameters():
                p.requires_grad = False
            if batch_idx > 0 and batch_idx % params.valid_interval == 0:
                logger.info(f"[Freeze] extractor speech encoder parameters at step {params.batch_idx_train}")
        elif params.batch_idx_train >= params.extrator_frozen_steps:
            for p in model.module.extractor.speech_encoder.parameters():
                p.requires_grad = True
            if batch_idx > 0 and batch_idx % params.valid_interval == 0:
                logger.info(f"[Unfreeze] extractor speech encoder unfreeze at step {params.batch_idx_train}")
            #params['extrator_frozen'] = False
        params.batch_idx_train += 1
        batch_size = params.batch_size

        optimizer.zero_grad()
        train_batch_nums.append(batch_idx)
        #batch_size = batch[0].shape[0]
        #params.batch_idx_train += 1
        
        loss, loss_info = compute_loss(
            model, batch, is_training=True, use_standard_bce=params.use_standard_bce, params=params
        )
        accelerator.backward(loss)  # instead of loss.backward()

        # 应用梯度裁剪
        grad_norm = None
        if params.gradient_clip > 0:
            if accelerator.sync_gradients:
                grad_norm = accelerator.clip_grad_norm_(
                    model.parameters(), max_norm=params.gradient_clip
                )
        #elif params.grad_clip:
        #    if accelerator.sync_gradients:
        #        grad_norm = accelerator.clip_grad_norm_(
        #            model.parameters(), max_norm=1.0
        #        )

        optimizer.step()
        scheduler.step()

        # log to tensorboard
        if writer and accelerator.is_main_process:
            for key, value in loss_info.items():
                writer.add_scalar(f"train/{key}", value, params.batch_idx_train)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], params.batch_idx_train)
            if grad_norm is not None:
                writer.add_scalar("train/grad_norm", grad_norm.item(), params.batch_idx_train)

        ## average checkpoint
        if (
            params.train_on_average
            and accelerator.is_main_process
            and params.batch_idx_train > 0
            and params.batch_idx_train % params.average_period == 0
        ):
            logger.info(
                f"Currently, model averaging is being used during the training process."
            )
            update_averaged_model(
                params=params,
                model_cur=model,
                model_avg=model_avg,
            )

        ## save and remove unuse checkpoint
        if (
            params.batch_idx_train > 0
            and params.batch_idx_train % params.save_every_n == 0
            and accelerator.is_main_process
        ):
            do_save_and_remove_once(
                params, model, model_avg, optimizer, scheduler, train_dl, scaler
            )
        if batch_idx % params.log_interval == 0:
            ## To align with the numbers in fairseq iter
            num_updates = 0
            if params.cur_epoch == 1:
                num_updates = batch_idx
            else:
                integer_multi_num = (
                    len(train_dl) - len(train_dl) % params.log_interval
                )  # 3128 - 3128%500=3000
                num_updates = (params.cur_epoch - 1) * integer_multi_num + batch_idx

            ## get grad_scale and lr
            # grad_scale = scale_result.new_scale
            grad_scale = ""
            cur_lr = scheduler.get_last_lr()[0]

            logger.info(
                f"[Train] - Epoch {params.cur_epoch}, "
                f"batch_idx_train: {params.batch_idx_train-1}, num_updates: {num_updates}, {loss_info}, "
                f"batch size: {batch_size}, grad_norm: {grad_norm}, grad_scale: {grad_scale}, "
                f"lr: {cur_lr}, "
            )
        # log end-of-epoch stats
        if batch_idx == len(train_dl) - 1:
            # grad_scale = scale_result.new_scale
            grad_scale = ""
            cur_lr = scheduler.get_last_lr()[0]
            logger.info(
                f"end of epoch {params.cur_epoch}, batch_idx: {batch_idx} "
                f"batch_idx_train: {params.batch_idx_train-1}, {loss_info}, "
                f"batch size: {batch_size}, grad_norm: {grad_norm}, grad_scale: {grad_scale}, "
                f"lr: {cur_lr}, "
            )
        if batch_idx > 0 and batch_idx % params.valid_interval == 0:
            logger.info("Computing validation loss")
            # 计算当前训练集的DER
            #current_train_der = tot_loss["DER"] / len(train_batch_nums) if len(train_batch_nums) > 0 else 0.0
            valid_info = compute_validation_loss(
                params=params,
                model=model,
                valid_dl=valid_dl,
                batch_idx_train=params.batch_idx_train,
                writer=writer,
                #train_der=current_train_der,  # 传入训练集DER
            )
            model.train()
            logger.info(
                f"[Eval] - Epoch {params.cur_epoch}, batch_idx_train: {params.batch_idx_train-1}, "
                f" validation: {valid_info}"
            )
            logger.info(
                f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated()//1000000}MB"
            )
    loss_value = tot_loss["loss"] / len(train_batch_nums)
    params.train_loss = loss_value

    if params.train_loss < params.best_train_loss:
        params.best_train_epoch = params.cur_epoch
        params.best_train_loss = params.train_loss

    der_value = tot_loss["DER"] / len(train_batch_nums)
    if der_value < params.best_train_der:
        params.best_train_epoch = params.cur_epoch
        params.best_train_der = der_value
    
    
def is_real(batch):
    return torch.any(batch[-1] == 0)

def train_one_epoch_multi(
    params: AttributeDict,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    accelerator,
    train_dl_simu: torch.utils.data.DataLoader,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    model_avg: Optional[nn.Module] = None,
    writer: Optional[SummaryWriter] = None,
) -> bool:
    """Train the model for one epoch.

    The training loss from the mean of all frames is saved in
    `params.train_loss`. It runs the validation process every
    `params.valid_interval` batches.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The model for training.
      optimizer:
        The optimizer we are using.
      train_dl_simu:
        Dataloader for the simu training dataset.
      train_dl:
        Dataloader for the training dataset.
      valid_dl:
        Dataloader for the validation dataset.
    """
    assert train_dl_simu is not None, f"train_dl_simu is None"
    assert train_dl is not None, f"train_dl is None"

    if hasattr(model, 'module'):
        model.module.cur_epoch = params.cur_epoch
    else:
        model.cur_epoch = params.cur_epoch
    model.train()

    real_tot_loss = MetricsTracker()
    simu_tot_loss = MetricsTracker()
    tot_loss = MetricsTracker()
    
    train_batch_nums = []

    # index 0: for real dataset
    # index 1: for simu dataset
    # This sets the probabilities for choosing which datasets
    dl_weights = [1 - params.real_dataset_prob, params.real_dataset_prob]

    iter_real = iter(train_dl)
    iter_simu = iter(train_dl_simu)

    batch_idx = 0

    while True:
        idx = random.Random.choices((0, 1), weights=dl_weights, k=1)[0]
        dl = iter_real if idx == 0 else iter_simu

        try:
            batch = next(dl)
        except StopIteration:
            break

        batch_idx += 1

#    for batch_idx, batch in enumerate(train_dl):
#        # 每个batch前判断是否需要冻结/解冻extrator
#        if  params.batch_idx_train < params.extrator_frozen_steps:
#            for p in model.module.extractor.speech_encoder.parameters():
#                p.requires_grad = False
#            if batch_idx > 0 and batch_idx % params.valid_interval == 0:
#                logger.info(f"[Freeze] extractor speech encoder parameters at step {params.batch_idx_train}")
#        elif params.batch_idx_train >= params.extrator_frozen_steps:
#            for p in model.module.extractor.speech_encoder.parameters():
#                p.requires_grad = True
#            if batch_idx > 0 and batch_idx % params.valid_interval == 0:
#                logger.info(f"[Unfreeze] extractor speech encoder unfreeze at step {params.batch_idx_train}")
            #params['extrator_frozen'] = False
        params.batch_idx_train += 1
        batch_size = params.batch_size

        optimizer.zero_grad()
        train_batch_nums.append(batch_idx)
        real = is_real(batch) 
        loss, loss_info = compute_loss(
            model, batch, is_training=True, use_standard_bce=params.use_standard_bce, params=params
        )
         # summary stats
        tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + loss_info
        if real:
            real_tot_loss = (
                real_tot_loss * (1 - 1 / params.reset_interval)
            ) + loss_info
            #prefix = "real"  # for logging only
        else:
            simu_tot_loss = (
                simu_tot_loss * (1 - 1 / params.reset_interval)
            ) + loss_info
            #prefix = "simu"

        accelerator.backward(loss)  # instead of loss.backward()

        # 应用梯度裁剪
        grad_norm = None
        if params.gradient_clip > 0:
            if accelerator.sync_gradients:
                grad_norm = accelerator.clip_grad_norm_(
                    model.parameters(), max_norm=params.gradient_clip
                )

        optimizer.step()
        scheduler.step()

        # log to tensorboard
        if writer and accelerator.is_main_process:
            for key, value in loss_info.items():
                writer.add_scalar(f"train/{key}", value, params.batch_idx_train)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], params.batch_idx_train)
            writer.add_scalar("train/real_loss", real_tot_loss["loss"], params.batch_idx_train)
            writer.add_scalar("train/simu_loss", simu_tot_loss["loss"], params.batch_idx_train)
            if grad_norm is not None:
                writer.add_scalar("train/grad_norm", grad_norm.item(), params.batch_idx_train)

        ## average checkpoint
        if (
            params.train_on_average
            and accelerator.is_main_process
            and params.batch_idx_train > 0
            and params.batch_idx_train % params.average_period == 0
        ):
            logger.info(
                f"Currently, model averaging is being used during the training process."
            )
            update_averaged_model(
                params=params,
                model_cur=model,
                model_avg=model_avg,
            )

        ## save and remove unuse checkpoint
        if (
            params.batch_idx_train > 0
            and params.batch_idx_train % params.save_every_n == 0
            and accelerator.is_main_process
        ):
            do_save_and_remove_once(
                params, model, model_avg, optimizer, scheduler, train_dl, scaler
            )
        if batch_idx % params.log_interval == 0:
            ## To align with the numbers in fairseq iter
            num_updates = 0
            if params.cur_epoch == 1:
                num_updates = batch_idx
            else:
                integer_multi_num = (
                    len(train_dl) - len(train_dl) % params.log_interval
                )  # 3128 - 3128%500=3000
                num_updates = (params.cur_epoch - 1) * integer_multi_num + batch_idx

            ## get grad_scale and lr
            # grad_scale = scale_result.new_scale
            grad_scale = ""
            cur_lr = scheduler.get_last_lr()[0]

            logger.info(
                f"[Train] - Epoch {params.cur_epoch}, "
                f"batch_idx_train: {params.batch_idx_train-1}, num_updates: {num_updates}, {loss_info}, "
                f"batch size: {batch_size}, grad_norm: {grad_norm}, grad_scale: {grad_scale}, "
                f"lr: {cur_lr}, "
            )
        # log end-of-epoch stats
        if batch_idx == len(train_dl) - 1:
            # grad_scale = scale_result.new_scale
            grad_scale = ""
            cur_lr = scheduler.get_last_lr()[0]
            logger.info(
                f"end of epoch {params.cur_epoch}, batch_idx: {batch_idx} "
                f"batch_idx_train: {params.batch_idx_train-1}, {loss_info}, "
                f"batch size: {batch_size}, grad_norm: {grad_norm}, grad_scale: {grad_scale}, "
                f"lr: {cur_lr}, "
            )
        if batch_idx > 0 and batch_idx % params.valid_interval == 0:
            logger.info("Computing validation loss")
            # 计算当前训练集的DER
            #current_train_der = tot_loss["DER"] / len(train_batch_nums) if len(train_batch_nums) > 0 else 0.0
            valid_info = compute_validation_loss(
                params=params,
                model=model,
                valid_dl=valid_dl,
                batch_idx_train=params.batch_idx_train,
                writer=writer,
                #train_der=current_train_der,  # 传入训练集DER
            )
            model.train()
            logger.info(
                f"[Eval] - Epoch {params.cur_epoch}, batch_idx_train: {params.batch_idx_train-1}, "
                f" validation: {valid_info}"
            )
            logger.info(
                f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated()//1000000}MB"
            )
    loss_value = tot_loss["loss"] / len(train_batch_nums)
    params.train_loss = loss_value

    if params.train_loss < params.best_train_loss:
        params.best_train_epoch = params.cur_epoch
        params.best_train_loss = params.train_loss

    der_value = tot_loss["DER"] / len(train_batch_nums)
    if der_value < params.best_train_der:
        params.best_train_epoch = params.cur_epoch
        params.best_train_der = der_value

def load_model_params(
    ckpt: str, model: nn.Module, init_modules: List[str] = None, strict: bool = True
):
    """Load model params from checkpoint
    Args:
        ckpt (str): Path to the checkpoint
        model (nn.Module): model to be loaded
        init_modules (list[str]): List of modules to be initialized

    """
    logger.info(f"Loading checkpoint from {ckpt}")
    checkpoint = torch.load(ckpt, map_location="cpu")

    # if module list is empty, load the whole model from ckpt
    if not init_modules:
        if next(iter(checkpoint["model"])).startswith("module."):
            logger.info("Loading checkpoint saved by DDP")

            dst_state_dict = model.state_dict()
            src_state_dict = checkpoint["model"]
            for key in dst_state_dict.keys():
                src_key = "{}.{}".format("module", key)
                dst_state_dict[key] = src_state_dict.pop(src_key)
            assert len(src_state_dict) == 0
            model.load_state_dict(dst_state_dict, strict=strict)
        else:
            model.load_state_dict(checkpoint["model"], strict=strict)
    else:
        src_state_dict = checkpoint["model"]
        dst_state_dict = model.state_dict()
        for module in init_modules:
            logger.info(f"Loading parameters starting with prefix {module}")
            src_keys = [
                k for k in src_state_dict.keys() if k.startswith(module.strip() + ".")
            ]
            dst_keys = [
                k for k in dst_state_dict.keys() if k.startswith(module.strip() + ".")
            ]
            assert set(src_keys) == set(dst_keys)  # two sets should match exactly
            for key in src_keys:
                dst_state_dict[key] = src_state_dict.pop(key)

        model.load_state_dict(dst_state_dict, strict=strict)

    return None                                                    
def load_checkpoint_if_available(
    params: AttributeDict,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    model_avg: Optional[nn.Module] = None,
) -> None:
    """Load checkpoint from file.

    If params.start_epoch is positive, it will load the checkpoint from
    `params.start_epoch - 1`. Otherwise, this function does nothing.

    Apart from loading state dict for `model`, `optimizer` and `scheduler`,
    it also updates `best_train_epoch`, `best_train_loss`, `best_valid_epoch`,
    and `best_valid_loss` in `params`.

    Args:
      params:
        The return value of :func:`get_params`.
      model:
        The training model.
      optimizer:
        The optimizer that we are using.
      scheduler:
        The learning rate scheduler we are using.
    Returns:
      Return None.
    """
    if params.start_batch > 0:
        filename = params.exp_dir / f"checkpoint-{params.start_batch}.pt"
    elif params.start_epoch > 1:
        filename = params.exp_dir / f"epoch-{params.start_epoch}.pt"
    else:
        return None
    assert filename.is_file(), f"{filename} does not exist!"
    saved_params = load_checkpoint(
        filename,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        model_avg=model_avg,
    )

    keys = [
        "best_train_epoch",
        "best_valid_epoch",
        "batch_idx_train",
        "best_train_loss",
        "best_train_der",
        "best_valid_loss",
        "best_valid_der",
    ]
    for k in keys:
        params[k] = saved_params[k]

    return saved_params

def build_spk2int(*textgrid_dirs: str):
    """
    统计给定TextGrid目录下的说话人，生成spk2int。
    支持多个目录联合统计。
    """
    spk_ids = set()
    for textgrid_dir in textgrid_dirs:
        tg_dir = Path(textgrid_dir)
        for tg_file in tg_dir.glob('*.TextGrid'):
            try:
                tg = textgrid.TextGrid.fromFile(str(tg_file))
                for tier in tg:
                    if tier.name.strip():
                        spk_ids.add(tier.name[-9:]) # 保证和AlimeetingDiarDataset一致
            except Exception as e:
                logging.warning(f"Could not process {tg_file}: {e}")
    spk2int = {spk: i for i, spk in enumerate(sorted(list(spk_ids)))}
    logger.info(f"Found {len(spk2int)} unique speakers in the provided set.")
    logger.info(f"spk2int: {spk2int}, spk_ids: {spk_ids}")
    return spk2int

def broadcast_spk2int(args, accelerator):
    if accelerator.is_main_process:
        spk2int = build_spk2int(args.train_textgrid_dir, args.valid_textgrid_dir)
        spk2int_bytes = pickle.dumps(spk2int)
        spk2int_tensor = torch.ByteTensor(list(spk2int_bytes)).to(accelerator.device)
        length_tensor = torch.LongTensor([spk2int_tensor.numel()]).to(accelerator.device)
    else:
        spk2int_tensor = None
        length_tensor = torch.LongTensor([0]).to(accelerator.device)
    # 广播长度
    length_tensor = accelerator.broadcast(length_tensor, 0)
    # 分配空间
    if not accelerator.is_main_process:
        spk2int_tensor = torch.empty(length_tensor.item(), dtype=torch.uint8, device=accelerator.device)
    # 广播内容
    spk2int_tensor = accelerator.broadcast(spk2int_tensor, 0)
    # 反序列化
    spk2int_bytes = bytes(spk2int_tensor.cpu().tolist())
    spk2int = pickle.loads(spk2int_bytes)
    return spk2int

def build_train_dl(args, spk2int): 
    logger.info("Building train dataloader with training spk2int...")
    train_dataset = AlimeetingDiarDataset(
        wav_dir=args.train_wav_dir,
        textgrid_dir=args.train_textgrid_dir,
        sample_rate=16000,
        frame_shift=0.04, # 25fps to match vad_out_len
        musan_path=args.musan_path,
        rir_path=args.rir_path,
        noise_ratio=args.noise_ratio,
        window_sec=args.window_sec,
        window_shift_sec=args.window_shift_sec
    )
    vad_out_len = args.vad_out_len if hasattr(args, 'vad_out_len') else 200
    def collate_fn_wrapper(batch):
        wavs, labels, spk_ids_list, fbanks, labels_len = train_dataset.collate_fn(batch, vad_out_len=vad_out_len)
        max_spks_in_batch = labels.shape[1]
        spk_label_indices = torch.full((len(spk_ids_list), max_spks_in_batch), -1, dtype=torch.long)
        for i, spk_id_sample in enumerate(spk_ids_list):
            for j, spk_id in enumerate(spk_id_sample):
                if spk_id and spk_id in spk2int:
                    spk_label_indices[i, j] = spk2int[spk_id]
        return fbanks, labels, spk_label_indices, labels_len
    train_dl = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn_wrapper,
        num_workers=4,  # 建议：4~8，避免超过CPU核心数
        pin_memory=True,
        persistent_workers=True  # 避免每epoch重建进程
    )   
    return train_dl

def build_train_dl_with_local_spk2int(args,): 
    logger.info("Building train dataloader with training spk2int...")
    train_dataset = AlimeetingDiarDataset(
        wav_dir=args.train_wav_dir,
        textgrid_dir=args.train_textgrid_dir,
        sample_rate=16000,
        frame_shift=0.04, # 25fps to match vad_out_len
        musan_path=args.musan_path,
        rir_path=args.rir_path,
        noise_ratio=args.noise_ratio,
        window_sec=args.window_sec,
        window_shift_sec=args.window_shift_sec
    )
    vad_out_len = args.vad_out_len if hasattr(args, 'vad_out_len') else 200
    def collate_fn_wrapper(batch):
        wavs, labels, spk_ids_list, fbanks, labels_len = train_dataset.collate_fn(batch, vad_out_len=vad_out_len)
        # 构造spk2int
        all_spk = set()
        for spk_ids in spk_ids_list:
            all_spk.update([s for s in spk_ids if s is not None])
        spk2int = {spk: i for i, spk in enumerate(sorted(list(all_spk)))}
        logger.info(f"train set spk2int len: {len(spk2int)} in fn build_train_dl_with_local_spk2int")
        max_spks_in_batch = labels.shape[1]
        spk_label_indices = torch.full((len(spk_ids_list), max_spks_in_batch), -1, dtype=torch.long)
        for i, spk_id_sample in enumerate(spk_ids_list):
            for j, spk_id in enumerate(spk_id_sample):
                if spk_id and spk_id in spk2int:
                    spk_label_indices[i, j] = spk2int[spk_id]
        return fbanks, labels, spk_label_indices, labels_len
    train_dl = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn_wrapper,
        num_workers=4,  # 建议：4~8，避免超过CPU核心数
        pin_memory=True,
        persistent_workers=True  # 避免每epoch重建进程
    )   
    return train_dl

def build_valid_dl(args, spk2int): 
    logger.info("Building valid dataloader with training spk2int...")
    valid_dataset = AlimeetingDiarDataset(
        wav_dir=args.valid_wav_dir,
        textgrid_dir=args.valid_textgrid_dir,
        sample_rate=16000,
        frame_shift=0.04, # 25fps to match vad_out_len
        musan_path=args.musan_path,
        rir_path=args.rir_path,
        noise_ratio=0.0,
        window_sec=args.window_sec,
        window_shift_sec=args.window_shift_sec
    )
    vad_out_len = args.vad_out_len if hasattr(args, 'vad_out_len') else 200
    def collate_fn_wrapper(batch):
        wavs, labels, spk_ids_list, fbanks, labels_len = valid_dataset.collate_fn(batch, vad_out_len=vad_out_len)
        max_spks_in_batch = labels.shape[1]
        spk_label_indices = torch.full((len(spk_ids_list), max_spks_in_batch), -1, dtype=torch.long)
        for i, spk_id_sample in enumerate(spk_ids_list):
            for j, spk_id in enumerate(spk_id_sample):
                if spk_id and spk_id in spk2int:
                    spk_label_indices[i, j] = spk2int[spk_id]
        return fbanks, labels, spk_label_indices, labels_len
    valid_dl = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_wrapper,
        num_workers=4,  # 建议：4~8，避免超过CPU核心数
        pin_memory=True,
        persistent_workers=True  # 避免每epoch重建进程
    ) 
    return valid_dl

def build_valid_dl_with_local_spk2int(args): 
    logger.info("Building valid dataloader with training spk2int...")
    valid_dataset = AlimeetingDiarDataset(
        wav_dir=args.valid_wav_dir,
        textgrid_dir=args.valid_textgrid_dir,
        sample_rate=16000,
        frame_shift=0.04, # 25fps to match vad_out_len
        musan_path=args.musan_path,
        rir_path=args.rir_path,
        noise_ratio=0.0,
        window_sec=args.window_sec,
        window_shift_sec=args.window_shift_sec
    )
    vad_out_len = args.vad_out_len if hasattr(args, 'vad_out_len') else 200
    def collate_fn_wrapper(batch):
        wavs, labels, spk_ids_list, fbanks, labels_len = valid_dataset.collate_fn(batch, vad_out_len=vad_out_len)
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
    valid_dl = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_wrapper,
        num_workers=4,  # 建议：4~8，避免超过CPU核心数
        pin_memory=True,
        persistent_workers=True  # 避免每epoch重建进程
    ) 
    return valid_dl

def build_test_dl(args, spk2int):
    logger.info("Building test dataloader with training spk2int...")
    test_dataset = AlimeetingDiarDataset(
        wav_dir=args.test_wav_dir,
        textgrid_dir=args.test_textgrid_dir,
        sample_rate=16000,
        frame_shift=0.04,
        musan_path=None,
        rir_path=None,
        noise_ratio=0.0,
        window_sec=args.window_sec,
        window_shift_sec=args.window_shift_sec
    )
    def collate_fn_wrapper(batch):
        wavs, labels, spk_ids_list, fbanks, labels_len = test_dataset.collate_fn(batch)
        max_spks_in_batch = labels.shape[1]
        spk_label_indices = torch.full((len(spk_ids_list), max_spks_in_batch), -1, dtype=torch.long)
        for i, spk_id_sample in enumerate(spk_ids_list):
            for j, spk_id in enumerate(spk_id_sample):
                if spk_id and spk_id in spk2int:
                    spk_label_indices[i, j] = spk2int[spk_id]
        return fbanks, labels, spk_label_indices, labels_len
    test_dl = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_wrapper,
        num_workers=4,  # 建议：4~8，避免超过CPU核心数
        pin_memory=True,
        persistent_workers=True  # 避免每epoch重建进程
    )
    return test_dl
#def vad_func(wav,sr):
#    fsmn_vad_model = AutoModel(model="fsmn-vad", model_revision="v2.0.4", disable_update=True)
#    if wav.dtype != np.int16:
#        wav = (wav * 32767).astype(np.int16)
#        result = fsmn_vad_model.generate(wav, fs=sr)
#        time_stamp = result[0]['value']
#        return time_stamp # in ms
#
#def spktochunks(args):
#    voxceleb2_dataset_dir=args.voxceleb2_dataset_dir
#    wavscp =  f"{args.voxceleb2_dataset_dir}/wav.scp"
#    spk2utt = f"{args.voxceleb2_dataset_dir}/spk2utt"
#    spk2wav = defaultdict(list)
#    wav2scp = {}
#    with open(wavscp,'r')as fscp:
#        for line in fscp:
#            line = line.strip().split()
#            key = line[0]
#            wav2scp[key] = line[1]
#
#    with open(spk2utt, 'r')as fspk:
#        for line in fspk:
#            line = line.strip().split()
#            key = line[0]
#            paths = [wav2scp[i] for i in line[1:]]
#            if key in spk2wav:
#                spk2wav[key].append(paths)
#            else:
#                spk2wav[key] = paths
#
#    spk2chunks=defaultdict(list)
#    for spk_id in spk2wav.keys():
#        for wav_path in spk2wav[spk_id]:
#            wav, sr = sf.read(wav_path)
#            if sr != 16000:
#                wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
#            time_stamp_list = vad_func(wav,sr=16000)
#            # in ms ->(/1000) in second ->(*16000) in sample points
#            speech_chunks = [wav[int(s*16):int(e*16)] for s, e in time_stamp_list]
#            if spk_id in spk2chunks:
#                spk2chunks[spk_id].append(speech_chunks)
#            else:
#                spk2chunks[spk_id] = speech_chunks
#
# def spktochunks(args):
#     """原始版本的spktochunks函数 - 已移除"""

# def spktochunks_fast(args, max_speakers=None, max_files_per_speaker=None, use_cache=None):
#     """内存优化的加速版本spktochunks函数 - 已移除"""

def process_batch(batch, spk2chunks, process=None, max_memory_mb=None, sub_batch_size=None, strict_memory_check=False):
    """处理一批说话人的音频数据"""
    import os
    import gc
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # 记录批次开始时间
    batch_start_time = time.time()
    
    # 收集当前批次的所有任务
    all_tasks = []
    for spk_id, speaker_tasks in batch:
        all_tasks.extend(speaker_tasks)
    
    logger.info(f"处理批次: {len(batch)} 个说话人，{len(all_tasks)} 个音频文件")
    
    # 如果任务数量过多，分批处理以避免内存溢出
    if sub_batch_size is None:
        sub_batch_size = 20  # 减少默认子批次大小，避免内存溢出
    sub_batches = [all_tasks[i:i + sub_batch_size] 
                   for i in range(0, len(all_tasks), sub_batch_size)]
    
    logger.info(f"将 {len(all_tasks)} 个任务分成 {len(sub_batches)} 个子批次处理")
    
    def process_audio_file_simple(task):
        """简化的音频处理函数，减少内存使用"""
        try:
            wav_path = task['wav_path']
            time_stamp_list = task['time_stamp_list']
            
            if not os.path.exists(wav_path):
                return None
            
            # 读取音频文件
            wav, sr = sf.read(wav_path)
            if sr != 16000:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            
            # 处理音频片段
            speech_chunks = []
            for s, e in time_stamp_list:
                chunk = wav[int(s*16):int(e*16)].copy()  # 使用copy避免引用原数组
                speech_chunks.append(chunk)
            
            # 立即释放wav数据
            del wav
            
            # 强制垃圾回收
            gc.collect()
            
            return {
                'spk_id': task['spk_id'],
                'chunks': speech_chunks
            }
            
        except Exception as e:
            logger.warning(f"处理音频文件失败 {task['wav_path']}: {e}")
            return None
    
    # 使用较少的线程以控制内存
    max_workers = min(1, os.cpu_count() or 1)  # 进一步减少线程数，避免内存竞争
    
    # 动态调整子批次大小
    current_sub_batch_size = sub_batch_size
    
    # 逐个处理子批次
    for sub_batch_idx, sub_batch in enumerate(sub_batches):
        sub_batch_start_time = time.time()
        # 检查内存使用
        if process and max_memory_mb:
            current_memory = process.memory_info().rss / 1024 / 1024
            if current_memory > max_memory_mb * 0.8:  # 提前在80%时进行垃圾回收
                logger.warning(f"子批次 {sub_batch_idx+1}/{len(sub_batches)} 前内存使用: {current_memory:.1f} MB，强制垃圾回收")
                gc.collect()
                current_memory = process.memory_info().rss / 1024 / 1024
                if current_memory > max_memory_mb * 0.95:  # 如果超过95%，强制跳过
                    logger.error(f"垃圾回收后内存仍然超限 {current_memory:.1f} MB，跳过子批次 {sub_batch_idx+1}")
                    continue  # 跳过这个子批次
                elif current_memory > max_memory_mb:
                    if strict_memory_check:
                        logger.error(f"垃圾回收后内存仍然超限 {current_memory:.1f} MB，跳过子批次 {sub_batch_idx+1}")
                        continue  # 跳过这个子批次
                    else:
                        logger.warning(f"垃圾回收后内存仍然超限 {current_memory:.1f} MB，但继续处理子批次 {sub_batch_idx+1}")
                        
                        # 动态减少后续子批次大小
                        if current_sub_batch_size > 5:
                            current_sub_batch_size = max(5, current_sub_batch_size // 2)
                            logger.info(f"动态调整子批次大小为: {current_sub_batch_size}")
        
        logger.info(f"处理子批次 {sub_batch_idx+1}/{len(sub_batches)}: {len(sub_batch)} 个音频文件")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交任务
            futures = [executor.submit(process_audio_file_simple, task) for task in sub_batch]
            
            # 收集结果
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        spk2chunks[result['spk_id']].extend(result['chunks'])
                        # 立即释放结果数据
                        del result['chunks']
                except Exception as e:
                    logger.error(f"处理任务失败: {e}")
        
        # 每个子批次完成后立即清理
        del futures
        gc.collect()
        
        # 强制清理内存
        import sys
        if hasattr(sys, 'getrefcount'):
            # 强制清理引用计数
            for _ in range(3):
                gc.collect()
        
        # 报告进度
        sub_batch_time = time.time() - sub_batch_start_time
        if process:
            current_memory = process.memory_info().rss / 1024 / 1024
            logger.info(f"子批次 {sub_batch_idx+1}/{len(sub_batches)} 完成，耗时: {sub_batch_time:.1f}秒，当前内存: {current_memory:.1f} MB")
        
        # 显示子批次进度
        if (sub_batch_idx + 1) % 5 == 0 or (sub_batch_idx + 1) == len(sub_batches):
            elapsed_time = time.time() - batch_start_time
            progress = ((sub_batch_idx + 1) / len(sub_batches)) * 100
            if sub_batch_idx > 0:
                avg_time_per_sub_batch = elapsed_time / (sub_batch_idx + 1)
                remaining_sub_batches = len(sub_batches) - (sub_batch_idx + 1)
                eta_seconds = remaining_sub_batches * avg_time_per_sub_batch
                eta_minutes = eta_seconds / 60
                
                if eta_minutes >= 1:
                    eta_str = f"{eta_minutes:.1f}分钟"
                else:
                    eta_str = f"{eta_seconds:.0f}秒"
                
                logger.info(f"批次进度: {sub_batch_idx+1}/{len(sub_batches)} ({progress:.1f}%) - 预计剩余: {eta_str}")
    
    # 批次处理完成后立即清理
    batch_total_time = time.time() - batch_start_time
    logger.info(f"批次处理完成，总耗时: {batch_total_time:.1f}秒")
    del all_tasks, sub_batches
    gc.collect()

def spktochunks_lazy(args, max_speakers=None, max_files_per_speaker=None):
    """
    内存优化版本的spktochunks函数
    使用串行处理和内存管理来避免OOM
    """
    import os
    import gc
    import time
    from collections import defaultdict
    
    logger.info("使用内存优化模式处理spktochunks...")
    
    # 记录开始时间
    start_time = time.time()
    
    # 使用更保守的内存管理策略
    spk2chunks = defaultdict(list)
    speaker_count = 0
    
    if args.compression_type == "gzip":
        with gzip.open(args.voxceleb2_spk2chunks_json, "rt", encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        spk_id = data["spk_id"]
                        
                        # 限制说话人数量（用于测试）
                        if max_speakers and speaker_count >= max_speakers:
                            break
                        
                        wav_paths = data["wav_paths"]
                        time_stamps = data["results"]
                        assert len(wav_paths) == len(time_stamps)
                        
                        # 限制每个说话人的文件数量（用于测试）
                        if max_files_per_speaker:
                            wav_paths = wav_paths[:max_files_per_speaker]
                            time_stamps = time_stamps[:max_files_per_speaker]
                        
                        # 串行处理，避免内存爆炸
                        for wav_path, time_stamp_list in zip(wav_paths, time_stamps):
                            try:
                                if not os.path.exists(wav_path):
                                    logger.warning(f"音频文件不存在: {wav_path}")
                                    continue
                                
                                wav, sr = sf.read(wav_path)
                                if sr != 16000:
                                    wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
                                
                                speech_chunks = [wav[int(s*16):int(e*16)] for s, e in time_stamp_list]
                                spk2chunks[spk_id].extend(speech_chunks)
                                
                                # 主动释放内存
                                del wav
                                
                            except Exception as e:
                                logger.warning(f"处理音频文件失败 {wav_path}: {e}")
                                continue
                        speaker_count += 1
                        
                        # 每处理10个说话人就强制垃圾回收
                        if speaker_count % 10 == 0:
                            gc.collect()
                            elapsed_time = time.time() - start_time
                            progress = (speaker_count / max_speakers * 100) if max_speakers else 0
                            logger.info(f"已处理 {speaker_count} 个说话人，当前说话人 {spk_id} 包含 {len(spk2chunks[spk_id])} 个音频文件，已用时间: {elapsed_time/60:.1f}分钟")
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"第{line_num}行JSON解析失败: {e}")
                    except Exception as e:
                        logger.error(f"处理说话人 {spk_id} 时发生错误: {e}")
                        continue
    
    # 最终垃圾回收
    gc.collect()
    total_time = time.time() - start_time
    total_minutes = total_time / 60
    logger.info(f"内存优化版本处理完成，共处理了 {speaker_count} 个说话人，总耗时: {total_minutes:.1f}分钟")
    return spk2chunks

def spktochunks_memory_safe(args, max_speakers=None, max_files_per_speaker=None):
    """
    超级内存安全版本的spktochunks函数
    专门用于内存非常受限的环境，会牺牲一些性能来确保不OOM
    """
    import os
    import gc
    import time
    import psutil
    from collections import defaultdict
    
    logger.info("使用超级内存安全模式处理spktochunks...")
    
    # 监控内存使用
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # 自动检测系统可用内存
    system_memory = psutil.virtual_memory()
    total_memory_mb = system_memory.total / 1024 / 1024
    available_memory_mb = system_memory.available / 1024 / 1024
    
    # 使用可用内存的40%（比fast版本更保守）
    max_memory_mb = min(available_memory_mb * 0.4, total_memory_mb * 0.4)
    max_memory_mb = int(max_memory_mb)  # 转换为整数
    
    logger.info(f"自动检测系统内存: 总计 {total_memory_mb:.0f} MB, 可用 {available_memory_mb:.0f} MB")
    logger.info(f"内存安全模式设置内存限制: {max_memory_mb} MB (可用内存的40%)")
    logger.info(f"初始内存使用: {initial_memory:.1f} MB")
    
    # 记录开始时间
    start_time = time.time()
    
    spk2chunks = defaultdict(list)
    speaker_count = 0
    
    if args.compression_type == "gzip":
        with gzip.open(args.voxceleb2_spk2chunks_json, "rt", encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        # 检查内存使用
                        current_memory = process.memory_info().rss / 1024 / 1024
                        if current_memory > max_memory_mb:
                            logger.warning(f"内存使用超过限制 {current_memory:.1f} MB > {max_memory_mb} MB，强制垃圾回收")
                            gc.collect()
                            current_memory = process.memory_info().rss / 1024 / 1024
                            if current_memory > max_memory_mb:
                                logger.error(f"垃圾回收后内存仍然超限 {current_memory:.1f} MB，停止处理")
                                break
                        
                        data = json.loads(line)
                        spk_id = data["spk_id"]
                        
                        # 限制说话人数量
                        if max_speakers and speaker_count >= max_speakers:
                            break
                        
                        wav_paths = data["wav_paths"]
                        time_stamps = data["results"]
                        assert len(wav_paths) == len(time_stamps)
                        
                        # 限制每个说话人的文件数量
                        if max_files_per_speaker:
                            wav_paths = wav_paths[:max_files_per_speaker]
                            time_stamps = time_stamps[:max_files_per_speaker]
                        
                        # 串行处理，每个文件立即释放
                        for wav_path, time_stamp_list in zip(wav_paths, time_stamps):
                            try:
                                if not os.path.exists(wav_path):
                                    continue
                                
                                wav, sr = sf.read(wav_path)
                                if sr != 16000:
                                    wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
                                
                                speech_chunks = [wav[int(s*16):int(e*16)] for s, e in time_stamp_list]
                                spk2chunks[spk_id].extend(speech_chunks)
                                
                                # 立即释放音频数据
                                del wav, speech_chunks
                                
                            except Exception as e:
                                logger.warning(f"处理音频文件失败 {wav_path}: {e}")
                                continue
                        
                        speaker_count += 1
                        
                        # 每处理5个说话人就检查内存和垃圾回收
                        if speaker_count % 5 == 0:
                            gc.collect()
                            current_memory = process.memory_info().rss / 1024 / 1024
                            elapsed_time = time.time() - start_time
                            progress = (speaker_count / max_speakers * 100) if max_speakers else 0
                            logger.info(f"已处理 {speaker_count} 个说话人，当前内存: {current_memory:.1f} MB，已用时间: {elapsed_time/60:.1f}分钟")
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"第{line_num}行JSON解析失败: {e}")
                    except Exception as e:
                        logger.error(f"处理说话人时发生错误: {e}")
                        continue
    
    # 最终清理
    gc.collect()
    final_memory = process.memory_info().rss / 1024 / 1024
    total_time = time.time() - start_time
    total_minutes = total_time / 60
    logger.info(f"内存安全版本处理完成，共处理了 {speaker_count} 个说话人，总耗时: {total_minutes:.1f}分钟")
    logger.info(f"最终内存使用: {final_memory:.1f} MB (增加: {final_memory - initial_memory:.1f} MB)")
    return spk2chunks

#    lines = gzip.open(args.voxceleb2_spk2chunks_json,'rt', encoding='utf-8').read().splitlines()
#    spk2chunks = defaultdict(list)
#    for line in tqdm(lines, desc=f"lines: "):
#         dict = json.loads(line)
#         spk_id = dict["spk_id"]
#         wav_paths = dict["wav_paths"]
#         time_stamps = dict["result"]
#         assert len(wav_paths) == len(time_stamps), f"len(wav_paths): {len(wav_paths)} vs len(time_stamps): {len(time_stamps)}"
#         for wav_path, time_stamp_list in zip(wav_paths, time_stamps):
#             wav, sr = sf.read(wav_path)
#             if sr != 16000:
#                 wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
#             # time_stamp_list is start /end , its unit is millisecond
#             # in ms ->(/1000) in second ->(*16000) in sample points
#             speech_chunks = [wav[int(s*16):int(e*16)] for s, e in time_stamp_list] 
#             if spk_id in spk2chunks:
#                 spk2chunks[spk_id].append(speech_chunks)
#             else:
#                 spk2chunks[spk_id] = speech_chunks
#
#    return spk2chunks

def build_global_spk2int(args):
    """
    统计给定TextGrid目录下的说话人，生成spk2int。
    支持多个目录联合统计。
    """
    spk_ids = set()
    # 1. real dataset  speaker i.e. alimeeting
    textgrid_dirs=[args.train_textgrid_dir, args.valid_textgrid_dir]
    for textgrid_dir in textgrid_dirs:
        tg_dir = Path(textgrid_dir)
        for tg_file in tg_dir.glob('*.TextGrid'):
            try:
                tg = textgrid.TextGrid.fromFile(str(tg_file))
                for tier in tg:
                    if tier.name.strip():
                        spk_ids.add(tier.name[-9:]) # 保证和AlimeetingDiarDataset一致
            except Exception as e:
                logger.warning(f"Could not process {tg_file}: {e}")

    # 2. simu dataset speaker i.e. we use voxceleb2 audio to simulate mix audio, so I use speakers of voxceleb2.
    utt2spk = f"{args.voxceleb2_dataset_dir}/utt2spk"
    with open(utt2spk, 'r') as f:
        for line in f:
            line = line.strip().split()
            spkid = line[1]
            spk_ids.add(spkid)


    spk2int = {spk: i for i, spk in enumerate(sorted(list(spk_ids)))}
    logger.info(f"Found {len(spk2int)} unique speakers in the provided set.")
    logger.info(f"spk2int: {spk2int}, spk_ids: {spk_ids}")
    return spk2int

#def build_simu_data_train_dl(args, spk2int, use_fast_version=True, max_speakers=None, max_files_per_speaker=None):
def build_simu_data_train_dl(args,spk2int):
    """
    构建模拟数据训练数据加载器
    
    Args:
        args: 参数对象
        spk2int: 说话人到整数的映射
        use_fast_version: 是否使用加速版本
        max_speakers: 最大处理说话人数量（用于测试）
        max_files_per_speaker: 每个说话人最大文件数量（用于测试）
    """
    logger.info("Building simu data train dataloader with training spk2int...")
    
    # 检查是否启用懒加载模拟数据模式
    if hasattr(args, 'use_lazy_simu') and args.use_lazy_simu:
        logger.info("启用懒加载模拟数据模式，跳过spktochunks预处理")
        train_dataset = SimuDiarMixer(
            spk2chunks=None,  # 懒加载模式下不需要预先加载
            voxceleb2_spk2chunks_json=args.voxceleb2_spk2chunks_json,
            sample_rate=16000,
            frame_length=0.025, # 25ms
            frame_shift=0.04, # 25fps(1s audio 25 labels, 8s audio 200 labels) to match vad_out_len, vad_out_len=200
            num_mel_bins=80,
            max_mix_len=8.0, # 8s
            min_silence=0.0,
            max_silence=4.0,
            min_speakers=1,
            max_speakers=3,
            target_overlap=0.2,
            musan_path=args.musan_path,
            rir_path=args.rir_path,
            noise_ratio=args.noise_ratio,
        )
        vad_out_len = args.vad_out_len if hasattr(args, 'vad_out_len') else 200
        def collate_fn_wrapper(batch):
            wavs, labels, spk_ids_list, fbanks, labels_len, data_source= train_dataset.collate_fn(batch, vad_out_len=vad_out_len)
            max_spks_in_batch = labels.shape[1] # b,spks,vad_out_len
            spk_label_indices = torch.full((len(spk_ids_list), max_spks_in_batch), -1, dtype=torch.long)
            for i, spk_id_sample in enumerate(spk_ids_list):
                for j, spk_id in enumerate(spk_id_sample):
                    if spk_id and spk_id in spk2int:
                        spk_label_indices[i, j] = spk2int[spk_id]
            return fbanks, labels, spk_label_indices, labels_len, data_source
        train_dl = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn_wrapper,
            num_workers=4,  # 建议：4~8，避免超过CPU核心数
            pin_memory=True,
            persistent_workers=True  # 避免每epoch重建进程
        )
        return train_dl
    
    # 选择使用哪个版本的spktochunks函数
#    if use_fast_version:
#        if hasattr(args, 'use_memory_safe') and args.use_memory_safe:
#            try:
#                logger.info("使用超级内存安全版本")
#                spk2chunks = spktochunks_memory_safe(args, max_speakers, max_files_per_speaker)
#                logger.info("内存安全版本初始化成功")
#            except Exception as e:
#                logger.warning(f"内存安全版本失败，回退到懒加载版本: {e}")
#                try:
#                    logger.info("尝试使用懒加载版本")
#                    spk2chunks = spktochunks_lazy(args, max_speakers, max_files_per_speaker)
#                    logger.info("懒加载版本初始化成功")
#                except Exception as e2:
#                    logger.warning(f"懒加载版本也失败，回退到加速版本: {e2}")
#                    logger.info("使用加速版本")
#                    spk2chunks = spktochunks_fast(args, max_speakers, max_files_per_speaker)
#        elif hasattr(args, 'use_lazy_loading') and args.use_lazy_loading:
#            try:
#                logger.info("尝试使用懒加载版本")
#                spk2chunks = spktochunks_lazy(args, max_speakers, max_files_per_speaker)
#                logger.info("懒加载版本初始化成功")
#            except Exception as e:
#                logger.warning(f"懒加载版本失败，回退到加速版本: {e}")
#                logger.info("使用加速版本")
#                spk2chunks = spktochunks_fast(args, max_speakers, max_files_per_speaker)
#        else:
#            logger.info("使用加速版本")
#            spk2chunks = spktochunks_fast(args, max_speakers, max_files_per_speaker)
#    else:
#        logger.info("使用原始版本")
#        spk2chunks = spktochunks(args)
#    
#    train_dataset = SimuDiarMixer(
#        spk2chunks=spk2chunks,
#        sample_rate=16000,
#        frame_length=0.025, # 25ms
#        frame_shift=0.04, # 25fps(1s audio 25 labels, 8s audio 200 labels) to match vad_out_len, vad_out_len=200
#        num_mel_bins=80,
#        max_mix_len=8.0, # 8s
#        min_silence=0.0,
#        max_silence=4.0,
#        min_speakers=1,
#        max_speakers=3,
#        target_overlap=0.2,
#        musan_path=args.musan_path,
#        rir_path=args.rir_path,
#        noise_ratio=args.noise_ratio,
#    )
#    vad_out_len = args.vad_out_len if hasattr(args, 'vad_out_len') else 200
#    def collate_fn_wrapper(batch):
#        wavs, labels, spk_ids_list, fbanks, labels_len = train_dataset.collate_fn(batch, vad_out_len=vad_out_len)
#        max_spks_in_batch = labels.shape[1] # b,spks,vad_out_len
#        spk_label_indices = torch.full((len(spk_ids_list), max_spks_in_batch), -1, dtype=torch.long)
#        for i, spk_id_sample in enumerate(spk_ids_list):
#            for j, spk_id in enumerate(spk_id_sample):
#                if spk_id and spk_id in spk2int:
#                    spk_label_indices[i, j] = spk2int[spk_id]
#        return fbanks, labels, spk_label_indices, labels_len
#    train_dl = DataLoader(
#        train_dataset,
#        batch_size=args.batch_size,
#        shuffle=True,
#        collate_fn=collate_fn_wrapper,
#        num_workers=4,  # 建议：4~8，避免超过CPU核心数
#        pin_memory=True,
#        persistent_workers=True  # 避免每epoch重建进程
#
#    )
#    return train_dl

def main():
    parser = get_parser()
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    params = get_params(args)
    params.update(vars(args))
    #logging.info(f"params: {params}")
    fix_random_seed(params.seed)

    # accelerator must be created before any DDP stuff
    from accelerate import (
        Accelerator,
        DDPCommunicationHookType,
        DistributedDataParallelKwargs,
    )
    ddp_kwargs = DistributedDataParallelKwargs(
        comm_hook=DDPCommunicationHookType.FP16, find_unused_parameters=True
    )
    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs],
        mixed_precision="fp16",
        project_dir=args.exp_dir,
    )

    # 构建global spk2int (用真实的训练集和验证集和模拟的训练集)
    spk2int = build_global_spk2int(args)
    logger.info(f"spk2int: {spk2int}")
    params.n_all_speakers = len(spk2int)
    
    
    # build train/valid dataloader
    if args.train_stage==1:
        # using simulated training set as trainset, dev set of alimeeting as devset.
        train_dl_simu = build_simu_data_train_dl(
            args, spk2int, 
            #use_fast_version=args.use_fast_spktochunks,
            #max_speakers=args.max_speakers_test,
            #max_files_per_speaker=args.max_files_per_speaker_test
        )
        valid_dl = build_valid_dl(args, spk2int)
    elif args.train_stage==2:
        # using 80% simulated training set as trainset and 20% train set of alimeeting as trainset, dev set of alimeeting as devset.
        train_dl_simu = build_simu_data_train_dl(
            args, spk2int,
            #use_fast_version=args.use_fast_spktochunks,
            #max_speakers=args.max_speakers_test,
            #max_files_per_speaker=args.max_files_per_speaker_test
        )
        train_dl = build_train_dl(args, spk2int)
        valid_dl = build_valid_dl(args, spk2int)
    elif args.train_stage==3:
        train_dl = build_train_dl(args, spk2int)
        valid_dl = build_valid_dl(args, spk2int)

    writer: Optional[SummaryWriter] = None
    if accelerator.is_main_process and params.tensorboard:
        writer = SummaryWriter(log_dir=f"{args.exp_dir}/tensorboard")

    gradient_accumulation = 1
    # Note: scale_window is not used in the current code, but kept for reference
    scale_window = max(int(2**14 / accelerator.num_processes / gradient_accumulation), 1)
    logger.info(f"The scale window is set to {scale_window}.")
    logger.info(f"params: {params}")
    # Model
    model = SSNDModel(
        speaker_pretrain_model_path=params.speaker_pretrain_model_path,
        extractor_model_type=params.extractor_model_type,
        feat_dim=params.feat_dim,
        emb_dim=params.emb_dim,
        q_det_aux_dim=params.q_det_aux_dim,
        q_rep_aux_dim=params.q_rep_aux_dim,
        d_model=params.d_model,
        nhead=params.nhead,
        d_ff=params.d_ff,
        num_layers=params.num_layers,
        max_speakers=params.max_speakers,
        vad_out_len=params.vad_out_len,
        arcface_margin=params.arcface_margin,
        arcface_scale=params.arcface_scale,
        pos_emb_dim=params.pos_emb_dim,
        max_seq_len=params.max_seq_len,
        n_all_speakers=params.n_all_speakers,
        mask_prob=params.mask_prob,
        training=True,
        arcface_weight=params.arcface_weight,
        bce_gamma=params.bce_gamma,
        bce_alpha=params.bce_alpha,
        decoder_dropout=params.decoder_dropout,
        extractor_dropout=params.extractor_dropout,
        #out_bias=params.out_bias,
    )
    # 强制初始化DetectionDecoder输出层bias为0
    #with torch.no_grad():
    #    model.det_decoder.out_proj.bias.fill_(0.0)

    logger.info(f"model: {model}")
    num_param = sum([p.numel() for p in model.parameters()])
    logger.info(f"Number of model parameters: {num_param/1e6} M")

    assert params.save_every_n >= params.average_period
    model_avg: Optional[nn.Module] = None
    if accelerator.is_main_process:  # it is same as rank == 0
        # model_avg is only used with rank 0
        model_avg = copy.deepcopy(model).to(torch.float64)

    # load model parameters for model fine-tuning
    if params.do_finetune:
        assert params.start_epoch == 1, "Fine-tune must start from epoch 1"
        modules = params.init_modules.split(",") if params.init_modules else None
        checkpoints = load_model_params(
            ckpt=params.finetune_ckpt, model=model, init_modules=modules, strict=False
        )
        # Need to update the model_avg if use initialisation
        if accelerator.is_main_process:  # it is same as rank == 0
            # model_avg is only used with rank 0
            model_avg = copy.deepcopy(model).to(torch.float64)
    else:
        # resuming training
        checkpoints = load_checkpoint_if_available(
            params=params, model=model, model_avg=model_avg
        )  
    
    if True:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    ## get optimizer, scheduler
    optimizer, scheduler = get_optimizer_scheduler(params, model)

    ## accelerated model, optimizer, scheduler ,train_dl, valid_dl
    if args.train_stage==1:
        model, optimizer, scheduler, train_dl_simu, valid_dl = accelerator.prepare(
            model, optimizer, scheduler, train_dl_simu, valid_dl
        )
        train_dl=train_dl_simu
    elif args.train_stage==2:
        model, optimizer, scheduler, train_dl_simu, train_dl, valid_dl = accelerator.prepare(
            model, optimizer, scheduler, train_dl_simu, train_dl,valid_dl
        )
    elif args.train_stage==3:
        model, optimizer, scheduler, train_dl, valid_dl = accelerator.prepare(
            model, optimizer, scheduler, train_dl,valid_dl
        )
    # logging.info(f"After accelerator: model: {model}")
    scaler: Optional[GradScaler] = None
    logger.info(f"start training from epoch {params.start_epoch}")
    logger.info(f"Train set grouped total_num_itrs = {len(train_dl)}")

    # fix_random_seed(params.seed) # fairseq1 seed=1337 # this may be not correct at here.
    #extrator_frozen_steps = 100000  # 前10000步冻结extrator
    #extrator_frozen = True

    for epoch in range(params.start_epoch, params.num_epochs + 1):
        # fix_random_seed(params.seed + epoch-1) # fairseq1 seed=1337
        params.cur_epoch = epoch
        if args.train_stage==2:
            train_one_epoch_multi(
                params=params,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                train_dl_simu=train_dl_simu,
                train_dl=train_dl,
                valid_dl=valid_dl,
                accelerator=accelerator,
                model_avg=model_avg,
                writer=writer,
            )
            if accelerator.is_main_process:
                save_checkpoint(
                    params=params,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )
        else:
            train_one_epoch(
                params=params,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                #train_dl_simu=train_dl_simu,
                train_dl=train_dl,
                valid_dl=valid_dl,
                accelerator=accelerator,
                model_avg=model_avg,
                writer=writer,
            )
            if accelerator.is_main_process:
                save_checkpoint(
                    params=params,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )  
        #if params.batch_idx_train >= extrator_frozen_steps and extrator_frozen:
        #    extrator_frozen = False
        #    for p in model.module.extractor.speech_encoder.parameters():
        #        p.requires_grad = True
        #    logging.info(f"[Unfreeze] extractor解冻 at step {params.batch_idx_train}")
    if writer:
        writer.close()
    logger.info("Done!")
    if accelerator.num_processes > 1:
        torch.distributed.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()                                                                                                                                                                                                                           
