import argparse
import copy
import logging
import warnings
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, Optional, Tuple, Union, List

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
import textgrid

from ssnd_model import SSNDModel
from alimeeting_diar_dataset import AlimeetingDiarDataset
import pickle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)

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
    parser.add_argument("--weight-decay",type=float, default=0.001, help= "AdamW optimizer weight_decay")
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
        weight_decay=params.weight_decay,  # 从0.01增加到0.05
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
):
    """
    batch: (fbanks, labels, spk_label_idx, labels_len)
    """
    with torch.set_grad_enabled(is_training):
        fbanks, labels, spk_label_idx, labels_len = batch  # [B, T, F], [B, N, T], [B, N], [B], N is num of speakers,
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
        if is_training and B > 0:
            print(f"[LOSS DIAG] vad_pred.shape={vad_pred.shape}, padded_vad_labels.shape={padded_vad_labels.shape}")
            print(f"[LOSS DIAG] vad_pred[0, :, :10]={vad_pred[0, :, :10].detach().cpu().numpy()}")
            print(f"[LOSS DIAG] padded_vad_labels[0, :, :10]={padded_vad_labels[0, :, :10].detach().cpu().numpy()}")
            print(f"[DIAG] loss: {loss.item()}, bce_loss: {bce_loss.item()}, arcface_loss: {arcface_loss.item()}")
            # 添加预测概率的统计信息
            vad_probs = torch.sigmoid(vad_pred)
            print(f"[DIAG] vad_probs.mean(): {vad_probs.mean().item()}, vad_probs.std(): {vad_probs.std().item()}")
            print(f"[DIAG] vad_probs.max(): {vad_probs.max().item()}, vad_probs.min(): {vad_probs.min().item()}")
            
            # 添加VAD预测分布分析
            vad_probs_flat = vad_probs.flatten()
            positive_preds = vad_probs_flat > 0.5
            print(f"[VAD DIAG] 预测为正样本的比例: {positive_preds.float().mean().item():.4f}")
            print(f"[VAD DIAG] 真实正样本比例: {padded_vad_labels.float().mean().item():.4f}")
            
            # 分析每个说话人的预测
            for i in range(min(3, vad_probs.shape[1])):  # 只看前3个说话人
                spk_probs = vad_probs[0, i, :]
                spk_labels = padded_vad_labels[0, i, :]
                spk_positive_preds = spk_probs > 0.5
                print(f"[VAD DIAG] Speaker {i}: 预测正样本比例={spk_positive_preds.float().mean().item():.4f}, 真实正样本比例={spk_labels.float().mean().item():.4f}")
        # DER 计算
        outs_prob = torch.sigmoid(vad_pred).detach().cpu().numpy()
        print(f"[DER DIAG] outs_prob.shape={outs_prob.shape}, padded_vad_labels.shape={padded_vad_labels.shape}, labels_len={labels_len}")
        print(f"[DER DIAG] labels_len.sum()={labels_len.sum() if hasattr(labels_len, 'sum') else labels_len}")
        mi, fa, cf, acc, der = model.module.calc_diarization_result(
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
) -> MetricsTracker:
    """Run the validation process."""
    model.eval()

    tot_loss = MetricsTracker()
    batch_nums = []
    tot_loss_valid = 0
    for batch_idx, batch in enumerate(valid_dl):
        # 新增详细打印和断言
        try:
            fbanks, labels, spk_label_idx, labels_len = batch
            #print(f"[VAL DIAG] batch_idx={batch_idx}, fbanks.shape={getattr(fbanks, 'shape', None)}, labels.shape={getattr(labels, 'shape', None)}, spk_label_idx.shape={getattr(spk_label_idx, 'shape', None)}, labels_len={labels_len}")
            #print(f"[VAL DIAG] batch_idx={batch_idx}, labels_len.sum()={labels_len.sum().item() if hasattr(labels_len, 'sum') else labels_len}")
            #print(f"[VAL DIAG] batch_idx={batch_idx}, labels[0, :, :10]={labels[0, :, :10] if hasattr(labels, 'shape') and labels.shape[0] > 0 else 'N/A'}")
            #print(f"[VAL DIAG] batch_idx={batch_idx}, spk_label_idx[0]={spk_label_idx[0] if hasattr(spk_label_idx, 'shape') and spk_label_idx.shape[0] > 0 else 'N/A'}")
            assert fbanks is not None and labels is not None and spk_label_idx is not None and labels_len is not None, f"Batch {batch_idx} has None values!"
            assert hasattr(labels_len, 'sum') and labels_len.sum().item() > 0, f"Batch {batch_idx} has zero valid frames! labels_len={labels_len}"
            assert hasattr(labels, 'shape') and labels.shape[0] > 0, f"Batch {batch_idx} labels is empty!"
        except Exception as e:
            print(f"[VAL ERROR] batch_idx={batch_idx}, error: {e}")
            raise
        loss, loss_info = compute_loss(
            model=model,
            batch=batch,
            is_training=False,
        )
        loss = loss.detach()  # 确保无梯度
        batch_nums.append(batch_idx)
        # assert loss.requires_grad is False  # 移除该断言
        tot_loss_valid = tot_loss_valid + loss_info["loss"]
        tot_loss = tot_loss + loss_info

        if batch_idx == 0:
            fbanks, labels, spk_label_idx, labels_len = batch
            #print(f'[VAL DIAG] labels.shape: {labels.shape}, spk_label_idx.shape: {spk_label_idx.shape}, labels_len: {labels_len}')
            #print(f'[VAL DIAG] labels[0, :, :10]: {labels[0, :, :10]}')
            #print(f'[VAL DIAG] spk_label_idx[0]: {spk_label_idx[0]}')

    for item in tot_loss.keys():
        tot_loss[item] = tot_loss[item] / len(batch_nums)

    if writer: # this is main process
        for key, value in tot_loss.items():
            writer.add_scalar(f"valid/{key}", value, batch_idx_train)

    loss_value = tot_loss["loss"]
    if loss_value < params.best_valid_loss:
        params.best_valid_epoch = params.cur_epoch
        params.best_valid_loss = loss_value

    der_value = tot_loss["DER"]
    if der_value < params.best_valid_der:
        params.best_valid_epoch = params.cur_epoch
        params.best_valid_der = der_value

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
    # extrator冻结/解冻逻辑
    #extrator_frozen_steps = 100000
    #extrator_frozen = params.get('extrator_frozen', True)
    for batch_idx, batch in enumerate(train_dl):
        # 每个batch前判断是否需要冻结/解冻extrator
        if  params.batch_idx_train < params.extrator_frozen_steps:
            for p in model.module.extractor.speech_encoder.parameters():
                p.requires_grad = False
            logging.info(f"[Freeze] extractor speech encoder parameters at step {params.batch_idx_train}")
        elif params.batch_idx_train >= params.extrator_frozen_steps:
            for p in model.module.extractor.speech_encoder.parameters():
                p.requires_grad = True
            logging.info(f"[Unfreeze] extractor speech encoder unfreeze at step {params.batch_idx_train}")
            #params['extrator_frozen'] = False
        params.batch_idx_train += 1
        batch_size = params.batch_size

        optimizer.zero_grad()
        train_batch_nums.append(batch_idx)
        loss, loss_info = compute_loss(
            model=model,
            batch=batch,
            is_training=True,
        )
        accelerator.backward(loss)  # instead of loss.backward()

        # grad clip(todo run)
        grad_norm = None
        if params.grad_clip:
            if accelerator.sync_gradients:
                grad_norm = accelerator.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )

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
            logging.info(
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

            logging.info(
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
            logging.info(
                f"end of epoch {params.cur_epoch}, batch_idx: {batch_idx} "
                f"batch_idx_train: {params.batch_idx_train-1}, {loss_info}, "
                f"batch size: {batch_size}, grad_norm: {grad_norm}, grad_scale: {grad_scale}, "
                f"lr: {cur_lr}, "
            )
        if batch_idx > 0 and batch_idx % params.valid_interval == 0:
            logging.info("Computing validation loss")
            valid_info = compute_validation_loss(
                params=params,
                model=model,
                valid_dl=valid_dl,
                batch_idx_train=params.batch_idx_train,
                writer=writer,
            )
            model.train()
            logging.info(
                f"[Eval] - Epoch {params.cur_epoch}, batch_idx_train: {params.batch_idx_train-1}, "
                f" validation: {valid_info}"
            )
            logging.info(
                f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated()//1000000}MB"
            )
        # 诊断：打印dataloader输出的shape和部分内容
        if batch_idx == 0:
            fbanks, labels, spk_label_idx, labels_len = batch
            print(f"[DIAG] Dataloader batch[0] fbanks.shape: {fbanks.shape}, labels.shape: {labels.shape}, spk_label_idx.shape: {spk_label_idx.shape}, labels_len: {labels_len}")
            print(f"[DIAG] Dataloader batch[0] labels[0, :, :10]: {labels[0, :, :10]}")
            print(f"[DIAG] Dataloader batch[0] spk_label_idx[0]: {spk_label_idx[0]}")
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
    logging.info(f"Loading checkpoint from {ckpt}")
    checkpoint = torch.load(ckpt, map_location="cpu")

    # if module list is empty, load the whole model from ckpt
    if not init_modules:
        if next(iter(checkpoint["model"])).startswith("module."):
            logging.info("Loading checkpoint saved by DDP")

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
            logging.info(f"Loading parameters starting with prefix {module}")
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
    logging.info(f"Found {len(spk2int)} unique speakers in the provided set.")
    logging.info(f"spk2int: {spk2int}, spk_ids: {spk_ids}")
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
    logging.info("Building train dataloader with training spk2int...")
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
        collate_fn=collate_fn_wrapper
    )   
    return train_dl

def build_train_dl_with_local_spk2int(args,): 
    logging.info("Building train dataloader with training spk2int...")
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
        logging.info(f"train set spk2int len: {len(spk2int)} in fn build_train_dl_with_local_spk2int")
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
        collate_fn=collate_fn_wrapper
    )   
    return train_dl

def build_valid_dl(args, spk2int): 
    logging.info("Building valid dataloader with training spk2int...")
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
        collate_fn=collate_fn_wrapper
    ) 
    return valid_dl

def build_valid_dl_with_local_spk2int(args): 
    logging.info("Building valid dataloader with training spk2int...")
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
        collate_fn=collate_fn_wrapper
    ) 
    return valid_dl

def build_test_dl(args, spk2int):
    logging.info("Building test dataloader with training spk2int...")
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
        collate_fn=collate_fn_wrapper
    )
    return test_dl

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

    # 构建spk2int (用训练集和验证集联合)
    spk2int = build_spk2int(args.train_textgrid_dir, args.valid_textgrid_dir)
    logging.info(f"spk2int: {spk2int}")
    params.n_all_speakers = len(spk2int)
    
    
    # build train/valid dataloader
    train_dl = build_train_dl(args, spk2int)
    valid_dl = build_valid_dl(args, spk2int)
    # build train/vaild dataloader with local spk2int
    #train_dl = build_train_dl_with_local_spk2int(args)
    #valid_dl = build_valid_dl_with_local_spk2int(args)
    #batch = next(iter(train_dl))
    #_, _, spk_label_idx, _ = batch
    #params.n_all_speakers = int((spk_label_idx.max() + 1).item()) if (spk_label_idx >= 0).any() else args.max_speakers
    #if args.test_textgrid_dir and args.test_wav_dir:
    #    test_dl = build_test_dl(args, spk2int)

    writer: Optional[SummaryWriter] = None
    if accelerator.is_main_process and params.tensorboard:
        writer = SummaryWriter(log_dir=f"{args.exp_dir}/tensorboard")

    gradient_accumulation = 1
    # Note: scale_window is not used in the current code, but kept for reference
    scale_window = max(int(2**14 / accelerator.num_processes / gradient_accumulation), 1)
    logging.info(f"The scale window is set to {scale_window}.")
    logging.info(f"params: {params}")
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
        #out_bias=params.out_bias,
    )
    # 强制初始化DetectionDecoder输出层bias为0
    #with torch.no_grad():
    #    model.det_decoder.out_proj.bias.fill_(0.0)

    logging.info(f"model: {model}")
    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param/1e6} M")

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
    model, optimizer, scheduler, train_dl, valid_dl = accelerator.prepare(
        model, optimizer, scheduler, train_dl, valid_dl
    )
    # logging.info(f"After accelerator: model: {model}")
    scaler: Optional[GradScaler] = None
    logging.info(f"start training from epoch {params.start_epoch}")
    logging.info(f"Train set grouped total_num_itrs = {len(train_dl)}")

    # fix_random_seed(params.seed) # fairseq1 seed=1337 # this may be not correct at here.
    #extrator_frozen_steps = 100000  # 前10000步冻结extrator
    #extrator_frozen = True

    for epoch in range(params.start_epoch, params.num_epochs + 1):
        # fix_random_seed(params.seed + epoch-1) # fairseq1 seed=1337
        params.cur_epoch = epoch
        train_one_epoch(
            params=params,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            train_dl=train_dl,
            valid_dl=valid_dl,
            accelerator=accelerator,
            model_avg=model_avg,
            writer=writer,
        )
        if accelerator.is_main_process:
            save_checkpoint(
                params=params,                                                                                                                                                                                    model=model,
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
    logging.info("Done!")
    if accelerator.num_processes > 1:
        torch.distributed.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()                                                                                                                                                                                                                           
