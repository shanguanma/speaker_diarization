#!/usr/bin/env python3
# Author: Duo MA
# Email: maduo@cuhk.edu.cn

"""
for example, using two gpus to train this model without grad clip norm and no freeze speech encoder
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_unfreeze_with_musan
   CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12673 ts_vad2/train_accelerate_ddp2.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --freeze-updates 0\
    --grad-clip false\
    --musan-path $musan_path \
    --exp-dir $exp_dir



    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1
    musan_path=/mntcephfs/lee_dataset/asr/musan
    rir_path=/mntcephfs/lee_dataset/asr/RIRS_NOISES
    exp_dir=/mntcephfs/lab_data/maduo/exp/speaker_diarization/ts_vad2/ts_vad2_two_gpus_unfreeze_with_musan_and_rirs
   CUDA_VISIABLE_DEVICES=0,1 accelerate launch --main_process_port 12683 ts_vad2/train_accelerate_ddp2.py \
    --world-size 2 \
    --num-epochs 20\
    --start-epoch 1\
    --freeze-updates 0\
    --grad-clip false\
    --musan-path $musan_path \
    --rir-path $rir_path \
    --exp-dir $exp_dir


"""

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

from build_datasets import load_dataset
from build_datasets import TSVADDataConfig
from model import TSVADModel
from model import TSVADConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)
LRSchedulerType = Union[torch.optim.lr_scheduler._LRScheduler]


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of GPUs for DDP training.",
    )

    parser.add_argument(
        "--tensorboard",
        type=str2bool,
        default=True,
        help="Should various information be logged in tensorboard.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=30,
        help="Number of epochs to train.",
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
        default=4000,
        help="number warmup iters of training ",
    )
    parser.add_argument(
        "--freeze-updates",
        type=int,
        default=4000,
        help="number freeze speech_encoder  iters of training ",
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
        "--start-epoch",
        type=int,
        default=1,
        help="""Resume training from this epoch. It should be positive.
        If larger than 1, it will load checkpoint from
        exp-dir/epoch-{start_epoch-1}.pt
        """,
    )
    parser.add_argument(
        "--seed",
        type=int,
        # default=42,
        default=1337,
        help="The seed for random generators intended for reproducibility",
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        default="zipformer/exp",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
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
    parser.add_argument(
        "--grad-clip",
        type=str2bool,
        default=False,
        help="whether grad clip norm at traing stage",
    )
    parser.add_argument(
        "--feature-grad-mult",
        type=float,
        default=0.1,
        help="""wavlm default config is 0.1,
        it will effect ConvFeatureExtractionModel of wavlm model or hubert model,
        if it is setted to 0.0, ConvFeatureExtractionModel will
        be freezed in tsvad model training stage.
        if it is setted to 0.1,parameters of ConvFeatureExtractionModel of
        wavlm will updated in tsvad model training stage.
        """,
    )
    parser.add_argument("--lr", type=float, default=2e-4, help="adamw init lr rate.")
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
    parser.add_argument("--enhanced-audio-dir",type=str, help="use zipenhancer speech enhancement model or sherpa_onnx gtcrn model to generate audio directory")
    add_data_arguments(parser)
    add_model_arguments(parser)
    add_data_model_common_arguments(parser)
    add_finetune_arguments(parser)
    return parser
def add_data_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--musan-path",
        type=none_or_str,
        nargs="?",
        default=None,
        help="musan noise wavform directory path",
    )
    parser.add_argument(
        "--rir-path",
        type=none_or_str,
        nargs="?",
        default=None,
        help="rir noise wavform directory path",
    )
    parser.add_argument(
        "--spk-path",
        type=str,
        default="/mntcephfs/lab_data/maduo/model_hub/ts_vad/spk_embed/alimeeting/SpeakerEmbedding",
        help="target speaker embedding path",
    )

    parser.add_argument(
        "--speaker-embedding-name-dir",
        type=str,
        default="cam++_en_zh_advanced_feature_dir",
        help="specify speaker embedding directory name",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="/mntcephfs/lab_data/maduo/datasets/alimeeting",  # oracle target audio and labels path
        help="path to target audio and mixture labels root directory.",
    )
    parser.add_argument("--dataset-name",type=str,default="magicdata-ramc",help="dataset name", )
    parser.add_argument("--max-num-speaker", type=int, default=4, help="support max number of speaker in ts_vad")
    parser.add_argument("--enhance-ratio", type=float, default=0.0, help="if >0, will add speech enhance audio augment")
    
    return parser

def add_data_model_common_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--speech-encoder-type",
        type=str,
        default="CAM++",
        help="speech encoder arch ",
    )
    parser.add_argument(
        "--speaker-embed-dim",
        type=int,
        default=192,
        help="target speaker model output feature dimension and is also tsvad speech encoder output feature dimension ",
    )
    parser.add_argument(
        "--rs-len",
        type=int,
        default=4,
        help="mix audio lenght of per sample in ts vad model",
    )
    parser.add_argument(
        "--segment-shift",
        type=float,
        default=1,
        help="mix audio segment shift stride of per sample in ts vad model",
    )
    parser.add_argument("--single-backend-type",type=str, default="transformer",help="choice from `transformer` , `mamba`, `mamba_v2` or `mamba2`")
    parser.add_argument("--multi-backend-type",type=str, default="transformer",help="choice from `transformer` , `mamba`, `mamba_v2` or `mamba2` ")
    return parser

def add_model_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--speech-encoder-path",
        type=str,
        default="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt",
        help="speech encoder pretrain model path",
    )
    parser.add_argument(
        "--select-encoder-layer-nums",
        type=int,
        default=6,
        help="""it will select transformer encoder layer of wavlm model.i.e. --select-encoder-layer-nums 6, means that we only use cnn front and first 6 transformer layer of wavlm in tsvad model.""",
    )
    parser.add_argument(
        "--wavlm-fuse-feat-post-norm",
        type=str2bool,
        default=False,
        help="""if true, it will apply layernorm on weight sum of all transformer layer feature in pretrained wavlm model""",
    )
    parser.add_argument(
        "--speech-encoder-config",
        type=str,
        default="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wav-bert2.0/config.json",
        help="""this config is only used to instantiate wav-bert 2.0 model, this model is used at Seamless model."""
    )
    parser.add_argument("--num-transformer-layer",type=int,default=2, help="""single_backend or multi_backend number of layers""")
    parser.add_argument("--d-state",type=int,default=64,help="""d_state of mamba2 network""")
    parser.add_argument("--expand",type=int,default=4,help="""expand of mamba2 network""")
    parser.add_argument("--ots-vad-style", type=str, default="", help="choice it from `v1`")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate in model")
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


def get_params() -> AttributeDict:
    params = AttributeDict(
        {
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_der": float("inf"),
            "best_valid_der": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "log_interval": 500,  # same as fairseq
            "reset_interval": 200,
            # "valid_interval": 1500,  # same as fairseq
            "valid_interval": 500,
            # "ignore_id": -1,
            # "label_smoothing": 0.1,
            "batch_size": 64,
        }
    )
    return params


def calculate_loss(outs, labels, labels_len):
    total_loss = 0
    for i in range(labels_len.size(0)):
        total_loss += F.binary_cross_entropy_with_logits(
            outs[i, :, : labels_len[i]], labels[i, :, : labels_len[i]]
        )
    return total_loss / labels_len.size(0)


def get_optimizer_scheduler(params, model, world_size):
    from torch.optim import AdamW

    optimizer = AdamW(
        model.parameters(),
        lr=params.lr,
        betas=(0.9, 0.98),
        eps=1e-08,
        weight_decay=0.01,
    )
    if params.lr_type=="PolynomialDecayLR":
        # optimizer = AdamW(model.parameters(),lr=5e-5,betas=(0.9, 0.98)) # same as fairseq2
        from polynomial import PolynomialDecayLR

        scheduler = PolynomialDecayLR(
            optimizer, params.max_updates, params.warmup_updates, power=1.0
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


def compute_loss(
    model: Union[nn.Module, DDP],
    batch: dict,
    is_training: bool,
    batch_idx_train: int,
):
    with torch.set_grad_enabled(is_training):
        ref_speech = batch["net_input"]["ref_speech"]
        target_speech = batch["net_input"]["target_speech"]
        labels = batch["net_input"]["labels"]
        labels_len = batch["net_input"]["labels_len"]
        outs = model(
            ref_speech=ref_speech,
            target_speech=target_speech,
            labels=labels,
            num_updates=batch_idx_train,
        )
        loss = calculate_loss(outs=outs, labels=labels, labels_len=labels_len)

        ## public logger
        outs_prob = torch.nn.functional.sigmoid(outs)
        # convert tensor to numpy
        # logging.info(f"outs_prob requries_grad: {outs_prob.requries_grad}")
        outs_prob = outs_prob.data.cpu().numpy()
        mi, fa, cf, acc, der = model.module.calc_diarization_result(
            # mi, fa, cf, acc, der = model.calc_diarization_result(
            outs_prob.transpose((0, 2, 1)),
            labels.transpose(1, 2),
            labels_len,
        )

    assert loss.requires_grad == is_training
    info = {}
    info["loss"] = loss.detach().cpu().item()
    info["DER"] = der
    info["ACC"] = acc
    info["MI"] = mi
    info["FA"] = fa
    info["CF"] = cf
    return loss, info


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
        loss, loss_info = compute_loss(
            model=model,
            batch=batch,
            is_training=False,
            batch_idx_train=batch_idx_train,
        )
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

# 分布式训练专用配置
#prof = torch.profiler.profile(
#    activities=[torch.profiler.ProfilerActivity.CPU,
#        torch.profiler.ProfilerActivity.CUDA,],
#    schedule=torch.profiler.schedule(
#        wait=2,
#        warmup=1,
#        active=3,
#        repeat=2
#    ),
#    with_flops=True,  # 计算浮点运算量
#    with_modules=True  # 显示调用模块信息
#)

#from memory_profiler import profile
#@profile
def train_one_epoch(
    params: AttributeDict,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LRSchedulerType,
    scaler: GradScaler,
    accelerator,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    model_avg: Optional[nn.Module] = None,
    writer: Optional[SummaryWriter] = None,
) -> None:
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
    model.train()

    tot_loss = MetricsTracker()
    train_batch_nums = []
    # only for debug
    #prof= torch.profiler.profile(
    #    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    #    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/ts_vad'),
    #    record_shapes=True,
    #    profile_memory=True,
    #    with_stack=True,)
    #prof.start()
    for batch_idx, batch in enumerate(train_dl):
        #prof.step()
        params.batch_idx_train += 1
        batch_size = params.batch_size

        optimizer.zero_grad()
        train_batch_nums.append(batch_idx)
        loss, loss_info = compute_loss(
            model=model,
            batch=batch,
            is_training=True,
            batch_idx_train=params.batch_idx_train,
        )
        accelerator.backward(loss)  # instead of loss.backward()

        # grad clip(todo run)
        grad_norm = None
        if params.grad_clip:
            if accelerator.sync_gradients:
                grad_norm = accelerator.clip_grad_norm_(
                    model.parameters(), max_norm=2.0
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
                f"end of epoch {params.cur_epoch}, last batch_idx of trainset: {batch_idx} "
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
    #prof.stop()
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
        "best_valid_loss",
        "best_train_der",
        "best_valid_der",
    ]
    for k in keys:
        params[k] = saved_params[k]

    return saved_params

#from memory_profiler import profile
#@profile
def main(args):
    params = get_params()
    params.update(vars(args))
    logging.info(f"params: {params}")
    ## set seed firstly, it will effect batch and lr,
    fix_random_seed(params.seed)  # fairseq1 seed=1337
    world_size = params.world_size
    # data part
    data_cfg = TSVADDataConfig()
    data_cfg.musan_path = params.musan_path
    data_cfg.rir_path = params.rir_path
    data_cfg.speech_encoder_type = params.speech_encoder_type
    data_cfg.spk_path = params.spk_path
    data_cfg.speaker_embedding_name_dir = params.speaker_embedding_name_dir
    data_cfg.data_dir = params.data_dir
    data_cfg.speaker_embed_dim = params.speaker_embed_dim
    data_cfg.max_num_speaker = params.max_num_speaker
    data_cfg.rs_len = params.rs_len
    data_cfg.segment_shift = params.segment_shift
    data_cfg.dataset_name = params.dataset_name
    data_cfg.enhance_ratio = params.enhance_ratio
    data_cfg.enhanced_audio_dir = params.enhanced_audio_dir

    logging.info(f"final data_cfg: {data_cfg}")
    if params.dataset_name=="alimeeting":
        valid_dataset = load_dataset(data_cfg, "Eval")
        train_dataset = load_dataset(data_cfg, "Train")
    elif params.dataset_name=="magicdata-ramc":
        valid_dataset = load_dataset(data_cfg, "dev")
        train_dataset = load_dataset(data_cfg, "train")

    valid_dl = DataLoader(
        dataset=valid_dataset,  # the dataset instance
        batch_size=params.batch_size,  # automatic batching
        drop_last=False,  # drops the last incomplete batch in case the dataset size is not divisible by 64
        shuffle=False,  # shuffles the dataset before every epoch
        collate_fn=valid_dataset.collater,
        sampler=None,
    )

    train_dl = DataLoader(
        dataset=train_dataset,  # the dataset instance
        batch_size=params.batch_size,  # automatic batching
        drop_last=False,  # drops the last incomplete batch in case the dataset size is not divisible by 64
        # shuffle=(train_sampler is None),                 # shuffles the dataset before every epoch
        shuffle=True,
        collate_fn=train_dataset.collater,
        sampler=None,
    )

    from accelerate import (
        Accelerator,
        DDPCommunicationHookType,
        DistributedDataParallelKwargs,
    )

    gradient_accumulation = 1
    scale_window = max(int(2**14 / world_size / gradient_accumulation), 1)
    logging.info(f"The scale window is set to {scale_window}.")
    scaler_kwargs = GradScalerKwargs(
        init_scale=128.0,
        growth_factor=2.0,
        backoff_factor=1 / 2.0,
        growth_interval=scale_window,
        enabled=True,
    )

    ddp_kwargs = DistributedDataParallelKwargs(
        comm_hook=DDPCommunicationHookType.FP16, find_unused_parameters=True
    )
    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs, scaler_kwargs],
        mixed_precision="fp16",
        project_dir=args.exp_dir,
    )

    model_cfg = TSVADConfig()
    # here ,modified model cfg
    model_cfg.speech_encoder_type = params.speech_encoder_type
    model_cfg.speech_encoder_path = params.speech_encoder_path
    model_cfg.speaker_embed_dim = params.speaker_embed_dim
    model_cfg.freeze_speech_encoder_updates = params.freeze_updates
    model_cfg.feature_grad_mult = params.feature_grad_mult
    model_cfg.select_encoder_layer_nums = (
        params.select_encoder_layer_nums
    )  # only for speech_encoder_type=="WavLm"
    model_cfg.wavlm_fuse_feat_post_norm = (
        params.wavlm_fuse_feat_post_norm
    )  # only for self.speech_encoder_type == "WavLM_weight_sum"
    model_cfg.speech_encoder_config=params.speech_encoder_config # only for wav-bert2 ssl model
    model_cfg.single_backend_type=params.single_backend_type
    model_cfg.multi_backend_type=params.multi_backend_type
    model_cfg.num_transformer_layer=params.num_transformer_layer
    model_cfg.d_state = params.d_state
    model_cfg.expand = params.expand
    model_cfg.ots_vad_style = params.ots_vad_style
    model_cfg.dropout = params.dropout



    logging.info(f"model_cfg: {model_cfg}")
    model = TSVADModel(cfg=model_cfg, task_cfg=data_cfg)
    #model.speech_encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    logging.info(f"model: {model}")
    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

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
        assert params.start_epoch > 0, params.start_epoch
        checkpoints = load_checkpoint_if_available(
            params=params, model=model, model_avg=model_avg
        )
    ## this is very important. it will solve the error:
    # python3.11/site-packages/torch/autograd/init.py", line 251, in backward
    # Variable._execution_engine.run_backward( # Calls into the C++ engine to run the backward pass
    # RuntimeError: Expected to mark a variable ready only once.
    # This error is caused by one of the following reasons:
    # 1) Use of a module parameter outside the forward function.
    # Please make sure model parameters are not shared across multiple concurrent
    # forward-backward passes. or try to use _set_static_graph() as a workaround
    # if this module graph does not change during training loop.
    # 2) Reused parameters in multiple reentrant backward passes.
    # For example, if you use multiple checkpoint functions to wrap the same part of your model,
    # it would result in the same set of parameters been used by different reentrant
    # backward passes multiple times, and hence marking a variable ready multiple times.
    # DDP does not support such use cases in default.
    # You can try to use _set_static_graph() as a workaround
    # if your module graph does not change over iterations.
    # Parameter at index 557 with name speech_encoder.xvector.block3.tdnnd16.linear1.weight
    # has been marked as ready twice. This means that multiple autograd engine hooks
    # have fired for this particular parameter during this iteration.

    # the below combine ddp find_unused_parameters=True in accelerate package.
    # it will solve the strange error.
    #if True:
    #    from functools import partial

    #    notfailing_checkpoint = partial(
    #        torch.utils.checkpoint.checkpoint, use_reentrant=False
    #    )
    #    torch.utils.checkpoint.checkpoint = notfailing_checkpoint
    #    model.gradient_checkpointing_enable()

    if params.speech_encoder_type!="w2v-bert2":
        if True:
            from functools import partial

            notfailing_checkpoint = partial(
                torch.utils.checkpoint.checkpoint, use_reentrant=False
            )
            torch.utils.checkpoint.checkpoint = notfailing_checkpoint
            model.gradient_checkpointing_enable()
    else:
        if True:
            model.speech_encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    #if True:
    #    if params.speech_encoder_type=="CAM++":
    #        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    #    else:
    #    model.speech_encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    #    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    ## get optimizer, scheduler
    optimizer, scheduler = get_optimizer_scheduler(params, model, world_size)

    ## accelerated model, optimizer, scheduler ,train_dl, valid_dl
    model, optimizer, scheduler, train_dl, valid_dl = accelerator.prepare(
        model, optimizer, scheduler, train_dl, valid_dl
    )
    
    writer: Optional[SummaryWriter] = None
    if accelerator.is_main_process and params.tensorboard:
        writer = SummaryWriter(log_dir=f"{args.exp_dir}/tensorboard")

    # logging.info(f"After accelerator: model: {model}")
    scaler: Optional[GradScaler] = None
    logging.info(f"start training from epoch {params.start_epoch}")
    logging.info(f"Train set grouped total_num_itrs = {len(train_dl)}")

    # fix_random_seed(params.seed) # fairseq1 seed=1337 # this may be not correct at here.
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
                params=params,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
            )
        # early stop(TODO) Duo Ma
        # Assume `should_do_breakpoint` is a custom defined function that returns a conditional,
        # and that conditional might be true only on process 1
        # if should_do_breakpoint(loss):
        #    accelerator.set_breakpoint()

        # if params.batch_idx_train>=params.max_updates:
        #    logging.info(f"batch_idx_train >= {params.max_updates}, stop training")
        #    break
    if writer:
        writer.close()
    logging.info("Done!")
    if world_size > 1:
        torch.distributed.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    main(args)
