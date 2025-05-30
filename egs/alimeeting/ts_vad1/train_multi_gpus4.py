#!/usr/bin/env python3
# Author: Duo MA
# Email: maduo@cuhk.edu.cn
"""
python3 ts_vad1/train2.py --world-size 1 --num-epochs 30 --start-epoch 1 --use-fp16 1 --exp-dir ts_vad1/exp2
"""

import argparse
import copy
import logging
import warnings
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, Optional, Tuple, Union

#import optim
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

from lhotse.utils import fix_random_seed
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from dataclasses import dataclass

from checkpoint import load_checkpoint, remove_checkpoints
from checkpoint import save_checkpoint as save_checkpoint_impl
from checkpoint import (
    save_checkpoint_with_global_batch_idx,
)
from dist import cleanup_dist, setup_dist
from utils import (
    AttributeDict,
    MetricsTracker,
    setup_logger,
    str2bool,
)

from datasets import load_dataset
from model4 import TSVADModel
from gradient import clip_gradient_norm

from contextlib import AbstractContextManager, nullcontext

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
        "--master-port",
        type=int,
        default=12354,
        help="Master port to use for DDP training.",
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
        default=42,
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
        "--use-fp16",
        type=str2bool,
        default=True,
        help="Whether to use half precision training.",
    )
    parser.add_argument(
        "--musan-path",
        type=str,
        default=None,
        help="musan noise path",
    )
    parser.add_argument(
        "--rir-path",
        type=str,
        default=None,
        help="rir noise path",
    )
    return parser

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
            "log_interval": 500, # same as fairseq
            "reset_interval": 200,
            #"valid_interval": 1500,  # same as fairseq
            "valid_interval": 500,
            #"ignore_id": -1,
            #"label_smoothing": 0.1,
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
def compute_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    batch: dict,
    is_training: bool,
    #batch_idx_train: int,
):
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device
    with torch.set_grad_enabled(is_training):
        ref_speech = batch["net_input"]["ref_speech"].to(device)
        target_speech = batch["net_input"]["target_speech"].to(device)
        labels = batch["net_input"]["labels"].to(device)
        labels_len = batch["net_input"]["labels_len"].to(device)
        #loss, net_output = model(ref_speech=ref_speech,target_speech=target_speech,labels=labels,labels_len=labels_len,num_updates=batch_idx_train)
        #loss = net_output["losses"]["diar"]
        #outs, outs_prob = model(ref_speech=ref_speech,target_speech=target_speech,labels=labels,labels_len=labels_len)
        #loss = model.module.calculate_loss(outs, labels, labels_len)
        outs = model(ref_speech=ref_speech,target_speech=target_speech,labels=labels,labels_len=labels_len)
        loss = calculate_loss(outs=outs, labels=labels, labels_len=labels_len)

        ## public logger
        outs_prob=torch.nn.functional.sigmoid(outs)
        # convert tensor to numpy
        #logging.info(f"outs_prob requries_grad: {outs_prob.requries_grad}")
        outs_prob = outs_prob.data.cpu().numpy()
        mi, fa, cf, acc, der = model.module.calc_diarization_result(
            outs_prob.transpose((0, 2, 1)), labels.transpose(1, 2), labels_len
        )

    assert loss.requires_grad == is_training
    info={}
    info["loss"] = loss.detach().cpu().item()
    info["DER"] = der
    info["ACC"] = acc
    info['MI']  = mi
    info['FA']  = fa
    info['CF']  = cf
    return loss, info

def compute_validation_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    valid_dl: torch.utils.data.DataLoader,
    #batch_idx_train: int=0,
    world_size: int = 1,
) -> MetricsTracker:
    """Run the validation process."""
    model.eval()

    tot_loss = MetricsTracker()
    batch_nums=[]
    tot_loss_valid=0
    for batch_idx, batch in enumerate(valid_dl):
        loss, loss_info = compute_loss(
            params=params,
            model=model,
            batch=batch,
            is_training=False,
            #batch_idx_train=batch_idx_train,
           )
        batch_nums.append(batch_idx)
        assert loss.requires_grad is False
        tot_loss_valid = tot_loss_valid + loss_info["loss"]
        tot_loss = tot_loss + loss_info

    for item in tot_loss.keys():
        tot_loss[item]=tot_loss[item]/len(batch_nums)

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
    rank: int = 0,
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
    if rank != 0:
        return
    filename = params.exp_dir / f"epoch-{params.cur_epoch}.pt"
    save_checkpoint_impl(
        filename=filename,
        model=model,
        params=params,
        optimizer=optimizer,
        scheduler=scheduler,
        rank=rank,
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
        filename = params.exp_dir / f"epoch-{params.start_epoch-1}.pt"
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

def _maybe_no_sync(batch_idx:int,num_batches:int,world_size:int,model:nn.Module)->AbstractContextManager[None]:
    if batch_idx < num_batches - 1 and world_size > 1:
        return model.no_sync()  # type: ignore[no-any-return]
    return nullcontext()


#def train_one_epoch(
#    params: AttributeDict,
#    model: nn.Module,
#    optimizer: torch.optim.Optimizer,
#    scheduler: LRSchedulerType,
#    scaler: GradScaler,
#    train_dl: torch.utils.data.DataLoader,
#    valid_dl: torch.utils.data.DataLoader,
#    tb_writer: Optional[SummaryWriter] = None,
#    world_size: int = 1,
#    rank: int = 0,
#    model_avg: Optional[nn.Module] = None,
#) -> None:
#    model.train()
#
#    #tot_loss = MetricsTracker()
#    #train_batch_nums=[]
#    ddp_loss = torch.zeros(2).to(rank)
#    for batch_idx, batch in enumerate(train_dl):
#        params.batch_idx_train += 1
#        optimizer.zero_grad()
#        with torch.cuda.amp.autocast(enabled=params.use_fp16):
#                loss, loss_info = compute_loss(
#                    params=params,
#                    model=model,
#                    batch=batch,
#                    is_training=True,
#                    batch_idx_train=params.batch_idx_train,
#                )
#        scaler.scale(loss).backward()
#
#        # grad clip
#        grad_norm = clip_gradient_norm(
#                model, max_norm=None
#        )
#
#        # 在调用scaler.step()之前确保所有进程同步
#        #torch.distributed.barrier()
#
#        # update parameter of model
#        scaler.step(optimizer)
#        scaler.update()
#
#        # update scheduler
#        scheduler.step()
#
#        ddp_loss[0] += loss.item()
#        ddp_loss[1] += 1 # because my loss has reduced sum.
#        logging.info(f"batch_idx: {batch_idx}, batch_idx_train: {params.batch_idx_train}, loss: {loss}")
#    torch.distributed.all_reduce(ddp_loss, op=torch.distributed.ReduceOp.SUM)
#    if rank == 0:
#        logging.info('Train Epoch: {} \tLoss: {:.6f}'.format( ddp_loss[0] / ddp_loss[1]))
#
#
def train_one_epoch(
    params: AttributeDict,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LRSchedulerType,
    scaler: GradScaler,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    tb_writer: Optional[SummaryWriter] = None,
    world_size: int = 1,
    rank: int = 0,
    model_avg: Optional[nn.Module] = None,

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
      graph_compiler:
        It is used to convert transcripts to FSAs.
      train_dl:
        Dataloader for the training dataset.
      valid_dl:
        Dataloader for the validation dataset.
      tb_writer:
        Writer to write log messages to tensorboard.
      world_size:
        Number of nodes in DDP training. If it is 1, DDP is disabled.
    """
    model.train()

    tot_loss = MetricsTracker()
    train_batch_nums=[]
    #from torch.cuda.amp import GradScaler, autocast
    #scaler = GradScaler()
    ddp_loss = torch.zeros(2).to(rank)
    for batch_idx, batch in enumerate(train_dl):
        params.batch_idx_train += 1
        batch_size = params.batch_size
        # # when word_size>1,
        ## solve the errors:
        ## RuntimeError: Expected to mark a variable ready only once.
        # This error is caused by one of the following reasons:
        # 1) Use of a module parameter outside the forward function.
        # Please make sure model parameters are not shared across multiple concurrent forward-backward passes.
        # or try to use _set_static_graph() as a workaround if this module graph does not change during training loop.
        # 2) Reused parameters in multiple reentrant backward passes.
        # For example, if you use multiple checkpoint functions to wrap the same part of your model,
        # it would result in the same set of parameters been used by different reentrant backward passes
        # multiple times, and hence marking a variable ready multiple times. DDP does not support such use cases in default.
        # You can try to use _set_static_graph() as a workaround if your module graph does not change over iterations.
        # Parameter at index 557 with name speech_encoder.xvector.block3.tdnnd16.linear1.weight has been marked as ready twice.
        # This means that multiple autograd engine hooks have fired for this particular parameter during this iteration.

        # the model._set_static_graph() method is not working for me
        # (Duo Ma) note: I add the below one line code to solve the errors at training ddp model when world_size>1.
#        with _maybe_no_sync(batch_idx, len(train_dl),world_size,model):
#            with torch.cuda.amp.autocast(enabled=params.use_fp16):
#                loss, loss_info = compute_loss(
#                    params=params,
#                    model=model,
#                    batch=batch,
#                    is_training=True,
#                    batch_idx_train=params.batch_idx_train,
#                )
#        train_batch_nums.append(batch_idx)
#        # summary stats
#        tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + loss_info
#
#        # NOTE: We use reduction==sum and loss is computed over utterances
#        # in the batch and there is no normalization to it so far.
#        scaler.backward(loss)
#        # normalize
#        # clip
#        scaler.unscale_gradients_()
#        grad_norm = clip_gradient_norm(
#                model, max_norm=None
#        )
#        # Update the parameters.
#        _, scale_result = scaler.run_optimizer_step(params.batch_idx_train)
#        if not scale_result.overflow:
#            scheduler.step()
#
#        # set grad to zero
#        optimizer.zero_grad(set_to_none=True)
        # 清除旧的梯度信息
        optimizer.zero_grad()
        train_batch_nums.append(batch_idx)
        with torch.cuda.amp.autocast(enabled=params.use_fp16):
                loss, loss_info = compute_loss(
                    params=params,
                    model=model,
                    batch=batch,
                    is_training=True,
                    #batch_idx_train=params.batch_idx_train,
                )
        # 反向传播之前
        # 冻结参数
        #if params.batch_idx_train < params.freeze_updates:
        #    for param in model.module.speech_encoder.parameters():
                #logging.info(f"batch_idx_train: {params.batch_idx_train},model.module.speech_encoder.parameters()  grad should set False")
        #        param.requires_grad = False
        #else:
        #    for param in model.module.speech_encoder.parameters():
        #        #logging.info(f"batch_idx_train: {params.batch_idx_train},model.module.speech_encoder.parameters()  grad should set True")
        #        param.requires_grad = True
        #torch.distributed.barrier()
        # 使用动态损失缩放进行反向传播
        scaler.scale(loss).backward()

        # grad clip
        grad_norm = clip_gradient_norm(
                model, max_norm=None
        )

        # 在调用scaler.step()之前确保所有进程同步
        #torch.distributed.barrier()

        # update parameter of model
        scaler.step(optimizer)
        scaler.update()

        # update scheduler
        scheduler.step()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += 1

    #dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
        # 冻结参数
        #if params.batch_idx_train < params.freeze_updates:
        #    for param in model.module.speech_encoder.parameters():
                #logging.info(f"batch_idx_train: {params.batch_idx_train},model.module.speech_encoder.parameters()  grad should set False")
        #        param.requires_grad = False

        #else:
        #    for param in model.module.speech_encoder.parameters():
                #logging.info(f"batch_idx_train: {params.batch_idx_train},model.module.speech_encoder.parameters()  grad should set True")
        #        param.requires_grad = True

        #torch.distributed.barrier()
        #scaler._grad_scaler.scale(loss).backward()
        #scaler.unscale_gradients_()
        #grad_norm = clip_gradient_norm(
        #        model, max_norm=None
        #)
        # 运行优化器步骤和更新缩放因子
        #_, scale_result = scaler.run_optimizer_step(params.batch_idx_train)
        #scheduler.step()
        # set grad to zero
        #optimizer.zero_grad(set_to_none=True)
        # store batch level checkpoint
        if (
            params.batch_idx_train > 0
            and params.batch_idx_train % params.save_every_n == 0
        ):
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
                rank=rank,
            )
            remove_checkpoints(
                out_dir=params.exp_dir,
                topk=params.keep_last_k,
                rank=rank,
            )
        ## dynamic scaler
        #if batch_idx % 1 == 0:
        #    scaler_result = dynamic_scaler(params.batch_idx_train,scaler)
        if batch_idx % params.log_interval == 0:
            ## To align with the numbers in fairseq iter
            num_updates=0
            if params.cur_epoch==1:
                num_updates=batch_idx
            else:
                integer_multi_num = len(train_dl) - len(train_dl) % params.log_interval # 3128 - 3128%500=3000
                num_updates = (params.cur_epoch - 1)*integer_multi_num + batch_idx

            ## get grad_scale and lr
            #grad_scale = scale_result.new_scale
            grad_scale =""
            cur_lr = scheduler.get_last_lr()[0]

            logging.info(
                f"[Train] - Epoch {params.cur_epoch}, "
                f"batch_idx_train: {params.batch_idx_train-1}, num_updates: {num_updates}, {loss_info}, "
                f"batch size: {batch_size}, grad_norm: {grad_norm}, grad_scale: {grad_scale}, "
                f"lr: {cur_lr:.2e}, "
            )

        # log end-of-epoch stats
        #if batch_idx == params.cur_epoch *(len(train_dl) - len(train_dl) % params.log_interval) + len(train_dl) % params.log_interval:
        if batch_idx == len(train_dl) -1:
            #grad_scale = scale_result.new_scale
            grad_scale =""
            cur_lr = scheduler.get_last_lr()[0]
            logging.info(
                f"end of epoch {params.cur_epoch}, batch_idx: {batch_idx} "
                f"batch_idx_train: {params.batch_idx_train-1}, {loss_info}, "
                f"batch size: {batch_size}, grad_norm: {grad_norm}, grad_scale: {grad_scale}, "
                f"lr: {cur_lr:.2e}, "
            )
        #if batch_idx % params.log_interval == 0:
            #if tb_writer is not None:
            #    loss_info.write_summary(
            ##        tb_writer, "train/current_", params.batch_idx_train
            #    )
            #    tot_loss.write_summary(tb_writer, "train/tot_", params.batch_idx_train)

        if batch_idx > 0 and batch_idx % params.valid_interval == 0:
            logging.info("Computing validation loss")
            valid_info = compute_validation_loss(
                params=params,
                model=model,
                valid_dl=valid_dl,
                world_size=world_size,
                #batch_idx_train=params.batch_idx_train,
            )
            model.train()
            logging.info(f"[Eval] - Epoch {params.cur_epoch}, batch_idx_train: {params.batch_idx_train-1} validation: {valid_info}")
            logging.info(
                f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated()//1000000}MB"
            )
            #if tb_writer is not None:
            #    valid_info.write_summary(
            #        tb_writer, "train/valid_", params.batch_idx_train
            #    )
    torch.distributed.all_reduce(ddp_loss, op=torch.distributed.ReduceOp.SUM)
    if rank == 0:
        logging.info('Train Epoch: {} \tLoss: {:.6f}'.format(params.cur_epoch, ddp_loss[0] / ddp_loss[1]))
    loss_value = tot_loss["loss"] /len(train_batch_nums)
    params.train_loss = loss_value

    if params.train_loss < params.best_train_loss:
        params.best_train_epoch = params.cur_epoch
        params.best_train_loss = params.train_loss

    der_value = tot_loss["DER"] /len(train_batch_nums)
    if der_value < params.best_train_der:
        params.best_train_epoch = params.cur_epoch
        params.best_train_der = der_value

def get_optimizer_scheduler(params, model, world_size):
    from adamw import AdamW
    #optimizer = AdamW(model.parameters(),lr=2e-4,betas=(0.9, 0.98),eps=1e-08, weight_decay=0.01) # same as fairseq
    optimizer: Optional[torch.optim.Optimizer] = None
    if world_size >1: ## DDP or FSDP
        # when it will be the below error:,You should check if your optimizer excludes parameters without gradients.
        # You must manually exclude network parameters without gradients by
        # adding `filter(lambda p: p.requires_grad, model.parameters()) `or `filter(lambda p: p.requires_grad, model.module.parameters())`
        # python3.11/site-packages/torch/autograd/__init__.py", line 251, in backward
        #    Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
        #RuntimeError: Expected to mark a variable ready only once. This error is caused by one of the following reasons:
        # 1) Use of a module parameter outside the `forward` function.
        # Please make sure model parameters are not shared across multiple concurrent forward-backward passes.
        # or try to use _set_static_graph() as a workaround if this module graph does not change during training loop.
        # 2) Reused parameters in multiple reentrant backward passes.
        # For example, if you use multiple `checkpoint` functions to wrap the same part of your model,
        # it would result in the same set of parameters been used by different reentrant backward passes multiple times,
        # and hence marking a variable ready multiple times. DDP does not support such use cases in default.
        # You can try to use _set_static_graph() as a workaround if your module graph does not change over iterations.
        #Parameter at index 557 with name speech_encoder.xvector.block3.tdnnd16.linear1.weight has been marked as ready twice.
        # This means that multiple autograd engine  hooks have fired for this particular parameter during this iteration.
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.module.parameters()),lr=2e-4,betas=(0.9, 0.98),eps=1e-08, weight_decay=0.01)
    else: # single GPU
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),lr=2e-4,betas=(0.9, 0.98),eps=1e-08, weight_decay=0.01)
    #optimizer = AdamW(model.parameters(),lr=5e-5,betas=(0.9, 0.98)) # same as fairseq2
    from polynomial import PolynomialDecayLR
    scheduler = PolynomialDecayLR(optimizer,params.max_updates,params.warmup_updates,power=1.0)
    return optimizer, scheduler

def get_scaler(world_size,optimizer):
    #from dynamic_loss_scaler2 import  DynamicLossScaler
    from dynamic_loss_scaler3 import  DynamicLossScaler
    #from gang import setup_root_gang
    from logging_me import get_log_writer

    log = get_log_writer(__name__)
    #root_gang = setup_root_gang(log, monitored=False)
    #scaler = DynamicLossScaler(optimizer,root_gang,sharded=world_size>1,init_scale=128.0,min_scale=0.0001,gradient_accumulation=1,enabled=True)
    #scaler = DynamicLossScaler(optimizer,world_size,sharded=world_size>1,init_scale=128.0,min_scale=0.0001,gradient_accumulation=1,enabled=True)
    from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
    gradient_accumulation=1
    scale_window = max(int(2**14 / world_size / gradient_accumulation), 1)
    log.info(f"The scale window is set to {scale_window}.")

    scaler = ShardedGradScaler(
                init_scale=128.0,
                growth_factor=2.0,
                backoff_factor=1 / 2.0,
                growth_interval=scale_window,
                enabled=True,
                process_group=torch.distributed.group.WORLD,)
    return scaler

def _are_close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1.3e-6, abs_tol=1e-5)
def dynamic_scaler(step:int,scaler,_min_scale=0.0001):
    old_scale = scaler.get_scale()

    scaler.update()

    new_scale = scaler.get_scale()

    if _are_close(old_scale, new_scale):
        return LossScaleResult(old_scale, new_scale)

    if new_scale > old_scale:
        logging.info("No gradient overflow detected in the last {} step(s) after step {}, increasing loss scale from {:g} to {:g}.", self._scale_window, step_nr, old_scale, new_scale)  # fmt: skip

        return LossScaleResult(old_scale, new_scale)

    if _min_scale > new_scale:
        scaler.update(self._min_scale)

        if _are_close(old_scale, self._min_scale):
            logging.error("Overflow detected at step {}, ignoring gradient, loss scale is already at minimum ({:g}). Your loss is probably exploding. Try lowering the learning rate, using gradient clipping, or increasing the batch size.", step_nr, self._min_scale)  # fmt: skip
        else:
            logging.error("Overflow detected at step {}, ignoring gradient, decreasing loss scale from {:g} to {:g} (minimum). Your loss is probably exploding. Try lowering the learning rate, using gradient clipping, or increasing the batch size.", step_nr, old_scale, self._min_scale)  # fmt: skip

        return LossScaleResult(
            old_scale, new_scale, overflow=True, min_reached=True
        )
    else:
        logging.info("Overflow detected at step {}, ignoring gradient, decreasing loss scale from {:g} to {:g}.", step_nr, old_scale, new_scale)  # fmt: skip

        return LossScaleResult(old_scale, new_scale, overflow=True)

@dataclass(frozen=True)
class LossScaleResult:
    """Holds the result of a loss scale operation."""

    old_scale: float
    """The scale before the optimizer step."""

    new_scale: float
    """The scale after the optimizer step."""

    overflow: bool = False
    """If ``True``, the loss has overflowed."""

    min_reached: bool = False
    """If ``True``, the scale has been decreased to its minimum value."""
def run(rank, world_size, args):
    """
    Args:
      rank:
        It is a value between 0 and `world_size-1`, which is
        passed automatically by `mp.spawn()` in :func:`main`.
        The node with rank 0 is responsible for saving checkpoint.
      world_size:
        Number of GPUs for DDP training.
      args:
        The return value of get_parser().parse_args()
    """
    params = get_params()
    params.update(vars(args))

    fix_random_seed(params.seed)
    if world_size > 1:
        setup_dist(rank, world_size, params.master_port)

    setup_logger(f"{params.exp_dir}/log/log-train")
    logging.info("Training started")

    if args.tensorboard and rank == 0:
        tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
    else:
        tb_writer = None
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
    logging.info(f"Device: {device}")

    logging.info(params)

    logging.info("About to create model")

    # data part
    from datasets import TSVADDataConfig
    data_cfg = TSVADDataConfig()
    data_cfg.musan_path=args.musan_path
    data_cfg.rir_path=args.rir_path
    print(f"data_cfg: {data_cfg}")
    valid_dataset = load_dataset(data_cfg,'Eval')
    train_dataset = load_dataset(data_cfg,'Train')
    from torch.utils.data.distributed import DistributedSampler
    valid_sampler = DistributedSampler(valid_dataset,num_replicas=world_size, rank=rank) if world_size>1 else None
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) if world_size>1 else None

    valid_dl = DataLoader(dataset=valid_dataset, # the dataset instance
                                 batch_size=params.batch_size, # automatic batching
                                 drop_last=False,               # drops the last incomplete batch in case the dataset size is not divisible by 64
                                 shuffle=(valid_sampler is None),                 # shuffles the dataset before every epoch
                                 collate_fn=valid_dataset.collater,
                                 sampler=valid_sampler,
                                )

    train_dl = DataLoader(dataset=train_dataset, # the dataset instance
                                 batch_size=params.batch_size, # automatic batching
                                 drop_last=False,               # drops the last incomplete batch in case the dataset size is not divisible by 64
                                 shuffle=(train_sampler is None),                 # shuffles the dataset before every epoch
                                 collate_fn=train_dataset.collater,
                                 sampler=train_sampler,
                                )


    model = TSVADModel()
    from model2 import TSVADConfig
    model_cfg=TSVADConfig()

    #model.load_speaker_encoder(model_cfg.speech_encoder_path,device=device, module_name="speech_encoder")
    ## refer from https://github.com/huggingface/transformers/issues/21381#issuecomment-1666498410
    #             https://github.com/huggingface/transformers/issues/23018#issuecomment-1792375578
    #             https://github.com/huggingface/transformers/blob/a22ff36e0e347d3d0095cccd931cbbd12b14e86a/src/transformers/modeling_utils.py#L2323
    #if True:
    #    from functools import partial
    #    notfailing_checkpoint = partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)
    #    torch.utils.checkpoint.checkpoint = notfailing_checkpoint
    #    model.gradient_checkpointing_enable()
    #model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    logging.info(f"model: {model}")
    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    model_avg: Optional[nn.Module] = None
    checkpoints = load_checkpoint_if_available(
        params=params, model=model, model_avg=model_avg
    )

    #torch.cuda.set_device(rank)


    #init_start_event = torch.cuda.Event(enable_timing=True)
    #init_end_event = torch.cuda.Event(enable_timing=True)
    model.to(device)
    if world_size > 1:
        #logging.info("Using FSDP")
        #from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        #model = FSDP(model.cuda(rank))
        logging.info("Using DDP")
        # when find_unused_parameters=True,
        # it can ignore variables in the forward output that do not participate in the loss calculation
        #model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    # create optimizer and scheduler
    optimizer, scheduler = get_optimizer_scheduler(params, model,world_size)

    if checkpoints and "optimizer" in checkpoints:
        logging.info("Loading optimizer state dict")
        optimizer.load_state_dict(checkpoints["optimizer"])

    if (
        checkpoints
        and "scheduler" in checkpoints
        and checkpoints["scheduler"] is not None
    ):
        logging.info("Loading scheduler state dict")
        scheduler.load_state_dict(checkpoints["scheduler"])

    # we use fp16 to train network, so we use fariseq2 style scaler,it is more better than offical pytorch and icefall
    scaler = get_scaler(world_size,optimizer)

    #init_start_event.record()

    logging.info(f"start training from epoch {params.start_epoch}")
    logging.info(f"Train set grouped total_num_itrs = {len(train_dl)}")
    for epoch in range(params.start_epoch, params.num_epochs+1):
        fix_random_seed(params.seed + epoch-1) # fairseq1 seed=1337

        params.cur_epoch = epoch

        train_one_epoch(
            params=params,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            train_dl=train_dl,
            valid_dl=valid_dl,
            tb_writer=tb_writer,
            world_size=world_size,
            model_avg=model_avg,
        )
        save_checkpoint(
            params=params,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            rank=rank,
        )

        if params.batch_idx_train>=params.max_updates:
            logging.info(f"batch_idx_train >= {params.max_updates}, stop training")
            break
    logging.info("Done!")
    #init_end_event.record()
    #if rank == 0:
    #    print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
    #    print(f"{model}")
    #if True:
    #    # use a barrier to make sure training is done on all ranks
    #    torch.distributed.barrier()
    #    states = model.state_dict()
    #    if rank == 0:
    #        torch.save(states, "sb.pt")
    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()

def main():
    parser = get_parser()
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)
    ## for debug multi gpus FSDP train
    #import os
    #os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
    #os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"


    world_size = args.world_size
    assert world_size >= 1
    if world_size > 1:
        mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run(rank=0, world_size=1, args=args)



torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()

