#!/usr/bin/env python3
# Author: Duo MA
# Email: maduo@cuhk.edu.cn

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
#from torch.nn.utils import clip_grad_norm_

from lhotse.utils import fix_random_seed
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from dataclasses import dataclass
#from lightning.fabric import Fabric
from accelerate import Accelerator

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


def get_optimizer_scheduler(params, model, world_size):
    #from adamw import AdamW
    from torch.optim import AdamW
    #optimizer = AdamW(model.parameters(),lr=2e-4,betas=(0.9, 0.98),eps=1e-08, weight_decay=0.01) # same as fairseq
    optimizer: Optional[torch.optim.Optimizer] = None
    if world_size >1: ## DDP or FSDP
        #optimizer = AdamW(filter(lambda p: p.requires_grad, model.module.parameters()),lr=2e-4,betas=(0.9, 0.98),eps=1e-08, weight_decay=0.01)
        #optimizer = AdamW(model.parameters(),lr=2e-4,betas=(0.9, 0.98),eps=1e-08, weight_decay=0.01)
        optimizer = AdamW(model.parameters(),lr=2e-4,betas=(0.9, 0.98),eps=1e-08, weight_decay=0.01)
    else: # single GPU
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),lr=2e-4,betas=(0.9, 0.98),eps=1e-08, weight_decay=0.01)
    #optimizer = AdamW(model.parameters(),lr=5e-5,betas=(0.9, 0.98)) # same as fairseq2
    from polynomial import PolynomialDecayLR
    scheduler = PolynomialDecayLR(optimizer,params.max_updates,params.warmup_updates,power=1.0)
    return optimizer, scheduler

def compute_loss(
    #params: AttributeDict,
    model: Union[nn.Module, DDP],
    batch: dict,
    is_training: bool,
    #batch_idx_train: int,
):
    #device = model.device if isinstance(model, DDP) else next(model.parameters()).device
    with torch.set_grad_enabled(is_training):
        ref_speech = batch["net_input"]["ref_speech"]
        target_speech = batch["net_input"]["target_speech"]
        labels = batch["net_input"]["labels"]
        labels_len = batch["net_input"]["labels_len"]
        #loss, net_output = model(ref_speech=ref_speech,target_speech=target_speech,labels=labels,labels_len=labels_len,num_updates=batch_idx_train)        #loss = net_output["losses"]["diar"]
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
        #mi, fa, cf, acc, der = model.calc_diarization_result(
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
    #world_size: int = 1,
) -> MetricsTracker:
    """Run the validation process."""
    model.eval()

    tot_loss = MetricsTracker()
    batch_nums=[]
    tot_loss_valid=0
    for batch_idx, batch in enumerate(valid_dl):
        loss, loss_info = compute_loss(
            #params=params,
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
    #rank: int = 0,
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
    #if rank != 0:
    #    return
    filename = params.exp_dir / f"epoch-{params.cur_epoch}.pt"
    save_checkpoint_impl(
        filename=filename,
        model=model,
        params=params,
        optimizer=optimizer,
        scheduler=scheduler,
        #rank=rank,
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


def train_one_epoch(
    params: AttributeDict,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LRSchedulerType,
    scaler: GradScaler,
    accelerator,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    #tb_writer: Optional[SummaryWriter] = None,
    #world_size: int = 1,
    #rank: int = 0,
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
      #world_size:
      #  Number of nodes in DDP training. If it is 1, DDP is disabled.
    """
    model.train()

    tot_loss = MetricsTracker()
    train_batch_nums=[]
    for batch_idx, batch in enumerate(train_dl):
        params.batch_idx_train += 1
        batch_size = params.batch_size

        optimizer.zero_grad()
        train_batch_nums.append(batch_idx)
        #with torch.cuda.amp.autocast(enabled=params.use_fp16):
        loss, loss_info = compute_loss(model=model,batch=batch,is_training=True,
            #batch_idx_train=params.batch_idx_train,
        )
        accelerator.backward(loss)  # instead of loss.backward()
        # grad clip
        #grad_norm = clip_gradient_norm(
        #        model, max_norm=None
        #)
        # refer : https://lightning.ai/docs/fabric/stable/api/fabric_methods.html
        #grad_norm = fabric.clip_gradients(model, optimizer,max_norm=2.0,)
        grad_norm=None
        optimizer.step()
        scheduler.step()
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
                #rank=rank,
            )
            remove_checkpoints(
                out_dir=params.exp_dir,
                topk=params.keep_last_k,
                #rank=rank,
            )

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

            #logging.info(
            print(
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
            #logging.info(
            print(
                f"end of epoch {params.cur_epoch}, batch_idx: {batch_idx} "
                f"batch_idx_train: {params.batch_idx_train-1}, {loss_info}, "
                f"batch size: {batch_size}, grad_norm: {grad_norm}, grad_scale: {grad_scale}, "
                f"lr: {cur_lr:.2e}, "
            )
        if batch_idx > 0 and batch_idx % params.valid_interval == 0:
            logging.info("Computing validation loss")
            valid_info = compute_validation_loss(
                params=params,
                model=model,
                valid_dl=valid_dl,
                #world_size=world_size,
                #batch_idx_train=params.batch_idx_train,
            )
            model.train()
            print(f"[Eval] - Epoch {params.cur_epoch}, batch_idx_train: {params.batch_idx_train-1} validation: {valid_info}")
            print(
                f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated()//1000000}MB"
            )
    #if rank == 0:
    #    logging.info('Train Epoch: {} \tLoss: {:.6f}'.format(params.cur_epoch, ddp_loss[0] / ddp_loss[1]))
    loss_value = tot_loss["loss"] /len(train_batch_nums)
    params.train_loss = loss_value

    if params.train_loss < params.best_train_loss:
        params.best_train_epoch = params.cur_epoch
        params.best_train_loss = params.train_loss

    der_value = tot_loss["DER"] /len(train_batch_nums)
    if der_value < params.best_train_der:
        params.best_train_epoch = params.cur_epoch
        params.best_train_der = der_value

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

def main(args):
    params = get_params()
    params.update(vars(args))
    world_size=2
    # data part
    from datasets import TSVADDataConfig
    data_cfg = TSVADDataConfig()
    data_cfg.musan_path=args.musan_path
    data_cfg.rir_path=args.rir_path
    print(f"data_cfg: {data_cfg}")
    valid_dataset = load_dataset(data_cfg,'Eval')
    train_dataset = load_dataset(data_cfg,'Train')
    #from torch.utils.data.distributed import DistributedSampler
    #valid_sampler = DistributedSampler(valid_dataset,num_replicas=world_size, rank=rank) if world_size>1 else None
    #train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) if world_size>1 else None

    valid_dl = DataLoader(dataset=valid_dataset, # the dataset instance
                                 batch_size=params.batch_size, # automatic batching
                                 drop_last=False,               # drops the last incomplete batch in case the dataset size is not divisible by 64
                                 shuffle=False,                 # shuffles the dataset before every epoch
                                 collate_fn=valid_dataset.collater,
                                 sampler=None,
                                )

    train_dl = DataLoader(dataset=train_dataset, # the dataset instance
                                 batch_size=params.batch_size, # automatic batching
                                 drop_last=False,               # drops the last incomplete batch in case the dataset size is not divisible by 64
                                 #shuffle=(train_sampler is None),                 # shuffles the dataset before every epoch
                                 shuffle=True,
                                 collate_fn=train_dataset.collater,
                                 sampler=None,
                                )

    #from lightning.fabric import Fabric
    #from lightning.fabric.strategies import DDPStrategy
    #fabric = Fabric(accelerator="cuda",strategy=DDPStrategy(find_unused_parameters=True),precision="16-true",devices=world_size,num_nodes=1,)
    #fabric = Fabric(accelerator="cuda",strategy="ddp_find_unused_parameters_true",precision="16-true",devices=world_size,num_nodes=1,)
    #fabric.launch()
    #accelerator = Accelerator(mixed_precision='fp16',project_dir=args.exp_dir)
    from accelerate import Accelerator, DDPCommunicationHookType, DistributedDataParallelKwargs

    ddp_kwargs = DistributedDataParallelKwargs(comm_hook=DDPCommunicationHookType.FP16,find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs],mixed_precision='fp16',project_dir=args.exp_dir)

    from model4 import TSVADConfig
    model_cfg=TSVADConfig()
    model = TSVADModel(cfg=model_cfg)
    model_avg: Optional[nn.Module] = None
    checkpoints = load_checkpoint_if_available(
        params=params, model=model, model_avg=model_avg
    )
    if True:
        from functools import partial
        notfailing_checkpoint = partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)
        torch.utils.checkpoint.checkpoint = notfailing_checkpoint
        model.gradient_checkpointing_enable()
    #model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer, scheduler = get_optimizer_scheduler(params, model,world_size)
    model, optimizer, scheduler ,train_dl, valid_dl = accelerator.prepare(model, optimizer,scheduler,train_dl, valid_dl)
    # here ,modified model cfg
    # init the model directly on the device and with parameters in half-precision
    #with fabric.init_module():
    #    model =TSVADModel(cfg=model_cfg)

    # create optimizer and scheduler
    #optimizer, scheduler = get_optimizer_scheduler(params, model,world_size)
    #model,optimizer = fabric.setup(model,optimizer)
    #fabric.print(f"optimizer: {optimizer}, scheduler: {scheduler}")
    #valid_dl = fabric.setup_dataloaders(valid_dl)
    #train_dl = fabric.setup_dataloaders(train_dl)

    #print(f"valid_dl: {valid_dl}, train_dl: {train_dl}")
    #for batch_idx, batch in enumerate(train_dl):
    #    if batch_idx < 1:
    #        print(f"batch: {batch}")
    #    else:
    #        break

    #for batch_idx, batch in enumerate(train_dl):
    #    loss, loss_info = compute_loss(model=model, batch=batch,is_training=True)
    #    model.train()
    scaler: Optional[GradScaler]=None
    print(f"start training from epoch {params.start_epoch}")
    print(f"Train set grouped total_num_itrs = {len(train_dl)}")
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
            #tb_writer=tb_writer,
            #world_size=world_size,
            accelerator=accelerator,
            model_avg=model_avg,
        )
        save_checkpoint(
            params=params,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            #rank=rank,
        )

        if params.batch_idx_train>=params.max_updates:
            # Print only on the main process
            print(f"batch_idx_train >= {params.max_updates}, stop training")
            break
    logging.info("Done!")
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)
    main(args)
