#!/usr/bin/env python3
# Author: Duo MA
# Email: maduo@cuhk.edu.cn

import os
import argparse
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


from torch.optim.lr_scheduler import StepLR

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


#class Net(nn.Module):
#    def __init__(self):
#        super(Net, self).__init__()
#        self.conv1 = nn.Conv2d(1, 32, 3, 1)
#        self.conv2 = nn.Conv2d(32, 64, 3, 1)
#        self.dropout1 = nn.Dropout(0.25)
#        self.dropout2 = nn.Dropout(0.5)
#        self.fc1 = nn.Linear(9216, 128)
#        self.fc2 = nn.Linear(128, 10)
#
#    def forward(self, x):
#        #print(f"input shape: {x.shape}")
#        # x: (B,1,24,24)
#        x = self.conv1(x)
#        x = F.relu(x)
#        x = self.conv2(x)
#        x = F.relu(x)
#        x = F.max_pool2d(x, 2) # B,64,12,12
#        #print(f"F.max_pool2d(x, 2), {x.shape}")
#        x = self.dropout1(x)
#        x = torch.flatten(x, 1) # B,9216
#        #print(f"torch.flatten(x, 1), {x.shape}")
#        x = self.fc1(x)
#        x = F.relu(x)
#        x = self.dropout2(x)
#
#        x = self.fc2(x)
#        output = F.log_softmax(x, dim=1) #B,C
#        #print(f"output shape: {output.shape}")
#        return output
#
from cam_pplus_wespeaker1 import CAMPPlus
import logging
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 80, 3, 1)
        self.speech_encoder = CAMPPlus(feat_dim=80, embedding_size=192,speech_encoder=True)
        speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt"
        self.load_speaker_encoder(speech_encoder_path,device=torch.device("cpu"), module_name="speech_encoder")
        self.fc2 = nn.Linear(173056, 10)
        self.speech_encoder_path=speech_encoder_path
    def forward(self,x):
        #self.load_speaker_encoder(self.speech_encoder_path,device=torch.device("cpu"), module_name="speech_encoder")
        x = self.conv1(x) # B,C,W,H
        B,C,_,_ = x.shape
        x = x.reshape(B,C,-1)
        x = x.permute(0,2,1)
        # B,T,F -> B,512,T/2
        x = self.speech_encoder(x)
        #x = x.permute(0,2,1)
        x = torch.flatten(x, 1)
        #print(f"after flatten shape, {x.shape}")
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        #print(f"output shape: {output.shape}")
        return output

    def load_speaker_encoder(self, model_path, device, module_name="speech_encoder"):
        loadedState = torch.load(model_path, map_location=device)
        selfState = self.state_dict()
        for name, param in loadedState.items():
            origname = name
            if (
                module_name == "speech_encoder"
                and hasattr(self.speech_encoder, "bn1")
                and isinstance(self.speech_encoder.bn1, BatchNorm1D)
                and ".".join(name.split(".")[:-1]) + ".running_mean" in loadedState
            ):
                name = ".".join(name.split(".")[:-1]) + ".bn." + name.split(".")[-1]

            name = f"{module_name}." + name

            if name not in selfState:
                print("%s is not in the model." % origname)
                continue
            if selfState[name].size() != loadedState[origname].size():
                sys.stderr.write(
                    "Wrong parameter length: %s, model: %s, loaded: %s"
                    % (origname, selfState[name].size(), loadedState[origname].size())
                )
                continue
            selfState[name].copy_(param)
def get_scaler(world_size):
    #from dynamic_loss_scaler2 import  DynamicLossScaler
    #from dynamic_loss_scaler3 import  DynamicLossScaler
    #from gang import setup_root_gang
    #from logging_me import get_log_writer

    #log = get_log_writer(__name__)
    #root_gang = setup_root_gang(log, monitored=False)
    #scaler = DynamicLossScaler(optimizer,root_gang,sharded=world_size>1,init_scale=128.0,min_scale=0.0001,gradient_accumulation=1,enabled=True)
    #scaler = DynamicLossScaler(optimizer,world_size,sharded=world_size>1,init_scale=128.0,min_scale=0.0001,gradient_accumulation=1,enabled=True)
    from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
    gradient_accumulation=1
    scale_window = max(int(2**14 / world_size / gradient_accumulation), 1)
    import logging
    logging.info(f"The scale window is set to {scale_window}.")

    scaler = ShardedGradScaler(
                init_scale=128.0,
                growth_factor=2.0,
                backoff_factor=1 / 2.0,
                growth_interval=scale_window,
                enabled=True,
                process_group=torch.distributed.group.WORLD,)
    return scaler
from torch import Tensor
from torch.nn import Module
from torch.nn.utils import clip_grad_norm_
def clip_gradient_norm(
    module: Module, max_norm: float | None, norm_type: float = 2.0
) -> Tensor:
    """Clip the gradient norms ``module``.

    :param module:
        The module whose gradients to clip.
    :param max_norm:
        The maximum norm.
    :param norm_type:
        The type of the used p-norm.
    """
    if max_norm is None:
        max_norm = torch.inf

    if isinstance(module, FSDP):
        if not module.check_is_root():
            raise ValueError("`module` must be the root FSDP module.")

        return module.clip_grad_norm_(max_norm, norm_type)

    return clip_grad_norm_(  # type: ignore[no-any-return]
        module.parameters(), max_norm, norm_type, error_if_nonfinite=False
    )

def train(args, model, rank, world_size, train_loader, optimizer,scheduler, scaler, epoch, sampler=None):
    model.train()
    ddp_loss = torch.zeros(2).to(rank)
    if sampler:
        sampler.set_epoch(epoch)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(rank), target.to(rank)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            output = model(data)
            loss = F.nll_loss(output, target, reduction='sum')

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
        #loss.backward()
        #optimizer.step()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(data)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, ddp_loss[0] / ddp_loss[1]))


def test(model, rank, world_size, test_loader):
    model.eval()
    correct = 0
    ddp_loss = torch.zeros(3).to(rank)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(rank), target.to(rank)
            output = model(data)
            ddp_loss[0] += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            ddp_loss[1] += pred.eq(target.view_as(pred)).sum().item()
            ddp_loss[2] += len(data)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    if rank == 0:
        test_loss = ddp_loss[0] / ddp_loss[2]
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, int(ddp_loss[1]), int(ddp_loss[2]),
            100. * ddp_loss[1] / ddp_loss[2]))


def fsdp_main(rank, world_size, args):
    setup(rank, world_size)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset1 = datasets.MNIST('../data', train=True, download=True,
                        transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                        transform=transform)

    sampler1 = DistributedSampler(dataset1, rank=rank, num_replicas=world_size, shuffle=True)
    sampler2 = DistributedSampler(dataset2, rank=rank, num_replicas=world_size)

    train_kwargs = {'batch_size': args.batch_size, 'sampler': sampler1}
    test_kwargs = {'batch_size': args.test_batch_size, 'sampler': sampler2}
    cuda_kwargs = {'num_workers': 2,
                    'pin_memory': True,
                    'shuffle': False}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=100
    )
    torch.cuda.set_device(rank)


    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    model = Net().to(rank)

    #model = FSDP(model)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    #model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    scaler = get_scaler(world_size)
    init_start_event.record()
    for epoch in range(1, args.epochs + 1):
        train(args, model, rank, world_size, train_loader, optimizer,scheduler,scaler, epoch, sampler=sampler1)
        test(model, rank, world_size, test_loader)
        scheduler.step()

    init_end_event.record()

    if rank == 0:
        print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
        print(f"{model}")

    if args.save_model:
        # use a barrier to make sure training is done on all ranks
        dist.barrier()
        states = model.state_dict()
        if rank == 0:
            torch.save(states, "mnist_cnn.pt")

    cleanup()

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(fsdp_main,
        args=(WORLD_SIZE, args),
        nprocs=WORLD_SIZE,
        join=True)


