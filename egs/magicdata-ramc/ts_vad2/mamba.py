# Copyright (c) 2024 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Alexis Plaquet, 2024

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.block import Block
from mamba_ssm.models.mixer_seq_simple import _init_weights
from mamba_ssm.ops.triton.layer_norm import RMSNorm

# copy and modified from https://github.com/nttcslab-sp/mamba-diarization/blob/5bb1214296479543989446f7acee5a4bce4cd466/src/plaqntt/modules/mamba.py
class MambaBlockV2(nn.Module):
    """
    Parametrized bidirectional Mamba block from SPMamba https://github.com/JusperLee/SPMamba/blob/main/look2hear/models/SPMamba.py.
    Under Apache License 2.0 (not provided in the original repository).


    # note: 2024-12-18, Block class api required offer mlp_cls
    it required nvcc ,if you haven't it, you can install it via conda , i.e. you install pytorch on cuda11.8, you can `conda install nvidia/label/cuda-11.8.0::cuda-nvcc`
    you install other version via url `https://anaconda.org/nvidia/cuda-nvcc`
    you need to install mamba-ssm via `pip install mamba-ssm  --no-build-isolation`
    """

    def __init__(self, in_channels, n_layer=1, d_state=16, d_conv=4, expand=4, rmsnorm_eps=1e-5, bidirectional=False):
        super(MambaBlockV2, self).__init__()
        self.forward_blocks = nn.ModuleList([])
        for i in range(n_layer):
            self.forward_blocks.append(
                Block(
                    in_channels,
                    mlp_cls=nn.Identity,
                    mixer_cls=partial(Mamba, layer_idx=i, d_state=d_state, d_conv=d_conv, expand=expand),
                    norm_cls=partial(RMSNorm, eps=rmsnorm_eps),
                    fused_add_norm=False,
                )
            )
        if bidirectional:
            self.backward_blocks = nn.ModuleList([])
            for i in range(n_layer):
                self.backward_blocks.append(
                    Block(
                        in_channels,
                        mlp_cls=nn.Identity,
                        mixer_cls=partial(Mamba, layer_idx=i, d_state=d_state, d_conv=d_conv, expand=expand),
                        norm_cls=partial(RMSNorm, eps=rmsnorm_eps),
                        fused_add_norm=False,
                    )
                )

        self.apply(partial(_init_weights, n_layer=n_layer))

    def forward(self, input):
        for_residual = None
        forward_f = input.clone()
        #print(f"forward_f device: {forward_f.device}")
        for block in self.forward_blocks:
            forward_f, for_residual = block(forward_f, for_residual, inference_params=None)
        residual = (forward_f + for_residual) if for_residual is not None else forward_f

        if hasattr(self, "backward_blocks"):
            back_residual = None
            backward_f = torch.flip(input, [1])
            for block in self.backward_blocks:
                backward_f, back_residual = block(backward_f, back_residual, inference_params=None)
            back_residual = (backward_f + back_residual) if back_residual is not None else backward_f

            back_residual = torch.flip(back_residual, [1])
            #if self.bidirectional_merging == "concat":
            residual = torch.cat([residual, back_residual], -1)

        return residual


# reference from https://github.com/nttcslab-sp/mamba-diarization/blob/master/src/plaqntt/modules/resblock.py
#                https://github.com/nttcslab-sp/mamba-diarization/blob/master/src/plaqntt/pyannote_audio/sdresblocks.py
class MambaBlock(nn.Module):
    """
    Parametrized bidirectional Mamba block from SPMamba https://github.com/JusperLee/SPMamba/blob/main/look2hear/models/SPMamba.py.
    Under Apache License 2.0 (not provided in the original repository).


    # note: 2024-12-18, Block class api required offer mlp_cls
    it required nvcc ,if you haven't it, you can install it via conda , i.e. you install pytorch on cuda11.8, you can `conda install nvidia/label/cuda-11.8.0::cuda-nvcc`
    you install other version via url `https://anaconda.org/nvidia/cuda-nvcc`
    you need to install mamba-ssm via `pip install mamba-ssm  --no-build-isolation`
    """

    def __init__(self, in_channels, n_layer=1, d_state=64, d_conv=4, expand=2, rmsnorm_eps=1e-5, bidirectional=True, bidirectional_merging= "add",):
        super(MambaBlock, self).__init__()
        self.bidirectional_merging=bidirectional_merging
        self.forward_blocks = nn.ModuleList([])
        for i in range(n_layer):
            self.forward_blocks.append(
                Block(
                    in_channels,
                    mlp_cls=nn.Identity,
                    mixer_cls=partial(Mamba, layer_idx=i, d_state=d_state, d_conv=d_conv, expand=expand),
                    norm_cls=partial(RMSNorm, eps=rmsnorm_eps),
                    fused_add_norm=False,
                )
            )
        if bidirectional:
            self.backward_blocks = nn.ModuleList([])
            for i in range(n_layer):
                self.backward_blocks.append(
                    Block(
                        in_channels,
                        mlp_cls=nn.Identity,
                        mixer_cls=partial(Mamba, layer_idx=i, d_state=d_state, d_conv=d_conv, expand=expand),
                        norm_cls=partial(RMSNorm, eps=rmsnorm_eps),
                        fused_add_norm=False,
                    )
                )

        self.apply(partial(_init_weights, n_layer=n_layer))

    def forward(self, input):
        for_residual = None
        forward_f = input.clone()
        #print(f"forward_f device: {forward_f.device}")
        for block in self.forward_blocks:
            forward_f, for_residual = block(forward_f, for_residual, inference_params=None)
        residual = (forward_f + for_residual) if for_residual is not None else forward_f

        if hasattr(self, "backward_blocks"):
            back_residual = None
            backward_f = torch.flip(input, [1])
            for block in self.backward_blocks:
                backward_f, back_residual = block(backward_f, back_residual, inference_params=None)
            back_residual = (backward_f + back_residual) if back_residual is not None else backward_f

            back_residual = torch.flip(back_residual, [1])
            if self.bidirectional_merging == "concat":
                residual = torch.cat([residual, back_residual], -1)
            elif self.bidirectional_merging == "add":
                residual += back_residual
            elif self.bidirectional_merging == "mul":
                residual = torch.mul(residual, back_residual)

        return residual


class Mamba2BlockV2(nn.Module):
    """
    Parametrized bidirectional Mamba block from SPMamba https://github.com/JusperLee/SPMamba/blob/main/look2hear/models/SPMamba.py.
    Under Apache License 2.0 (not provided in the original repository).

    # note: 2024-12-18, Block class api required offer mlp_cls
    it required nvcc ,if you haven't it, you can install it via conda , i.e. you install pytorch on cuda11.8, you can `conda install nvidia/label/cuda-11.8.0::cuda-nvcc`
    you install other version via url `https://anaconda.org/nvidia/cuda-nvcc`
   
    you need to install mamba-ssm via `pip install mamba-ssm  --no-build-isolation`
    if you occur the errors:ImportError: /home/maduo/.conda/envs/speaker_diarization/lib/python3.11/site-packages/selective_scan_cuda.cpython-311-x86_64-linux-gnu.so: undefined symbol: _ZN3c104cuda9SetDeviceEi
    you need to uninstall it via `pip uninstall mamba-ssm` and install it again via `pip install mamba-ssm --no-cache-dir --no-build-isolation`

    you need to install causal-conv1d via ` pip install 'causal-conv1d==1.2.1' --no-cache-dir --no-build-isolation`

    It requires that the input of forward() feature dimension is a multiple of 8. In fact, this requirement comes from causal_conv1d
    """

    def __init__(self, in_channels, n_layer=1, d_state=64, d_conv=4, expand=2, rmsnorm_eps=1e-5, bidirectional=False):
        super(Mamba2BlockV2, self).__init__()
        self.forward_blocks = nn.ModuleList([])
        for i in range(n_layer):
            self.forward_blocks.append(
                Block(
                    in_channels,
                    mlp_cls=nn.Identity,
                    mixer_cls=partial(Mamba2, layer_idx=i, d_state=d_state, d_conv=d_conv, expand=expand),
                    norm_cls=partial(RMSNorm, eps=rmsnorm_eps),
                    fused_add_norm=False,
                )
            )
        if bidirectional:
            self.backward_blocks = nn.ModuleList([])
            for i in range(n_layer):
                self.backward_blocks.append(
                    Block(
                        in_channels,
                        mlp_cls=nn.Identity,
                        mixer_cls=partial(Mamba2, layer_idx=i, d_state=d_state, d_conv=d_conv, expand=expand),
                        norm_cls=partial(RMSNorm, eps=rmsnorm_eps),
                        fused_add_norm=False,
                    )
                )

        self.apply(partial(_init_weights, n_layer=n_layer))

    def forward(self, input):
        for_residual = None
        forward_f = input.clone()
        #print(f"forward_f device: {forward_f.device}")
        for block in self.forward_blocks:
            forward_f, for_residual = block(forward_f, for_residual, inference_params=None)
        residual = (forward_f + for_residual) if for_residual is not None else forward_f

        if hasattr(self, "backward_blocks"):
            back_residual = None
            backward_f = torch.flip(input, [1])
            for block in self.backward_blocks:
                backward_f, back_residual = block(backward_f, back_residual, inference_params=None)
            back_residual = (backward_f + back_residual) if back_residual is not None else backward_f

            back_residual = torch.flip(back_residual, [1])
            residual = torch.cat([residual, back_residual], -1)

        return residual

if __name__ == "__main__":
    input=torch.randn(64,200,192).to("cuda")
    print(f"input device: {input.device}")
    model=MambaBlockV2(192,n_layer=2,d_state=64,d_conv=4,expand=4,bidirectional=True).to("cuda")
    model2=Mamba2BlockV2(256,n_layer=2,d_state=64,d_conv=4,expand=4,bidirectional=True).to("cuda")
    output = model(input)
    for name, v in model2.named_parameters():
        print(f"name: {name}, v: {v.shape}")
 
    print(f"output shape: {output.shape}")
    input2=torch.randn(64,200,256).to("cuda")
    output2 = model2(input2)
    print(f"output2 shape: {output2.shape}")
    model3 = MambaBlock(192,n_layer=2,d_state=64,d_conv=4,expand=4,bidirectional=True,bidirectional_merging= "concat",).to("cuda")
    output3 = model3(input)
    print(f"output3 shape: {output3.shape}")
