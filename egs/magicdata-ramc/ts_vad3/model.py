# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, field
import sys
import contextlib
from typing import Optional
from collections import defaultdict
from argparse import Namespace
import math
import numpy as np
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from typing import Any, Callable
#from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel

from cam_pplus_wespeaker import CAMPPlus
from wavlm import WavLM, WavLMConfig
from resnet_wespeaker import ResNet34
from samresnet_wespeaker import SimAM_ResNet34_ASP
from ecapa_tdnn_wespeaker import ECAPA_TDNN_GLOB_c1024
from ecapa_tdnn import ECAPA_TDNN
from whisper_encoder import ModelDimensions
from whisper_encoder import WhisperEncoder
from redimnet_wespeaker import (
    ReDimNetB0,
    ReDimNetB1,
    ReDimNetB2,
    ReDimNetB3,
    ReDimNetM,
    ReDimNetS,
)
from ERes2NetV2 import ERes2NetV2
# from mamba import MambaBlockV2, MambaBlock, Mamba2BlockV2
try:
    from ts_vad2.mamba import MambaBlockV2, MambaBlock, Mamba2BlockV2
except ImportError:
    MambaBlockV2, MambaBlock, Mamba2BlockV2 = None, None, None

import os
sys.path.append(os.path.dirname(__file__))
print(os.path.dirname(__file__))
from redimnet.redimnet.model import ReDimNetWrap


logger = logging.getLogger(__name__)


@dataclass
class TSVADConfig:
    speech_encoder_path: str = (
        "/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt"
    )
    """path to pretrained speech encoder path."""
    speech_encoder_type: str = "CAM++"
    """choice it  from CAM++ or WavLM as speech encoder of our TSVAD"""
    freeze_speech_encoder_updates: int = 4000
    """updates to freeze speech encoder."""

    speaker_encoder_type: str = "CAM++"
    """speaker encoder(it is used to extract target speaker embedding) of our TSVAD"""

    speaker_encoder_path: str = (
        "/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt"
    )
    """path to pretrained speech encoder path."""

    fusion_type: str = "att_wo_linear"
    """cross attention between mix audio frame embedding and target speaker frame embedding, i.e. choice it from `att_wo_linear,att_w_linear` """

    num_attention_head: int = 4
    """number of attention head in transformer"""

    num_transformer_layer: int = 2
    """number of transformer layer """

    transformer_embed_dim: int = 384
    """transformer dimension."""

    transformer_ffn_embed_dim: int = 1536
    """transformer ffn dimension"""

    speaker_embed_dim: int = 192
    """speaker embedding dimension."""

    dropout: float = 0.1
    """dropout prob"""

    use_spk_embed: bool = True
    """whether to use speaker embedding"""

    feature_grad_mult: float = 0.1
    """ wavlm default config is 0.1,
        it will effect ConvFeatureExtractionModel of wavlm model,
        if it is setted to 0.0, ConvFeatureExtractionModel will
        be freezed in tsvad model training stage.
        if it is setted to 0.1,parameters of ConvFeatureExtractionModel of
        wavlm will updated in tsvad model training stage.
    """


    whisper_n_mels: int = 80
    """
    whisper-large-v2 : n_mels=80
    whisper-large-v3 : n_mels=128, The other dimensions of the model are the same as whisper-large-v2
    """

    select_encoder_layer_nums: int = 6
    """it will select transformer encoder layer of wavlm model.
    i.e. --select-encoder-layer-nums 6,
    means that we only use cnn front and first 6 transformer layer of wavlm in tsvad model.
    """


    wavlm_fuse_feat_post_norm: bool = False
    """
    if it is true, it will apply layer norm on weight sum of all transformer layer in pretrained wavlm model

    """

    speech_encoder_config: str = (
        "/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/wav-bert2.0/config.json"
    )
    """
    this config is only used to instantiate wav-bert 2.0 model, this model is used at Seamless model.
    """

    single_backend_type: str = "transformer"
    """single backend type choices from `transformer or mamba or mamba_v2`"""

    multi_backend_type: str = "transformer"
    """multi backend type choices from `transformer or mamba or mamba_v2`"""

    d_state: int = 64
    """d_state of mamba2 """

    expand: int = 4
    """expand of mamba2"""

    label_rate: int = 25
    """default is 25, on label is 40ms,  for redimnet, I use one label is 10ms, means that label_rate is 100"""

    fusion_case: str = ""
    """choices it from `without_speaker_utt_embed, `"""
    without_speaker_utt_embed: bool = False
    """if true, I will use fusion embed as target speaker embed, not use utterance-level embedding"""

model_cfg = TSVADConfig()

class FusionModule(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, q,k,v):
        return F.scaled_dot_product_attention(q,k,v,
            dropout_p=(self.p if self.training else 0.0))

class FusionModule2(nn.Module):
    def __init__(self, p: float=0.1, d_model: int=512):
        super().__init__()
        self.p = p
        self.linear_q = nn.Linear(d_model,d_model)
        self.linear_k = nn.Linear(d_model,d_model)
        self.linear_v = nn.Linear(d_model,d_model)
    def forward(self, q,k,v):
        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)
        return F.scaled_dot_product_attention(q,k,v,
            dropout_p=(self.p if self.training else 0.0))
class SpeechFeatUpsample2(nn.Module):
    def __init__(self, speaker_embed_dim: int, upsample: int, model_dim: int = 2560):
        super(SpeechFeatUpsample2, self).__init__()
        self.speaker_embed_dim = speaker_embed_dim
        # here model_dim means it is feature dimension  before pool layer of resnet34_wespeaker or samresnet model dimension
        self.up = nn.ConvTranspose1d(
            model_dim,
            speaker_embed_dim,
            5,
            stride=upsample,
            padding=2,
            output_padding=1,
        )
        self.batchnorm = BatchNorm1D(num_features=speaker_embed_dim)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)  # (B,F,T) -> (B,D,2T)
        x = self.batchnorm(x)
        x = self.act(x)
        return x  # (B,D,2T)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class BatchNorm1D(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.bn = nn.BatchNorm1d(*args, **kwargs)

    def forward(self, input):
        if torch.sum(torch.isnan(input)) == 0:
            output = self.bn(input)
        else:
            output = input
        return output


from datasets import TSVADDataConfig

data_cfg = TSVADDataConfig()


class TSVADModel(nn.Module):
    def __init__(
        self, cfg=model_cfg, task_cfg=data_cfg, device=torch.device("cpu")
    ) -> None:
        super(TSVADModel, self).__init__()
        self.gradient_checkpointing = False
        self.freeze_speech_encoder_updates = cfg.freeze_speech_encoder_updates
        self.use_spk_embed = cfg.use_spk_embed
        self.rs_dropout = nn.Dropout(p=cfg.dropout)
        self.label_rate = task_cfg.label_rate
        self.support_variable_number_speakers = (
            task_cfg.support_variable_number_speakers
        )
        self.without_speaker_utt_embed = cfg.without_speaker_utt_embed
        self.fusion_case = cfg.fusion_case
        logger.info(f"in the model class init, self.fusion_case: {self.fusion_case}")
        self.dropout = cfg.dropout
        # assert (
        #    self.label_rate == cfg.label_rate
        # ), f"self.label_rate is {elf.label_rate} not support!"
        self.label_rate == cfg.label_rate
        self.speech_encoder_type = cfg.speech_encoder_type
        self.speech_encoder_path = cfg.speech_encoder_path
        self.speaker_encoder_type = cfg.speaker_encoder_type
        self.speaker_encoder_path = cfg.speaker_encoder_path
        self.fusion_type = cfg.fusion_type

        self.wavlm_fuse_feat_post_norm = cfg.wavlm_fuse_feat_post_norm
        print(f"self.wavlm_fuse_feat_post_norm: {self.wavlm_fuse_feat_post_norm}")
        self.max_num_speaker = task_cfg.max_num_speaker
        sample_times = 16000 / task_cfg.sample_rate
        self.select_encoder_layer_nums = (
            cfg.select_encoder_layer_nums
        )  # only for wav-bert2
        ## create speech encoder
        (
            self.speech_encoder,
            self.speech_down_or_up,
            self.pretrain_speech_encoder_dim,
            self.weights,
            self.wavlm_encoder_num_layer,
            self.wavlmproj,
            self.wavlmlnorm,
        ) = self.create_speech_encoder(sample_times, cfg, device)

        ## create speaker encoder
        self.speaker_encoder, self.pretrain_speaker_encoder_dim = self.create_speaker_encoder(cfg, device)
        assert self.pretrain_speaker_encoder_dim == self.pretrain_speech_encoder_dim, f"self.pretrain_speaker_encoder_dim: {self.pretrain_speaker_encoder_dim},self.pretrain_speech_encoder_dim: {self.pretrain_speech_encoder_dim}!!"

        # create fusion encoder
        #self.fusion_encoder = self.creat_fusion_encoder(cfg)

        # projection
        if cfg.speaker_embed_dim * 2 != cfg.transformer_embed_dim:
            self.proj_layer = nn.Linear(
                cfg.speaker_embed_dim * 2, cfg.transformer_embed_dim
            )
        else:
            self.proj_layer = None

        self.single_backend_type = cfg.single_backend_type
        self.single_backend, self.pos_encoder, self.backend_down = (
            self.create_single_backend(cfg, task_cfg)
        )

        self.multi_backend_type = cfg.multi_backend_type
        self.multi_backend, self.fc, self.multi_backend_proj = (
            self.create_multi_backend(cfg)
        )

        # fixed speaker encoder
        for param in self.speaker_encoder.parameters():
            param.requires_grad = False

        # self.fc = nn.Linear(cfg.transformer_embed_dim, self.max_num_speaker)
        if self.fusion_type == "att_w_linear":
            self.linear_q = nn.Linear(self.pretrain_speaker_encoder_dim, self.pretrain_speaker_encoder_dim)
            self.linear_k = nn.Linear(self.pretrain_speaker_encoder_dim, self.pretrain_speaker_encoder_dim)
            self.linear_v = nn.Linear(self.pretrain_speaker_encoder_dim, self.pretrain_speaker_encoder_dim)
        if self.fusion_case=="fusion_embed_as_mix_embed_w_utt_embed":
            self.fusion_linear = nn.Linear(self.pretrain_speaker_encoder_dim*self.max_num_speaker, self.pretrain_speaker_encoder_dim)
    #def creat_fusion_encoder(self,cfg):
    #    self.fusion_encoder = nn.Module = None
    #    if self.fusion_type == "att_wo_linear":
    #        self.fusion_encoder = FusionModule(p=cfg.dropout)
    #    elif self.fusion_type == "att_w_linear":
    #        self.fusion_encoder = FusionModule2(p=cfg.dropout,d_model=self.pretrain_speaker_encoder_dim)
    #    return self.fusion_encoder

    def create_single_backend(self, cfg, task_cfg):
        self.single_backend_proj_for_mamba2: nn.Module = None
        self.single_backend: Optional[nn.Module] = None
        self.pos_encoder: Optional[nn.Module] = None
        self.backend_down: Optional[nn.Module] = None
        if cfg.single_backend_type == "transformer":
            self.pos_encoder = PositionalEncoding(
                cfg.transformer_embed_dim,
                dropout=cfg.dropout,
                max_len=(task_cfg.rs_len * self.label_rate),
            )
            self.single_backend = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=cfg.transformer_embed_dim,
                    dim_feedforward=cfg.transformer_ffn_embed_dim,
                    nhead=cfg.num_attention_head,
                    dropout=cfg.dropout,
                ),
                num_layers=cfg.num_transformer_layer,
            )
            if self.support_variable_number_speakers:
                self.backend_down = None
            else:
                self.backend_down = nn.Sequential(
                    nn.Conv1d(
                        cfg.transformer_embed_dim * self.max_num_speaker,
                        cfg.transformer_embed_dim,
                        5,
                        stride=1,
                        padding=2,
                    ),
                    BatchNorm1D(num_features=cfg.transformer_embed_dim),
                    nn.ReLU(),
                )
        elif cfg.single_backend_type == "conformer":
            self.pos_encoder = PositionalEncoding(
                cfg.transformer_embed_dim,
                dropout=cfg.dropout,
                max_len=(task_cfg.rs_len * self.label_rate),
            )
            self.single_backend = torchaudio.models.Conformer(
                input_dim=cfg.transformer_embed_dim,
                num_heads=cfg.num_attention_head,
                ffn_dim=cfg.transformer_ffn_embed_dim,
                num_layers=cfg.num_transformer_layer,
                depthwise_conv_kernel_size=31,
            )
            self.backend_down = nn.Sequential(
                nn.Conv1d(
                    cfg.transformer_embed_dim * self.max_num_speaker,
                    cfg.transformer_embed_dim,
                    5,
                    stride=1,
                    padding=2,
                ),
                BatchNorm1D(num_features=cfg.transformer_embed_dim),
                nn.ReLU(),
            )
        elif cfg.single_backend_type == "mamba":
            self.pos_encoder = PositionalEncoding(
                cfg.transformer_embed_dim,
                dropout=cfg.dropout,
                max_len=(task_cfg.rs_len * self.label_rate),
            )
            ## because I use concat resual, so self.single_backend output feature dimension equal to cfg.transformer_embed_dim *2
            self.single_backend = MambaBlockV2(
                cfg.transformer_embed_dim,
                n_layer=cfg.num_transformer_layer,
                d_state=64,
                d_conv=4,
                expand=4,
                bidirectional=True,
            )
            self.backend_down = nn.Sequential(
                nn.Conv1d(
                    2 * cfg.transformer_embed_dim * self.max_num_speaker,
                    cfg.transformer_embed_dim,
                    5,
                    stride=1,
                    padding=2,
                ),
                BatchNorm1D(num_features=cfg.transformer_embed_dim),
                nn.ReLU(),
            )
        elif cfg.single_backend_type == "mamba_v2":
            self.pos_encoder = PositionalEncoding(
                cfg.transformer_embed_dim,
                dropout=cfg.dropout,
                max_len=(task_cfg.rs_len * self.label_rate),
            )
            ## because I use "add" resual, so self.single_backend output feature dimension equal to cfg.transformer_embed_dim
            self.single_backend = MambaBlock(
                cfg.transformer_embed_dim,
                n_layer=cfg.num_transformer_layer,
                d_state=64,
                d_conv=4,
                expand=2,
                bidirectional=True,
                bidirectional_merging="add",
            )
            self.backend_down = nn.Sequential(
                nn.Conv1d(
                    cfg.transformer_embed_dim * self.max_num_speaker,
                    cfg.transformer_embed_dim,
                    5,
                    stride=1,
                    padding=2,
                ),
                BatchNorm1D(num_features=cfg.transformer_embed_dim),
                nn.ReLU(),
            )
        elif cfg.single_backend_type == "mamba2":
            self.pos_encoder = PositionalEncoding(
                cfg.transformer_embed_dim,
                dropout=cfg.dropout,
                max_len=(task_cfg.rs_len * self.label_rate),
            )
            # causal_conv1d  channel must be multiples of 8  , So I select 384=192*2 as model dimension.
            self.single_backend = Mamba2BlockV2(
                cfg.transformer_embed_dim,
                n_layer=cfg.num_transformer_layer,
                d_state=cfg.d_state,
                d_conv=4,
                expand=cfg.expand,
                bidirectional=True,
            )

            self.backend_down = nn.Sequential(
                nn.Conv1d(
                    2 * cfg.transformer_embed_dim * self.max_num_speaker,
                    cfg.transformer_embed_dim,
                    5,
                    stride=1,
                    padding=2,
                ),
                BatchNorm1D(num_features=cfg.transformer_embed_dim),
                nn.ReLU(),
            )

        return self.single_backend, self.pos_encoder, self.backend_down

    def create_multi_backend(self, cfg):
        self.multi_backend: Optional[nn.Module] = None
        self.fc: Optional[nn.Module] = None
        self.multi_backend_proj: nn.Module = None
        if cfg.multi_backend_type == "transformer":
            self.multi_backend = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=cfg.transformer_embed_dim,
                    dim_feedforward=cfg.transformer_ffn_embed_dim,
                    nhead=cfg.num_attention_head,
                    dropout=cfg.dropout,
                ),
                num_layers=cfg.num_transformer_layer,
            )
            if self.support_variable_number_speakers:
                self.fc = nn.Linear(cfg.transformer_embed_dim, 1)
            else:
                self.fc = nn.Linear(cfg.transformer_embed_dim, self.max_num_speaker)
        elif cfg.multi_backend_type == "conformer":
            self.multi_backend = torchaudio.models.Conformer(
                input_dim=cfg.transformer_embed_dim,
                num_heads=cfg.num_attention_head,
                ffn_dim=cfg.transformer_ffn_embed_dim,
                num_layers=cfg.num_transformer_layer,
                depthwise_conv_kernel_size=31,
            )

            if self.support_variable_number_speakers:
                self.fc = nn.Linear(cfg.transformer_embed_dim, 1)
            else:
                self.fc = nn.Linear(cfg.transformer_embed_dim, self.max_num_speaker)

        elif cfg.multi_backend_type == "mamba":
            self.multi_backend = MambaBlockV2(
                cfg.transformer_embed_dim,
                n_layer=cfg.num_transformer_layer,
                d_state=64,
                d_conv=4,
                expand=4,
                bidirectional=True,
            )
            self.fc = nn.Linear(2 * cfg.transformer_embed_dim, self.max_num_speaker)

        elif cfg.multi_backend_type == "mamba_v2":
            self.multi_backend = MambaBlock(
                cfg.transformer_embed_dim,
                n_layer=cfg.num_transformer_layer,
                d_state=64,
                d_conv=4,
                expand=2,
                bidirectional=True,
                bidirectional_merging="add",
            )
            self.fc = nn.Linear(cfg.transformer_embed_dim, self.max_num_speaker)

        elif cfg.multi_backend_type == "mamba2":
            self.multi_backend = Mamba2BlockV2(
                cfg.transformer_embed_dim,
                n_layer=cfg.num_transformer_layer,
                d_state=cfg.d_state,
                d_conv=4,
                expand=cfg.expand,
                bidirectional=True,
            )
            self.multi_backend_proj = nn.Linear(
                2 * cfg.transformer_embed_dim, cfg.transformer_embed_dim
            )

            if self.support_variable_number_speakers:
                self.fc = nn.Linear(cfg.transformer_embed_dim, 1)
            else:
                self.fc = nn.Linear(cfg.transformer_embed_dim, self.max_num_speaker)

        return self.multi_backend, self.fc, self.multi_backend_proj

    def create_speaker_encoder(self,cfg, device):
        self.speaker_encoder: Optional[nn.Module] = None
        pretrain_speaker_encoder_dim: int = None
        if self.speaker_encoder_type=="CAM++" or self.speaker_encoder_type=="CAM++_per":
            self.speaker_encoder = CAMPPlus(feat_dim=80, embedding_size=192)
            self.speaker_encoder.train()
            self.load_speaker_encoder(cfg.speech_encoder_path, device=device, module_name="speaker_encoder")
            pretrain_speaker_encoder_dim = 512 # it is dimension of last layer of before pooling layer.

        return self.speaker_encoder, pretrain_speaker_encoder_dim


    def create_speech_encoder(self, sample_times, cfg, device):
        self.speech_encoder: Optional[nn.Module] = None
        self.speech_down_or_up: Optional[nn.Module] = None
        pretrain_speech_encoder_dim: int = None
        self.weights: Optional[nn.Module] = None  # only for WavLM_weight_sum
        wavlm_encoder_num_layer: int = None  # only for WavLM_weight_sum
        self.wavlmlnorm: Optional[nn.Module] = None  # only for WavLM_weight_sum
        self.wavlmproj: Optional[nn.Module] = None  # only for WavLM_weight_sum
        if self.speech_encoder_type == "CAM++":
            self.speech_encoder = CAMPPlus(feat_dim=80, embedding_size=192)
            self.speech_encoder.train()
            self.load_speaker_encoder(
                cfg.speech_encoder_path, device=device, module_name="speech_encoder"
            )
            stride = int(2 // sample_times) if self.label_rate == 25 else 1
            pretrain_speech_encoder_dim = 512
            self.speech_down_or_up = nn.Sequential(
                nn.Conv1d(
                    pretrain_speech_encoder_dim,
                    cfg.speaker_embed_dim,
                    5,
                    stride=stride,
                    padding=2,
                ),
                BatchNorm1D(num_features=cfg.speaker_embed_dim),
                nn.ReLU(),
            )
        elif self.speech_encoder_type == "ERes2NetV2_COMMON":
            self.speech_encoder = ERes2NetV2(feat_dim=80, embedding_size=192, m_channels=64, baseWidth=26, scale=2, expansion=2)
            self.speech_encoder.train()
            self.load_speaker_encoder(
                cfg.speech_encoder_path, device=device, module_name="speech_encoder"
            )
            pretrain_speech_encoder_dim=10240
            #no downsample, stride=1, label_rate of last layer of pool layer is 13, means that 1s audio has 13 frames
            self.speech_down_or_up = nn.Sequential(
                nn.Conv1d(
                    pretrain_speech_encoder_dim,
                    cfg.speaker_embed_dim,
                    5,
                    stride=1,
                    padding=2,
                ),
                BatchNorm1D(num_features=cfg.speaker_embed_dim),
                nn.ReLU(),
            )

        elif self.speech_encoder_type == "ERes2NetV2_w24s4ep4_COMMON":
            self.speech_encoder = ERes2NetV2(feat_dim=80, embedding_size=192, m_channels=64, baseWidth=24, scale=4, expansion=4)
            self.speech_encoder.train()
            self.load_speaker_encoder(
                cfg.speech_encoder_path, device=device, module_name="speech_encoder"
            )
            pretrain_speech_encoder_dim=20480
            # # no downsample, stride=1, label_rate is 13, means that 1s audio has 13 frames
            self.speech_down_or_up = nn.Sequential(
                nn.Conv1d(
                    pretrain_speech_encoder_dim,
                    cfg.speaker_embed_dim,
                    5,
                    stride=1,
                    padding=2,
                ),
                BatchNorm1D(num_features=cfg.speaker_embed_dim),
                nn.ReLU(),
            )
        # ReDimNetB2_offical
        elif self.speech_encoder_type == "ReDimNetB2_offical":
            model_name = f'b2-vox2-ft_lm.pt'
            url=f"https://github.com/IDRnD/ReDimNet/releases/download/latest/{model_name}"
            full_state_dict = torch.hub.load_state_dict_from_url(url, progress=True)

            model_config = full_state_dict['model_config']
            state_dict = full_state_dict['state_dict']
            logger.info(f"ReDimNetB2_offical model_config: {model_config}")
            self.speech_encoder = ReDimNetWrap(**model_config)
            self.speech_encoder.load_state_dict(state_dict)
            pretrain_speech_encoder_dim = 1152
            # no downsample, stride=1, label_rate is 67
            self.speech_down_or_up = nn.Sequential(
                nn.Conv1d(
                    pretrain_speech_encoder_dim,
                    cfg.speaker_embed_dim,
                    5,
                    stride=1,
                    padding=2,
                ),
                BatchNorm1D(num_features=cfg.speaker_embed_dim),
                nn.ReLU(),
            )

        elif self.speech_encoder_type == "ReDimNetB0":
            self.speech_encoder = ReDimNetB0(
                feat_dim=60, embed_dim=192, pooling_func="ASTP"
            )
            self.speech_encoder.train()
            self.load_speaker_encoder(
                cfg.speech_encoder_path, device=device, module_name="speech_encoder"
            )
            # Because input and output of ReDimNetB2 are not subsample, so I need to subsample rate is 4 to match label_rate.
            stride = int(4 // sample_times) if self.label_rate == 25 else 1
            pretrain_speech_encoder_dim = 600
            self.speech_down_or_up = nn.Sequential(
                nn.Conv1d(
                    pretrain_speech_encoder_dim,
                    cfg.speaker_embed_dim,
                    5,
                    stride=stride,
                    padding=2,
                ),
                BatchNorm1D(num_features=cfg.speaker_embed_dim),
                nn.ReLU(),
            )
        elif self.speech_encoder_type == "ReDimNetB1":
            self.speech_encoder = ReDimNetB1(
                feat_dim=72, embed_dim=192, pooling_func="ASTP"
            )
            self.speech_encoder.train()
            self.load_speaker_encoder(
                cfg.speech_encoder_path, device=device, module_name="speech_encoder"
            )
            # Because input and output of ReDimNetB2 are not subsample, so I need to subsample rate is 4 to match label_rate.
            stride = int(4 // sample_times) if self.label_rate == 25 else 1
            pretrain_speech_encoder_dim = 864
            self.speech_down_or_up = nn.Sequential(
                nn.Conv1d(
                    pretrain_speech_encoder_dim,
                    cfg.speaker_embed_dim,
                    5,
                    stride=stride,
                    padding=2,
                ),
                BatchNorm1D(num_features=cfg.speaker_embed_dim),
                nn.ReLU(),
            )
        elif self.speech_encoder_type == "ReDimNetB2":
            self.speech_encoder = ReDimNetB2(
                feat_dim=72, embed_dim=192, pooling_func="ASTP"
            )
            self.speech_encoder.train()
            self.load_speaker_encoder(
                cfg.speech_encoder_path, device=device, module_name="speech_encoder"
            )
            # Because input and output of ReDimNetB2 are not subsample, so I need to subsample rate is 4 to match label_rate.
            stride = int(4 // sample_times) if self.label_rate == 25 else 1
            pretrain_speech_encoder_dim = 1152
            self.speech_down_or_up = nn.Sequential(
                nn.Conv1d(
                    pretrain_speech_encoder_dim,
                    cfg.speaker_embed_dim,
                    5,
                    stride=stride,
                    padding=2,
                ),
                BatchNorm1D(num_features=cfg.speaker_embed_dim),
                nn.ReLU(),
            )
        elif self.speech_encoder_type == "ReDimNetB2_layernorm":
            self.speech_encoder = ReDimNetB2(
                feat_dim=72, embed_dim=192, pooling_func="ASTP"
            )
            self.speech_encoder.train()
            self.load_speaker_encoder(
                cfg.speech_encoder_path, device=device, module_name="speech_encoder"
            )
            # Because input and output of ReDimNetB2 are not subsample, so I need to subsample rate is 4 to match label_rate.
            stride = int(4 // sample_times) if self.label_rate == 25 else 1
            pretrain_speech_encoder_dim = 1152
            self.speech_down_or_up = nn.Sequential(
                nn.Conv1d(
                    pretrain_speech_encoder_dim,
                    cfg.speaker_embed_dim,
                    5,
                    stride=stride,
                    padding=2,
                ),
                nn.LayerNorm(cfg.speaker_embed_dim),
                nn.ReLU(),
            )
        elif self.speech_encoder_type == "ReDimNetB3":
            self.speech_encoder = ReDimNetB3(
                feat_dim=72, embed_dim=192, pooling_func="ASTP"
            )
            self.speech_encoder.train()
            self.load_speaker_encoder(
                cfg.speech_encoder_path, device=device, module_name="speech_encoder"
            )
            # Because input and output of ReDimNetB3 are not subsample, so I need to subsample rate is 4 to match label_rate.
            stride = int(4 // sample_times) if self.label_rate == 25 else 1
            pretrain_speech_encoder_dim = 1152
            self.speech_down_or_up = nn.Sequential(
                nn.Conv1d(
                    pretrain_speech_encoder_dim,
                    cfg.speaker_embed_dim,
                    5,
                    stride=stride,
                    padding=2,
                ),
                BatchNorm1D(num_features=cfg.speaker_embed_dim),
                nn.ReLU(),
            )
        elif self.speech_encoder_type == "ReDimNetM":
            self.speech_encoder = ReDimNetM(
                feat_dim=72, embed_dim=192, pooling_func="ASTP"
            )
            self.speech_encoder.train()
            self.load_speaker_encoder(
                cfg.speech_encoder_path, device=device, module_name="speech_encoder"
            )
            # Because input and output of ReDimNetM are not subsample, so I need to subsample rate is 4 to match label_rate.
            stride = int(4 // sample_times) if self.label_rate == 25 else 1
            pretrain_speech_encoder_dim = 1728
            self.speech_down_or_up = nn.Sequential(
                nn.Conv1d(
                    pretrain_speech_encoder_dim,
                    cfg.speaker_embed_dim,
                    5,
                    stride=stride,
                    padding=2,
                ),
                BatchNorm1D(num_features=cfg.speaker_embed_dim),
                nn.ReLU(),
            )

        elif self.speech_encoder_type == "ReDimNetS":
            self.speech_encoder = ReDimNetS(
                feat_dim=72, embed_dim=192, pooling_func="ASTP"
            )
            self.speech_encoder.train()
            self.load_speaker_encoder(
                cfg.speech_encoder_path, device=device, module_name="speech_encoder"
            )
            # Because input and output of ReDimNetS are not subsample, so I need to subsample rate is 4 to match label_rate.
            stride = int(4 // sample_times) if self.label_rate == 25 else 1
            pretrain_speech_encoder_dim = 1152
            self.speech_down_or_up = nn.Sequential(
                nn.Conv1d(
                    pretrain_speech_encoder_dim,
                    cfg.speaker_embed_dim,
                    5,
                    stride=stride,
                    padding=2,
                ),
                BatchNorm1D(num_features=cfg.speaker_embed_dim),
                nn.ReLU(),
            )
        elif self.speech_encoder_type == "w2v-bert2":
            # Its input is 80-dim fbank(i.e.B,T,D),however it reshape to (B,T//2,D*2)
            # It is kaldi fbank(10ms(160 samples) frame shift, 25ms(400samples) windows lens),
            # So origin fbank frame rate is 100,and after reshape operation, fbank frame rate is 50
            # so input shape of  Wav2Vec2BertModel is (B,T//2,D*2), frame rate is 50
            # checkpoint = torch.load(cfg.speech_encoder_path,map_location=device)
            from transformers import Wav2Vec2BertConfig, Wav2Vec2BertModel

            conf = Wav2Vec2BertConfig.from_pretrained(
                cfg.speech_encoder_config, hidden_act="swish"
            )
            conf.num_hidden_layers = cfg.select_encoder_layer_nums
            # self.speech_encoder = Wav2Vec2BertModel(conf) ##  Initializing a model (with random weights)
            # self.speech_encoder.train()
            # from safetensors.torch import load_model
            # load_model(self.speech_encoder,cfg.speech_encoder_path,strict=False)
            # Instead of model.load_state_dict(load_file("model.safetensors"))
            self.speech_encoder = Wav2Vec2BertModel.from_pretrained(
                cfg.speech_encoder_path, config=conf
            )
            # self.speech_encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            pretrain_speech_encoder_dim = conf.output_hidden_size

            # self.speech_encoder.load_state_dict(checkpoint["model"], strict=False)
            # w2v-bert2.0 output frame rate 50,so I need to downsample to 25.
            self.speech_down_or_up = nn.Sequential(
                nn.Conv1d(
                    pretrain_speech_encoder_dim,
                    cfg.speaker_embed_dim,
                    5,
                    stride=int(2 // sample_times),
                    padding=2,
                ),
                BatchNorm1D(num_features=cfg.speaker_embed_dim),
                nn.ReLU(),
            )
        elif self.speech_encoder_type == "hubert":
            # hubert chinese model is from https://huggingface.co/TencentGameMate/chinese-hubert-large/tree/main
            # maduo note: I follow pretrain mask setting because I did not reset the relevant mask parameters
            from transformers import HubertConfig, HubertModel

            conf = HubertConfig.from_pretrained(cfg.speech_encoder_config)
            conf.num_hidden_layers = cfg.select_encoder_layer_nums
            self.speech_encoder = HubertModel.from_pretrained(
                cfg.speech_encoder_path, config=conf
            )
            if cfg.feature_grad_mult == 0:  # freeze cnn front in tsvad train stage
                self.speech_encoder.feature_extractor._freeze_parameters()

            pretrain_speech_encoder_dim = conf.hidden_size
            # wav-bert2.0 output frame rate 50,so I need to downsample to 25.
            self.speech_down_or_up = nn.Sequential(
                nn.Conv1d(
                    pretrain_speech_encoder_dim,
                    cfg.speaker_embed_dim,
                    5,
                    stride=int(2 // sample_times),
                    padding=2,
                ),
                BatchNorm1D(num_features=cfg.speaker_embed_dim),
                nn.ReLU(),
            )
        elif self.speech_encoder_type == "WavLM":
            checkpoint = torch.load(cfg.speech_encoder_path, map_location=device)
            wavlm_cfg = WavLMConfig(checkpoint["cfg"])
            wavlm_cfg.encoder_layers = cfg.select_encoder_layer_nums  # default =6
            wavlm_cfg.feature_grad_mult = cfg.feature_grad_mult
            pretrain_speech_encoder_dim = wavlm_cfg.encoder_embed_dim
            self.speech_encoder = WavLM(
                wavlm_cfg
            )  #  Initializing a model (with random weights)
            self.speech_encoder.train()
            self.speech_encoder.load_state_dict(checkpoint["model"], strict=False)
            # wavlm output frame rate 50,so I need to downsample to 25.
            self.speech_down_or_up = nn.Sequential(
                nn.Conv1d(
                    pretrain_speech_encoder_dim,
                    cfg.speaker_embed_dim,
                    5,
                    stride=int(2 // sample_times),
                    padding=2,
                ),
                BatchNorm1D(num_features=cfg.speaker_embed_dim),
                nn.ReLU(),
            )
        elif self.speech_encoder_type == "WavLM_weight_sum":
            checkpoint = torch.load(cfg.speech_encoder_path, map_location=device)
            wavlm_cfg = WavLMConfig(checkpoint["cfg"])
            wavlm_cfg.feature_grad_mult = cfg.feature_grad_mult
            pretrain_speech_encoder_dim = wavlm_cfg.encoder_embed_dim
            self.speech_encoder = WavLM(wavlm_cfg)
            self.speech_encoder.train()
            self.speech_encoder.load_state_dict(checkpoint["model"], strict=False)
            # print(f"wavlm_cfg.encoder_layers: {wavlm_cfg.encoder_layers}")
            wavlm_encoder_num_layer = wavlm_cfg.encoder_layers
            # self.weights = nn.Parameter(torch.zeros(wavlm_encoder_num_layer))

            # refer from: `Leveraging Self-Supervised Learning for Speaker Diarization`
            #       code: https://github.com/BUTSpeechFIT/DiariZen/blob/main/diarizen/models/eend/model_wavlm_conformer.py#L182
            if self.wavlm_fuse_feat_post_norm:
                self.weights = nn.Linear(wavlm_encoder_num_layer, 1, bias=False)
                self.wavlmproj = nn.Linear(
                    pretrain_speech_encoder_dim, cfg.speaker_embed_dim
                )
                self.wavlmlnorm = nn.LayerNorm(cfg.speaker_embed_dim)
            else:
                self.weights = nn.Parameter(torch.zeros(wavlm_encoder_num_layer))
            # wavlm output frame rate 50,so I need to downsample to 25.
            self.speech_down_or_up = nn.Sequential(
                nn.Conv1d(
                    (
                        cfg.speaker_embed_dim
                        if self.wavlm_fuse_feat_post_norm
                        else pretrain_speech_encoder_dim
                    ),
                    cfg.speaker_embed_dim,
                    5,
                    stride=int(2 // sample_times),
                    padding=2,
                ),
                BatchNorm1D(num_features=cfg.speaker_embed_dim),
                nn.ReLU(),
            )
        elif self.speech_encoder_type == "whisper":
            state_dict = torch.load(cfg.speech_encoder_path, map_location=device)
            ## convert hugging face whisper weight name into my whisper encoder model
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("model.encoder.", "", 1)
                new_state_dict[new_key] = v

            whisper_cfg = ModelDimensions()
            whisper_cfg.n_mels = cfg.whisper_n_mels
            pretrain_speech_encoder_dim = whisper_cfg.n_audio_state * (
                whisper_cfg.layer_ed - whisper_cfg.layer_st + 1
            )
            self.speech_encoder = WhisperEncoder(whisper_cfg)
            self.speech_encoder.train()
            self.speech_encoder.load_state_dict(new_state_dict, strict=False)
            # whisper output frame rate 50,so I need to downsample to 25.
            self.speech_down_or_up = nn.Sequential(
                nn.Conv1d(
                    pretrain_speech_encoder_dim,
                    cfg.speaker_embed_dim,
                    5,
                    stride=int(2 // sample_times),
                    padding=2,
                ),
                BatchNorm1D(num_features=cfg.speaker_embed_dim),
                nn.ReLU(),
            )

        # resnet_wespeaker
        elif self.speech_encoder_type == "resnet34_wespeaker":
            self.speech_encoder = ResNet34(
                feat_dim=80,
                embed_dim=256,
                pooling_func="TSTP",
                two_emb_layer=False,
                speech_encoder=True,
            )
            self.speech_encoder.train()
            self.load_speaker_encoder(
                cfg.speech_encoder_path, device=device, module_name="speech_encoder"
            )
            # input of cam++ model is fbank, means that 1s has 100 frames
            # we set target label rate is 25, means that 1s has 25 frames
            # resnet34_wespeaker model downsample scale is 8, so frame rate  is 12.5, so I should set stride equal to 2.
            upsample = 2
            ## the input shape of self.speech_up except is (B,F,T)
            # self.speech_up = SpeechFeatUpsample(speaker_embed_dim=cfg.speaker_embed_dim, upsample=upsample)
            self.speech_down_or_up = SpeechFeatUpsample2(
                speaker_embed_dim=cfg.speaker_embed_dim,
                upsample=upsample,
                model_dim=2560,
            )
        # samresnet_wespeaker
        elif self.speech_encoder_type == "simam_resnet34_wespeaker":
            self.speech_encoder = SimAM_ResNet34_ASP(
                in_planes=64,
                embed_dim=256,
                acoustic_dim=80,
                dropout=0,
                speech_encoder=True,
            )
            self.speech_encoder.train()
            self.load_speaker_encoder(
                cfg.speech_encoder_path, device=device, module_name="speech_encoder"
            )
            # input of cam++ model is fbank, means that 1s has 100 frames
            # we set target label rate is 25, means that 1s has 25 frames
            # samresnet_wespeaker model downsample scale is 8, so frame rate is 12.5, so I should set stride equal to 2.
            upsample = 2
            ## the input shape of self.speech_up except is (B,F,T)
            self.speech_down_or_up = SpeechFeatUpsample2(
                speaker_embed_dim=cfg.speaker_embed_dim,
                upsample=upsample,
                model_dim=5120,
            )
        # ecapa_wespeaker
        elif self.speech_encoder_type == "ecapa_channel_1024_wespeaker":
            # ECAPA_TDNN_GLOB_c1024(feat_dim=80,embed_dim=192,pooling_func='ASTP',speech_encoder=True).train()

            self.speech_encoder = ECAPA_TDNN_GLOB_c1024(
                feat_dim=80, embed_dim=192, pooling_func="ASTP", speech_encoder=True
            )
            self.speech_encoder.train()
            self.load_speaker_encoder(
                cfg.speech_encoder_path, device=device, module_name="speech_encoder"
            )
            stride = int(4 // sample_times)
            pretrain_speech_encoder_dim = 1536  # # here 1536 means it is feature dimension  before pool layer of ecapa_wespeaker model dimension
            ## the input shape of self.speech_down except is (B,F,T)
            self.speech_down_or_up = nn.Sequential(
                nn.Conv1d(
                    pretrain_speech_encoder_dim,
                    cfg.speaker_embed_dim,
                    5,
                    stride=stride,
                    padding=2,
                ),
                BatchNorm1D(num_features=cfg.speaker_embed_dim),
                nn.ReLU(),
            )
        # ecapa opensource, i.e. https://github.com/TaoRuijie/ECAPA-TDNN/blob/main/model.py
        # maybe add it.
        # elif self.speech_encoder_type == "ecapa_channel_1024":
        # it is wavform as input of network, fbank is inside of network. it
        #     self.speech_encoder = ECAPA_TDNN(1024,speech_encoder=True)
        return (
            self.speech_encoder,
            self.speech_down_or_up,
            pretrain_speech_encoder_dim,
            self.weights,
            wavlm_encoder_num_layer,
            self.wavlmproj,
            self.wavlmlnorm,
        )
    #def model_parameters(self):
    #    return[*self.speech_encoder.parameters(),
    #           *
    #            ]
    def non_speech_encoder_parameters(self):

        if self.multi_backend_proj is not None:

            return [
                *self.speech_down_or_up.parameters(),
                *self.pos_encoder.parameters(),
                *self.single_backend.parameters(),
                *self.backend_down.parameters(),
                *self.multi_backend.parameters(),
                *self.multi_backend_proj.parameters(),# only for multi_backend=="mamba2"
                *self.fc.parameters(),
            ]
        else:
            return [
                *self.speech_down_or_up.parameters(),
                *self.pos_encoder.parameters(),
                *self.single_backend.parameters(),
                *self.backend_down.parameters(),
                *self.multi_backend.parameters(),
                *self.fc.parameters(),
            ]
    def fusion_att(self,mix_frame_emb, target_speaker_frame_emb):
        """
        args:
            mix_frame_emb: shape(B,F,T')
            target_speaker_frame_emb: shape(B*num_speakers,F,T')
        return:
            fusion_emb : shape(B,F,T')


        """
        query = mix_frame_emb
        query = query.permute(0,2,1).unsqueeze(1) #(B,1,T',F)


        _,F1,T1 = target_speaker_frame_emb.shape
        speaker_frame_emb= target_speaker_frame_emb.view(-1,self.max_num_speaker,F1,T1)

        speaker_frame_emb = speaker_frame_emb.permute(0,1,3,2)
        B,_,T1,F1 = speaker_frame_emb.shape
        speaker_frame_emb = speaker_frame_emb.reshape(B,-1,F1) #(B,num_speakers*T1, F1)

        key = speaker_frame_emb
        key = key.unsqueeze(1) # (B,1, num_speakers*T1, F1)
        value = speaker_frame_emb
        value = value.unsqueeze(1) # (B,1, num_speakers*T1, F1)
        #print(f"query shape: {query.shape}, key shape: {key.shape}, value shape: {value.shape}")
        #with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
        fusion_emb = F.scaled_dot_product_attention(query,key,value, dropout_p=(self.dropout if self.training else 0.0)) # (B,1,T',F)

        fusion_emb = fusion_emb.squeeze(1).transpose(1, 2)  # (B,T',F)->(B,F,T')
        return  fusion_emb


    def fusion_single_head_att_per_speaker(self, mix_frame_emb, target_speaker_frame_emb):
        """
        args:
            mix_frame_emb: shape(B,F,T'')
            target_speaker_frame_emb: shape(B, num_speakers,F,T')
        return:
            fusion_emb : shape(B,num_speakers,F,T'')


        """
        query = mix_frame_emb
        query = query.permute(0,2,1).unsqueeze(1) #(B,1,T'',F)
        fusion_embs = []
        for i in range(self.max_num_speaker):
            ts_frame_emb = target_speaker_frame_emb[:,i,:,:]
            ts_frame_emb = ts_frame_emb.unsqueeze(1).permute(0,1,3,2) #(B,1,T',F)
            fusion_emb = F.scaled_dot_product_attention(query,ts_frame_emb,ts_frame_emb, dropout_p=(self.dropout if self.training else 0.0)) # (B,1,,T'',F)
            fusion_embs.append(fusion_emb.squeeze(1))
        fusion_embs = torch.stack(fusion_embs,dim=1).permute(0,1,3,2) #(B,self.max_num_speaker,F,T'')
        return fusion_embs




    def fusion_att_multi_head(self,mix_frame_emb, target_speaker_frame_emb, num_heads: int=4):
        """
        args:
            mix_frame_emb: shape(B,F,T')
            target_speaker_frame_emb: shape(B*num_speakers,F,T')
        return:
            fusion_emb : shape(B,F,T')


        """
        query = mix_frame_emb

        query = query.permute(0,2,1) # (B,T',F)
        dim = query.size(2)//num_heads
        B = query.size(0)
        T = query.size(1)
        query = query.reshape(B,num_heads,T,dim) # (B,num_heads,T',F//num_heads)


        _,F1,T1 = target_speaker_frame_emb.shape
        speaker_frame_emb= target_speaker_frame_emb.view(-1,self.max_num_speaker,F1,T1)

        speaker_frame_emb = speaker_frame_emb.permute(0,1,3,2)
        B,_,T1,F1 = speaker_frame_emb.shape
        speaker_frame_emb = speaker_frame_emb.reshape(B,-1,F1) #(B,num_speakers*T1, F1)
        time = speaker_frame_emb.size(1)
        speaker_frame_emb = speaker_frame_emb.reshape(B, num_heads, time, F1//num_heads)



        key = speaker_frame_emb
        #key = key.unsqueeze(1) # (B,1, num_speakers*T1, F1)
        value = speaker_frame_emb
        #value = value.unsqueeze(1) # (B,1, num_speakers*T1, F1)
        #print(f"query shape: {query.shape}, key shape: {key.shape}, value shape: {value.shape}")
        #with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
        fusion_emb = F.scaled_dot_product_attention(query,key,value, dropout_p=(self.dropout if self.training else 0.0)) # (B,num_heads,,T',F//num_heads)
        fusion_emb = fusion_emb.reshape(B,T,-1)
        fusion_emb = fusion_emb.transpose(1, 2)  # (B,T',F)->(B,F,T')
        return  fusion_emb


    def fusion_att_linear(self,mix_frame_emb, target_speaker_frame_emb):
        """
        args:
            mix_frame_emb: shape(B,F,T')
            target_speaker_frame_emb: shape(B*num_speakers,F,T')
        return:
            fusion_emb : shape(B,F,T')


        """
        query = mix_frame_emb

        query = query.permute(0,2,1).unsqueeze(1) #(B,1,T',F)
        query = self.linear_q(query)


        _,F1,T1 = target_speaker_frame_emb.shape
        speaker_frame_emb= target_speaker_frame_emb.view(-1,self.max_num_speaker,F1,T1)

        speaker_frame_emb = speaker_frame_emb.permute(0,1,3,2)
        B,_,T1,F1 = speaker_frame_emb.shape
        speaker_frame_emb = speaker_frame_emb.reshape(B,-1,F1) #(B,num_speakers*T1, F1)

        key = speaker_frame_emb
        key = key.unsqueeze(1) # (B,1, num_speakers*T1, F1)
        key = self.linear_k(key)
        value = speaker_frame_emb
        value = value.unsqueeze(1) # (B,1, num_speakers*T1, F1)
        value = self.linear_v(value)
        #print(f"query shape: {query.shape}, key shape: {key.shape}, value shape: {value.shape}")
        #with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
        fusion_emb = F.scaled_dot_product_attention(query,key,value, dropout_p=(self.dropout if self.training else 0.0)) # (B,1,T',F)

        fusion_emb = fusion_emb.squeeze(1).transpose(1, 2)  # (B,T',F)->(B,F,T')
        return  fusion_emb


    def forward_common3(self, ref_speech: torch.Tensor, target_speech: torch.Tensor, labels: torch.Tensor,fix_encoder: bool = True,num_updates: int = 0):
        B = ref_speech.size(0)
        #T = ref_speech.size(1)
        D = ref_speech.size(2)
        max_len = labels.size(-1)
        fix_encoder = num_updates < self.freeze_speech_encoder_updates
        if self.speech_encoder_type == "CAM++":
            # its input should be Fbank(80-dim), shape(B,T,80)
            with torch.no_grad() if fix_encoder else contextlib.ExitStack():
                before_pool_x = self.speech_encoder(ref_speech, get_time_out=True)
        mix_emb = before_pool_x #(B,F,T')
        if  self.speaker_encoder_type == "CAM++_per":
            # its input should be Fbank(80-dim),shape(B,num_speakers,T_,80)
            B,_,T_,D = target_speech.shape
            ts_frames = []
            for i in range(self.max_num_speaker):
                ts_input = target_speech[:,i,:,:] # (B,T_,80)
                with torch.no_grad():
                    before_pool_target_speaker = self.speaker_encoder(ts_input, get_time_out=True)
                    ts_frames.append(before_pool_target_speaker)
            speaker_frame_emb = torch.stack(ts_frames,dim=1) # (B, num_speakers,F,T2)
        if self.fusion_type == "fusion_att_per_speaker_wo_utt_emb":
             fusion_emb = self.fusion_single_head_att_per_speaker(mix_emb, speaker_frame_emb) # #(B,self.max_num_speaker,F, T'')

        fusion_time = fusion_emb.size(3)
        fusion_emb = fusion_emb.reshape(B,-1,fusion_time) # (B,self.max_num_speaker*F, T'')
        fusion_emb = self.fusion_linear(fusion_emb.permute(0,2,1)) # (B,T'',F)


        # downsample mix embed
        x = self.speech_down_or_up(fusion_emb.permute(0,2,1)) # #(B, F1,T1)

         # ## process gap of mix audio frame len and label len
        gap = x.size(-1) - max_len
        assert (
            abs(x.size(-1) - max_len) <= 3
            ), f"label and ref_speech(mix speech) diff: {x.size(-1)-max_len}, label len: {max_len}, ref_speech len: {x.size(-1)}"
        # padding audio len to label len
        if gap == -1:
            x = nn.functional.pad(x, (0, 1))
        elif gap == -2:
            x = nn.functional.pad(x, (0, 2))
        elif gap == -3:
            x = nn.functional.pad(x, (0, 3))
        # cut audio len to label len
        x = x[:, :, :max_len]  # (B,F1,T1),
        mix_embeds = x.transpose(1, 2)  # (B,T1,F1)


        # prepared utt-level target speaker
        #print(f"target_speech: {target_speech.shape}")
        if self.speaker_encoder_type == "CAM++_per":
            xs_t = []
            for i in range(self.max_num_speaker):
                x_tp = target_speech[:,i,:,:]
                x_out = self.speaker_encoder.forward(x_tp) #(B,T_,80) ->(B,192)
                xs_t.append(x_out)
            target_utt_embs = torch.stack(xs_t,dim=1) # B, 4, 192
        #target_speech = x_t.view(B, self.max_num_speaker, -1)  # B, 4, 192
        else:
            x_t = self.speaker_encoder.forward(target_speech) # (B*num_speaker,192)
            target_utt_embs = x_t.view(B, self.max_num_speaker, -1)  # B, 4, 192

        # the below codes, D=F'', T=T''

        ts_embeds = self.rs_dropout(target_utt_embs)  # B, 4, D
        #print(f"ts_embeds shape: {ts_embeds.shape}")

        # combine target embedding and mix embedding
        # Extend ts_embeds for time alignemnt
        ts_embeds = ts_embeds.unsqueeze(2)  # B, 4, 1, D
        # repeat T to cat mix frame-level information
        ts_embeds = ts_embeds.repeat(1, 1, mix_embeds.shape[1], 1)  # B, 4, T, D
        #B, _, T, _ = ts_embeds.shape

        #print(f"mix_embeds shape: {mix_embeds.shape}")
        # Transformer for single speaker
        # assume self.max_num_speaker==4,
        cat_embeds = []
        for i in range(self.max_num_speaker):
            ts_embed = ts_embeds[:, i, :, :]  # B,T,D
            cat_embed = torch.cat((ts_embed, mix_embeds), 2)  # B,T,2D
            if self.proj_layer is not None:
                cat_embed = self.proj_layer(cat_embed)
            #print(f"cat_embed: shape: {cat_embed.shape}")  # (B,T,384)
            cat_embed = cat_embed.transpose(0, 1)  # T, B, 2D
            if self.single_backend_type == "conformer":
                cat_embed = cat_embed.transpose(0, 1)  # B,T, 2D
                lengths = torch.tensor(
                    [cat_embed.size(1) for i in cat_embed],
                    dtype=torch.int32,
                    device=cat_embed.device,
                )
                cat_embed = cat_embed.transpose(0, 1)  # T,B 2D
                cat_embed = self.pos_encoder(cat_embed)
                cat_embed = cat_embed.transpose(0, 1)  # B,T, F
                # print(f"before single_backend, cat_embed shape: {cat_embed.shape}")
                cat_embed, _ = self.single_backend(cat_embed, lengths)  # B,T, F
                # print(f"after single_backend, cat_embed shape: {cat_embed.shape}")
            else:
                cat_embed = self.pos_encoder(cat_embed)
                cat_embed = self.single_backend(cat_embed)  # T, B, F
                cat_embed = cat_embed.transpose(0, 1)  # B, T, F

            cat_embeds.append(cat_embed)
        cat_embeds = torch.stack(cat_embeds)  # 4, B, T, F
        cat_embeds = torch.permute(cat_embeds, (1, 0, 3, 2))  # B,4,F,T
        cat_embeds_time =  cat_embeds.size(-1)
        # Combine the outputs
        cat_embeds = cat_embeds.reshape(B, -1, cat_embeds_time)  # B, 4 * F, T

        # cat multi forward
        #B, _, T = cat_embeds.size()
        # Downsampling
        cat_embeds = self.backend_down(cat_embeds)  # B, F, T'
        # Transformer for multiple speakers
        cat_embeds = self.pos_encoder(torch.permute(cat_embeds, (2, 0, 1)))
        if self.multi_backend_type == "conformer":
            cat_embeds = cat_embeds.transpose(0, 1)  # B, T', F
            lengths = torch.tensor(
                [cat_embeds.size(1) for i in cat_embeds],
                dtype=torch.int32,
                device=cat_embeds.device,
            )
            # print(f"before multi_backend, cat_embeds shape: {cat_embeds.shape},lengths shape: {lengths.shape}")
            cat_embeds, _ = self.multi_backend(cat_embeds, lengths)  # B, T', F
            # print(f"after multi_backend, cat_embeds shape: {cat_embeds.shape}")
        else:
            cat_embeds = self.multi_backend(cat_embeds)  # T', B, F
            cat_embeds = cat_embeds.transpose(0, 1)  # B,T',F

        if self.multi_backend_type == "mamba2":
            cat_embeds = self.multi_backend_proj(cat_embeds)
        outs = self.fc(cat_embeds)  # B T' 4
        outs = outs.transpose(1, 2)  # B,4,T'
        return outs


    def forward_common2(self, ref_speech: torch.Tensor, target_speech: torch.Tensor, labels: torch.Tensor,fix_encoder: bool = True,num_updates: int = 0):
        B = ref_speech.size(0)
        #T = ref_speech.size(1)
        D = ref_speech.size(2)
        max_len = labels.size(-1)
        fix_encoder = num_updates < self.freeze_speech_encoder_updates
        if self.speech_encoder_type == "CAM++":
            # its input should be Fbank(80-dim), shape(B,T,80)
            with torch.no_grad() if fix_encoder else contextlib.ExitStack():
                before_pool_x = self.speech_encoder(ref_speech, get_time_out=True)
        mix_emb = before_pool_x #(B,F,T')
        if  self.speaker_encoder_type == "CAM++_per":
            # its input should be Fbank(80-dim),shape(B,num_speakers,T_,80)
            B,_,T_,D = target_speech.shape
            ts_frames = []
            for i in range(self.max_num_speaker):
                ts_input = target_speech[:,i,:,:] # (B,T_,80)
                with torch.no_grad():
                    before_pool_target_speaker = self.speaker_encoder(ts_input, get_time_out=True)
                    ts_frames.append(before_pool_target_speaker)
            speaker_frame_emb = torch.stack(ts_frames,dim=1) # (B, num_speakers,F,T2)
        if self.fusion_type == "fusion_att_per_speaker_wo_utt_emb":
             fusion_emb = self.fusion_single_head_att_per_speaker(mix_emb, speaker_frame_emb) # #(B,self.max_num_speaker,F, T'')


        # downsample mix embed
        x = self.speech_down_or_up(mix_emb)#(B, F1,T1)
        fusion_dim = fusion_emb.size(2)
        fusion_time = fusion_emb.size(3)
        ts_embeds = self.speech_down_or_up(fusion_emb.reshape(-1,fusion_dim,fusion_time))#(B*self.max_num_speaker, F1,T1)
        fusion_down_dim = ts_embeds.size(1)
        fusion_down_time = ts_embeds.size(2)
        ts_embeds = ts_embeds.reshape(B,-1,fusion_down_time,fusion_down_dim)
        #print(f"ts_embeds shape: {ts_embeds.shape}")
        # ## process gap of mix audio frame len and label len
        gap = x.size(-1) - max_len
        assert (
            abs(x.size(-1) - max_len) <= 3
            ), f"label and ref_speech(mix speech) diff: {x.size(-1)-max_len}, label len: {max_len}, ref_speech len: {x.size(-1)}"
        # padding audio len to label len
        if gap == -1:
            x = nn.functional.pad(x, (0, 1))
        elif gap == -2:
            x = nn.functional.pad(x, (0, 2))
        elif gap == -3:
            x = nn.functional.pad(x, (0, 3))
        # cut audio len to label len
        x = x[:, :, :max_len]  # (B,F1,T1),
        mix_embeds = x.transpose(1, 2)  # (B,T1,F1)
        #print(f"mix_embeds shape: {mix_embeds.shape}")
        # Transformer for single speaker
        # assume self.max_num_speaker==4,
        cat_embeds = []
        for i in range(self.max_num_speaker):
            ts_embed = ts_embeds[:, i, :, :]  # B,T,D
            cat_embed = torch.cat((ts_embed, mix_embeds), 2)  # B,T,2D
            if self.proj_layer is not None:
                cat_embed = self.proj_layer(cat_embed)
            #print(f"cat_embed: shape: {cat_embed.shape}")  # (B,T,384)
            cat_embed = cat_embed.transpose(0, 1)  # T, B, 2D
            if self.single_backend_type == "conformer":
                cat_embed = cat_embed.transpose(0, 1)  # B,T, 2D
                lengths = torch.tensor(
                    [cat_embed.size(1) for i in cat_embed],
                    dtype=torch.int32,
                    device=cat_embed.device,
                )
                cat_embed = cat_embed.transpose(0, 1)  # T,B 2D
                cat_embed = self.pos_encoder(cat_embed)
                cat_embed = cat_embed.transpose(0, 1)  # B,T, F
                # print(f"before single_backend, cat_embed shape: {cat_embed.shape}")
                cat_embed, _ = self.single_backend(cat_embed, lengths)  # B,T, F
                # print(f"after single_backend, cat_embed shape: {cat_embed.shape}")
            else:
                cat_embed = self.pos_encoder(cat_embed)
                cat_embed = self.single_backend(cat_embed)  # T, B, F
                cat_embed = cat_embed.transpose(0, 1)  # B, T, F

            cat_embeds.append(cat_embed)
        cat_embeds = torch.stack(cat_embeds)  # 4, B, T, F
        cat_embeds = torch.permute(cat_embeds, (1, 0, 3, 2))  # B,4,F,T
        cat_embeds_time =  cat_embeds.size(-1)
        # Combine the outputs
        cat_embeds = cat_embeds.reshape(B, -1, cat_embeds_time)  # B, 4 * F, T

        # cat multi forward
        #B, _, T = cat_embeds.size()
        # Downsampling
        cat_embeds = self.backend_down(cat_embeds)  # B, F, T'
        # Transformer for multiple speakers
        cat_embeds = self.pos_encoder(torch.permute(cat_embeds, (2, 0, 1)))
        if self.multi_backend_type == "conformer":
            cat_embeds = cat_embeds.transpose(0, 1)  # B, T', F
            lengths = torch.tensor(
                [cat_embeds.size(1) for i in cat_embeds],
                dtype=torch.int32,
                device=cat_embeds.device,
            )
            # print(f"before multi_backend, cat_embeds shape: {cat_embeds.shape},lengths shape: {lengths.shape}")
            cat_embeds, _ = self.multi_backend(cat_embeds, lengths)  # B, T', F
            # print(f"after multi_backend, cat_embeds shape: {cat_embeds.shape}")
        else:
            cat_embeds = self.multi_backend(cat_embeds)  # T', B, F
            cat_embeds = cat_embeds.transpose(0, 1)  # B,T',F

        if self.multi_backend_type == "mamba2":
            cat_embeds = self.multi_backend_proj(cat_embeds)
        outs = self.fc(cat_embeds)  # B T' 4
        outs = outs.transpose(1, 2)  # B,4,T'
        return outs


    def forward_common(self, ref_speech: torch.Tensor, target_speech: torch.Tensor, labels: torch.Tensor,fix_encoder: bool = True,num_updates: int = 0):
        B = ref_speech.size(0)
        T = ref_speech.size(1)
        D = ref_speech.size(2)
        max_len = labels.size(-1)
        fix_encoder = num_updates < self.freeze_speech_encoder_updates
        if self.speech_encoder_type == "CAM++":
            # its input should be Fbank(80-dim), shape(B,T,80)
            with torch.no_grad() if fix_encoder else contextlib.ExitStack():
                before_pool_x = self.speech_encoder(ref_speech, get_time_out=True)
        mix_emb = before_pool_x #(B,F,T')

        if self.speaker_encoder_type == "CAM++":
            # its input should be Fbank(80-dim),shape(B,num_speakers,T_,80)
            B,_,T_,D = target_speech.shape
            target_speech = target_speech.reshape(-1,T_,D)
            with torch.no_grad():
                before_pool_target_speaker = self.speaker_encoder(target_speech, get_time_out=True)
            speaker_frame_emb =  before_pool_target_speaker #(B* num_speakers,F,T')

        elif self.speaker_encoder_type == "CAM++_per":
            # its input should be Fbank(80-dim),shape(B,num_speakers,T_,80)
            B,_,T_,D = target_speech.shape
            ts_frames = []
            for i in range(self.max_num_speaker):
                ts_input = target_speech[:,i,:,:] # (B,T_,80)
                with torch.no_grad():
                    before_pool_target_speaker = self.speaker_encoder(ts_input, get_time_out=True) #(B,F, T_')
                    ts_frames.append(before_pool_target_speaker)
            speaker_frame_emb = torch.stack(ts_frames,dim=1) #(B, num_speakers,F, T_')
            speaker_dim = speaker_frame_emb.size(2)
            speaker_time= speaker_frame_emb.size(3)
            speaker_frame_emb = speaker_frame_emb.reshape(-1,speaker_dim,speaker_time) #  (B* num_speakers,F,T')

        # context function
        if self.fusion_type == "att_wo_linear":
            fusion_emb = self.fusion_att(mix_emb, speaker_frame_emb) # (B,F,T')
        elif self.fusion_type == "att_w_linear":
            fusion_emb = self.fusion_att_linear(mix_emb, speaker_frame_emb) # (B,F,T')
        elif self.fusion_type == "att_wo_linear_multi_head":
            fusion_emb = self.fusion_att_multi_head(mix_emb, speaker_frame_emb)

        # downsample or upsample
        x = self.speech_down_or_up(fusion_emb)#(B, F'',T'')

        # ## process gap of mix audio frame len and label len
        gap = x.size(-1) - max_len
        assert (
            abs(x.size(-1) - max_len) <= 3
            ), f"label and ref_speech(mix speech) diff: {x.size(-1)-max_len}, label len: {max_len}, ref_speech len: {x.size(-1)}"
        # padding audio len to label len
        if gap == -1:
            x = nn.functional.pad(x, (0, 1))
        elif gap == -2:
            x = nn.functional.pad(x, (0, 2))
        elif gap == -3:
            x = nn.functional.pad(x, (0, 3))
        # cut audio len to label len
        x = x[:, :, :max_len]  # (B,F'',T''),
        mix_embeds = x.transpose(1, 2)  # (B,T'',F'')


        #print(f"target_speech: {target_speech.shape}")
        if self.speaker_encoder_type == "CAM++_per":
            xs_t = []
            for i in range(self.max_num_speaker):
                x_tp = target_speech[:,i,:,:]
                x_out = self.speaker_encoder.forward(x_tp) #(B,T_,80) ->(B,192)
                xs_t.append(x_out)
            target_speech = torch.stack(xs_t,dim=1) # B, 4, 192
        #target_speech = x_t.view(B, self.max_num_speaker, -1)  # B, 4, 192
        else:
            x_t = self.speaker_encoder.forward(target_speech) # (B*num_speaker,192)
            target_speech = x_t.view(B, self.max_num_speaker, -1)  # B, 4, 192

        # the below codes, D=F'', T=T''

        ts_embeds = self.rs_dropout(target_speech)  # B, 4, D
        #print(f"ts_embeds shape: {ts_embeds.shape}")

        # combine target embedding and mix embedding
        # Extend ts_embeds for time alignemnt
        ts_embeds = ts_embeds.unsqueeze(2)  # B, 4, 1, D
        # repeat T to cat mix frame-level information
        ts_embeds = ts_embeds.repeat(1, 1, mix_embeds.shape[1], 1)  # B, 4, T, D
        B, _, T, _ = ts_embeds.shape
        #print(f"ts_embeds shape: {ts_embeds.shape} split")

        # Transformer for single speaker
        # assume self.max_num_speaker==4,
        cat_embeds = []
        for i in range(self.max_num_speaker):
            ts_embed = ts_embeds[:, i, :, :]  # B,T,D
            cat_embed = torch.cat((ts_embed, mix_embeds), 2)  # B,T,2D
            if self.proj_layer is not None:
                cat_embed = self.proj_layer(cat_embed)
            logging.debug(f"cat_embed: shape: {cat_embed.shape}")  # (B,T,384)
            cat_embed = cat_embed.transpose(0, 1)  # T, B, 2D
            if self.single_backend_type == "conformer":
                cat_embed = cat_embed.transpose(0, 1)  # B,T, 2D
                lengths = torch.tensor(
                    [cat_embed.size(1) for i in cat_embed],
                    dtype=torch.int32,
                    device=cat_embed.device,
                )
                cat_embed = cat_embed.transpose(0, 1)  # T,B 2D
                cat_embed = self.pos_encoder(cat_embed)
                cat_embed = cat_embed.transpose(0, 1)  # B,T, F
                # print(f"before single_backend, cat_embed shape: {cat_embed.shape}")
                cat_embed, _ = self.single_backend(cat_embed, lengths)  # B,T, F
                # print(f"after single_backend, cat_embed shape: {cat_embed.shape}")
            else:
                cat_embed = self.pos_encoder(cat_embed)
                cat_embed = self.single_backend(cat_embed)  # T, B, F
                cat_embed = cat_embed.transpose(0, 1)  # B, T, F

            cat_embeds.append(cat_embed)
        cat_embeds = torch.stack(cat_embeds)  # 4, B, T, F
        cat_embeds = torch.permute(cat_embeds, (1, 0, 3, 2))  # B,4,F,T
        # Combine the outputs
        cat_embeds = cat_embeds.reshape(B, -1, T)  # B, 4 * F, T

        # cat multi forward
        B, _, T = cat_embeds.size()
        # Downsampling
        cat_embeds = self.backend_down(cat_embeds)  # B, F, T'
        # Transformer for multiple speakers
        cat_embeds = self.pos_encoder(torch.permute(cat_embeds, (2, 0, 1)))
        if self.multi_backend_type == "conformer":
            cat_embeds = cat_embeds.transpose(0, 1)  # B, T', F
            lengths = torch.tensor(
                [cat_embeds.size(1) for i in cat_embeds],
                dtype=torch.int32,
                device=cat_embeds.device,
            )
            # print(f"before multi_backend, cat_embeds shape: {cat_embeds.shape},lengths shape: {lengths.shape}")
            cat_embeds, _ = self.multi_backend(cat_embeds, lengths)  # B, T', F
            # print(f"after multi_backend, cat_embeds shape: {cat_embeds.shape}")
        else:
            cat_embeds = self.multi_backend(cat_embeds)  # T', B, F
            cat_embeds = cat_embeds.transpose(0, 1)  # B,T',F

        if self.multi_backend_type == "mamba2":
            cat_embeds = self.multi_backend_proj(cat_embeds)
        outs = self.fc(cat_embeds)  # B T' 4
        outs = outs.transpose(1, 2)  # B,4,T'
        return outs

    def forward(
        self,
        ref_speech: torch.Tensor,
        target_speech: torch.Tensor,
        labels: torch.Tensor,
        num_updates: int,
    ):

        if  self.without_speaker_utt_embed:
            self.fusion_case=="fusion_embed_as_target_embed_and_wo_utt_embed"
            outs = self.forward_common2(
                ref_speech=ref_speech,
                target_speech=target_speech,
                labels=labels,
                num_updates=num_updates,
            )
        elif not self.without_speaker_utt_embed:

            if  self.fusion_case=="fusion_embed_as_mix_embed_w_utt_embed":
                outs = self.forward_common3(
                    ref_speech=ref_speech,
                    target_speech=target_speech,
                    labels=labels,
                    num_updates=num_updates,
                )
            else:
                outs = self.forward_common(
                    ref_speech=ref_speech,
                    target_speech=target_speech,
                    labels=labels,
                    num_updates=num_updates,
                )
        return outs

    def infer(
        self,
        ref_speech: torch.Tensor,
        target_speech: torch.Tensor,
        labels: torch.Tensor,
        labels_len: torch.Tensor,
        num_updates: int = 0,
        file_path=None,
        speaker_ids=None,
        start=None,
        # inference:bool=True,
    ):
        outs = self.forward(
            ref_speech=ref_speech,
            target_speech=target_speech,
            labels=labels,
            num_updates=num_updates,
        )
        # print(f"outs shape: {outs.shape}")
        loss = self.calculate_loss(outs, labels, labels_len)
        result = {"losses": {"diar": loss}}

        activate = nn.Sigmoid()
        outs_prob = activate(outs)
        outs_prob = outs_prob.data.cpu().numpy()
        mi, fa, cf, acc, der = self.calc_diarization_result(
            outs_prob.transpose((0, 2, 1)), labels.transpose(1, 2), labels_len
        )
        result["labels_len"] = labels_len
        result["DER"] = der
        result["ACC"] = acc
        result["MI"] = mi
        result["FA"] = fa
        result["CF"] = cf

        res_dict = defaultdict(lambda: defaultdict(list))
        B, _, _ = outs.shape
        for b in range(B):
            for t in range(labels_len[b]):
                n = max(speaker_ids[b])
                for i in range(n):
                    id = speaker_ids[b][i]
                    name = file_path[b]
                    out = outs_prob[b, i, t]
                    t0 = start[b]
                    res_dict[str(name) + "-" + str(id)][t0 + t].append(out)

        return result, res_dict

    # @staticmethod
    def calc_diarization_error(self, pred, label, length):
        # Note (jiatong): Credit to https://github.com/hitachi-speech/EEND

        (batch_size, max_len, num_output) = label.size()
        # mask the padding part
        mask = np.zeros((batch_size, max_len, num_output))
        for i in range(batch_size):
            mask[i, : length[i], :] = 1

        # pred and label have the shape (batch_size, max_len, num_output)
        label_np = label.data.cpu().numpy().astype(int)
        # pred_np = (pred.data.cpu().numpy() > 0).astype(int)
        pred_np = (pred > 0.5).astype(int)
        label_np = label_np * mask
        pred_np = pred_np * mask
        length = length.data.cpu().numpy()

        # compute speech activity detection error
        n_ref = np.sum(label_np, axis=2)
        n_sys = np.sum(pred_np, axis=2)
        speech_scored = float(np.sum(n_ref > 0))
        speech_miss = float(np.sum(np.logical_and(n_ref > 0, n_sys == 0)))
        speech_falarm = float(np.sum(np.logical_and(n_ref == 0, n_sys > 0)))

        # compute speaker diarization error
        speaker_scored = float(np.sum(n_ref))
        speaker_miss = float(np.sum(np.maximum(n_ref - n_sys, 0)))
        speaker_falarm = float(np.sum(np.maximum(n_sys - n_ref, 0)))
        n_map = np.sum(np.logical_and(label_np == 1, pred_np == 1), axis=2)
        speaker_error = float(np.sum(np.minimum(n_ref, n_sys) - n_map))
        correct = float(1.0 * np.sum((label_np == pred_np) * mask) / num_output)
        num_frames = np.sum(length)
        return (
            correct,
            num_frames,
            speech_scored,
            speech_miss,
            speech_falarm,
            speaker_scored,
            speaker_miss,
            speaker_falarm,
            speaker_error,
        )

    # @staticmethod
    def calc_diarization_result(self, outs_prob, labels, labels_len):
        # DER
        (
            correct,
            num_frames,
            speech_scored,
            speech_miss,
            speech_falarm,
            speaker_scored,
            speaker_miss,
            speaker_falarm,
            speaker_error,
        ) = self.calc_diarization_error(outs_prob, labels, labels_len)

        if speech_scored == 0 or speaker_scored == 0:
            logger.warn("All labels are zero")
            return 0, 0, 0, 0, 0
        if speech_scored == 0 or speaker_scored == 0:
            logger.warn("All labels are zero")
            return 0, 0, 0, 0, 0

        _, _, mi, fa, cf, acc, der = (
            speech_miss / speech_scored,
            speech_falarm / speech_scored,
            speaker_miss / speaker_scored,
            speaker_falarm / speaker_scored,
            speaker_error / speaker_scored,
            correct / num_frames,
            (speaker_miss + speaker_falarm + speaker_error) / speaker_scored,
        )
        return mi, fa, cf, acc, der

    def calculate_loss(self, outs, labels, labels_len):
        total_loss = 0
        loss_fn = torch.nn.BCEWithLogitsLoss()
        for i in range(labels_len.size(0)):

            total_loss += loss_fn(
                outs[i, :, : labels_len[i]], labels[i, :, : labels_len[i]]
            )
        return total_loss / labels_len.size(0)

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
                logger.warn("%s is not in the model." % origname)
                continue
            if selfState[name].size() != loadedState[origname].size():
                sys.stderr.write(
                    "Wrong parameter length: %s, model: %s, loaded: %s"
                    % (origname, selfState[name].size(), loadedState[origname].size())
                )
                continue
            selfState[name].copy_(param)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        Activates gradient checkpointing for the current model.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".

        We pass the `__call__` method of the modules instead of `forward` because `__call__` attaches all the hooks of
        the module. https://discuss.pytorch.org/t/any-different-between-model-input-and-model-forward-input/3690/2

        Args:
            gradient_checkpointing_kwargs (dict, *optional*):
                Additional keyword arguments passed along to the `torch.utils.checkpoint.checkpoint` function.
        """
        # if not self.supports_gradient_checkpointing:
        #    raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")

        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {"use_reentrant": True}
        import functools

        gradient_checkpointing_func = functools.partial(
            checkpoint, **gradient_checkpointing_kwargs
        )

        # For old GC format (transformers < 4.35.0) for models that live on the Hub
        # we will fall back to the overwritten `_set_gradient_checkpointing` method
        # _is_using_old_format = "value" in inspect.signature(self._set_gradient_checkpointing).parameters

        # if not _is_using_old_format:
        self._set_gradient_checkpointing(
            enable=True, gradient_checkpointing_func=gradient_checkpointing_func
        )

    def _set_gradient_checkpointing(
        self, enable: bool = True, gradient_checkpointing_func: Callable = checkpoint
    ):
        is_gradient_checkpointing_set = False

        # Apply it on the top-level module in case the top-level modules supports it
        # for example, LongT5Stack inherits from `PreTrainedModel`.
        if hasattr(self, "gradient_checkpointing"):
            self._gradient_checkpointing_func = gradient_checkpointing_func
            self.gradient_checkpointing = enable
            is_gradient_checkpointing_set = True

        for module in self.modules():
            if hasattr(module, "gradient_checkpointing"):
                module._gradient_checkpointing_func = gradient_checkpointing_func
                module.gradient_checkpointing = enable
                is_gradient_checkpointing_set = True

        if not is_gradient_checkpointing_set:
            raise ValueError(
                f"{self.__class__.__name__} is not compatible with gradient checkpointing. Make sure all the architecture support it by setting a boolean attribute"
                " `gradient_checkpointing` to modules of the model that uses checkpointing."
            )


if __name__ == "__main__":
    """Build a new model instance."""
    model_cfg = TSVADConfig()
    model_cfg.single_backend_type = "mamba2"
    model_cfg.d_state = 256
    model = TSVADModel(cfg=model_cfg, task_cfg=data_cfg, device=torch.device("cpu"))
    print(model)
    x = torch.zeros(10, 398, 80)  # B,T,F
    out = model.speech_encoder(x)
    print(f"out shape: {out.shape}")
    num_params = sum(param.numel() for param in model.parameters())
    for name, v in model.named_parameters():
        print(f"name: {name}, v: {v.shape}")
    print("{} M".format(num_params / 1e6))  # 6.61M
    data_cfg = TSVADDataConfig()
    print(vars(data_cfg))
