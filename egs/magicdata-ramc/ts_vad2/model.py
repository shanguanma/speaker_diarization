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


model_cfg = TSVADConfig()


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
        # assert (
        #    self.label_rate == cfg.label_rate
        # ), f"self.label_rate is {elf.label_rate} not support!"
        self.label_rate == cfg.label_rate
        self.speech_encoder_type = cfg.speech_encoder_type
        self.speech_encoder_path = cfg.speech_encoder_path
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

        # self.fc = nn.Linear(cfg.transformer_embed_dim, self.max_num_speaker)

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
            #no downsample, stride=1, label_rate is 13, means that 1s audio has 13 frames
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
    def forward_common(
        self,
        ref_speech: torch.Tensor,
        target_speech: torch.Tensor,
        labels: torch.Tensor,
        # labels_len: torch.Tensor,
        fix_encoder: bool = True,
        num_updates: int = 0,
    ):
        B = ref_speech.size(0)
        T = ref_speech.size(1)
        # print(f"ref_speech shape: {ref_speech.shape}")
        max_len = labels.size(-1)
        fix_encoder = num_updates < self.freeze_speech_encoder_updates
        if self.speech_encoder_type == "WavLM":
            # it should be wavform input, shape (B,T), fbank_input of ts_vad_dataset.py should be set False
            with torch.no_grad() if fix_encoder else contextlib.ExitStack():
                x = self.speech_encoder.extract_features(ref_speech)[0]
            x = x.view(
                B, -1, self.pretrain_speech_encoder_dim
            )  # B, 50 * T, self.pretrain_speech_encoder_dim
            x = x.transpose(1, 2)  # (B,D,T)
            x = self.speech_down_or_up(x)
        elif self.speech_encoder_type == "WavLM_weight_sum":
            # it should be wavform input, shape (B,T), fbank_input of ts_vad_dataset.py should be set False
            with torch.no_grad() if fix_encoder else contextlib.ExitStack():
                # print(x[0][0].shape)#(B,T,D), it is output of last layer of transformer
                # print(len(x[0][1])) # when output_layer=12,ret_layer_results=True, it is 13,
                # Because the first element of ret_layer_results is the result of pos_conv,
                # and the second element is the output of the first layer of transformer,
                # Each element here is actually a tuple. The first element of the tuple is hidden state, and the second element is attention_state.
                x = self.speech_encoder.extract_features(
                    ref_speech,
                    output_layer=self.wavlm_encoder_num_layer,
                    ret_layer_results=True,
                )
            hss = [hs for hs, att in x[0][1]]
            # print(f"hss len: {len(hss)}")
            stacked_hs = torch.stack(hss[1:], dim=-1)  # (T,B,D,12)

            if self.wavlm_fuse_feat_post_norm:
                x = self.weights(stacked_hs)  # (T,B,D,1)
                x = torch.squeeze(x, -1)  # (T,B,D)
                x = x.permute(1, 0, 2)  # (B,T,D)
                x = self.wavlmproj(x)  # (B,T,D)
                x = self.wavlmlnorm(x)  # (B,T,D)
                x = x.permute(0, 2, 1)  # (B,D,T)
                x = self.speech_down_or_up(x)
            else:
                _, T, B, D = stacked_hs.shape
                # print(f"stacked_hs.shape: {stacked_hs.shape}")
                stacked_hs = stacked_hs.view(stacked_hs.shape[-1], -1)  # (12,T*B*D)
                # print(f"stacked_hs shap: {stacked_hs.shape}")
                norm_weights = F.softmax(self.weights, dim=-1)
                # print(f"norm_weights.unsqueeze(-1) shape: {norm_weights.unsqueeze(-1).shape}")
                weighted_hs = (norm_weights.unsqueeze(-1) * stacked_hs).sum(dim=0)
                weighted_hs = weighted_hs.view(B, T, D)
                x = weighted_hs.permute(0, 2, 1)  # (B,D,T)
                x = self.speech_down_or_up(x)
        elif self.speech_encoder_type == "whisper":
            # it should be wavform input, shape (B,T), fbank_input of ts_vad_dataset.py should be set False
            # whisper output frame rate 50,so I need to downsample to 25.
            # whisper encoder output  shape (B,T,D)
            with torch.no_grad() if fix_encoder else contextlib.ExitStack():
                x = self.speech_encoder(ref_speech)
            x = x.transpose(1, 2)  # (B,T,D)-> (B,D,T)
            x = self.speech_down_or_up(x)  # (B,D,T/2)

        elif self.speech_encoder_type == "w2v-bert2":
            # its input fbank feature(80-dim)
            with torch.no_grad() if fix_encoder else contextlib.ExitStack():
                batch_size, num_frames, num_channels = ref_speech.shape
                ref_speech = torch.reshape(
                    ref_speech, (batch_size, num_frames // 2, num_channels * 2)
                )  # (batch_size, T, 160)
                x = self.speech_encoder(ref_speech, output_hidden_states=True)  #
                x = x.hidden_states[
                    -1
                ]  # it is equal to x = x['hidden_states'][self.select_encoder_layer_nums]
            x = x.transpose(1, 2)  # (B,T,D)->(B,D,T)
            x = self.speech_down_or_up(x)
        elif self.speech_encoder_type == "hubert":
            # its input is wavform. we use pretrain mask setting again
            with torch.no_grad() if fix_encoder else contextlib.ExitStack():
                batch_size, raw_sequence_length = ref_speech.shape
                x = self.speech_encoder(ref_speech, output_hidden_states=True)
                x = x.hidden_states[-1]  # (B,T,D)
            x = x.transpose(1, 2)  # (B,T,D)->(B,D,T)
            x = self.speech_down_or_up(x)
        elif (
            self.speech_encoder_type == "ReDimNetB0"
            or self.speech_encoder_type == "ReDimNetB1"
            or self.speech_encoder_type == "ReDimNetB2"
            or self.speech_encoder_type == "ReDimNetB2_layernorm"
            or self.speech_encoder_type == "ReDimNetB3"
            or self.speech_encoder_type == "ReDimNetM"
            or self.speech_encoder_type == "ReDimNetS"
            or self.speech_encoder_type == "ReDimNetB2_offical"
        ):
            with torch.no_grad() if fix_encoder else contextlib.ExitStack():
                # its input melbank feature(72-dim)
                #print(f"input shape of speech_encoder: {ref_speech.shape}")
                x = self.speech_encoder.get_frame_level_feat(ref_speech)  # (B,T,D)
            x = x.transpose(1, 2)  # (B,T,D)->(B,D,T)
            # print(f"output shape of speech_encoder: {x.shape}")
            x = self.speech_down_or_up(x)  # (B,D,T) -> (B,F,T)
            # print(f"output shape of self.speech_down_or_up(x): {x.shape}")
        elif self.speech_encoder_type == "ERes2NetV2_COMMON" or self.speech_encoder_type=="ERes2NetV2_w24s4ep4_COMMON" and self.label_rate==13:
            with torch.no_grad() if fix_encoder else contextlib.ExitStack():
                # its input fbank feature(80-dim)
                x = self.speech_encoder.get_frame_level_feat(ref_speech) # (B,D,T)
             # no downsample, its frame rate is 13.
             x = self.speech_down_or_up(x)  # (B,D,T) -> (B,F,T)
        elif self.speech_encoder_type == "ERes2NetV2_COMMON" or self.speech_encoder_type=="ERes2NetV2_w24s4ep4_COMMON" and self.label_rate==25:
            with torch.no_grad() if fix_encoder else contextlib.ExitStack():
                # its input fbank feature(80-dim)
                x = self.speech_encoder.get_frame_level_feat_frame_rate25(ref_speech) # (B,D,T)
             # no downsample, its frame rate is 13.
             x = self.speech_down_or_up(x)  # (B,D,T) -> (B,F,T)
        else:
            with torch.no_grad() if fix_encoder else contextlib.ExitStack():
                # its input fbank feature(80-dim)
                x = self.speech_encoder(ref_speech, get_time_out=True)
            # print(f"x shape: {x.shape}") # B,F',T'
            x = self.speech_down_or_up(x)
        # print(f"x shape: {x.shape}, max_len: {max_len}")

        ## process gap of audio len and label len
        gap = x.size(-1) - max_len
        assert (
            abs(x.size(-1) - max_len) <= 3
        ), f"label and ref_speech(mix speech) diff: {x.size(-1)-max_len}"
        # padding audio len to label len
        if gap == -1:
            x = nn.functional.pad(x, (0, 1))
        elif gap == -2:
            x = nn.functional.pad(x, (0, 2))
        elif gap == -3:
            x = nn.functional.pad(x, (0, 3))
        # cut audio len to label len
        x = x[:, :, :max_len]  # (B,D,T)
        # print(f"after padding audio len shape: {x.shape}")

        # assert (
        #    x.size(-1) - max_len <= 2 and x.size(-1) - max_len >= -1
        # ), f"label and ref_speech(mix speech) diff: {x.size(-1)-max_len}"
        # if x.size(-1) - max_len == -1:
        #    x = nn.functinal.pad(x, (0, 1))
        # x = x[:, :, :max_len]  # (B,D,T)
        mix_embeds = x.transpose(1, 2)  # (B,T,D)

        # target speech
        ts_embeds = self.rs_dropout(target_speech)  # B, 4, D

        # combine target embedding and mix embedding
        # Extend ts_embeds for time alignemnt
        ts_embeds = ts_embeds.unsqueeze(2)  # B, 4, 1, D
        # repeat T to cat mix frame-level information
        ts_embeds = ts_embeds.repeat(1, 1, mix_embeds.shape[1], 1)  # B, 4, T, D
        B, _, T, _ = ts_embeds.shape

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

    def forward_common_variable_number_speakers(
        self,
        ref_speech: torch.Tensor,
        target_speech: torch.Tensor,
        labels: torch.Tensor,
        # labels_len: torch.Tensor,
        fix_encoder: bool = True,
        num_updates: int = 0,
    ):
        B = ref_speech.size(0)
        T = ref_speech.size(1)
        # print(f"ref_speech shape: {ref_speech.shape}")
        max_len = labels.size(-1)
        fix_encoder = num_updates < self.freeze_speech_encoder_updates
        if self.speech_encoder_type == "WavLM":
            # it should be wavform input, shape (B,T), fbank_input of ts_vad_dataset.py should be set False
            with torch.no_grad() if fix_encoder else contextlib.ExitStack():
                x = self.speech_encoder.extract_features(ref_speech)[0]
            x = x.view(
                B, -1, self.pretrain_speech_encoder_dim
            )  # B, 50 * T, self.pretrain_speech_encoder_dim
            x = x.transpose(1, 2)  # (B,D,T)
            x = self.speech_down_or_up(x)
        elif self.speech_encoder_type == "WavLM_weight_sum":
            # it should be wavform input, shape (B,T), fbank_input of ts_vad_dataset.py should be set False
            with torch.no_grad() if fix_encoder else contextlib.ExitStack():
                # print(x[0][0].shape)#(B,T,D), it is output of last layer of transformer
                # print(len(x[0][1])) # when output_layer=12,ret_layer_results=True, it is 13,
                # Because the first element of ret_layer_results is the result of pos_conv,
                # and the second element is the output of the first layer of transformer,
                # Each element here is actually a tuple. The first element of the tuple is hidden state, and the second element is attention_state.
                x = self.speech_encoder.extract_features(
                    ref_speech,
                    output_layer=self.wavlm_encoder_num_layer,
                    ret_layer_results=True,
                )
            hss = [hs for hs, att in x[0][1]]
            # print(f"hss len: {len(hss)}")
            stacked_hs = torch.stack(hss[1:], dim=-1)  # (T,B,D,12)

            if self.wavlm_fuse_feat_post_norm:
                x = self.weights(stacked_hs)  # (T,B,D,1)
                x = torch.squeeze(x, -1)  # (T,B,D)
                x = x.permute(1, 0, 2)  # (B,T,D)
                x = self.wavlmproj(x)  # (B,T,D)
                x = self.wavlmlnorm(x)  # (B,T,D)
                x = x.permute(0, 2, 1)  # (B,D,T)
                x = self.speech_down_or_up(x)
            else:
                _, T, B, D = stacked_hs.shape
                # print(f"stacked_hs.shape: {stacked_hs.shape}")
                stacked_hs = stacked_hs.view(stacked_hs.shape[-1], -1)  # (12,T*B*D)
                # print(f"stacked_hs shap: {stacked_hs.shape}")
                norm_weights = F.softmax(self.weights, dim=-1)
                # print(f"norm_weights.unsqueeze(-1) shape: {norm_weights.unsqueeze(-1).shape}")
                weighted_hs = (norm_weights.unsqueeze(-1) * stacked_hs).sum(dim=0)
                weighted_hs = weighted_hs.view(B, T, D)
                x = weighted_hs.permute(0, 2, 1)  # (B,D,T)
                x = self.speech_down_or_up(x)
        elif self.speech_encoder_type == "whisper":
            # it should be wavform input, shape (B,T), fbank_input of ts_vad_dataset.py should be set False
            # whisper output frame rate 50,so I need to downsample to 25.
            # whisper encoder output  shape (B,T,D)
            with torch.no_grad() if fix_encoder else contextlib.ExitStack():
                x = self.speech_encoder(ref_speech)
            x = x.transpose(1, 2)  # (B,T,D)-> (B,D,T)
            x = self.speech_down_or_up(x)  # (B,D,T/2)

        elif self.speech_encoder_type == "w2v-bert2":
            # its input fbank feature(80-dim)
            with torch.no_grad() if fix_encoder else contextlib.ExitStack():
                batch_size, num_frames, num_channels = ref_speech.shape
                ref_speech = torch.reshape(
                    ref_speech, (batch_size, num_frames // 2, num_channels * 2)
                )  # (batch_size, T, 160)
                x = self.speech_encoder(ref_speech, output_hidden_states=True)  #
                x = x.hidden_states[
                    -1
                ]  # it is equal to x = x['hidden_states'][self.select_encoder_layer_nums]
            x = x.transpose(1, 2)  # (B,T,D)->(B,D,T)
            x = self.speech_down_or_up(x)
        elif self.speech_encoder_type == "hubert":
            # its input is wavform. we use pretrain mask setting again
            with torch.no_grad() if fix_encoder else contextlib.ExitStack():
                batch_size, raw_sequence_length = ref_speech.shape
                x = self.speech_encoder(ref_speech, output_hidden_states=True)
                x = x.hidden_states[-1]  # (B,T,D)
            x = x.transpose(1, 2)  # (B,T,D)->(B,D,T)
            x = self.speech_down_or_up(x)
        elif (
            self.speech_encoder_type == "ReDimNetB0"
            or self.speech_encoder_type == "ReDimNetB1"
            or self.speech_encoder_type == "ReDimNetB2"
            or self.speech_encoder_type == "ReDimNetB3"
            or self.speech_encoder_type == "ReDimNetM"
            or self.speech_encoder_type == "ReDimNetS"
        ):
            with torch.no_grad() if fix_encoder else contextlib.ExitStack():
                # its input melbank feature(72-dim)
                # print(f"input shape of speech_encoder: {ref_speech.shape}")
                x = self.speech_encoder.get_frame_level_feat(ref_speech)  # (B,T,D)
            x = x.transpose(1, 2)  # (B,T,D)->(B,D,T)
            # print(f"output shape of speech_encoder: {x.shape}")
            x = self.speech_down_or_up(x)  # (B,D,T) -> (B,F,T)
            # print(f"output shape of self.speech_down_or_up(x): {x.shape}")
        else:
            with torch.no_grad() if fix_encoder else contextlib.ExitStack():
                # its input fbank feature(80-dim)
                x = self.speech_encoder(ref_speech, get_time_out=True)
            # print(f"x shape: {x.shape}") # B,F',T'
            x = self.speech_down_or_up(x)
        # print(f"x shape: {x.shape}, max_len: {max_len}")

        ## process gap of audio len and label len
        gap = x.size(-1) - max_len
        assert (
            abs(x.size(-1) - max_len) <= 2
        ), f"label and ref_speech(mix speech) diff: {x.size(-1)-max_len}"
        # padding audio len to label len
        if gap == -1:
            x = nn.functional.pad(x, (0, 1))
        elif gap == -2:
            x = nn.functional.pad(x, (0, 2))
        # cut audio len to label len
        x = x[:, :, :max_len]  # (B,D,T)

        mix_embeds = x.transpose(1, 2)  # (B,T,D)

        # target speech
        ts_embeds = self.rs_dropout(
            target_speech
        )  # B, max_num_speaker in current batch, D

        # combine target embedding and mix embedding
        # Extend ts_embeds for time alignemnt
        ts_embeds = ts_embeds.unsqueeze(2)  # B, max_num_speaker in current batch, 1, D
        # repeat T to cat mix frame-level information
        ts_embeds = ts_embeds.repeat(
            1, 1, mix_embeds.shape[1], 1
        )  # B, max_num_speaker in current batch, T, D
        B, num_speaker, T, _ = ts_embeds.shape

        # Transformer for single speaker
        # assume self.max_num_speaker==4,
        cat_embeds = []
        for i in range(num_speaker):
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
        cat_embeds = torch.stack(cat_embeds)  # num_speaker, B, T, F
        num_speaker, B, T, F = cat_embeds.shape
        # cat_embeds = torch.permute(cat_embeds, (1, 0, 3, 2))  # B,num_speaker,F,T
        # Combine the outputs
        cat_embeds = cat_embeds.permute(0, 1, 3, 2)  #  num_speaker, B,  F, T
        cat_embeds = cat_embeds.reshape(-1, F, T)  # num_speaker* B, F, T

        # cat multi forward
        # B, _, T = cat_embeds.size()
        # Downsampling
        # cat_embeds = self.backend_down(cat_embeds)  # B, F, T
        # Transformer for multiple speakers
        cat_embeds = self.pos_encoder(
            torch.permute(cat_embeds, (2, 0, 1))
        )  # T,B*num_speaker,F
        if self.multi_backend_type == "conformer":
            cat_embeds = cat_embeds.transpose(0, 1)  # B*num_speaker, T, F
            lengths = torch.tensor(
                [cat_embeds.size(1) for i in cat_embeds],
                dtype=torch.int32,
                device=cat_embeds.device,
            )
            # print(f"before multi_backend, cat_embeds shape: {cat_embeds.shape},lengths shape: {lengths.shape}")
            cat_embeds, _ = self.multi_backend(
                cat_embeds, lengths
            )  # B*num_speaker, T, F
            cat_embeds = cat_embeds.view(B, num_speaker, T, -1)
            cat_embeds = cat_embeds.permute(0, 2, 1, 3)  # B,T,num_speaker, F
            # print(f"after multi_backend, cat_embeds shape: {cat_embeds.shape}")
        else:
            cat_embeds = self.multi_backend(cat_embeds)  # T, B*num_speaker, F
            # cat_embeds = cat_embeds.transpose(0, 1)  # B*num_speaker,T,F
            cat_embeds = cat_embeds.view(T, B, num_speaker, -1)  # T,B, num_speaker, F
            cat_embeds = cat_embeds.permute(1, 0, 2, 3)  # B,T,num_speaker, F

        outs = self.fc(cat_embeds).squeeze(-1)  # B, T, num_speaker
        outs = outs.transpose(1, 2)  # B,num_speaker,T
        return outs

    def forward(
        self,
        ref_speech: torch.Tensor,
        target_speech: torch.Tensor,
        labels: torch.Tensor,
        num_updates: int,
    ):
        if not self.support_variable_number_speakers:
            outs = self.forward_common(
                ref_speech=ref_speech,
                target_speech=target_speech,
                labels=labels,
                num_updates=num_updates,
            )
        else:
            outs = self.forward_common_variable_number_speakers(
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
        outs = self.forward_common(
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
