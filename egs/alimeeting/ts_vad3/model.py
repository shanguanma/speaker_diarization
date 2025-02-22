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

#from ts_vad_dataset import FBank
import torchaudio.compliance.kaldi as kaldi
from cam_pplus_wespeaker import CAMPPlus, CAMPPlusFrame
from wavlm import WavLM, WavLMConfig
from resnet_wespeaker import ResNet34
from samresnet_wespeaker import SimAM_ResNet34_ASP
from ecapa_tdnn_wespeaker import ECAPA_TDNN_GLOB_c1024
from ecapa_tdnn import ECAPA_TDNN
from whisper_encoder import ModelDimensions
from whisper_encoder import WhisperEncoder

logger = logging.getLogger(__name__)


@dataclass
class TSVADConfig:
    speech_encoder_type: str = "w2v-bert2"
    """choice it  from CAM++, WavLM , w2v-bert2 or mhubert-147 as speech encoder of our TSVAD"""

    speech_encoder_config: str = (
        "/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/config.json"
    )
    """
    this config is only used to instantiate huggingface format model(i.e.w2v-bert2.0, mhubert-147, Chinese hubert).
    """
    speech_encoder_path: str = (
        "/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/w2v-bert2.0/model.safetensors"
    )
    """path to pretrained speech encoder path."""
    freeze_speech_encoder_updates: int = 4000
    """train number of step <4000, freeze speech encoder. train number of step> 4000, speech encoder start update."""

    speaker_encoder_type: str = "CAM++"
    """choice it  from CAM++ as speaker encoder of our TSVAD"""

    speaker_encoder_path: str = (
        "/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt"
    )
    """path to pretrained speaker encoder path."""

    freeze_speaker_encoder_updates: int = 62600
    """
    if >=62600,At alimeeting dataset,one epoch has 1565 steps when batch_size=64,so 40 epochs have 62600 steps.
        in other words, I will always freeze speaker encoder when our tsvad is training.
    if <62600,speaker encoder start update.
    """

    fuse_fbank_feat: bool = False
    """if it is true, at fbank feat level, target speaker and mix speech interact with each other
       if it is false, at fbank feat level, target speaker  will not help mix speech
    """

    fuse_speaker_embedding_feat: bool = False
    """if it is true, at embedding feat level, target speaker and mix speech interact with each other
       if it is false, at embedding feat level, target speaker will not help mix speech
    """

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
    # freeze_speech_encoder_updates: (TODO)
    # freeze_speaker_encoder: bool = True
    # """
    # if it true, we will freeze speaker encoder,when we get target speaker embedding online on training our tsvad model.
    # if it false,we joint speaker encoder and speech encoder to train our tsvad model.
    # """
    n_head: int = 4
    """fuse feat module attention head number"""
    fuse_attn_type: str= "native"
    """choise from `native, attn, attn2` """

model_cfg = TSVADConfig()

class Rotary(torch.nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.inv_freq = None
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=x.device).float() / self.dim))
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)


class FBank(object):
    def __init__(
        self,
        n_mels,
        sample_rate,
        mean_nor: bool = False,
    ):
        self.n_mels = n_mels
        self.mean_nor = mean_nor
        self.sample_rate = sample_rate

    def __call__(self, wav, dither=0.0):
        sr = 16000
        assert sr==self.sample_rate, f"sample_rate"
        if len(wav.shape) == 1:
            wav = wav.unsqueeze(0)
        # select single channel
        if wav.shape[0] > 1:
            wav = wav[0, :]
        assert len(wav.shape) == 2 and wav.shape[0] == 1
        #wav = wav * (1 << 15)
        logger.debug(f"in the model.py here wav should be target speech ,wav: {wav}, its shape: {wav.shape}")
        feat = kaldi.fbank(
            wav,
            num_mel_bins=self.n_mels,
            sample_frequency=sr,
            dither=dither,
        )
        logger.debug(f"in the model.py here feat should be target speech ,feat: {feat},its shape: {feat.shape}")
        # feat: [T, N]
        if self.mean_nor:
            feat = feat - feat.mean(0, keepdim=True)
        return feat


class FuseModule(nn.Module):
    def __init__(self,n_embd,cfg):
        super(FuseModule,self).__init__()
        assert n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.dropout = cfg.dropout
        self.q_attn = nn.Linear(n_embd, n_embd, bias=True)
        self.k_attn = nn.Linear(n_embd, n_embd, bias=True)
        self.v_attn = nn.Linear(n_embd, n_embd, bias=True)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=True)
        self.resid_dropout = nn.Dropout(self.dropout)
    def forward(self,speaker_feat,speech_feat):
        # speaker_feat: (B,4*T_,D) 
        # speech_feat: (B,T_hat,D)
        B = speaker_feat.size(0)
        D = speaker_feat.size(-1)
        T = speaker_feat.size(1)
        T_hat  = speech_feat.size(1)


        k = self.k_attn(speaker_feat)
        #print(f"k shape: {k.shape}")
        v = self.v_attn(speaker_feat)
        q = self.q_attn(speech_feat)
        query = q.view(B, T_hat, self.n_head, D//self.n_head).transpose(1,2) #(B, nh, T_hat,hs)
        key = k.view(B, T, self.n_head, D//self.n_head).transpose(1,2) #(B, nh, T,hs)
        value = v.view(B, T, self.n_head, D//self.n_head).transpose(1,2) #(B, nh, T,hs)
        att_feat = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=(self.dropout if self.training else 0.0),
        ) # (B, nh, T_hat,hs)
        y = att_feat.transpose(1, 2).contiguous().view(B, T_hat, D) # re-assemble all head outputs side by side
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        fuse_feat = torch.cat((y, speech_feat), dim=2)  # (B,T',2*D)
        return fuse_feat


class FuseModule2(nn.Module):
    def __init__(self,n_embd,cfg,n_head=4):
        super(FuseModule2,self).__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.dropout = cfg.dropout
        self.q_attn = nn.Linear(n_embd, n_head*n_embd, bias=False)
        self.k_attn = nn.Linear(n_embd, n_embd, bias=False)
        self.v_attn = nn.Linear(n_embd, n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(n_head*n_embd, n_embd, bias=False)
        self.resid_dropout = nn.Dropout(self.dropout)
    def forward(self,speaker_feat,speech_feat):
        # speaker_feat: (B,4*T_,D) # I will view 4 speaker as 4 heads
        # speech_feat: (B,T_hat,D)
        B = speaker_feat.size(0)
        D = speaker_feat.size(-1)
        T_hat  = speech_feat.size(1)
        T = speaker_feat.size(1)

        k = self.k_attn(speaker_feat)
        v = self.v_attn(speaker_feat)
        q = self.q_attn(speech_feat) # (B,T_hat,n_head*D)
        query = q.view(B, T_hat, self.n_head, D).transpose(1,2) #(B, nh, T_hat,hs)
        key = k.view(B, self.n_head, T//self.n_head, D)# (B, nh, T_,hs)
        value = v.view(B, self.n_head, T//self.n_head, D)# (B, nh, T_,hs)
        #k = q.view(B, T, cfg.h_head, D//cfg.n_head).transpose(1,2) #(B, nh, T,hs)
        #v = q.view(B, T, cfg.h_head, D//cfg.n_head).transpose(1,2) #(B, nh, T,hs)
        #print(f"in the FuseModule2 forward: query shape: {query.shape},key shape: {key.shape}") 
        att_feat = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=(self.dropout if self.training else 0.0),
        ) # (B, nh, T_hat,hs)
        y = att_feat.transpose(1, 2).contiguous().view(B, T_hat, self.n_head*D) # re-assemble all head outputs side by side
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        fuse_feat = torch.cat((y, speech_feat), dim=2)  # (B,T_hat,2*D)
        return fuse_feat




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
        self.gradient_checkpointing = False  # for DDP of accelerator
        self._tied_weights_keys = [] # for fsdp of accelerator
        self.freeze_speech_encoder_updates = cfg.freeze_speech_encoder_updates
        self.freeze_speaker_encoder_updates = cfg.freeze_speaker_encoder_updates
        self.speaker_encoder_type = cfg.speaker_encoder_type
        self.fuse_fbank_feat = cfg.fuse_fbank_feat
        self.fuse_speaker_embedding_feat = cfg.fuse_speaker_embedding_feat
        self.fuse_attn_type = cfg.fuse_attn_type


        # self.use_spk_embed = cfg.use_spk_embed
        self.use_spk_embed = (
            task_cfg.spk_path
        )  # spk_path is target speaker utterance embedding path or None.
        self.rs_dropout = nn.Dropout(p=cfg.dropout)
        self.label_rate = task_cfg.label_rate
        assert (
            self.label_rate == 25
        ), f"self.label_rate is {elf.label_rate} not support!"

        self.speech_encoder_type = cfg.speech_encoder_type
        # self.speech_encoder_path = cfg.speech_encoder_path
        self.wavlm_fuse_feat_post_norm = cfg.wavlm_fuse_feat_post_norm
        logger.info(f"self.wavlm_fuse_feat_post_norm: {self.wavlm_fuse_feat_post_norm}")
        self.max_num_speaker = task_cfg.max_num_speaker
        sample_times = 16000 / task_cfg.sample_rate
        # self.target_speaker_input_type = task_cfg.target_speaker_input_type

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

        if not self.use_spk_embed:
            (
                ## create speaker encoder
                ## Its main function is to interact the target speaker's multi-level information
                # with the reference speech (mixer speech)
                # so that the model can obtain multi-level information to help determine the speaker's timestamp.
                # target speaker is wavfrom input
                self.speaker_encoder,
                speaker_frame_embedding_dim,
            ) = self.create_speaker_encoder(sample_times, cfg, device)
            logger.info(f"input speaker model is wavfrom !!!")
            self.feature_extractor = FBank(
                80, sample_rate=task_cfg.sample_rate, mean_nor=True
            )
            if self.fuse_fbank_feat:
                if self.fuse_attn_type=="native":
                    self.fuse_fbank_module = None
                elif self.fuse_attn_type=="attn":
                    self.fuse_fbank_module=FuseModule(80,cfg)
                elif self.fuse_attn_type=="attn2":
                    self.fuse_fbank_module=FuseModule2(80,cfg)
                # project linear
                self.fuse_fbank_linear = nn.Linear(160, 80)
            else:
                self.fuse_fbank_linear = None


            if self.fuse_speaker_embedding_feat:
                if self.pretrain_speech_encoder_dim != speaker_frame_embedding_dim:
                    self.speaker_encoder_frame_dim_proj = nn.Linear(
                        speaker_frame_embedding_dim, self.pretrain_speech_encoder_dim
                    )
                else:
                    self.speaker_embedding_proj = None

                
                if self.fuse_attn_type=="native":
                    self.fuse_frame_module = None
                elif self.fuse_attn_type=="attn":
                    self.fuse_frame_module=FuseModule(self.pretrain_speech_encoder_dim,cfg)
                elif self.fuse_attn_type=="attn2":
                    self.fuse_frame_module=FuseModule2(self.pretrain_speech_encoder_dim,cfg)


                self.fuse_speaker_encoder_linear = nn.Linear(
                    self.pretrain_speech_encoder_dim * 2,
                    self.pretrain_speech_encoder_dim,
                )
            else:
                self.fuse_speaker_encoder_linear = None

            self.att_fuse_dropout = cfg.dropout
        else:
            # target speaker embedding as input. not use speaker encoder network,
            self.speaker_encoder = None
            self.speaker_down_or_up = None
            self.pretrain_speaker_encoder_dim = None
            self.feature_extractor = None
            self.fuse_fbank_linear = None
            self.fuse_speaker_encoder_linear = None

        # projection
        if cfg.speaker_embed_dim * 2 != cfg.transformer_embed_dim:
            self.proj_layer = nn.Linear(
                cfg.speaker_embed_dim * 2, cfg.transformer_embed_dim
            )
        else:
            self.proj_layer = None

        self.single_backend = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=cfg.transformer_embed_dim,
                dim_feedforward=cfg.transformer_ffn_embed_dim,
                nhead=cfg.num_attention_head,
                dropout=cfg.dropout,
            ),
            num_layers=cfg.num_transformer_layer,
        )
        self.pos_encoder = PositionalEncoding(
            cfg.transformer_embed_dim,
            dropout=cfg.dropout,
            max_len=(task_cfg.rs_len * self.label_rate),
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

        self.multi_backend = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=cfg.transformer_embed_dim,
                dim_feedforward=cfg.transformer_ffn_embed_dim,
                nhead=cfg.num_attention_head,
                dropout=cfg.dropout,
            ),
            num_layers=cfg.num_transformer_layer,
        )

        self.fc = nn.Linear(cfg.transformer_embed_dim, self.max_num_speaker)

    def create_speaker_encoder(self, sample_times, cfg, device):
        self.speaker_encoder: Optional[nn.Module] = None
        if self.speaker_encoder_type == "CAM++":
            self.speaker_encoder = CAMPPlus(feat_dim=80, embedding_size=192)
            self.speaker_encoder.load_state_dict(
                torch.load(cfg.speaker_encoder_path, map_location=device),strict=False
            ) 
            #self.load_speaker_encoder(
            #    cfg.speaker_encoder_path, device=device, module_name="speaker_encoder"
            #)
            speaker_embedding_dim = 512
        return (
            self.speaker_encoder,
            speaker_embedding_dim,
        )

    def create_speech_encoder(self, sample_times, cfg, device):
        self.speech_encoder: Optional[nn.Module] = None
        self.speech_down_or_up: Optional[nn.Module] = None
        pretrain_speech_encoder_dim: int = None
        self.weights: Optional[nn.Module] = None  # only for WavLM_weight_sum
        wavlm_encoder_num_layer: int = None  # only for WavLM_weight_sum
        self.wavlmlnorm: Optional[nn.Module] = None  # only for WavLM_weight_sum
        self.wavlmproj: Optional[nn.Module] = None  # only for WavLM_weight_sum
        if self.speech_encoder_type == "CAM++":
            self.speech_encoder = CAMPPlusFrame(feat_dim=80, embedding_size=192)
            #self.speech_encoder.train()
            #self.load_speaker_encoder(
            #    cfg.speech_encoder_path, device=device, module_name="speech_encoder"
            #)
            self.speech_encoder.load_state_dict(
                torch.load(cfg.speech_encoder_path, map_location=device),strict=False
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

    def forward_speech_encoder(
        self,
        ref_speech: torch.Tensor,
        labels: torch.Tensor,
        # speech_fbank_or_cnn_feat: torch.Tensor = None
        fix_speech_encoder: bool = True,
        num_updates: int = 0,
    ):
        B = ref_speech.size(0)
        T = ref_speech.size(1)
        # print(f"ref_speech shape: {ref_speech.shape}")
        max_len = labels.size(-1)
        fix_speech_encoder = num_updates < self.freeze_speech_encoder_updates
        speech_fbank_or_cnn_feat: torch.Tensor
        speech_encoder_output_feat: torch.Tensor

        if self.speech_encoder_type == "WavLM":
            # it should be wavform input, shape (B,T), fbank_input of ts_vad_dataset.py should be set False
            with torch.no_grad() if fix_speech_encoder else contextlib.ExitStack():
                out = self.speech_encoder.extract_features(
                    ref_speech,
                    output_layer=self.wavlm_encoder_num_layer,
                    ret_conv=True,
                    ret_layer_results=True,
                )
                speech_fbank_or_cnn_feat = out[0][0]  # (B,T,D)
                speech_encoder_output_feat = out[0][1][-1][
                    0
                ]  # you can see more detail about wavlm.py,shape: (T,B,D)
                speech_encoder_output_feat = speech_encoder_output_feat.permute(
                    1, 0, 2
                )  # (T,B,D) -> (B,T,D)
            x = speech_encoder_output_feat.view(
                -1, B, self.pretrain_speech_encoder_dim
            )  # B, , self.pretrain_speech_encoder_dim
            x = x.permute(1, 2, 0)  # (B,D,T)
            x = self.speech_down_or_up(x)
        elif self.speech_encoder_type == "WavLM_weight_sum":
            # it should be wavform input, shape (B,T), fbank_input of ts_vad_dataset.py should be set False
            with torch.no_grad() if fix_speech_encoder else contextlib.ExitStack():
                # print(x[0][0].shape)#(B,T,D), it is output of last layer of transformer
                # print(len(x[0][1])) # when output_layer=12,ret_layer_results=True, it is 13,
                # Because the first element of ret_layer_results is the result of pos_conv,
                # and the second element is the output of the first layer of transformer,
                # Each element here is actually a tuple. The first element of the tuple is hidden state, and the second element is attention_state.
                out = self.speech_encoder.extract_features(
                    ref_speech,
                    output_layer=self.wavlm_encoder_num_layer,
                    ret_conv=True,
                    ret_layer_results=True,
                )
                speech_fbank_or_cnn_feat = out[0][0]  # #(B,T,D)

            hss = [hs for hs, att in out[0][1]]
            # print(f"hss len: {len(hss)}")
            stacked_hs = torch.stack(hss[1:], dim=-1)  # (T,B,D,12)

            if self.wavlm_fuse_feat_post_norm:
                x = self.weights(stacked_hs)  # (T,B,D,1)
                x = torch.squeeze(x, -1)  # (T,B,D)
                x = x.permute(1, 0, 2)  # (B,T,D)
                x = self.wavlmproj(x)  # (B,T,D)
                x = self.wavlmlnorm(x)  # (B,T,D)
                speech_encoder_output_feat = x  # (B,T,D)
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
                speech_encoder_output_feat = x  # (B,T,D)
                x = weighted_hs.permute(0, 2, 1)  # (B,D,T)
                x = self.speech_down_or_up(x)
        ## (TODO) MADUO, maybe remove it and finetune, now I have no time to do it.
        elif self.speech_encoder_type == "whisper":
            # it should be wavform input, shape (B,T), fbank_input of ts_vad_dataset.py should be set False
            # whisper output frame rate 50,so I need to downsample to 25.
            # whisper encoder output  shape (B,T,D)
            with torch.no_grad() if fix_speech_encoder else contextlib.ExitStack():
                x = self.speech_encoder(ref_speech)
            x = x.transpose(1, 2)  # (B,T,D)-> (B,D,T)
            x = self.speech_down_or_up(x)  # (B,D,T/2)

        elif self.speech_encoder_type == "w2v-bert2":
            # its input fbank feature(80-dim)
            speech_fbank_or_cnn_feat = ref_speech  # (B,T,D)
            # B,T,F, w2v-bert2's expected input is dimension is 160, to achieve the same 50 frame rate as wav2vec2,
            # because w2v-bert2 Because the actual input of w2v-bert2.0 is kaldi's fbank
            # (its frame rate is 100, because the window length is 25ms, the window shift is 10ms, and mel_bin is 80),
            # then the frame number is halved and the dimension is doubled
            with torch.no_grad() if fix_speech_encoder else contextlib.ExitStack():
                batch_size, num_frames, num_channels = ref_speech.shape
                ref_speech = torch.reshape(
                    ref_speech, (batch_size, num_frames // 2, num_channels * 2)
                )  # (batch_size, T, 160)
                x = self.speech_encoder(ref_speech, output_hidden_states=True)  #
                x = x.hidden_states[
                    -1
                ]  # it is equal to x = x['hidden_states'][self.select_encoder_layer_nums]
                speech_encoder_output_feat = x  # (B,T,D)

            x = x.transpose(1, 2)  # (B,T,D)->(B,D,T)
            x = self.speech_down_or_up(x)

        elif self.speech_encoder_type == "hubert":
            # its input is wavform. we use pretrain mask setting again
            with torch.no_grad() if fix_speech_encoder else contextlib.ExitStack():
                batch_size, raw_sequence_length = ref_speech.shape

                speech_fbank_or_cnn_feat = self.speech_encoder.feature_extractor(
                    ref_speech
                )  # (B,D,T)
                speech_fbank_or_cnn_feat = speech_fbank_or_cnn_feat.permute(
                    0, 2, 1
                )  # (B,D,T) -> (B,T,D)

                x = self.speech_encoder(ref_speech, output_hidden_states=True)
                x = x.hidden_states[-1]  # (B,T,D)
                speech_encoder_output_feat = x  # (B,T,D)

            x = x.transpose(1, 2)  # (B,T,D)->(B,D,T)
            x = self.speech_down_or_up(x)
        elif self.speech_encoder_type == "CAM++":
            with torch.no_grad() if fix_speech_encoder else contextlib.ExitStack():
                # its input fbank feature(80-dim)
                speech_fbank_or_cnn_feat = ref_speech  # (B,T,D)
                _, x = self.speech_encoder(ref_speech)
                speech_encoder_output_feat = x.permute(0, 2, 1)  # (B,D,T)->(B,T,D)
            # print(f"x shape: {x.shape}") # B,D,T
            x = self.speech_down_or_up(x)
        else:
            with torch.no_grad() if fix_speech_encoder else contextlib.ExitStack():
                # its input fbank feature(80-dim)
                speech_fbank_or_cnn_feat = ref_speech  # (B,T,D)
                x = self.speech_encoder(ref_speech, get_time_out=True)
                speech_encoder_output_feat = x.permute(0, 2, 1)  # (B,D,T)->(B,T,D)
            # print(f"x shape: {x.shape}") # B,D,T
            x = self.speech_down_or_up(x)
        assert (
            x.size(-1) - max_len <= 2 and x.size(-1) - max_len >= -1
        ), f"label and ref_speech(mix speech) diff: {x.size(-1)-max_len}"
        if x.size(-1) - max_len == -1:
            x = nn.functinal.pad(x, (0, 1))
        x = x[:, :, :max_len]  # (B,D,T)
        speech_fbank_or_cnn_feat = speech_fbank_or_cnn_feat[:, :max_len, :]  # (B,T,D)
        speech_encoder_output_feat = speech_encoder_output_feat[
            :, :max_len, :
        ]  # (B,T,D)
        mix_embeds = x.transpose(1, 2)  # (B,T,D)
        speech_encoder_final_feat = mix_embeds
        return (
            speech_fbank_or_cnn_feat,
            speech_encoder_output_feat,
            speech_encoder_final_feat,
        )

    def forward_speaker_encoder(self, x, num_updates):
        # case1: target speaker embedding as input
        if self.use_spk_embed:
            # x: Bx4xD, D is speaker embedding dimension
            # utterance level speaker embedding
            return None, None, self.rs_dropout(x)
        # case2: target speaker wavform as input
        # x shape(B,4,T), if data_cfg.ts_len in dataset.py is = 6(second), so T = 16000*6=96000
        B, _, T = x.shape
        logger.debug(f"target speech: {x},its shape: {x.shape}")
        x = x.view(B * self.max_num_speaker, T)
        fbank = [self.feature_extractor(ts) for ts in x]
        # frame level fbank feature.
        fbank = torch.stack(fbank)  # (B*self.max_num_speaker,T,80)
        logger.debug(f"target speech fbank: {fbank},its shape: {fbank.shape}")
        # frame level speaker embedding
        speaker_fbank = fbank.view(B, -1, 80)
        fix_speaker_encoder = num_updates < self.freeze_speaker_encoder_updates
        if self.speaker_encoder_type == "CAM++":
            with torch.no_grad() if fix_speaker_encoder else contextlib.ExitStack():
                speaker_encoder_utt_feat, speaker_encoder_frame_feat = (
                    self.speaker_encoder(fbank)
                )  # (B*self.max_num_speaker,D) (B*self.max_num_speaker,D,T), D=512
                speaker_encoder_utt_feat = speaker_encoder_utt_feat.view(
                    B, self.max_num_speaker, -1
                )  # (B,4,D) # utterance level speaker embedding,D=512
                speaker_encoder_frame_feat = speaker_encoder_frame_feat.permute(
                    0, 2, 1
                )  # (B*self.max_num_speaker,T,D)
                E = speaker_encoder_frame_feat.size(-1)
                speaker_encoder_frame_feat = speaker_encoder_frame_feat.reshape(
                    B, -1, E
                )  # (B, self.max_num_speaker*T, D)
        return speaker_fbank, speaker_encoder_frame_feat, speaker_encoder_utt_feat

    def att_fuse_kernel(self, speaker_feat, speech_feat):
        B = speaker_feat.size(0)
        D = speaker_feat.size(-1)
        key = speaker_feat.view(B, -1, D).unsqueeze(1)  # (B,1,4*T,D)
        value = key
        query = speech_feat.unsqueeze(1)  # (B,1,T',D)
        # print(f"query shape: {query.shape}, key shape: {key.shape}")
        att_feat = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=(self.att_fuse_dropout if self.training else 0.0),
        )
        att_feat = att_feat.squeeze(1)  # (B,T',D)
        fuse_feat = torch.cat((att_feat, speech_feat), dim=2)  # (B,T',2*D)
        return fuse_feat

    def fuse_feat_speech_encoder_forward(
        self, target_speaker_speech, ref_speech, labels, num_updates: int = 0
    ):
        """
        target_speaker_speech:
           if self.use_spk_embed = False,
           it is wavfrom, shape:(B,self.max_num_speaker, 16000*6), 6 is data_cfg.ts_len
           if self.use_spk_embed = True
           it is utterance level speaker embedding, shape: (B,self.max_num_speaker,speaker_embed_dim)

        ref_speech: fbank feat, shape: (B,T',80)
        """

        speaker_fbank, speaker_encoder_frame_feat, speaker_encoder_utt_feat = (
            self.forward_speaker_encoder(target_speaker_speech, num_updates)
        )

        max_len = labels.size(-1)
        fix_speech_encoder = num_updates < self.freeze_speech_encoder_updates
        ## it will force reset args to avoid error.
        if speaker_fbank is None:
            self.fuse_fbank_feat = False
        if speaker_encoder_frame_feat is None:
            self.fuse_speaker_embedding_feat = False

        if self.fuse_fbank_feat:
            if self.fuse_attn_type=="attn" or self.fuse_attn_type=="attn2":
                fuse_fbank = self.fuse_fbank_module(speaker_fbank, ref_speech)
            else:
                fuse_fbank = self.att_fuse_kernel(speaker_fbank, ref_speech)
            fuse_fbank = self.fuse_fbank_linear(fuse_fbank)  # (B,T',80)
        else:
            fuse_fbank = ref_speech
        if self.speech_encoder_type == "w2v-bert2":
            with torch.no_grad() if fix_speech_encoder else contextlib.ExitStack():
                batch_size, num_frames, num_channels = fuse_fbank.shape
                fuse_fbank = torch.reshape(
                    fuse_fbank, (batch_size, num_frames // 2, num_channels * 2)
                )  # (batch_size, T'//2, 160)
                x = self.speech_encoder(fuse_fbank, output_hidden_states=True)  #
                x = x.hidden_states[
                    -1
                ]  # it is equal to x = x['hidden_states'][self.select_encoder_layer_nums],shape:(B,T'//2,D),D=1024
        elif self.speech_encoder_type == "CAM++":
            with torch.no_grad() if fix_speech_encoder else contextlib.ExitStack():
                # its input fbank feature(80-dim)
                batch_size, num_frames, num_channels = fuse_fbank.shape # (B,T,D)
                #utt_feat, frame_feat = self.speech_encoder(fuse_fbank) # utt_feat:(B,F), frame_feat:(B,D,T)
                frame_feat = self.speech_encoder(fuse_fbank) # utt_feat:(B,F), frame_feat:(B,D,T)
                x = frame_feat.permute(0,2,1) # (B,D,T) -> (B,T,D)
                #speaker_encoder_utt_feat = utt_feat

        if self.fuse_speaker_embedding_feat:
            speaker_encoder_frame_feat_dim = speaker_encoder_frame_feat.size(-1)
            speech_encoder_embedding_feat_dim = x.size(-1)
            if speaker_encoder_frame_feat_dim != speech_encoder_embedding_feat_dim:
                speaker_encoder_frame_feat = self.speaker_encoder_frame_dim_proj(
                    speaker_encoder_frame_feat
                )
            if self.fuse_attn_type=="attn" or self.fuse_attn_type=="attn2":
                fuse_emb_feat = self.fuse_frame_module(speaker_encoder_frame_feat, x)
            else:
                fuse_emb_feat = self.att_fuse_kernel(speaker_encoder_frame_feat, x)
            fuse_emb_feat = self.fuse_speaker_encoder_linear(
                fuse_emb_feat
            )  # (B,T'//2,2*D)-> (B,T'//2,D)
        else:
            fuse_emb_feat = x

        x = fuse_emb_feat.transpose(1, 2)  # (B,T'//2,D)->(B,D,T'//2)
        # it except B,D,T'//2 feat shape
        x = self.speech_down_or_up(x)
        assert (
            x.size(-1) - max_len <= 2 and x.size(-1) - max_len >= -1
        ), f"label and ref_speech(mix speech) diff: {x.size(-1)-max_len}"
        if x.size(-1) - max_len == -1:
            x = nn.functinal.pad(x, (0, 1))
            x = x[:, :, :max_len]  # (B,D,T'//2//2)
        mix_embeds = x.transpose(1, 2)  # (B,T'//2//2,D)

        return mix_embeds, speaker_encoder_utt_feat

    def forward_common(
        self,
        ref_speech: torch.Tensor,
        target_speech: torch.Tensor,
        labels: torch.Tensor,
        num_updates: int = 0,
    ):
        ## fuse target speaker feture reference (mixer) speech forward branch
        # mix_embeds shape: (B,100,192) when rs_len=4s,speaker_embed_dim=192,
        # speaker_encoder_utt_feat shape:(B,4,192), when self.max_num_speaker=4, speaker_embed_dim=192
        mix_embeds, speaker_encoder_utt_feat = self.fuse_feat_speech_encoder_forward(
            target_speech, ref_speech, labels, num_updates
        )
        logger.debug(
            f"mix_embeds: {mix_embeds},mix_embeds shape: {mix_embeds.shape},  speaker_encoder_utt_feat: {speaker_encoder_utt_feat}, speaker_encoder_utt_feat shape: {speaker_encoder_utt_feat.shape}"
        )
        # speaker_encoder_final_feat is utterance target speaker embedding
        ts_embeds = self.rs_dropout(speaker_encoder_utt_feat)  # B, 4, D

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
            cat_embed = cat_embed.transpose(0, 1)  # T, B, 2D
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
        cat_embeds = self.multi_backend(cat_embeds)  # T', B, F
        cat_embeds = cat_embeds.transpose(0, 1)  # B,T',F

        outs = self.fc(cat_embeds)  # B T' 4
        outs = outs.transpose(1, 2)  # B,4,T'
        return outs

    def forward(
        self,
        ref_speech: torch.Tensor,
        target_speech: torch.Tensor,
        labels: torch.Tensor,
        num_updates: int = 0,
    ):
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
    # for debug
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )
    model_cfg = TSVADConfig()
    model_cfg.fuse_speaker_embedding_feat = True
    model_cfg.fuse_fbank_feat = True
    # model_cfg.use_spk_embed=True
    data_cfg = TSVADDataConfig()
    # data_cfg.spk_path=None # means that we use speaker model online on our tsvad

    model = TSVADModel(cfg=model_cfg, task_cfg=data_cfg)
    print(model)
    # x = torch.zeros(10, 398, 80)  # B,T,F
    # out = model.speech_encoder(x)
    # print(f"out shape: {out.shape}")
    # B,T,F, w2v-bert2's expected input is dimension is 160, to achieve the same 50 frame rate as wav2vec2,
    # because w2v-bert2 Because the actual input of w2v-bert2.0 is kaldi's fbank
    # (its frame rate is 100, because the window length is 25ms, the window shift is 10ms, and mel_bin is 80),
    # then the frame number is halved and the dimension is doubled
    ref_speech = torch.zeros(10, 400, 80)  # rs_len=4s
    if data_cfg.spk_path:
        target_speech = torch.randn(10, 4, 192)  # ts_len=6s
    else:
        target_speech = torch.randn(10, 4, 96000)  # ts_len=6s
    labels = torch.randint(0, 2, (10, 4, 100))  # 25*rs_len=25*4=100
    # labels =
    model.eval()
    out1 = model(ref_speech, target_speech, labels)
    print(f"out1 shape: {out1.shape}")
    num_params = sum(param.numel() for param in model.parameters())
    for name, v in model.named_parameters():
        print(f"name: {name}, v: {v.shape}")
    print("{} M".format(num_params / 1e6))  # 6.61M
    data_cfg = TSVADDataConfig()
    print(vars(data_cfg))
