# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, field
import sys
import contextlib
from typing import Optional, Tuple, Union,Any, Callable
from collections import defaultdict
from argparse import Namespace
import math
import numpy as np
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.utils.checkpoint as ckpt
from torch.utils.checkpoint import checkpoint
from torch.nn import LayerNorm

from cam_pplus_wespeaker import CAMPPlus
#from cam_pplus_wespeaker_wo_batchnorm import CAMPPlus # only for compute left and right context of cam++
from transformer_chunk_streaming import TransformerEncoderLayer, MultiHeadedAttention, PositionwiseFeedForward
from mask import make_pad_mask
from mask import mask_to_bias
from mask import add_optional_chunk_mask

try:
    from ts_vad2_streaming.mamba import Mamba2BlockV2
except ImportError:
    Mamba2BlockV2 = None

#from train_accelerate_ddp import calculate_loss
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

    static_chunk_size: int = 0

    use_dynamic_chunk: bool = True

    use_dynamic_left_chunk: bool = False

    gradient_checkpointing: bool = False

    use_sdpa: bool = False

    max_num_speaker: int = 4
    
    single_backend_type: str = "transformer"
    """single backend type choices from `transformer or mamba or mamba_v2`"""

    d_state: int = 128
    """d_state of mamba2 """
    expand: int = 4
    """expand of mamba2"""

model_cfg = TSVADConfig()
from datasets import TSVADDataConfig

data_cfg = TSVADDataConfig()

def calculate_loss(outs, labels, labels_len):
    total_loss = 0
    
    #if outs.size(-1) < labels.size(-1):
    #    gap = labels.size(-1) - outs.size(-1)
    #    outs = torch.nn.functional.pad(outs,(0,gap))

    for i in range(labels_len.size(0)):
        total_loss += F.binary_cross_entropy_with_logits(
            outs[i, :, : labels_len[i]], labels[i, :, : labels_len[i]]
        )
    return total_loss / labels_len.size(0)

class TSVADModel(nn.Module):
    def __init__(
        self, cfg=model_cfg, task_cfg=data_cfg, device=torch.device("cpu")
    ) -> None:
        super(TSVADModel, self).__init__()
        self.gradient_checkpointing = False
        # self.freeze_speech_encoder_updates = cfg.freeze_speech_encoder_updates
        self.rs_dropout = nn.Dropout(p=cfg.dropout)  # only for target speaker embedding
        self.max_num_speaker = cfg.max_num_speaker
        self.speech_encoder_type = cfg.speech_encoder_type
        self.speech_encoder_path = cfg.speech_encoder_path
        self.single_backend_type = cfg.single_backend_type
        # warp cam++ and 1d conv downsample
        self.embed = Subsampling4(
            speech_encoder_path=cfg.speech_encoder_path,
            idim=80,  # fbank feature dimention
            odim=cfg.speaker_embed_dim,
        )
        if self.single_backend_type == "transformer":
            self.single_backend = torch.nn.ModuleList(
                [
                    TransformerEncoderLayer(
                        cfg.transformer_embed_dim,
                        MultiHeadedAttention(
                            cfg.num_attention_head, cfg.transformer_embed_dim, cfg.dropout
                        ),
                        PositionwiseFeedForward(
                            cfg.transformer_embed_dim,
                            cfg.transformer_ffn_embed_dim,
                            cfg.dropout,
                        ),
                        LayerNorm,
                        cfg.dropout,
                    )
                    for _ in range(cfg.num_transformer_layer)
                ]
            )
            self.backend_down= torch.nn.Sequential(
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
            self.pos_encoder = PositionalEncoding(cfg.transformer_embed_dim, cfg.dropout)
        
        elif self.single_backend_type == "mamba2":
            self.pos_encoder = PositionalEncoding(cfg.transformer_embed_dim, cfg.dropout)
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
        elif self.single_backend_type == "mamba2_unidirectional":
            self.pos_encoder = PositionalEncoding(cfg.transformer_embed_dim, cfg.dropout)
            # causal_conv1d  channel must be multiples of 8  , So I select 384=192*2 as model dimension.
            self.single_backend = Mamba2BlockV2(
                cfg.transformer_embed_dim,
                n_layer=cfg.num_transformer_layer,
                d_state=cfg.d_state,
                d_conv=4,
                expand=cfg.expand,
                bidirectional=False,
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
        

        self.multi_backend = torch.nn.ModuleList(
            [
                TransformerEncoderLayer(
                    cfg.transformer_embed_dim,
                    MultiHeadedAttention(
                        cfg.num_attention_head, cfg.transformer_embed_dim, cfg.dropout
                    ),
                    PositionwiseFeedForward(
                        cfg.transformer_embed_dim,
                        cfg.transformer_ffn_embed_dim,
                        cfg.dropout,
                    ),
                    LayerNorm,
                    cfg.dropout,
                )
                for _ in range(cfg.num_transformer_layer)
            ]
        )


        self.static_chunk_size = cfg.static_chunk_size
        self.use_dynamic_chunk = cfg.use_dynamic_chunk
        self.use_dynamic_left_chunk = cfg.use_dynamic_left_chunk
        self.gradient_checkpointing = cfg.gradient_checkpointing
        self.use_sdpa = cfg.use_sdpa

        self.max_num_speaker = cfg.max_num_speaker
        self.fc = nn.Linear(
            cfg.transformer_embed_dim, self.max_num_speaker
        )  # transformer_embed_dim = 2 * speaker_embed_dim

    def forward(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
        target_speech: torch.Tensor,
        labels: torch.Tensor,
        decoding_chunk_size: int = 0,
        num_decoding_left_chunks: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        it is used to streaming in training stage.
          Args:
            xs: padded input tensor (B, T, 80),  80 means fbank feature dimension of mix audio.
            xs_lens: input length (B), It is only used to provide a variable length pad mask and has no other use in streaming mode training


            decoding_chunk_size: decoding chunk size for dynamic chunk
                0: default for training, use random dynamic chunk.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
            num_decoding_left_chunks: number of left chunks, this is for decoding,
            the chunk size is decoding_chunk_size.
                >=0: use num_decoding_left_chunks
                <0: use all left chunks
        Returns:
            encoder output tensor xs, and subsampled masks
            xs: padded output tensor (B, T' ~= T/subsample_rate, D) D is speaker embedding dimension
            (TODO) MADUO check it.
            masks: torch.Tensor batch padding mask after subsample
                (B, 1, T' ~= T/subsample_rate)
        NOTE:
            We pass the `__call__` method of the modules instead of `forward` to the
            checkpointing API because `__call__` attaches all the hooks of the module.
            https://discuss.pytorch.org/t/any-different-between-model-input-and-model-forward-input/3690/2
        """
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, 1, T)

        xs, masks = self.embed(
            xs, masks, labels
        )  # xs: (B,T,D), mask: (B, 1, T/subsample_rate), D is speaker embedding dimension
        #print(f"masks shape: {masks.shape} in after self.embed")
        chunk_masks = add_optional_chunk_mask(
            xs,
            masks,
            self.use_dynamic_chunk,
            self.use_dynamic_left_chunk,
            decoding_chunk_size,
            self.static_chunk_size,
            num_decoding_left_chunks,
            # Since we allow up to 1s(100 frames) delay, the maximum
            # chunk_size is 100 / 4 = 25.
            max_chunk_size=int(100.0 / self.embed.subsampling_rate),
        )
        #print(f"xs shape: {xs.shape}")
        #print(f"chunk_masks shape: {chunk_masks.shape}")
        if self.use_sdpa:
            chunk_masks = mask_to_bias(chunk_masks, xs.dtype)
        if self.gradient_checkpointing and self.training:
            xs = self.forward_layers_checkpointed(xs, chunk_masks, target_speech)
        else:
            xs = self.forward_layers(xs, chunk_masks,target_speech)
        # Here we assume the mask is not changed in encoder layers, so just
        # return the masks before encoder layers, and the masks will be used
        # for cross attention with decoder later
        # (TODO) check 'masks'
        #return xs, masks
        return xs

    def forward_layers(
        self,
        xs: torch.Tensor,
        chunk_masks: torch.Tensor,
        target_speech: torch.Tensor,
        #labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args
           xs(torch.tensor) : after embedding feature, shape(B,T',D), T'~= T/subsample_rate, D is speaker embedding dimension
           chunk_masks(torch.tensor): (B,T',T')
           target_speech(torch.tensor): it is speaker utterance embedding, shape (B,4,D), 4 means max number of speakers
           #labels (torch.tensor): speaker activate label of mix audio (xs), shape(B,4,T'),

        Return
           outs(torch.tensor): shape(B,4,T')
        """
        # target speech
        ts_embeds = self.rs_dropout(target_speech)  # B, 4, D

        # combine target embedding and mix embedding
        # Extend ts_embeds for time alignemnt
        ts_embeds = ts_embeds.unsqueeze(2)  # B, 4, 1, D
        # repeat T to cat mix frame-level information
        ts_embeds = ts_embeds.repeat(1, 1, xs.shape[1], 1)  # B, 4, T', D
        B, _, T, _ = ts_embeds.shape

        # Transformer for single speaker
        # assume self.max_num_speaker==4,
        cat_embeds = []
        for i in range(self.max_num_speaker):
            ts_embed = ts_embeds[:, i, :, :]  # B,T',D
            cat_embed = torch.cat((ts_embed, xs), 2)  # B,T',2D
            # cat_embed = cat_embed.transpose(0, 1)  # T, B, 2D

            cat_embed, _ = self.pos_encoder(cat_embed)  # (B,T',2D)
            if self.single_backend_type == "transformer":
                for single_layer in self.single_backend:
                    cat_embed, chunk_masks, _ = single_layer(
                        cat_embed, chunk_masks
                    )  # (B,T',F) ,F=2D
            elif self.single_backend_type == "mamba2" or self.single_backend_type == "mamba2_unidirectional":
                cat_embed = self.single_backend(cat_embed)  # (B,T',F) ,F=2D
            #elif self.single_backend_type == "mamba2_unidirectional":
            #    cat_embed = self.single_backend(cat_embed)  # (B,T',F) ,F=2D
            # cat_embed = self.single_backend(cat_embed)  # T, B, F
            # cat_embed = cat_embed.transpose(0, 1)  # B, T, F
            cat_embeds.append(cat_embed)
        cat_embeds = torch.stack(cat_embeds)  # 4, B, T', F
        cat_embeds = torch.permute(cat_embeds, (1, 0, 3, 2))  # B,4,F,T'
        # Combine the outputs
        cat_embeds = cat_embeds.reshape(B, -1, T)  # B, 4 * F, T'

        # cat multi forward
        B, _, T = cat_embeds.size()
        # project layer, currently setting conv1d (kernel=5, strid=1, padding=2), it no downsample and just a dimension change
        
        cat_embeds = self.backend_down(cat_embeds)  # (B, 4 * F, T')->(B, F, T')

        # Transformer for multiple speakers
        # cat_embeds = self.pos_encoder(torch.permute(cat_embeds, (2, 0, 1)))
        cat_embeds, _ = self.pos_encoder(torch.permute(cat_embeds, (0, 2, 1)))
        for multi_layer in self.multi_backend:
            cat_embeds, chunk_masks, _ = multi_layer(cat_embeds, chunk_masks)  # (B,T',F)
        outs = self.fc(cat_embeds)  # B T' 4
        outs = outs.transpose(1, 2)  # B,4,T'
        return outs

    @torch.jit.unused
    def forward_layers_checkpointed(
        self,
        xs: torch.Tensor,
        chunk_masks: torch.Tensor,
        target_speech: torch.Tensor,
        #labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args
           xs(torch.tensor) : after embedding feature, shape(B,T',D), T'~= T/subsample_rate, D is speaker embedding dimension
           chunk_masks(torch.tensor): (B,T',T')
           target_speech(torch.tensor): it is speaker utterance embedding, shape (B,4,D), 4 means max number of speakers
           labels (torch.tensor): speaker activate label of mix audio (xs), shape(B,4,T'),
        Return
           outs(torch.tensor): shape(B,4,T')
        """
        # target speech
        ts_embeds = self.rs_dropout(target_speech)  # B, 4, D

        # combine target embedding and mix embedding
        # Extend ts_embeds for time alignemnt
        ts_embeds = ts_embeds.unsqueeze(2)  # B, 4, 1, D
        # repeat T to cat mix frame-level information
        ts_embeds = ts_embeds.repeat(1, 1, xs.shape[1], 1)  # B, 4, T', D
        B, _, T, _ = ts_embeds.shape

        # Transformer for single speaker
        # assume self.max_num_speaker==4,
        cat_embeds = []
        for i in range(self.max_num_speaker):
            ts_embed = ts_embeds[:, i, :, :]  # B,T',D
            cat_embed = torch.cat((ts_embed, xs), 2)  # B,T',2D
            # cat_embed = cat_embed.transpose(0, 1)  # T, B, 2D

            cat_embed, _ = self.pos_encoder(cat_embed)  # (B,T',2D)
            if self.single_backend_type == "transformer":
                for single_layer in self.single_backend:
                    cat_embed, chunk_masks, _ = ckpt.checkpoint(
                        single_layer.__call__, cat_embed, chunk_masks, use_reentrant=False
                    )  # (B,T',F)
            elif self.single_backend_type == "mamba2" or self.single_backend_type == "mamba2_unidirectional":
                cat_embed =  ckpt.checkpoint(self.single_backend.__call__,cat_embed, use_reentrant=False)

            #elif self.single_backend_type == "mamba2_unidirectional":
            #    cat_embed =  ckpt.checkpoint(self.single_backend.__call__,cat_embed, use_reentrant=False)
            # cat_embed = self.single_backend(cat_embed)  # T, B, F
            # cat_embed = cat_embed.transpose(0, 1)  # B, T, F
            cat_embeds.append(cat_embed)
        cat_embeds = torch.stack(cat_embeds)  # 4, B, T', F
        cat_embeds = torch.permute(cat_embeds, (1, 0, 3, 2))  # B,4,F,T'
        # Combine the outputs
        cat_embeds = cat_embeds.reshape(B, -1, T)  # B, 4 * F, T'
        
        # cat multi forward
        B, _, T = cat_embeds.size()

        # project layer, currently setting conv1d (kernel=5, strid=1, padding=2), it no downsample and just a dimension change
        #logger.info(f"before self.backend_down cat_embeds shape: {cat_embeds.shape}")
        cat_embeds = self.backend_down(cat_embeds)  # (B, 4 * F, T')->(B, F, T')
        #logger.info(f"after self.backend_down cat_embeds shape: {cat_embeds.shape}")
        # Transformer for multiple speakers
        # cat_embeds = self.pos_encoder(torch.permute(cat_embeds, (2, 0, 1)))
        cat_embeds, _ = self.pos_encoder(torch.permute(cat_embeds, (0, 2, 1)))
        #logger.info(f"after self.pos_encoder cat_embeds shape: {cat_embeds.shape}")
        #logger.info(f"chunk_masks shape: {chunk_masks.shape}")
        for multi_layer in self.multi_backend:
            cat_embeds, chunk_masks, _ = ckpt.checkpoint(
                multi_layer.__call__, cat_embeds, chunk_masks, use_reentrant=False
            )  # (B,T',F)
        outs = self.fc(cat_embeds)  # B T' 4
        outs = outs.transpose(1, 2)  # B,4,T'
        return outs

    def forward_chunk_by_chunk(
        self,
        xs: torch.Tensor,
        target_speech: torch.Tensor,
        labels: torch.Tensor,
        decoding_chunk_size: int = 0,
        num_decoding_left_chunks: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward input chunk by chunk with chunk_size like a streaming fashion, it is used at inference stage

        Here we should pay special attention to computation cache in the
        streaming style forward chunk by chunk. Two things should be taken
        into account for computation in the current network:
            1. transformer encoder layers output cache
            3. convolution in subsampling

        However, we don't implement subsampling cache for:
            1. We can control subsampling module to output the right result by
               overlapping input instead of cache left context, even though it
               wastes some computation, but subsampling only takes a very
               small fraction of computation in the whole model.
            2. Typically, there are several covolution layers with subsampling
               in subsampling module, it is tricky and complicated to do cache
               with different convolution layers with different subsampling
               rate.
            3. Currently, nn.Sequential is used to stack all the convolution
               layers in subsampling, we need to rewrite it to make it work
               with cache, which is not prefered.

        Args:
            xs (torch.Tensor):  (1, temp, 80),
            target_speech (torch.Tensor): (1,max_num_speaker,speaker_embed_dim)
            decoding_chunk_size (int): decoding chunk size for dynamic chunk
                0: default for training, use random dynamic chunk.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
            num_decoding_left_chunks: number of left chunks, this is for decoding,
            the chunk size is decoding_chunk_size.
                >=0: use num_decoding_left_chunks, i.e. 5
                <0: use all left chunks
        Returns:
        (MADUO) todo add return tensor details
        """
        assert decoding_chunk_size > 0
        # The model is trained by static or dynamic chunk
        assert self.static_chunk_size > 0 or self.use_dynamic_chunk
        subsampling = self.embed.subsampling_rate
        #context = self.embed.right_context + 1  # Add current frame
        stride = subsampling * decoding_chunk_size
        #decoding_window = (decoding_chunk_size - 1) * subsampling + context
        #num_frames = xs.size(1)
        all_s_att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0, self.max_num_speaker), device=xs.device)
        m_att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0), device=xs.device)

        outputs = []
        offset = 0
        required_cache_size = decoding_chunk_size * num_decoding_left_chunks
        #print(f"xs shape: {xs.shape}, labels shape: {labels.shape}")
        if xs.size(1) != subsampling*labels.size(-1):
            gap = subsampling*labels.size(-1) - xs.size(1)
            xs = torch.nn.functional.pad(xs.permute(0,2,1),(0,gap))
            xs = xs.permute(0,2,1) # (B,T,80), T=subsampling*label.size(-1)
        num_frames = xs.size(1)
        #print(f"xs shape: {xs.shape}, labels shape: {labels.shape} after pad")
        # Feed forward overlap input step by step
        #for cur in range(0, num_frames - context + 1, stride):
        for cur in range(0,num_frames ,stride):
            #end = min(cur + decoding_window, num_frames)
            #print(f"cur: {}")
            end = min(cur+stride, num_frames)
            #print(f"cur: {cur}, end: {end}")
            chunk_xs = xs[:, cur:end, :]
            #print(f"chunk_xs shape: {chunk_xs.shape}")
            end_subframe= math.ceil(end/4)
            cur_subframe=math.ceil(cur/4)
            chunk_labels = labels[:,:,cur_subframe:end_subframe]
            # y shape: (B,4,chunk_size+1)
            (y, all_s_att_cache, m_att_cache) = self.forward_chunk(
                chunk_xs,
                target_speech,
                chunk_labels,
                offset,
                required_cache_size,
                all_s_att_cache,
                m_att_cache,
            )

            outputs.append(y)
            offset += y.size(2)
        ys = torch.cat(outputs, 2) # (B,4,T')
        #masks = torch.ones((1, 1, ys.size(1)), device=ys.device, dtype=torch.bool)
        #return ys, masks
        return ys

    def forward_chunk_by_chunk_temp(
        self,
        xs: torch.Tensor,
        target_speech: torch.Tensor,
        labels: torch.Tensor,
        decoding_chunk_size: int = 0,
        num_decoding_left_chunks: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        assert decoding_chunk_size > 0
        # The model is trained by static or dynamic chunk
        assert self.static_chunk_size > 0 or self.use_dynamic_chunk
        subsampling = self.embed.subsampling_rate # 4
        #context = self.embed.right_context + 1  # Add current frame
        #stride = subsampling * decoding_chunk_size
        #decoding_window = (decoding_chunk_size - 1) * subsampling + context

        # version1
        #context = self.embed.right_context +1
        #stride = subsampling * decoding_chunk_size
        #decoding_window = (decoding_chunk_size - 1) * subsampling + context

        # version2
        embed_left_context = 7
        embed_conv_right_context = 3
        context = embed_conv_right_context
        stride = subsampling * decoding_chunk_size
        decoding_window = embed_conv_right_context + stride

        # version3

        embed_left_context = 0
        embed_conv_right_context = 3
        context = embed_conv_right_context
        stride = subsampling * decoding_chunk_size
        decoding_window = embed_conv_right_context + stride
        #embed_left_context = 0
        #embed_conv_right_context = 3
        #context = embed_conv_right_context
        #stride = subsampling * decoding_chunk_size
        #decoding_window = (decoding_chunk_size - 1) * subsampling + context
        #decoding_window=100
        #stride=97
        #num_frames = xs.size(1)
        all_s_att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0, self.max_num_speaker), device=xs.device)
        m_att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0), device=xs.device)

        outputs = []
        offset = 0
        required_cache_size = decoding_chunk_size * num_decoding_left_chunks
        #print(f"xs shape: {xs.shape}, labels shape: {labels.shape}")
        if xs.size(1) != subsampling*labels.size(-1):
            gap = subsampling*labels.size(-1) - xs.size(1)
            xs = torch.nn.functional.pad(xs.permute(0,2,1),(0,gap))
            xs = xs.permute(0,2,1) # (B,T,80), T=subsampling*label.size(-1)
        num_frames = xs.size(1)
        #print(f"xs shape: {xs.shape}, labels shape: {labels.shape} after pad")
        # Feed forward overlap input step by step
        prev_context = None
        # version1
        #for cur in range(0, num_frames - context + 1, stride):
        # version2
        #for cur in range(0, num_frames - embed_left_context + 1, stride):
        # version3
        for cur in range(0, num_frames - embed_left_context, stride):
        #for cur in range(0, num_frames ,stride):
            end = min(cur + decoding_window, num_frames)
            #print(f"cur: {}")
            #end = min(cur+stride, num_frames)
            #print(f"cur: {cur}, end: {end}")
            chunk_xs = xs[:, cur:end, :]
            #print(f"chunk_xs shape: {chunk_xs.shape}")
            end_subframe= math.ceil(end/4)
            cur_subframe=math.ceil(cur/4)
            chunk_labels = labels[:,:,cur_subframe:end_subframe]
            #print(f"chunk_labels shape: {chunk_labels.shape}")
            # >>> a = torch.randn(1,5,6)
            # >>> a
            # tensor([[[ 1.4701, -0.2330, -0.1054,  0.8062, -0.7554, -1.5689],
            # [ 0.9159, -0.3713, -0.6349,  0.6448,  0.6860, -1.7852],
            # [ 1.0536,  0.3931,  1.0254, -0.2983, -0.1255,  1.0534],
            # [-0.2505,  0.3323, -1.9381, -0.1602, -1.0488, -0.0893],
            # [-0.0214,  0.0647,  2.2859, -1.6917,  1.0128, -1.8118]]])
            # right context is store at prev_context
            # >>> a[:, -context+1:, :]
            # tensor([[[-0.2505,  0.3323, -1.9381, -0.1602, -1.0488, -0.0893],
            # [-0.0214,  0.0647,  2.2859, -1.6917,  1.0128, -1.8118]]])
            #>>> b = torch.randn(1,5,6)
            #>>> b
            #tensor([[[-0.7487, -0.9015, -0.6136,  0.1377,  0.7068, -0.1022],
            #        [-1.1049, -0.9435,  0.0613,  0.2431, -0.8176,  1.1741],
            #        [ 0.0483, -1.2091, -2.1040,  0.2774,  1.0841,  1.4549],
            #        [-1.9732, -1.5702,  0.4821,  1.4262, -0.3025,  1.2081],
            #        [-0.7191, -0.2478, -0.7409,  1.0952,  1.8095,  1.1998]]])
            # >>> b[:,:context-1, :] = a[:, -context+1:, :]
            # >>> b
            # tensor([[[-0.2505,  0.3323, -1.9381, -0.1602, -1.0488, -0.0893],
            #        [-0.0214,  0.0647,  2.2859, -1.6917,  1.0128, -1.8118],
            #        [ 0.0483, -1.2091, -2.1040,  0.2774,  1.0841,  1.4549],
            #        [-1.9732, -1.5702,  0.4821,  1.4262, -0.3025,  1.2081],
            #        [-0.7191, -0.2478, -0.7409,  1.0952,  1.8095,  1.1998]]])

            if prev_context is not None:
                # Use the previous context information
                chunk_xs[:, :context-1, :] = prev_context
            # y shape: (B,4,chunk_size+1)
            (y, all_s_att_cache, m_att_cache) = self.forward_chunk(
                chunk_xs,
                target_speech,
                chunk_labels,
                offset,
                required_cache_size,
                all_s_att_cache,
                m_att_cache,
            )
            offset += y.size(2)
            # Update the previous context information
            if end < num_frames:
                prev_context = chunk_xs[:, -context+1:, :]
            else:
                prev_context = None
            # Remove the context part from the output
            y = y[:,:, :decoding_chunk_size]
            print(f"y shape: {y.shape}")
            outputs.append(y)
        ys = torch.cat(outputs, 2) # (B,4,T')
        print(f"ys shape: {ys.shape}")
        #masks = torch.ones((1, 1, ys.size(1)), device=ys.device, dtype=torch.bool)
        #return ys, masks
        return ys

    def forward_chunk_by_chunk_temp1(
        self,
        xs: torch.Tensor,
        target_speech: torch.Tensor,
        labels: torch.Tensor,
        decoding_chunk_size: int = 0,
        num_decoding_left_chunks: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        assert decoding_chunk_size > 0
        # The model is trained by static or dynamic chunk
        assert self.static_chunk_size > 0 or self.use_dynamic_chunk
        subsampling = self.embed.subsampling_rate # 4
        all_s_att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0, self.max_num_speaker), device=xs.device)
        m_att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0), device=xs.device)

        outputs = []
        offset = 0
        #num_decoding_left_chunks=0
        required_cache_size = decoding_chunk_size * num_decoding_left_chunks
        print(f"xs shape: {xs.shape}, labels shape: {labels.shape}")
        if xs.size(1) != subsampling*labels.size(-1):
            gap = subsampling*labels.size(-1) - xs.size(1)
            xs = torch.nn.functional.pad(xs.permute(0,2,1),(0,gap))
            xs = xs.permute(0,2,1) # (B,T,80), T=subsampling*label.size(-1)
        num_frames = xs.size(1)

        stride= subsampling * decoding_chunk_size
        
        for cur in range(0, num_frames, stride):
            end = min(cur + stride, num_frames)
            #print(f"cur: {}")
            #end = min(cur+stride, num_frames)
            print(f"cur: {cur}, end: {end}")
            chunk_xs = xs[:, cur:end, :]
            print(f"chunk_xs shape: {chunk_xs.shape}")
            end_subframe= math.ceil(end/4)
            cur_subframe=math.ceil(cur/4)
            chunk_labels = labels[:,:,cur_subframe:end_subframe]
            # y shape: (B,4,chunk_size+1)
            (y, all_s_att_cache, m_att_cache) = self.forward_chunk(
                chunk_xs,
                target_speech,
                chunk_labels,
                offset,
                required_cache_size,
                all_s_att_cache,
                m_att_cache,
            )
            offset += y.size(2)
            # Update the previous context information
            #if end < num_frames:
            #    prev_context = chunk_xs[:, -context+1:, :]
            #else:
            #    prev_context = None
            # Remove the context part from the output
            #y = y[:,:, :decoding_chunk_size]
            print(f"y shape: {y.shape}")
            outputs.append(y)
        ys = torch.cat(outputs, 2) # (B,4,T')
        print(f"ys shape: {ys.shape}")
        return ys

    def forward_chunk(
        self,
        chunk_xs: torch.Tensor,
        target_speech: torch.Tensor,
        chunk_labels: torch.Tensor,
        offset: int,
        required_cache_size: int,
        all_s_att_cache: torch.Tensor = torch.zeros(0, 0, 0, 0, 4),
        m_att_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
        att_mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Forward just one chunk

        Args:
            chunk_xs (torch.Tensor): chunk input, with shape (b=1, time, mel-dim),
                where `time == (chunk_size - 1) * subsample_rate + \
                        subsample.right_context + 1`
            target_speech(torch.Tensor), it is speaker embedding feature ,shape(B,4,D)

            offset (int): current offset in encoder output time stamp
            required_cache_size (int): cache size required for next chunk
                computation
                =0: actual cache size
                <0: means all history cache is required
            all_s_att_cache (torch.Tensor): cache tensor for KEY & VALUE in
                single_backend transformer attention, with shape
                (elayers, head, cache_t1, d_k * 2,self.max_num_speaker), where
                `head * d_k == hidden-dim` and
                `cache_t1 == chunk_size * num_decoding_left_chunks`.

            m_att_cache (torch.Tensor): cache tensor for KEY & VALUE in
                multi_backend transformer attention, with shape
                (elayers, head, cache_t1, d_k * 2), where
                `head * d_k == hidden-dim` and
                `cache_t1 == chunk_size * num_decoding_left_chunks`.

        Returns:
            torch.Tensor: output of current input xs,
                with shape (b=1, chunk_size, hidden-dim).
            torch.Tensor: new attention cache required for next chunk, with
                dynamic shape (elayers, head, ?, d_k * 2,self.max_num_speaker)
                for singl_backend transformer network
                depending on required_cache_size.
            torch.Tensor: new attention cache required for next chunk, with
                dynamic shape (elayers, head, ?, d_k * 2)
                for multi_backend transformer network
                depending on required_cache_size.



        """
        assert chunk_xs.size(0) == 1
        # tmp_masks is just for embed network interface compatibility
        tmp_masks = torch.ones(
            1, chunk_xs.size(1), device=chunk_xs.device, dtype=torch.bool
        )
        tmp_masks = tmp_masks.unsqueeze(1)

        # NOTE(xcsong): Before embed, shape(chunk_xs) is (b=1, time, mel-dim)
        chunk_xs, _ = self.embed(chunk_xs, tmp_masks, chunk_labels)
        # NOTE(xcsong): After  embed, shape(chunk_xs) is (b=1, chunk_size, hidden-dim)
        elayers, cache_t1 = all_s_att_cache.size(0), all_s_att_cache.size(2)
        chunk_size = chunk_xs.size(1)
        attention_key_size = cache_t1 + chunk_size
        # pos_emb = self.embed.position_encoding(
        #    offset=offset - cache_t1, size=attention_key_size
        # )

        if required_cache_size < 0:
            next_cache_start = 0
        elif required_cache_size == 0:
            next_cache_start = attention_key_size
        else:
            next_cache_start = max(attention_key_size - required_cache_size, 0)

        chunk_xs, l_all_s_att_cache, l_m_att_cache = self.forward_chunk_layer(
            chunk_xs,
            target_speech,
            offset,
            cache_t1,
            all_s_att_cache,
            m_att_cache,
            att_mask,
            next_cache_start,
            elayers,
        )

        return (chunk_xs, l_all_s_att_cache, l_m_att_cache)

    def forward_chunk_layer(
        self,
        chunk_xs: torch.Tensor,
        target_speech: torch.Tensor,
        offset: int,
        cache_t1: int,
        all_s_att_cache: torch.Tensor,
        m_att_cache: torch.Tensor,
        att_mask: torch.Tensor,
        next_cache_start: int,
        elayers: int,
    ):
        """
        Forward one chunk for single_backend and multi_backend network.

        Args:
           chunk_xs(torch.Tensor), it is output of embed network(warp CAM++ speech encoder and 1D conv downsample layer), shape(B,T',D), T'~=T/subsample_rate
           target_speech(torch.Tensor), it is speaker embedding feature ,shape(B,4,D)
           offset
        """
        # target speech
        ts_embeds = self.rs_dropout(target_speech)  # B, 4, D

        # combine target embedding and mix embedding
        # Extend ts_embeds for time alignemnt
        ts_embeds = ts_embeds.unsqueeze(2)  # B, 4, 1, D
        # repeat T to cat mix frame-level information
        ts_embeds = ts_embeds.repeat(1, 1, chunk_xs.shape[1], 1)  # B, 4, T', D
        B, _, T, _ = ts_embeds.shape

        # Transformer for single speaker
        # assume self.max_num_speaker==4,
        cat_embeds = []
        l_all_s_att_cache = []
        for j in range(self.max_num_speaker):
            ts_embed = ts_embeds[:, j, :, :]  # B,T',D
            cat_embed = torch.cat((ts_embed, chunk_xs), 2)  # B,T',2D

            cat_embed, _ = self.pos_encoder(
                cat_embed, offset=offset - cache_t1
            )  # (B,T',2D)

            s_att_cache = []
            for i, s_layer in enumerate(self.single_backend):
                # NOTE(MADUO): Before layer.forward
                #   shape(all_s_att_cache[...,j]) is (num_layers, head, cache_t1, d_k * 2),
                #   shape(all_s_att_cache[...,j][i : i + 1,:,:,:]) is (1,head, cache_t1, d_k*2)
                if elayers == 0:
                    print(f"all_s_att_cache: {all_s_att_cache.shape} in 0 layer")
                    #NOTE(MADUO)
                    # >>> import torch
                    # >>> a = torch.zeros((0,0,0,0,4))
                    # >>> a[...,0]
                    # tensor([], size=(0, 0, 0, 0))
                    s_kv_cache = (all_s_att_cache[..., j], all_s_att_cache[..., j])
                else:
                    print(f"all_s_att_cache: {all_s_att_cache.shape} in not 0 layer")
                    i_kv_cache = all_s_att_cache[..., j][
                        i : i + 1]  # (1,head,cache_t1, dk*2) [1, 4, 17, 192]
                    print(f"i_kv_cache shape: {i_kv_cache.shape}")
                    size = int(all_s_att_cache.size(-2) / 2)
                    s_kv_cache = (
                        i_kv_cache[:, :, :, :size],
                        i_kv_cache[:, :, :, size:],
                    )  # i_kv_cache[:, :, :, :size] shape: (1,head,cache_t1, dk)
                print(f"s_kv_cache[0] shape: {s_kv_cache[0].shape}, s_kv_cache[1] shape: {s_kv_cache[1].shape}")
                cat_embed, _, new_s_kv_cache = s_layer(
                    cat_embed, att_mask, att_cache=s_kv_cache
                )  # (B,T',F) ,F=2D
                # stack d_k of k and v
                new_s_att_cache = torch.cat(new_s_kv_cache, dim=-1)  # k and v cat
                print(f"new_s_att_cache: {new_s_att_cache.shape}")
                # NOTE(MADUO): After layer.forward
                #   shape(new_s_att_cache) is (1, head, attention_key_size, d_k * 2),
                s_att_cache.append(new_s_att_cache[:, :, next_cache_start:, :])
                # stack num_layers
            s_att_cache = torch.cat(
                s_att_cache, dim=0
            )  # (num_layers,head,attention_key_size, d_k*2)
            #print(f"l_all_s_att_cache: {l_all_s_att_cache}")
            l_all_s_att_cache.append(s_att_cache)

            cat_embeds.append(cat_embed)
        # stack transformer for every speaker
        print(f"l_all_s_att_cache len : {len(l_all_s_att_cache)}")
        print(f"l_all_s_att_cache[0] shape: {l_all_s_att_cache[0].shape}")
        l_all_s_att_cache = torch.stack(
            l_all_s_att_cache, dim=-1
        )  # (num_layers,head,attention_key_size, d_k*2, 4)
        print(f"l_all_s_att_cache shape: {l_all_s_att_cache.shape}")
        cat_embeds = torch.stack(cat_embeds)  # 4, B, chunk_size+1, F
        print(f"single_backend output shape: {cat_embeds.shape}") #
        cat_embeds = torch.permute(cat_embeds, (1, 0, 3, 2))  # B,4,F,T'
        # Combine the outputs
        cat_embeds = cat_embeds.reshape(B, -1, T)  # B, 4 * F, T'

        # cat multi forward
        B, _, T = cat_embeds.size()
        print(f"cat_embeds shape {cat_embeds.shape}")
        # project layer, currently setting conv1d (kernel=5, strid=1, padding=2), it no downsample and just a dimension change
        cat_embeds = self.backend_down(cat_embeds)  # (B, 4 * F, T')->(B, F, T')

        # Transformer for multiple speakers
        # cat_embeds = self.pos_encoder(torch.permute(cat_embeds, (2, 0, 1)))

        # (MADUO) TODO check offset.
        #cat_embeds, _ = self.pos_encoder(
        #    torch.permute(cat_embeds, (0, 2, 1)), offset=offset - cache_t1
        #)
        # (todo)
        cat_embeds = torch.permute(cat_embeds,(0,2,1))
        l_m_att_cache = []
        for h, m_layer in enumerate(self.multi_backend):
            # NOTE(MADU) Before layer.forward
            # shape(m_att_cache[h:h+1,:,:,:]) is (1,head,cache_t1,d_k*2)
            if elayers == 0:
                m_kv_cache = (m_att_cache, m_att_cache)
            else:
                h_kv_cache = m_att_cache[h : h + 1, :, :, :]
                size = m_att_cache.size(-1) // 2  # size is d_k
                m_kv_cache = (h_kv_cache[:, :, :, :size], h_kv_cache[:, :, :, size:])
            # expect input cat_embeds shape: (B,chunk_size+1, transformer_embed_dim)
            cat_embeds, _, new_m_kv_cache = m_layer(
                cat_embeds, att_mask, att_cache=m_kv_cache,
            )
            new_m_kv_cache = torch.cat(new_m_kv_cache, dim=-1)
            logger.debug(f"new_m_kv_cache: {new_m_kv_cache.shape}")
            # NOTE(MADUO): After layer.forward
            # shape(new_m_kv_cache) is (1,head,attention_key_size, d_k * 2)
            l_m_att_cache.append(new_m_kv_cache[:, :, next_cache_start:, :])
        # stack num_layers
        # NOTE(MADUO): shape(l_m_att_cache) is (num_layers, head, ?, d_k * 2),
        #   ? may be larger than cache_t1, it depends on required_cache_size
        l_m_att_cache = torch.cat(l_m_att_cache, dim=0)
        cat_embeds = self.fc(cat_embeds)  # (B,T',F) ->  (B, T', 4)
        cat_embeds = cat_embeds.transpose(1, 2)  # (B,T', 4) ->(B,4,T')
        return cat_embeds, l_all_s_att_cache, l_m_att_cache

    def infer(
        self,
        ref_speech: torch.Tensor,
        ref_speech_len: torch.Tensor, # it is only for mask pad in training stage.
        target_speech: torch.Tensor,
        labels: torch.Tensor,
        labels_len: torch.Tensor,
        file_path=None,
        speaker_ids=None,
        start=None,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
        # inference:bool=True,
    ):
        if simulate_streaming and decoding_chunk_size > 0:
            # The 'masks' is fake mask
            outs = self.forward_chunk_by_chunk(
                xs=ref_speech,
                target_speech=target_speech,
                labels=labels,
                decoding_chunk_size=decoding_chunk_size,
                num_decoding_left_chunks=num_decoding_left_chunks,
            )
        else:
            # The 'masks' here just indicates the invalid part of the sentence with unequal pad lengths.
            outs = self.forward(
                xs=ref_speech,
                xs_lens=ref_speech_len,
                target_speech=target_speech,
                labels = labels,
                decoding_chunk_size=decoding_chunk_size,
                num_decoding_left_chunks=num_decoding_left_chunks,
            )

        print(f"outs shape: {outs.shape}")
        if outs.size(-1) < labels.size(-1):
            labels = labels[:,:,:outs.size(-1)]
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
    def infer_debug(self,
            ref_speech: torch.Tensor,
            ref_speech_len: torch.Tensor, # it is only for mask pad in training stage.
            target_speech: torch.Tensor,
            labels: torch.Tensor,
            labels_len: torch.Tensor,
            file_path=None,
            speaker_ids=None,
            start=None,
            decoding_chunk_size: int = 0,
            num_decoding_left_chunks: int = -1,
            simulate_streaming: bool = False,
        ):
        with torch.set_grad_enabled(False):
            if simulate_streaming and decoding_chunk_size > 0:
                # The 'masks' is fake mask
                #  it contains version1,2,3
                #outs = self.forward_chunk_by_chunk_temp(
                outs = self.forward_chunk_by_chunk_temp1(
                    xs=ref_speech,
                    target_speech=target_speech,
                    labels=labels,
                    decoding_chunk_size=decoding_chunk_size,
                    num_decoding_left_chunks=num_decoding_left_chunks,
                )
            else:
                outs = self.forward(
                    xs=ref_speech,
                    xs_lens=ref_speech_len,
                    target_speech=target_speech,
                    labels=labels,
                    decoding_chunk_size=decoding_chunk_size,
                    num_decoding_left_chunks=num_decoding_left_chunks,
                )
            #print(f"outs shape: {outs.shape} in fn infer_debug") #  torch.Size([1, 4, 100])
            if outs.size(-1) < labels.size(-1):
                gap = labels.size(-1) - outs.size(-1)
                outs = torch.nn.functional.pad(outs,(0,gap))
            #print(f"pad outs shape: {outs.shape} in fn infer_debug") #  torch.Size([1, 4, 100])
            loss = calculate_loss(outs=outs, labels=labels, labels_len=labels_len)
            print(f"loss is {loss} in fn infer_debug")
            ## public logger
            outs_prob = torch.nn.functional.sigmoid(outs)
            # convert tensor to numpy
            # logging.info(f"outs_prob requries_grad: {outs_prob.requries_grad}")
            outs_prob = outs_prob.data.cpu().numpy()
            mi, fa, cf, acc, der = self.calc_diarization_result(
                # mi, fa, cf, acc, der = model.calc_diarization_result(
                outs_prob.transpose((0, 2, 1)), # (B,T',4)
                labels.transpose(1, 2), #(B,T',4)
                labels_len,
            )

        assert loss.requires_grad == False
        info = {}
        info["loss"] = loss.detach().cpu().item()
        info["DER"] = der
        info["ACC"] = acc
        info["MI"] = mi
        info["FA"] = fa
        info["CF"] = cf

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
        return info, res_dict
    #def calculate_loss(outs, labels, labels_len):
    #total_loss = 0
    #for i in range(labels_len.size(0)):
    #    total_loss += F.binary_cross_entropy_with_logits(
    #        outs[i, :, : labels_len[i]], labels[i, :, : labels_len[i]]
    #    )
    #return total_loss / labels_len.size(0)

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


class PositionalEncoding(torch.nn.Module):
    """Positional encoding.

    :param int d_model: embedding dim
    :param float dropout_rate: dropout rate
    :param int max_len: maximum input length

    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
    """

    def __init__(
        self,
        d_model: int,
        dropout_rate: float,
        max_len: int = 5000,
        reverse: bool = False,
    ):
        """Construct an PositionalEncoding object."""
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.max_len = max_len

        pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(
        self, x: torch.Tensor, offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, ...)
            offset (int, torch.tensor): position offset

        Returns:
            torch.Tensor: Encoded tensor. Its shape is (batch, time, ...)
            torch.Tensor: for compatibility to RelPositionalEncoding
        """

        pos_emb = self.position_encoding(offset, x.size(1), False)
        #print(f"x * self.xscale: {x * self.xscale.shape}")
        x = x * self.xscale
        #print(f"x shape: {x.shape}, pos_emb shape: {pos_emb.shape}") # x shape: torch.Size([64, 100, 384]), pos_emb shape: torch.Size([1, 100, 384])
        x = x + pos_emb
        return self.dropout(x), self.dropout(pos_emb)

    def position_encoding(
        self, offset: Union[int, torch.Tensor], size: int, apply_dropout: bool = True
    ) -> torch.Tensor:
        """For getting encoding in a streaming fashion

        Attention!!!!!
        we apply dropout only once at the whole utterance level in a none
        streaming way, but will call this function several times with
        increasing input size in a streaming scenario, so the dropout will
        be applied several times.

        Args:
            offset (int or torch.tensor): start offset
            size (int): required size of position encoding

        Returns:
            torch.Tensor: Corresponding encoding
        """
        # How to subscript a Union type:
        #   https://github.com/pytorch/pytorch/issues/69434
        if isinstance(offset, int):
            assert offset + size <= self.max_len
            pos_emb = self.pe[:, offset : offset + size]
        elif isinstance(offset, torch.Tensor) and offset.dim() == 0:  # scalar
            assert offset + size <= self.max_len
            pos_emb = self.pe[:, offset : offset + size]
        else:  # for batched streaming decoding on GPU
            assert torch.max(offset) + size <= self.max_len
            index = offset.unsqueeze(1) + torch.arange(0, size).to(
                offset.device
            )  # B X T
            flag = index > 0
            # remove negative offset
            index = index * flag
            pos_emb = F.embedding(index, self.pe[0])  # B X T X d_model

        if apply_dropout:
            pos_emb = self.dropout(pos_emb)
        return pos_emb


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


class Subsampling4(nn.Module):
    """
    warp CAM++ and 1D conv subsampling (to 1/4 length).
    it is not postion encoding operation.
    """

    def __init__(self, speech_encoder_path, idim: int = 80, odim: int = 192, device=torch.device("cpu")):
        super(Subsampling4, self).__init__()
        self.speech_encoder = CAMPPlus(feat_dim=idim, embedding_size=odim)
        self.speech_encoder.train()
        self.load_speaker_encoder(
            speech_encoder_path, device=device, module_name="speech_encoder"
        )
        pretrain_speech_encoder_dim = (
            512  # it is feature dimension before pool layer of cam++ network
        )
        self.speech_down_or_up = nn.Sequential(
            nn.Conv1d(
                pretrain_speech_encoder_dim,
                odim,
                5,
                stride=2,
                padding=2,
            ),
            BatchNorm1D(num_features=odim),
            #torch.nn.BatchNorm1d(num_features=odim),
            nn.ReLU(),
        )
        # (TODO) check the number of right context
        self.subsampling_rate = 4
        self.right_context = 6

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor, labels: torch.Tensor, offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time). it is generated by fn ~make_pad_mask()
            labels (torch.Tensor): Input tensor(#batch,4, time//4)
        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.
        """

        #max_len = labels.size(-1)
        #print(f"all_max_len : {all_max_len}")
        # its input fbank feature(80-dim)
        #print(f"input x shape: {x.shape}") # (64,998,80)
        x = self.speech_encoder(x, get_time_out=True)
        # print(f"x shape: {x.shape}") # B,F',T'
        x = self.speech_down_or_up(x)
        #print(f"x shape {x.shape}") #(64,192,250)

        #### label offset and match x
        #size = x.size(-1)
        #print(f"offset: {offset}")
        #if offset==250:
        #    labels = labels[:,:,offset-size: offset]
        #elif
        #    labels = labels[:,:,offset:offset+size]
        #if offset+size <=250: # 250 means that 10s have 250 frames
        #    labels = labels[:,:,offset:offset+size]
        max_len = labels.size(-1)
        #print(f"x.size(-1) : {x.size(-1)}, max_len: {max_len}") #x.size(-1) : 100, max_len: 100
        assert (
            x.size(-1) - max_len <= 2 and x.size(-1) - max_len >= -1
        ), f"label and ref_speech(mix speech) diff: {x.size(-1)-max_len}"
        if x.size(-1) - max_len == -1:
            x = nn.functinal.pad(x, (0, 1))
        x = x[:, :, :max_len]  # (B,D,T)
        #print(f"x shape {x.shape} after pad") # x shape torch.Size([64, 192, 100]) after pad
        x = x.transpose(1, 2)  # (B,T,D)
        x_mask = x_mask[:, :, 2::2][:, :, 2::2] # (64,1,248)
        if x_mask.size(-1)!= x.size(-2): # x.size(-2) > x_mask.size(-1)
            gap = x.size(-2) - x_mask.size(-1)
            x_mask = torch.nn.functional.pad(x_mask,(0,gap))
        return x, x_mask
    def forward_test(self,x):
        print(f"cam++ input x shape: {x.shape}")
        x = self.speech_encoder(x, get_time_out=True)
        print(f"cam++ output x shape: {x.shape}") # B,F',T'
        x = self.speech_down_or_up(x)
        print(f"downsample output shape {x.shape}") #(64,192,250)
        return x
#    def offset_layer(self,offset: Union[int, torch.Tensor],size: int) -> torch.Tensor:
#        # How to subscript a Union type:
#        #   https://github.com/pytorch/pytorch/issues/69434
#        if isinstance(offset, int):
#            assert offset + size <= self.max_len
#            pos_emb = self.[:, offset:offset + size]
#        elif isinstance(offset, torch.Tensor) and offset.dim() == 0:  # scalar
#            assert offset + size <= self.max_len
#            pos_emb = self.pe[:, offset:offset + size]
#        return pos_emb

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


if __name__ == "__main__":
    speech_encoder_path="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin"
    subsample_model=Subsampling4(speech_encoder_path, idim=80, odim=192)

    for i in range(900,1000,1):
        #print(i)
        inp=torch.randn(1,i,80)
        out = subsample_model.forward_test(inp)
        #print(f"out: {out}")

    #inp=torch.randn(1,103,80)
    #out = subsample_model.forward_test(inp)


