# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, field
import sys
import contextlib
from typing import Optional, Tuple
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
from transformer_chunk_streaming import TransformerEncoderLayer
from mask import make_pad_mask
from mask import mask_to_bias
from mask import add_optional_chunk_mask

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

    static_chunk_size: int = 0

    use_dynamic_chunk: bool = False

    use_dynamic_left_chunk: bool = False

    gradient_checkpointing: bool = False

    use_sdpa: bool = False

    max_num_speaker: int = 4


model_cfg = TSVADConfig()
from datasets import TSVADDataConfig

data_cfg = TSVADDataConfig()


class TSVADModel(nn.Module):
    def __init__(
        self, cfg=model_cfg, task_cfg=data_cfg, device=torch.device("cpu")
    ) -> None:
        super(TSVADModel, self).__init__()
        self.gradient_checkpointing = False
        # self.freeze_speech_encoder_updates = cfg.freeze_speech_encoder_updates
        self.use_spk_embed = cfg.use_spk_embed
        self.rs_dropout = nn.Dropout(p=cfg.dropout)  # only for target speaker embedding

        self.speech_encoder_type = cfg.speech_encoder_type
        self.speech_encoder_path = cfg.speech_encoder_path
        # warp cam++ and 1d conv downsample
        self.embed = Subsampling4(
            speech_encoder_path=cfg.speech_encoder_path,
            idim=80,  # fbank feature dimention
            odim=cfg.speaker_embed_dim,
        )

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
                )
                for _ in range(cfg.num_transformer_layer)
            ]
        )
        self.pos_encoder = PositionalEncoding(cfg.transformer_embed_dim, cfg.dropout)
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
        xs_len: torch.Tensor,
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
            xs, masks
        )  # xs: (B,T,D), mask: (B, 1, T/subsample_rate), D is speaker embedding dimension
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
        if self.use_sdpa:
            chunk_masks = mask_to_bias(chunk_masks, xs.dtype)
        if self.gradient_checkpointing and self.training:
            xs = self.forward_layers_checkpointed(xs, chunk_masks)
        else:
            xs = self.forward_layers(xs, chunk_masks)
        # Here we assume the mask is not changed in encoder layers, so just
        # return the masks before encoder layers, and the masks will be used
        # for cross attention with decoder later
        # (TODO) check 'masks'
        return xs, masks
        pass

    def forward_layers(
        self,
        xs: torch.Tensor,
        chunk_masks: torch.Tensor,
        target_speech: torch.Tensor,
        labels: torch.Tensor,
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
        ts_embeds = ts_embeds.repeat(1, 1, mix_embeds.shape[1], 1)  # B, 4, T', D
        B, _, T, _ = ts_embeds.shape

        # Transformer for single speaker
        # assume self.max_num_speaker==4,
        cat_embeds = []
        for i in range(self.max_num_speaker):
            ts_embed = ts_embeds[:, i, :, :]  # B,T',D
            cat_embed = torch.cat((ts_embed, mix_embeds), 2)  # B,T',2D
            # cat_embed = cat_embed.transpose(0, 1)  # T, B, 2D

            cat_embed = self.pos_encoder(cat_embed)  # (B,T',2D)
            for single_layer in self.single_backend:
                cat_embed, chunk_masks, _, _ = single_layer(
                    cat_embed, chunk_masks
                )  # (B,T',F) ,F=2D
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
        cat_embeds = self.pos_encoder(torch.permute(cat_embeds, (0, 2, 1)))
        for multi_layer in self.multi_backend:
            cat_embed, chunk_masks, _, _ = multi_layer(
                cat_embed, chunk_masks
            )  # (B,T',F)
        outs = self.fc(cat_embeds)  # B T' 4
        outs = outs.transpose(1, 2)  # B,4,T'
        return outs

    @torch.jit.unused
    def forward_layers_checkpointed(
        self,
        xs: torch.Tensor,
        chunk_masks: torch.Tensor,
        target_speech: torch.Tensor,
        labels: torch.Tensor,
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
        ts_embeds = ts_embeds.repeat(1, 1, mix_embeds.shape[1], 1)  # B, 4, T', D
        B, _, T, _ = ts_embeds.shape

        # Transformer for single speaker
        # assume self.max_num_speaker==4,
        cat_embeds = []
        for i in range(self.max_num_speaker):
            ts_embed = ts_embeds[:, i, :, :]  # B,T',D
            cat_embed = torch.cat((ts_embed, mix_embeds), 2)  # B,T',2D
            # cat_embed = cat_embed.transpose(0, 1)  # T, B, 2D

            cat_embed = self.pos_encoder(cat_embed)  # (B,T',2D)
            for single_layer in self.single_backend:
                cat_embed, chunk_masks, _, _ = checkpoint.checkpoint(
                    single_layer.__call__, cat_embed, chunk_masks, use_reentrant=False
                )  # (B,T',F)
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
        cat_embeds = self.pos_encoder(torch.permute(cat_embeds, (0, 2, 1)))
        for multi_layer in self.multi_backend:
            cat_embed, chunk_masks, _, _ = checkpoint.checkpoint(
                multi_layer.__call__, cat_embed, chunk_masks, use_reentrant=False
            )  # (B,T',F)
        outs = self.fc(cat_embeds)  # B T' 4
        outs = outs.transpose(1, 2)  # B,4,T'
        return outs

    def forward_chunk_by_chunk(
        self,
        xs: torch.Tensor,
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
        context = self.embed.right_context + 1  # Add current frame
        stride = subsampling * decoding_chunk_size
        decoding_window = (decoding_chunk_size - 1) * subsampling + context
        num_frames = xs.size(1)
        att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0), device=xs.device)

        outputs = []
        offset = 0
        required_cache_size = decoding_chunk_size * num_decoding_left_chunks

        # Feed forward overlap input step by step
        for cur in range(0, num_frames - context + 1, stride):
            end = min(cur + decoding_window, num_frames)
            chunk_xs = xs[:, cur:end, :]

            (y, att_cache) = self.forward_chunk(
                chunk_xs, offset, required_cache_size, att_cache
            )

            outputs.append(y)
            offset += y.size(1)
        ys = torch.cat(outputs, 1)
        masks = torch.ones((1, 1, ys.size(1)), device=ys.device, dtype=torch.bool)
        return ys, masks

    def forward_chunk(
        self,
        chunk_xs: torch.Tensor,
        offset: int,
        required_cache_size: int,
        att_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
        att_mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Forward just one chunk

        Args:
            xs (torch.Tensor): chunk input, with shape (b=1, time, mel-dim),
                where `time == (chunk_size - 1) * subsample_rate + \
                        subsample.right_context + 1`
            offset (int): current offset in encoder output time stamp
            required_cache_size (int): cache size required for next chunk
                computation
                >=0: actual cache size
                <0: means all history cache is required
            att_cache (torch.Tensor): cache tensor for KEY & VALUE in
                transformer attention, with shape
                (elayers, head, cache_t1, d_k * 2), where
                `head * d_k == hidden-dim` and
                `cache_t1 == chunk_size * num_decoding_left_chunks`.

        Returns:
            torch.Tensor: output of current input xs,
                with shape (b=1, chunk_size, hidden-dim).
            torch.Tensor: new attention cache required for next chunk, with
                dynamic shape (elayers, head, ?, d_k * 2)
                depending on required_cache_size.
            torch.Tensor: new conformer cnn cache required for next chunk, with
                same shape as the original cnn_cache.

        """
        assert xs.size(0) == 1
        # tmp_masks is just for embed network interface compatibility
        tmp_masks = torch.ones(1, xs.size(1), device=xs.device, dtype=torch.bool)
        tmp_masks = tmp_masks.unsqueeze(1)
        
        # NOTE(xcsong): Before embed, shape(xs) is (b=1, time, mel-dim)
        xs, _ = self.embed(xs, tmp_masks, offset)
        # NOTE(xcsong): After  embed, shape(xs) is (b=1, chunk_size, hidden-dim)
        elayers, cache_t1 = att_cache.size(0), att_cache.size(2)
        chunk_size = xs.size(1)
        attention_key_size = cache_t1 + chunk_size
        pos_emb = self.embed.position_encoding(
            offset=offset - cache_t1, size=attention_key_size
        )
        if required_cache_size < 0:
            next_cache_start = 0
        elif required_cache_size == 0:
            next_cache_start = attention_key_size
        else:
            next_cache_start = max(attention_key_size - required_cache_size, 0)
        r_att_cache = []
    

        for i, layer in enumerate(self.encoders):
            # NOTE(xcsong): Before layer.forward
            #   shape(att_cache[i:i + 1]) is (1, head, cache_t1, d_k * 2),
            #   shape(cnn_cache[i])       is (b=1, hidden-dim, cache_t2)
            if elayers == 0:
                kv_cache = (att_cache, att_cache)
            else:
                i_kv_cache = att_cache[i : i + 1]
                size = att_cache.size(-1) // 2
                kv_cache = (i_kv_cache[:, :, :, :size], i_kv_cache[:, :, :, size:])
            xs, _, new_kv_cache, new_cnn_cache = layer(
                xs,
                att_mask,
                pos_emb,
                att_cache=kv_cache,
                cnn_cache=cnn_cache[i] if cnn_cache.size(0) > 0 else cnn_cache,
            )
            new_att_cache = torch.cat(new_kv_cache, dim=-1)
            # NOTE(xcsong): After layer.forward
            #   shape(new_att_cache) is (1, head, attention_key_size, d_k * 2),
            #   shape(new_cnn_cache) is (b=1, hidden-dim, cache_t2)
            r_att_cache.append(new_att_cache[:, :, next_cache_start:, :])
            r_cnn_cache.append(new_cnn_cache.unsqueeze(0))
        if self.normalize_before:
            xs = self.after_norm(xs)

        # NOTE(xcsong): shape(r_att_cache) is (elayers, head, ?, d_k * 2),
        #   ? may be larger than cache_t1, it depends on required_cache_size
        r_att_cache = torch.cat(r_att_cache, dim=0)
        # NOTE(xcsong): shape(r_cnn_cache) is (e, b=1, hidden-dim, cache_t2)
        r_cnn_cache = torch.cat(r_cnn_cache, dim=0)

        return (xs, r_att_cache, r_cnn_cache)

        pass


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
        x = x * self.xscale + pos_emb
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

    def __init__(self, speech_encoder_path, idim: int = 80, odim: int = 192):
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
                speaker_embed_dim,
                5,
                stride=2,
                padding=2,
            ),
            BatchNorm1D(num_features=speaker_embed_dim),
            nn.ReLU(),
        )
        # (TODO) check the number of right context
        self.subsampling_rate = 4
        self.right_context = 6

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time). it is generated by fn ~make_pad_mask()
            labels (torch.Tensor): Input tensor(#batch,time//4)
        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.
        """
        max_len = labels.size(-1)
        # its input fbank feature(80-dim)
        x = self.speech_encoder(x, get_time_out=True)
        # print(f"x shape: {x.shape}") # B,F',T'
        x = self.speech_down_or_up(x)
        assert (
            x.size(-1) - max_len <= 2 and x.size(-1) - max_len >= -1
        ), f"label and ref_speech(mix speech) diff: {x.size(-1)-max_len}"
        if x.size(-1) - max_len == -1:
            x = nn.functinal.pad(x, (0, 1))
        x = x[:, :, :max_len]  # (B,D,T)
        x = x.transpose(1, 2)  # (B,T,D)
        x_mask = x_mask[:, :, 2::2][:, :, 2::2]
        return x, x_mask

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
