#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor
from torch import nn

from typing import Iterable, Optional

import os
import hashlib
import whisper
import logging
import urllib.request
from dataclasses import dataclass

@dataclass
class ModelDimensions:
    n_mels: int=80
    n_audio_ctx: int=1500 # max_source_positions
    n_audio_state: int=1280 # d_model
    n_audio_head: int=20 # encoder_attention_heads
    n_audio_layer: int=24 # encoder_layers, I only instance first 24 layers of whisper.
    layer_st: int=16
    layer_ed: int=23

class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x, self.weight.to(
                x.dtype), None if self.bias is None else self.bias.to(
                x.dtype))


class Conv1d(nn.Conv1d):
    def _conv_forward(self, x: Tensor, weight: Tensor,
                      bias: Optional[Tensor]) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment *
                               torch.arange(channels // 2))
    scaled_time = torch.arange(
        length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None),
            # will prepend the cached kv tensors; otherwise,
            # perform key/value projections for self- or
            # cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once
            # and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(
            self,
            q: Tensor,
            k: Tensor,
            v: Tensor,
            mask: Optional[Tensor] = None):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int,
                 cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = MultiHeadAttention(
            n_state, n_head) if cross_attention else None
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(
                n_state, n_mlp), nn.GELU(), Linear(
                n_mlp, n_state))
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x),
                                    xa, kv_cache=kv_cache)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


class WhisperEncoder(nn.Module):
    def __init__(
            self,
            cfg: ModelDimensions,
            #n_mels: int=80,
            #n_state: int=1280, # d_model
            #n_head: int=20,
            #n_layer: int=24,
            #n_ctx: int=1500,
            #layer_st: int=16,
            #layer_ed: int=23
        ):
        """
        layer17-24(index:16-23) contain more speaker information.
        It comes from
        `Whisper-PMFA: Partial Multi-Scale Feature Aggregation for
        Speaker Verification using Whisper Models`

        the above finding is only in whisper-large-v2
        """
        super().__init__()
        self.n_mels = cfg.n_mels

        self.conv1 = Conv1d(cfg.n_mels, cfg.n_audio_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(
            cfg.n_audio_state,
            cfg.n_audio_state,
            kernel_size=3,
            stride=2,
            padding=1)
        self.register_buffer("positional_embedding", sinusoids(cfg.n_audio_ctx, cfg.n_audio_state))

        self.layers: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(cfg.n_audio_state, cfg.n_audio_head) for _ in range(cfg.n_audio_layer)]
        )

        # ------------------------ADD:add new layer norm------------------------
        self.ln_post2 = LayerNorm(cfg.n_audio_state * (cfg.layer_ed - cfg.layer_st + 1))
        #self.ln_post2 = LayerNorm(cfg.n_audio_state)

        self.layer_st = cfg.layer_st
        self.layer_ed = cfg.layer_ed

    def forward(self, wavs: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        ## wav-> log mel
        with torch.no_grad():
            processed_feats = []
            for i in range(wavs.size(0)):
                tf_tensor = wavs[i].unsqueeze(0).to(wavs.device)
                mat = whisper.log_mel_spectrogram(
                    tf_tensor.squeeze(), n_mels=self.n_mels)
                processed_feats.append(mat)

            x = torch.stack(processed_feats, dim=0).to(wavs.device) # (batch_size,n_mels,n_audio_ctx)

        # ---------------------------ADD------------------------
        # x:(batch_size,n_mels,n_ctx) -> (batch_size,n_ctx,n_mels)
        #print(f"before: x shape: {x.shape}")
        #x = x.permute(0, 2, 1)

        #x = x.squeeze(1)
        #print(f"x.squeeze(1): x.shape : {x.shape}")
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        # ------------Change:Tailor the positional_embedding----------
        assert x.shape[2:] == self.positional_embedding.shape[1:], \
            "incorrect audio shape"
        if self.positional_embedding.shape[0] > x.shape[1]:
            temp_positional_embedding = self.positional_embedding[:x.shape[1], :]
        elif self.positional_embedding.shape[0] < x.shape[1]:
            x = x[:, :self.positional_embedding.shape[0], :]
            temp_positional_embedding = self.positional_embedding
        else:
            temp_positional_embedding = self.positional_embedding

        x = (x + temp_positional_embedding).to(x.dtype)

        # ----------Change: Concat block outputs------
        out = []
        for i, block in enumerate(self.layers):
            x = block(x)
            if self.layer_st <= i <= self.layer_ed:
                out.append(x)

        xs = torch.cat(out, dim=-1)

        xs = self.ln_post2(xs)
        # (B,T,D), D=n_state * (layer_ed - layer_st + 1)=1280*(23-16+1)=10240
        return xs


if __name__ == "__main__":
   cfg=ModelDimensions()
   wav = torch.randn(2,16000)
   model = WhisperEncoder(cfg)
   out = model(wav)
   print(out.shape) #(2, 50, 10240)
   torch.save(model.state_dict(), f"./demo_whisper.pt")
   print(f"model : {model}")
