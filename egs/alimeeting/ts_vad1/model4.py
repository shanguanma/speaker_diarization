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
#from cam_pplus_wespeaker1 import CAMPPlus
from cam_pplus_wespeaker_fairseq import CAMPPlus
logger = logging.getLogger(__name__)


@dataclass
class TSVADConfig:
    speech_encoder_path: str = (
        "/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt"
    )
    """path to pretrained speech encoder path."""

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

    use_lstm_single: bool = False
    """use lstm for single"""

    # use_lstm_multi: bool = False
    # """use lstm for multi"""

    use_spk_embed: bool = True
    """whether to use speaker embedding"""

    # add_ind_proj: bool = False
    # """whether to add projection for each speaker"""


model_cfg = TSVADConfig()


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
        self.use_spk_embed = cfg.use_spk_embed
        self.rs_dropout = nn.Dropout(p=cfg.dropout)
        self.label_rate = task_cfg.label_rate
        sample_times = 16000 / task_cfg.sample_rate
        self.max_num_speaker = task_cfg.max_num_speaker
        #self.speech_encoder.train()
        #self.load_speaker_encoder(cfg.speech_encoder_path,device=device, module_name="speech_encoder")
        #self.speech_encoder = CAMPPlus(feat_dim=80, embedding_size=512)
        self.speech_encoder = CAMPPlus(feat_dim=80, embedding_size=192,speech_encoder=True)
        self.speech_encoder.train()
        self.load_speaker_encoder(cfg.speech_encoder_path,device=device, module_name="speech_encoder")
        stride = int(2 // sample_times) if self.label_rate == 25 else 1
        self.speech_down = nn.Sequential(
            nn.Conv1d(512, cfg.speaker_embed_dim, 5, stride=stride, padding=2),

            BatchNorm1D(num_features=cfg.speaker_embed_dim),
            nn.ReLU(),
        )
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

        # self.m = nn.Sigmoid()
        self.speech_encoder_path = cfg.speech_encoder_path

    def forward(
        self,
        ref_speech: torch.Tensor,
        target_speech: torch.Tensor,
        labels: torch.Tensor,
        labels_len: torch.Tensor,
    ):
        B = ref_speech.size(0)
        T = ref_speech.size(1)
        max_len = labels.size(-1)

        #self.load_speaker_encoder(self.speech_encoder_path, device=ref_speech.device, module_name="speech_encoder")
        #print(f"ref_speech shape: {ref_speech.shape}") # B,T,F
        x = self.speech_encoder(ref_speech,get_time_out=True)
        #print(f"x shape: {x.shape}") # B,F',T'
        x = self.speech_down(x)
        assert (
            x.size(-1) - max_len <= 2 and x.size(-1) - max_len >= -1
        ), f"label and ref_speech(mix speech) diff: {x.size(-1)-max_len}"
        if x.size(-1) - max_len == -1:
            x = nn.functinal.pad(x, (0, 1))
        x = x[:, :, :max_len]  # (B,D,T)
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
        #loss = self.calculate_loss(outs, labels, labels_len)
        #logging.info(f"in the forward: loss: {loss}, outs: {outs}")
        #return loss, outs
        return outs

    # def foward(self,ref_speech:torch.Tensor,target_speech:torch.Tensor,labels:torch.Tensor):
    #    outs = self.forward_common(ref_speech=ref_speech,target_speech=target_speech,labels=labels)
    #    return outs
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

    #def calculate_loss(self, outs, labels, labels_len):
    #    total_loss = 0
    #    for i in range(labels_len.size(0)):
    #        total_loss += F.binary_cross_entropy_with_logits(
    #            outs[i, :, : labels_len[i]], labels[i, :, : labels_len[i]]
    #        )
    #    return total_loss / labels_len.size(0)
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
        #if not self.supports_gradient_checkpointing:
        #    raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")

        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {"use_reentrant": True}
        import functools
        gradient_checkpointing_func = functools.partial(checkpoint, **gradient_checkpointing_kwargs)

        # For old GC format (transformers < 4.35.0) for models that live on the Hub
        # we will fall back to the overwritten `_set_gradient_checkpointing` method
        #_is_using_old_format = "value" in inspect.signature(self._set_gradient_checkpointing).parameters

        #if not _is_using_old_format:
        self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=gradient_checkpointing_func)
    def _set_gradient_checkpointing(self, enable: bool = True, gradient_checkpointing_func: Callable = checkpoint):
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

    model = TSVADModel()
    print(model)
    x = torch.zeros(10, 398, 80) # B,T,F
    out = model.speech_encoder(x)
    print(f"out shape: {out.shape}")
    num_params = sum(param.numel() for param in model.parameters())
    for name, v in model.named_parameters():
        print(f"name: {name}, v: {v.shape}")
    print("{} M".format(num_params / 1e6)) # 6.61M
    data_cfg = TSVADDataConfig()
    print(vars(data_cfg))
