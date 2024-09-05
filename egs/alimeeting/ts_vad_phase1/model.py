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
from torch import Tensor
from cam_pplus_wespeaker import  CAMPPlus

logger = logging.getLogger(__name__)

@dataclass
class TSVADConfig:
    speech_encoder_path: str ="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/en_zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt"
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

    use_lstm_multi: bool = False
    """use lstm for multi"""

    use_spk_embed: bool = True
    """whether to use speaker embedding"""

    add_ind_proj: bool = False
    """whether to add projection for each speaker"""

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
#@register_model("ts_vad", dataclass=TSVADConfig)
class TSVADModel(nn.Module):
    def __init__(
        self,
        cfg=model_cfg,
        task_cfg=data_cfg,
        device=torch.device("cpu")
    ) -> None:
        super().__init__()
        # Speaker Encoder
        self.use_spk_embed = cfg.use_spk_embed
        self.rs_dropout = nn.Dropout(p=cfg.dropout)

        self.label_rate = task_cfg.label_rate
        assert task_cfg.speech_encoder_type == "cam++", f"task_cfg.speech_encoder_type: {task_cfg.speech_encoder_type}"
        assert task_cfg.label_rate == 25,f"cam++ label rate should be 25, but now is {task_cfg.speech_encoder_type}"
        # Speech Encoder
        self.speech_encoder_type = task_cfg.speech_encoder_type
        sample_times = 16000 / task_cfg.sample_rate
        if (
            self.speech_encoder_type == "cam++"
        ):  # its input is fbank , it is extract from data/ts_vad_dataset.py
            # self.speech_encoder = CAMPPlus(feat_dim=80,embedding_size=192)# embedding_size is from pretrain model embedding_size
            # self.speech_encoder = CAMPPlus(feat_dim=80,embedding_size=192,speech_encoder=True)
            self.speech_encoder = CAMPPlus(feat_dim=80, embedding_size=512)
            self.speech_encoder.train()
            self.load_speaker_encoder(
                cfg.speech_encoder_path,device=device, module_name="speech_encoder"
            )
            # input of cam++ model is fbank, means that 1s has 100 frames
            # we set target label rate is 25, means that 1s has 25 frames
            # cam++ model downsample scale is 2, so frame rate is cam++ model output is 50, so I should set stride equal to 2.
            stride = int(2 // sample_times) if task_cfg.label_rate == 25 else 1
            ## the input shape of self.speech_down except is (B,F,T)
            self.speech_down = nn.Sequential(
                # here 512 means it is feature dimension  before pool layer of cam++(it is also from wespeaker ) model.
                nn.Conv1d(512, cfg.speaker_embed_dim, 5, stride=stride, padding=2),
                BatchNorm1D(num_features=cfg.speaker_embed_dim),
                        nn.ReLU(),
            )

        # Projection
        if cfg.speaker_embed_dim * 2 != cfg.transformer_embed_dim:
            self.proj_layer = nn.Linear(
                cfg.speaker_embed_dim * 2, cfg.transformer_embed_dim
            )
        else:
            self.proj_layer = None

        # TS-VAD Backend
        if cfg.use_lstm_single:
            self.single_backend = nn.LSTM(
                input_size=cfg.transformer_embed_dim,
                hidden_size=cfg.transformer_embed_dim // 2,
                num_layers=cfg.num_transformer_layer,
                dropout=cfg.dropout,
                bidirectional=True,
            )
        else:
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
                cfg.transformer_embed_dim * task_cfg.max_num_speaker,
                cfg.transformer_embed_dim,
                5,
                stride=1,
                padding=2,
            ),
            BatchNorm1D(num_features=cfg.transformer_embed_dim),
            nn.ReLU(),
        )
        if cfg.use_lstm_multi:
            self.multi_backend = nn.LSTM(
                input_size=cfg.transformer_embed_dim,
                hidden_size=cfg.transformer_embed_dim // 2,
                num_layers=cfg.num_transformer_layer,
                dropout=cfg.dropout,
                bidirectional=True,
            )
        else:
            self.multi_backend = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=cfg.transformer_embed_dim,
                    dim_feedforward=cfg.transformer_ffn_embed_dim,
                    nhead=cfg.num_attention_head,
                    dropout=cfg.dropout,
                ),
                num_layers=cfg.num_transformer_layer,
            )
        # final projection
        self.add_ind_proj = cfg.add_ind_proj
        if cfg.add_ind_proj:
            self.pre_fc = nn.ModuleList(
                [
                    nn.Linear(cfg.transformer_embed_dim, 192)
                    for _ in range(task_cfg.max_num_speaker)
                ]
            )
            self.fc = nn.Linear(192, 1)
        else:
            self.fc = nn.Linear(cfg.transformer_embed_dim, task_cfg.max_num_speaker)

        self.loss = nn.BCEWithLogitsLoss()
        self.m = nn.Sigmoid()

        # others
        self.label_rate = task_cfg.label_rate
        self.freeze_speech_encoder_updates = cfg.freeze_speech_encoder_updates
        self.max_num_speaker = task_cfg.max_num_speaker
        self.embed_input = task_cfg.embed_input
        self.scale_factor = 0.04 / task_cfg.embed_shift
        self.use_lstm_single = cfg.use_lstm_single
        self.use_lstm_multi = cfg.use_lstm_multi

    # B: batchsize, T: number of frames (1 frame = 0.04s)
    # Obtain the reference speech represnetation(it should be mix speech representation)
    def rs_forward(self, x, max_len, fix_encoder=False):  # B, 25 * T
        B = x.size(0)
        T = x.size(1)
        if (
            self.speech_encoder_type == "cam++"
        ):  # its input is fbank, it is extract from ts_vad/ts_vad_dataset.py
            with torch.no_grad() if fix_encoder else contextlib.ExitStack():
                x = self.speech_encoder(x, get_time_out=True)
                #logging.info(f"fix_encoder is {fix_encoder}, x grad should be {x.requires_grad}")
            x = self.speech_down(x)
        assert (
            x.size(-1) - max_len <= 2 and x.size(-1) - max_len >= -1
        ), f"label and input diff: {x.size(-1) - max_len}"
        if x.size(-1) - max_len == -1:
            x = nn.functional.pad(x, (0, 1))
        x = x[:, :, :max_len]  # (B,D,T)
        x = x.transpose(1, 2)  # (B,T,D)

        return x

    # Obtain the target speaker represnetation(utterance level speaker embedding)
    def ts_forward(self, x):  # B, 4, 80, T * 100
        if self.use_spk_embed:
            return self.rs_dropout(x)
        B, _, D, T = x.shape
        x = x.view(B * self.max_num_speaker, D, T)
        x = self.speaker_encoder.forward(x)
        x = x.view(B, self.max_num_speaker, -1)  # B, 4, 192
        return x

    # Combine for ts-vad results
    def cat_single_forward(self, rs_embeds, ts_embeds):
        # Extend ts_embeds for time alignemnt
        ts_embeds = ts_embeds.unsqueeze(2)  # B, 4, 1, 256
        ts_embeds = ts_embeds.repeat(
            1, 1, rs_embeds.shape[1], 1
        )  # B, 4, T, 256  ## repeat T to cat mix frame-level information
        B, _, T, _ = ts_embeds.shape

        # Transformer for single speaker
        cat_embeds = []
        for i in range(self.max_num_speaker):
            ts_embed = ts_embeds[:, i, :, :]  # B, T, 256
            cat_embed = torch.cat(
                (ts_embed, rs_embeds), 2
            )  # B, T, 256 + B, T, 256 -> B, T, 512
            if self.proj_layer is not None:
                cat_embed = self.proj_layer(cat_embed)
            cat_embed = cat_embed.transpose(0, 1)  # T, B, 512
            cat_embed = self.pos_encoder(cat_embed)
            if self.use_lstm_single:
                cat_embed, _ = self.single_backend(cat_embed)  # T, B, 512
            else:
                cat_embed = self.single_backend(cat_embed)  # T, B, 512
            cat_embed = cat_embed.transpose(0, 1)  # B, T, 512
            cat_embeds.append(cat_embed)
        cat_embeds = torch.stack(cat_embeds)  # 4, B, T, 384
        cat_embeds = torch.permute(cat_embeds, (1, 0, 3, 2))  # B, 4, 384, T
        # Combine the outputs
        cat_embeds = cat_embeds.reshape(B, -1, T)  # B, 4 * 384, T
        return cat_embeds

    def cat_multi_forward(self, cat_embeds):
        B, _, T = cat_embeds.size()
        # Downsampling
        cat_embeds = self.backend_down(cat_embeds)  # B, 384, T
        # Transformer for multiple speakers
        cat_embeds = self.pos_encoder(torch.permute(cat_embeds, (2, 0, 1)))
        if self.use_lstm_multi:
            cat_embeds, _ = self.multi_backend(cat_embeds)  # T, B, 384
        else:
            cat_embeds = self.multi_backend(cat_embeds)  # T, B, 384

        cat_embeds = cat_embeds.transpose(0, 1)

        return cat_embeds

    def calculate_loss(self, outs, labels, labels_len):
        total_loss = 0

        for i in range(labels_len.size(0)):
            total_loss += self.loss(
                outs[i, :, : labels_len[i]], labels[i, :, : labels_len[i]]
            )

        outs_prob = self.m(outs)
        outs_prob = outs_prob.data.cpu().numpy()

        return total_loss / labels_len.size(0), outs_prob

    def forward(
        self,
        ref_speech: torch.Tensor,
        target_speech: torch.Tensor,
        labels: torch.Tensor,
        labels_len: torch.Tensor = None,
        file_path=None,
        speaker_ids=None,
        start=None,
        extract_features:bool=False,
        fix_encoder:bool=True,
        num_updates:int=0,
        inference: bool=False,
    ):
        rs_embeds = self.rs_forward(
            ref_speech,
            labels.size(-1),
            fix_encoder=(
                num_updates < self.freeze_speech_encoder_updates and fix_encoder
            ),
        )

        ts_embeds = self.ts_forward(target_speech)
        cat_embeds = self.cat_single_forward(rs_embeds, ts_embeds)
        outs_pre = self.cat_multi_forward(cat_embeds)

        if self.add_ind_proj:
            outs_pre = torch.stack(
                [self.pre_fc[i](outs_pre) for i in range(self.max_num_speaker)], dim=1
            )
            outs = (
                self.fc(outs_pre.view(-1, labels.size(-1), 192))
                .view(ref_speech.size(0), self.max_num_speaker, labels.size(-1))
                .transpose(1, 2)
            )
        else:
            outs = self.fc(outs_pre)  # B T 3

        outs = outs.transpose(1, 2)  # B 3 T

        loss, outs_prob = self.calculate_loss(outs, labels, labels_len)
        result = {"losses": {"diar": loss}}

        mi, fa, cf, acc, der = self.calc_diarization_result(
            outs_prob.transpose((0, 2, 1)), labels.transpose(1, 2), labels_len
        )
        result["labels_len"] = labels_len
        result["DER"] = der
        result["ACC"] = acc
        result["MI"] = mi
        result["FA"] = fa
        result["CF"] = cf

        if extract_features:
            # if self.add_ind_proj:
            return outs_pre.transpose(1, 2), result, outs_prob
            # else:
            #     return outs, mi, fa, cf, acc, der

        if inference:
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

        return result


    def load_speaker_encoder(self, model_path,device, module_name="speaker_encoder"):
        loadedState = torch.load(model_path, map_location=device)
        selfState = self.state_dict()
        for name, param in loadedState.items():
            origname = name

            if (
                module_name == "speaker_encoder"
                and hasattr(self.speaker_encoder, "bn1")
                and isinstance(self.speaker_encoder.bn1, BatchNorm1D)
                and ".".join(name.split(".")[:-1]) + ".running_mean" in loadedState
            ):
                name = ".".join(name.split(".")[:-1]) + ".bn." + name.split(".")[-1]

            if (
                module_name == "speech_encoder"
                and hasattr(self.speech_encoder, "bn1")
                and isinstance(self.speech_encoder.bn1, BatchNorm1D)
                and ".".join(name.split(".")[:-1]) + ".running_mean" in loadedState
            ):
                name = ".".join(name.split(".")[:-1]) + ".bn." + name.split(".")[-1]

            if name.startswith("speaker_encoder"):
                name = name.replace("speaker_encoder", module_name)
            else:
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


    @staticmethod
    def calc_diarization_error(pred, label, length):
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

    @staticmethod
    def calc_diarization_result(outs_prob, labels, labels_len):
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
        ) = TSVADModel.calc_diarization_error(outs_prob, labels, labels_len)

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



if __name__ == "__main__":
    """Build a new model instance."""

    model = TSVADModel()
    print(model)
    data_cfg = TSVADDataConfig()
    print(vars(data_cfg))
