# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Modified by: Duo Ma
# Licensed under the MIT license.

import numpy as np
import math
import logging
from typing import List,Optional
from torch import Tensor
from itertools import permutations


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
from torchaudio.models import Conformer

from speaker_diarization.eend_eda.encoder_decoder_attractor import LstmEncoderDedecoderAttractor



class TransformerModel(nn.Module):
    def __init__(self, n_speakers, in_size, n_heads, n_units, n_layers, dim_feedforward=2048, dropout=0.5, has_pos=False):
        """ Self-attention-based diarization model.

        Args:
          n_speakers (int): Number of speakers in recording
          in_size (int): Dimension of input feature vector
          n_heads (int): Number of attention heads
          n_units (int): Number of units in a self-attention block
          n_layers (int): Number of transformer-encoder layers
          dropout (float): dropout ratio
        """
        super(TransformerModel, self).__init__()
        self.n_speakers = n_speakers
        self.in_size = in_size
        self.n_heads = n_heads
        self.n_units = n_units
        self.n_layers = n_layers
        self.has_pos = has_pos

        self.src_mask = None
        self.encoder = nn.Linear(in_size, n_units)
        self.encoder_norm = nn.LayerNorm(n_units)
        if self.has_pos:
            self.pos_encoder = PositionalEncoding(n_units, dropout)
        encoder_layers = TransformerEncoderLayer(n_units, n_heads, dim_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.decoder = nn.Linear(n_units, n_speakers)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.bias.data.zero_()
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, has_mask=False, activation=None):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != src.size(1):
                mask = self._generate_square_subsequent_mask(src.size(1)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        ilens = [x.shape[0] for x in src]
        src = nn.utils.rnn.pad_sequence(src, padding_value=-1, batch_first=True)

        # src: (B, T, E)
        src = self.encoder(src)
        src = self.encoder_norm(src)
        # src: (T, B, E)
        src = src.transpose(0, 1)
        if self.has_pos:
            # src: (T, B, E)
            src = self.pos_encoder(src)
        # output: (T, B, E)
        output = self.transformer_encoder(src, self.src_mask)
        # output: (B, T, E)
        output = output.transpose(0, 1)
        # output: (B, T, C)
        output = self.decoder(output)

        if activation:
            output = activation(output)

        output = [out[:ilen] for out, ilen in zip(output, ilens)]

        return output

    def get_attention_weight(self, src):
        # NOTE: NOT IMPLEMENTED CORRECTLY!!!
        attn_weight = []
        def hook(module, input, output):
            # attn_output, attn_output_weights = multihead_attn(query, key, value)
            # output[1] are the attention weights
            attn_weight.append(output[1])

        handles = []
        for l in range(self.n_layers):
            handles.append(self.transformer_encoder.layers[l].self_attn.register_forward_hook(hook))

        self.eval()
        with torch.no_grad():
            self.forward(src)

        for handle in handles:
            handle.remove()
        self.train()

        return torch.stack(attn_weight)


class PositionalEncoding(nn.Module):
    """Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional information to each time step of x
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerEdaModel(nn.Module):
    def __init__(self, n_speakers, in_size, n_heads, n_units, n_layers, dim_feedforward=2048, dropout=0.5, has_pos=False,diar_weight: float = 1.0,
        attractor_weight: float = 1.0,):
        """ Self-attention-based encoder decoder attractor diarization model.

        Args:
          n_speakers (int): Number of speakers in recording
          in_size (int): Dimension of input feature vector
          n_heads (int): Number of attention heads
          n_units (int): Number of units in a self-attention block
          n_layers (int): Number of transformer-encoder layers
          dropout (float): dropout ratio
        """
        super(TransformerEdaModel, self).__init__()
        self.n_speakers = n_speakers
        self.in_size = in_size
        self.n_heads = n_heads
        self.n_units = n_units
        self.n_layers = n_layers
        self.has_pos = has_pos

        self.src_mask = None
        self.encoder = nn.Linear(in_size, n_units)
        self.encoder_norm = nn.LayerNorm(n_units)
        if self.has_pos:
            self.pos_encoder = PositionalEncoding(n_units, dropout)

        ## why add batch_first=True, it can solve the below warning:
        # UserWarning: enable_nested_tensor is True,
        # but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first
        # was not True(use batch_first for better inference performance)
        # warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")

        encoder_layers = TransformerEncoderLayer(n_units, n_heads, dim_feedforward, dropout,batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)

        self.eda = LstmEncoderDedecoderAttractor(input_size=n_units,num_layers=1,dropout=0.1)
        #self.decoder = nn.Linear(n_units, n_speakers)
        self.diar_weight=diar_weight
        self.attractor_weight=attractor_weight
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.bias.data.zero_()
        self.encoder.weight.data.uniform_(-initrange, initrange)
        #self.eda.bias.data.zero_()
        #self.eda.weight.data.uniform_(-initrange, initrange)
    def forward_embedding(self, src: List[Tensor]):
        device = src[0].device # cuda:0
        #logging.info(f"src[0].device: {src[0].device} in forward_embedding function")
        ilens = [x.shape[0] for x in src] # [utt1_T,utt2_T,...], len(ilens) = B
        src = nn.utils.rnn.pad_sequence(src, padding_value=-1, batch_first=True)#(B,T,C)
        src = src.to(device)
        # src: (B, T, C) ->(B,T,E)
        src = self.encoder(src)
        src = self.encoder_norm(src)
        # emb: (B,T,E)
        emb = self.transformer_encoder(src, self.src_mask)
        # output: (B, T, E)
        #emb = output.transpose(0, 1)

        # Shuffle the chronological order of encoder_out, then calculate attractor
        encoder_out_shuffled = emb.clone()
        for i in range(len(ilens)):
            encoder_out_shuffled[i, : ilens[i], :] = emb[
                i, torch.randperm(ilens[i]), :
            ]

        return encoder_out_shuffled, emb ,ilens

    def forward(self, src: List[Tensor], ts: List[Tensor], has_mask=False, activation=None):
        """Forward

        """
        # forward embedding
        encoder_out_shuffled, emb, ilens = self.forward_embedding(src)
        device = emb.device
        ## pad target label into batch style
        ts = nn.utils.rnn.pad_sequence(ts, padding_value=-1, batch_first=True)#(B,T,S)
        #logging.info(f"ts shape: {ts.shape}")
        # ts: (B,T,S)
        input_zeros = torch.zeros(emb.size(0),ts.size(2)+1, encoder_out_shuffled.size(2)).to(device) #(B,S+1,E)
        ilens_tensor = torch.LongTensor(ilens).to(device)
        #attractor(B,S+1,E) , attractor_probs is list, its length is equal to B, every element size is equal to S+1
        attractor, attractor_probs = self.eda(encoder_out_shuffled,ilens_tensor,input_zeros)

        # Remove the final attractor which does not correspond to a speaker
        # Then multiply the attractors and encoder_out
        pred = torch.bmm(emb, attractor[:, :-1, :].permute(0, 2, 1)) # pred(B,T,S)


        ## compute loss
        loss_att = self.attractor_loss(attractor_probs, ts)
        loss_pit, perm_idx, perm_list, label_perm = self.pit_loss(
                pred, ts, ilens
            )
        loss = self.diar_weight * loss_pit + self.attractor_weight * loss_att
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
        ) = self.calc_diarization_error(pred, label_perm, ilens)
        sad_mr, sad_fr, mi, fa, cf, acc, der = (
                speech_miss / speech_scored,
                speech_falarm / speech_scored,
                speaker_miss / speaker_scored,
                speaker_falarm / speaker_scored,
                speaker_error / speaker_scored,
                correct / num_frames,
                (speaker_miss + speaker_falarm + speaker_error) / speaker_scored,
            )
        stats = dict(
            loss=loss.detach(),
            loss_att=loss_att.detach() if loss_att is not None else None,
            loss_pit=loss_pit.detach() if loss_pit is not None else None,
            sad_mr=sad_mr,
            sad_fr=sad_fr,
            mi=mi,
            fa=fa,
            cf=cf,
            acc=acc,
            der=der,
        )
        return loss, stats

    def infer(self,src: List[Tensor],infer_num_speakers=None,max_n_speakers=15, attractor_threshold=0.5):
        """
        NOTE(Duo Ma)
        Different from the training process,
        the number of speakers with zero input to the attractor in the inference process is determined
        based on the specified number or the threshold of the attractor.
        The number of speakers in the training process is determined
        based on the number of speakers corresponding to the target label.

        modified from https://github.com/hitachi-speech/EEND/blob/master/eend/chainer_backend/models.py#L453
                      https://github.com/BUTSpeechFIT/EEND/blob/main/eend/backend/models.py#L315

        Args:
            src: mixer speech feature input of network , shape List[Tensor]
            infer_num_speakers: if it is specified, it will select top num_speaker as real speaker ouput.
                                Because attractor offer a max number of speakers,
                                in reality, there may not be so many, so we need to use the real number of speakers to select.
            max_n_speaker: Set the maximum number of speakers that the attractor can support
            attractor_threshold: if infer_num_speakers is not specified, we will use it to
        """
        encoder_out_shuffled,emb, ilens = self.forward_embedding(src)
        device = emb.device
        ## attractor part
        # ts: (B,T,S)
        input_zeros = torch.zeros(emb.size(0),max_n_speakers, encoder_out_shuffled.size(2)).to(device) #(B,max_n_speakers,E)
        ilens_tensor = torch.LongTensor(ilens).to(device)

        #attractor(B,max_n_speakers,E) , attractor_probs is list, every emlement size is equal to max_n_speakers
        attractor, pattractor_probs = self.eda(encoder_out_shuffled,ilens_tensor,input_zeros)

        # Remove the final attractor which does not correspond to a speaker
        # Then multiply the attractors and encoder_out
        pred = torch.bmm(emb, attractor[:, :-1, :].permute(0, 2, 1)) # (B,T,max_n_speakers)

        ## apply sigmoid, get speaker probability
        #logit = torch.nn.functional.sigmoid(pred) # (B,T,max_n_speakers)
        logits = [torch.nn.functional.sigmoid(p)for p in pred] # for loop in B axis, [(t, max_n_speaker),...]

        ys_active = []
        for p, y in zip(pattractor_probs, logits): # for loop B axis
            if infer_num_speakers is not None:
                sorted_p, order = torch.sort(p, descending=True)
                ys_active.append(y[:, order[:infer_num_speakers]])
            elif attractor_threshold is not None:
                silence = np.where(p.data.to("cpu") < attractor_threshold)[0]
                n_spk = silence[0] if silence.size else None
                ys_active.append(y[:, :n_spk])
            else:
                NotImplementedError(
                    'infer_num_speakers or attractor_threshold has to be given.')
        return ys_active # [(T,n_spk)], because I will assume batch size=1.


    def attractor_loss(self, att_prob: torch.Tensor, label:torch.Tensor):
        """
        # att_prob shape (B,S+1,1),
        # label shape (B,T,S)
        """
        assert isinstance(label, Tensor), f"label: {label}"


        batch_size = label.size(0)
        device = label.device
        bce_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        # create attractor label [1, 1, ..., 1, 0]
        # att_label: (B, num_spk + 1, 1)
        att_label = torch.zeros(batch_size, label.size(2) + 1)
        att_label[:, : label.size(2)] = 1
        # list -> tensor
        #logging.info(f"att_prob: {att_prob}, torch.stack(att_prob): {torch.stack(att_prob)}, its shape: {torch.stack(att_prob).shape}")
        #(B,num_spk+1)
        att_prob = torch.stack(att_prob)
        loss = bce_loss(att_prob, att_label.to(device))
        loss = torch.mean(torch.mean(loss, dim=1))
        return loss

    def pit_loss_single_permute(self, pred: torch.Tensor, label: torch.Tensor, length: List[int]):
        """
        # pred shape (B,T,S)
        # label shape(B,T,S)
        """
        device = pred.device
        assert isinstance(length, List), f"length: {length}"
        bce_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        mask = self.create_length_mask(length, label.size(1), label.size(2)) # in order to remove pad part loss.
        mask = mask.to(device)
        loss = bce_loss(pred, label)
        loss = loss * mask
        loss = torch.sum(torch.mean(loss, dim=2), dim=1)
        loss = torch.unsqueeze(loss, dim=1)
        return loss

    def pit_loss(self, pred: torch.Tensor, label: torch.Tensor, lengths: List[int]):
        """
        pred: shape(B,T,S)
        label: shape(B,T,S)
        lengths: output frame length(actual number of frames without pad) of encoder(i.e. transformer encoder or blstm encoder)

        """
        # modified from https://github.com/espnet/espnet/blob/master/espnet2/diar/espnet_model.py
        num_output = label.size(2)
        permute_list = [np.array(p) for p in permutations(range(num_output))]
        device = pred.device
        loss_list = []
        for p in permute_list:
            label_perm = label[:, :, p]
            loss_perm = self.pit_loss_single_permute(pred, label_perm.to(device), lengths)
            loss_list.append(loss_perm)
        loss = torch.cat(loss_list, dim=1)
        min_loss, min_idx = torch.min(loss, dim=1)
        loss = torch.sum(min_loss) / torch.sum(torch.tensor(lengths)).float()
        batch_size = len(min_idx)
        label_list = []
        for i in range(batch_size):
            label_list.append(label[i, :, permute_list[min_idx[i]]].data.cpu().numpy())
        label_permute = torch.from_numpy(np.array(label_list)).float()
        return loss, min_idx, permute_list, label_permute

    def create_length_mask(self, length: List[int], max_len: int, num_output:int):
        batch_size = len(length)
        mask = torch.zeros(batch_size, max_len, num_output)
        for i in range(batch_size):
            mask[i, : length[i], :] = 1
        #mask = to_device(self, mask)
        return mask

    def calc_diarization_error(self, pred: torch.Tensor, label: torch.Tensor, length: List[int]):
        # modified from https://github.com/espnet/espnet/blob/master/espnet2/diar/espnet_model.py
        (batch_size, max_len, num_output) = label.size()
        # mask the padding part
        mask = np.zeros((batch_size, max_len, num_output))
        for i in range(batch_size):
            mask[i, : length[i], :] = 1

        # pred and label have the shape (batch_size, max_len, num_output)
        label_np = label.data.cpu().numpy().astype(int)
        pred_np = (pred.data.cpu().numpy() > 0).astype(int)
        label_np = label_np * mask
        pred_np = pred_np * mask
        #length = length.data.cpu().numpy()

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
        num_frames = float(np.sum(length))
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

class EendEdaModel(nn.Module):
    def __init__(self, n_speakers, in_size, n_heads, n_units, n_layers, dim_feedforward=2048, dropout=0.5,diar_weight: float = 1.0,
        attractor_weight: float = 1.0,encoder_type="transformer",eda_type="lstm"):
        """ Self-attention-based encoder decoder attractor diarization model.

        Args:
          n_speakers (int): Number of speakers in recording
          in_size (int): Dimension of input feature vector
          n_heads (int): Number of attention heads
          n_units (int): Number of units in a self-attention block
          n_layers (int): Number of transformer-encoder layers
          dropout (float): dropout ratio
        """
        super(EendEdaModel, self).__init__()
        self.n_speakers = n_speakers
        self.in_size = in_size
        self.n_heads = n_heads
        self.n_units = n_units
        self.n_layers = n_layers
        self.encoder_type=encoder_type
        self.eda_type=eda_type

        self.linear = nn.Linear(in_size, n_units)
        self.linear_norm = nn.LayerNorm(n_units)

        ## why add batch_first=True, it can solve the below warning:
        # UserWarning: enable_nested_tensor is True,
        # but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first
        # was not True(use batch_first for better inference performance)
        # warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
        if self.encoder_type=="transformer":
            encoder_layers = TransformerEncoderLayer(n_units, n_heads, dim_feedforward, dropout,batch_first=True)
            self.encoder = TransformerEncoder(encoder_layers, n_layers)
        elif self.encoder_type=="conformer":
            ## it expect two inputs first it is shape (B, T, input_dim),
            # second it is shape (B,) and i-th element representing number of valid frames for i-th batch element in input.
            ## its output is same to its input, it does not downsample.
            self.encoder = Conformer(input_dim=n_units,num_heads=n_heads,ffn_dim=dim_feedforward,num_layers=n_layers,depthwise_conv_kernel_size=31)
        else:
            raise NotImplementedError(f"encoder_type not support {self.encoder_type}!!!")


        if self.eda_type=="lstm":
            self.eda = LstmEncoderDedecoderAttractor(input_size=n_units,num_layers=1,dropout=0.1)
        else:
            raise NotImplementedError(f"eda_type not support {self.eda_type}!!!")

        self.diar_weight=diar_weight
        self.attractor_weight=attractor_weight

    def forward_embedding(self, src: List[Tensor]):
        ilens = [x.shape[0] for x in src] # [utt1_T,utt2_T,...], len(ilens) = B
        src = nn.utils.rnn.pad_sequence(src, padding_value=-1, batch_first=True)#(B,T,C)
        device = src.device
        # src: (B, T, C) ->(B,T,E)
        src = self.linear(src)
        src = self.linear_norm(src)
        if self.encoder_type=="transformer":
            # emb: (B,T,E)
            emb = self.encoder(src)
        elif self.encoder_type=="conformer":
            # emb: (B,T,E)
            ilens_tensor = torch.LongTensor(ilens).to(device)
            emb,_ = self.encoder(src,ilens_tensor)


        # Shuffle the chronological order of encoder_out, then calculate attractor
        encoder_out_shuffled = emb.clone()
        for i in range(len(ilens)):
            encoder_out_shuffled[i, : ilens[i], :] = emb[
                i, torch.randperm(ilens[i]), :
            ]

        return encoder_out_shuffled, emb ,ilens

    def forward(self, src: List[Tensor], ts: List[Tensor], has_mask=False, activation=None):
        """Forward

        """
        # forward embedding
        encoder_out_shuffled, emb, ilens = self.forward_embedding(src)
        device = emb.device
        ## pad target label into batch style
        ts = nn.utils.rnn.pad_sequence(ts, padding_value=-1, batch_first=True)#(B,T,S)
        #logging.info(f"ts shape: {ts.shape}")
        # ts: (B,T,S)
        input_zeros = torch.zeros(emb.size(0),ts.size(2)+1, encoder_out_shuffled.size(2)).to(device) #(B,S+1,E)
        ilens_tensor = torch.LongTensor(ilens).to(device)
        #attractor(B,S+1,E) , attractor_probs is list, its length is equal to B, every element size is equal to S+1
        attractor, attractor_probs = self.eda(encoder_out_shuffled,ilens_tensor,input_zeros)

        # Remove the final attractor which does not correspond to a speaker
        # Then multiply the attractors and encoder_out
        pred = torch.bmm(emb, attractor[:, :-1, :].permute(0, 2, 1)) # pred(B,T,S)


        ## compute loss
        loss_att = self.attractor_loss(attractor_probs, ts)
        loss_pit, perm_idx, perm_list, label_perm = self.pit_loss(
                pred, ts, ilens
            )
        loss = self.diar_weight * loss_pit + self.attractor_weight * loss_att
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
        ) = self.calc_diarization_error(pred, label_perm, ilens)
        sad_mr, sad_fr, mi, fa, cf, acc, der = (
                speech_miss / speech_scored,
                speech_falarm / speech_scored,
                speaker_miss / speaker_scored,
                speaker_falarm / speaker_scored,
                speaker_error / speaker_scored,
                correct / num_frames,
                (speaker_miss + speaker_falarm + speaker_error) / speaker_scored,
            )
        stats = dict(
            loss=loss.detach(),
            loss_att=loss_att.detach() if loss_att is not None else None,
            loss_pit=loss_pit.detach() if loss_pit is not None else None,
            sad_mr=sad_mr,
            sad_fr=sad_fr,
            mi=mi,
            fa=fa,
            cf=cf,
            acc=acc,
            der=der,
        )
        return loss, stats

    def infer(self,src: List[Tensor],infer_num_speakers=None,max_n_speakers=15, attractor_threshold=0.5):
        """
        NOTE(Duo Ma)
        Different from the training process,
        the number of speakers with zero input to the attractor in the inference process is determined
        based on the specified number or the threshold of the attractor.
        The number of speakers in the training process is determined
        based on the number of speakers corresponding to the target label.

        modified from https://github.com/hitachi-speech/EEND/blob/master/eend/chainer_backend/models.py#L453
                      https://github.com/BUTSpeechFIT/EEND/blob/main/eend/backend/models.py#L315

        Args:
            src: mixer speech feature input of network , shape List[Tensor]
            infer_num_speakers: if it is specified, it will select top num_speaker as real speaker ouput.
                                Because attractor offer a max number of speakers,
                                in reality, there may not be so many, so we need to use the real number of speakers to select.
            max_n_speaker: Set the maximum number of speakers that the attractor can support
            attractor_threshold: if infer_num_speakers is not specified, we will use it to
        """
        encoder_out_shuffled,emb, ilens = self.forward_embedding(src)
        device = emb.device
        ## attractor part
        # ts: (B,T,S)
        input_zeros = torch.zeros(emb.size(0),max_n_speakers, encoder_out_shuffled.size(2)).to(device) #(B,max_n_speakers,E)
        ilens_tensor = torch.LongTensor(ilens).to(device)

        #attractor(B,max_n_speakers,E) , attractor_probs is list, every emlement size is equal to max_n_speakers
        attractor, pattractor_probs = self.eda(encoder_out_shuffled,ilens_tensor,input_zeros)

        # Remove the final attractor which does not correspond to a speaker
        # Then multiply the attractors and encoder_out
        pred = torch.bmm(emb, attractor[:, :-1, :].permute(0, 2, 1)) # (B,T,max_n_speakers)

        ## apply sigmoid, get speaker probability
        #logit = torch.nn.functional.sigmoid(pred) # (B,T,max_n_speakers)
        logits = [torch.nn.functional.sigmoid(p)for p in pred] # for loop in B axis, [(t, max_n_speaker),...]

        ys_active = []
        for p, y in zip(pattractor_probs, logits): # for loop B axis
            if infer_num_speakers is not None:
                #sorted_p, order = torch.sort(p, descending=True)
                #ys_active.append(y[:, order[:infer_num_speakers]])
                ys_active.append(y[:,:infer_num_speakers])
            elif attractor_threshold is not None:
                silence = np.where(p.data.to("cpu") < attractor_threshold)[0]
                n_spk = silence[0] if silence.size else None
                ys_active.append(y[:, :n_spk])
            else:
                NotImplementedError(
                    'infer_num_speakers or attractor_threshold has to be given.')
        return ys_active # [(T,n_spk)], because I will assume batch size=1.

    def attractor_loss(self, att_prob: torch.Tensor, label:torch.Tensor):
        """
        # att_prob shape (B,S+1,1),
        # label shape (B,T,S)
        """
        assert isinstance(label, Tensor), f"label: {label}"


        batch_size = label.size(0)
        device = label.device
        bce_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        # create attractor label [1, 1, ..., 1, 0]
        # att_label: (B, num_spk + 1, 1)
        att_label = torch.zeros(batch_size, label.size(2) + 1)
        att_label[:, : label.size(2)] = 1
        # list -> tensor
        #logging.info(f"att_prob: {att_prob}, torch.stack(att_prob): {torch.stack(att_prob)}, its shape: {torch.stack(att_prob).shape}")
        #(B,num_spk+1)
        att_prob = torch.stack(att_prob)
        loss = bce_loss(att_prob, att_label.to(device))
        loss = torch.mean(torch.mean(loss, dim=1))
        return loss

    def pit_loss_single_permute(self, pred: torch.Tensor, label: torch.Tensor, length: List[int]):
        """
        # pred shape (B,T,S)
        # label shape(B,T,S)
        """
        device = pred.device
        assert isinstance(length, List), f"length: {length}"
        bce_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        mask = self.create_length_mask(length, label.size(1), label.size(2)) # in order to remove pad part loss.
        mask = mask.to(device)
        loss = bce_loss(pred, label)
        loss = loss * mask
        loss = torch.sum(torch.mean(loss, dim=2), dim=1)
        loss = torch.unsqueeze(loss, dim=1)
        return loss


    def pit_loss(self, pred: torch.Tensor, label: torch.Tensor, lengths: List[int]):
        """
        pred: shape(B,T,S)
        label: shape(B,T,S)
        lengths: output frame length(actual number of frames without pad) of encoder(i.e. transformer encoder or blstm encoder)

        """
        # modified from https://github.com/espnet/espnet/blob/master/espnet2/diar/espnet_model.py
        num_output = label.size(2)
        permute_list = [np.array(p) for p in permutations(range(num_output))]
        device = pred.device
        loss_list = []
        for p in permute_list:
            label_perm = label[:, :, p]
            loss_perm = self.pit_loss_single_permute(pred, label_perm.to(device), lengths)
            loss_list.append(loss_perm)
        loss = torch.cat(loss_list, dim=1)
        min_loss, min_idx = torch.min(loss, dim=1)
        loss = torch.sum(min_loss) / torch.sum(torch.tensor(lengths)).float()
        batch_size = len(min_idx)
        label_list = []
        for i in range(batch_size):
            label_list.append(label[i, :, permute_list[min_idx[i]]].data.cpu().numpy())
        label_permute = torch.from_numpy(np.array(label_list)).float()
        return loss, min_idx, permute_list, label_permute

    def create_length_mask(self, length: List[int], max_len: int, num_output:int):
        batch_size = len(length)
        mask = torch.zeros(batch_size, max_len, num_output)
        for i in range(batch_size):
            mask[i, : length[i], :] = 1
        #mask = to_device(self, mask)
        return mask

    def calc_diarization_error(self, pred: torch.Tensor, label: torch.Tensor, length: List[int]):
        # modified from https://github.com/espnet/espnet/blob/master/espnet2/diar/espnet_model.py
        (batch_size, max_len, num_output) = label.size()
        # mask the padding part
        mask = np.zeros((batch_size, max_len, num_output))
        for i in range(batch_size):
            mask[i, : length[i], :] = 1

        # pred and label have the shape (batch_size, max_len, num_output)
        label_np = label.data.cpu().numpy().astype(int)
        pred_np = (pred.data.cpu().numpy() > 0).astype(int)
        label_np = label_np * mask
        pred_np = pred_np * mask
        #length = length.data.cpu().numpy()

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
        num_frames = float(np.sum(length))
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


if __name__ == "__main__":
    import torch
    model = TransformerModel(5, 40, 4, 512, 2, 0.1)
    input = torch.randn(8, 500, 40)
    print("Model output:", model(input).size())
    print("Model attention:", model.get_attention_weight(input).size())
    print("Model attention sum:", model.get_attention_weight(input)[0][0][0].sum())
