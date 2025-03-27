import torch
import torch.nn as nn
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

from torch import Tensor
from typing import Optional, Any, Union, Callable

import copy
from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
## this file comes from https://github.com/Audio-WestlakeU/FS-EEND/blob/main/nnet/modules/merge_tfm_encoder.py
#                        https://github.com/Audio-WestlakeU/FS-EEND/blob/main/nnet/model/onl_tfm_enc_1dcnn_enc_linear_non_autoreg_pos_enc_l2norm.py
#  with some modifications
class OnlineTransformerDADiarization(nn.Module):
    def __init__(self, n_speakers, in_size, n_units, n_heads, enc_n_layers, dec_n_layers, dropout, has_mask, max_seqlen, dec_dim_feedforward, conv_delay=9, mask_delay=0, decom_kernel_size=64):
        """ Self-attention-based diarization model.

        Args:
          in_size (int): Dimension of input feature vector
          n_units (int): Number of units in a self-attention block
          n_heads (int): Number of attention heads
          n_layers (int): Number of transformer-encoder layers
          dropout (float): dropout ratio
        """
        super(OnlineTransformerDADiarization, self).__init__()
        self.n_speakers = n_speakers
        self.delay = conv_delay
        self.enc = MaskedTransformerEncoderModel(
            in_size, n_heads, n_units, enc_n_layers, dropout=dropout, has_mask=has_mask, max_seqlen=max_seqlen, mask_delay=mask_delay
        )
        self.dec = MaskedTransformerDecoderModel(
            in_size, n_heads, n_units, dec_n_layers, dim_feedforward=dec_dim_feedforward, dropout=dropout, has_mask=has_mask, max_seqlen=max_seqlen, mask_delay=mask_delay
        )
        self.cnn = nn.Conv1d(n_units, n_units, kernel_size=2 * conv_delay + 1, padding=9)

    def forward(self, src, tgt, ilens):
        n_speakers = [t.shape[1] for t in tgt]
        max_nspks = max(n_speakers)
        # emb: (B, T, E)
        emb = self.enc(src)
        B, T, D = emb.shape
        emb  = [e[:ilen] for e, ilen in zip(emb, ilens)]
        emb = nn.utils.rnn.pad_sequence(emb, padding_value=0, batch_first=True)
        emb: Tensor = self.cnn(emb.transpose(1,2)).transpose(1,2) # (B, T, D)
        emb = emb / torch.norm(emb, dim=-1, keepdim=True)
        attractors = self.dec(emb, max_nspks)
        attractors = attractors / torch.norm(attractors, dim=-1, keepdim=True)

        # Calculate emb consistency loss (cosine similarity)
        attn_map = emb.matmul(emb.transpose(-1, -2))
        # att_norm: (B, T, 1)
        attn_norm = torch.norm(emb, dim=-1, keepdim=True)
        attn_norm = attn_norm.matmul(attn_norm.transpose(-1, -2))
        attn_map = attn_map / (attn_norm + 1e-6)
        tgt_pad = [F.pad(t, (0, max_nspks-t.shape[1]), "constant", 0) for t in tgt]
        tgt_pad = nn.utils.rnn.pad_sequence(tgt_pad, padding_value=0, batch_first=True)
        label_map = tgt_pad.matmul(tgt_pad.transpose(-1, -2))
        tgt_norm = torch.norm(tgt_pad, dim=-1, keepdim=True)
        tgt_norm = tgt_norm.matmul(tgt_norm.transpose(-1, -2))
        label_map = label_map / (tgt_norm + 1e-6)
        emb_consis_loss = F.mse_loss(attn_map, label_map)

        # output: (B, T, C)
        output = torch.matmul(emb.unsqueeze(dim=-2), attractors.transpose(-1, -2)).squeeze(dim=-2)
        output = [out[:ilen, :n_spk] for out, ilen, n_spk in zip(output, ilens, n_speakers)]

        emb = [e[:ilen] for e, ilen in zip(emb, ilens)]
        attractors = [attr[:ilen, 1:n_spk] for attr, ilen, n_spk in zip(attractors, ilens, n_speakers)]
        return output, emb_consis_loss, emb, attractors
    
    def test(self, src, ilens, max_nspks=6):
        # emb: (B, T, E)
        emb = self.enc(src)
        B, T, D = emb.shape
        emb  = [e[:ilen] for e, ilen in zip(emb, ilens)]
        emb = nn.utils.rnn.pad_sequence(emb, padding_value=0, batch_first=True)
        emb: Tensor = self.cnn(emb.transpose(1,2)).transpose(1,2) # (B, T, D)
        emb = emb / torch.norm(emb, dim=-1, keepdim=True)
        attractors = self.dec(emb, max_nspks)
        attractors = attractors / torch.norm(attractors, dim=-1, keepdim=True)
        
        # output: (B, T, C)
        output = torch.matmul(emb.unsqueeze(dim=-2), attractors.transpose(-1, -2)).squeeze(dim=-2)
        output = [out[:ilen] for out, ilen in zip(output, ilens)]

        emb = [e[:ilen] for e, ilen in zip(emb, ilens)]
        attractors = [attr[:ilen] for attr, ilen in zip(attractors, ilens)]
        return output, emb, attractors


class MaskedTransformerDecoderModel(nn.Module):
    def __init__(self, in_size, n_heads, n_units, n_layers, dim_feedforward, dropout=0.5, has_mask=False, max_seqlen = 500, has_pos=False, mask_delay=0):
        super(MaskedTransformerDecoderModel, self).__init__()
        self.in_size = in_size
        self.n_heads = n_heads
        self.n_units = n_units
        self.n_layers = n_layers
        self.has_pos = has_pos
        self.has_mask = has_mask
        self.max_seqlen = max_seqlen
        self.mask_delay = mask_delay

        self.encoder = nn.Linear(in_size, n_units)
        self.encoder_norm = nn.LayerNorm(n_units)

        self.pos_enc = PositionalEncoding(n_units, dropout)
        self.convert = nn.Linear(n_units * 2, n_units)
        decoder_layers = TransformerEncoderFusionLayer(n_units, n_heads, dim_feedforward, dropout, batch_first=True)
        #self.attractor_decoder = TransformerEncoder(decoder_layers, n_layers)
        self.attractor_decoder = torch.nn.ModuleList([decoder_layers for _ in range(n_layers)])

    def _generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device), diagonal=-self.mask_delay) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, emb: Tensor, max_nspks: int, activation: Optional[Callable]=None):
        pos_enc = self.pos_enc(emb, max_nspks) # (B, T, C, D)
        attractors_init: Tensor = self.convert(torch.cat([emb.unsqueeze(dim=2).repeat(1, 1, max_nspks, 1), pos_enc], dim=-1))
        
        t_mask = self._generate_square_subsequent_mask(emb.shape[1], emb.device)
        attractors = attractors_init
        for layer in self.attractor_decoder:
            attractors = layer(attractors, t_mask)
        #attractors = self.attractor_decoder(attractors_init, t_mask)
        return attractors

class MaskedTransformerEncoderModel(nn.Module):
    def __init__(self, in_size, n_heads, n_units, n_layers, dim_feedforward=2048, dropout=0.5, has_mask=False, max_seqlen = 500, has_pos=False, mask_delay=0):
        """ Self-attention-based diarization model.

        Args:
          n_speakers (int): Number of speakers in recording
          in_size (int): Dimension of input feature vector
          n_heads (int): Number of attention heads
          n_units (int): Number of units in a self-attention block
          n_layers (int): Number of transformer-encoder layers
          dropout (float): dropout ratio
        """
        super(MaskedTransformerEncoderModel, self).__init__()
        self.in_size = in_size
        self.n_heads = n_heads
        self.n_units = n_units
        self.n_layers = n_layers
        self.has_pos = has_pos
        self.has_mask = has_mask
        self.max_seqlen = max_seqlen
        self.mask_delay = mask_delay

        self.bn = nn.BatchNorm1d(in_size)
        self.encoder = nn.Linear(in_size, n_units)
        self.encoder_norm = nn.LayerNorm(n_units)
        if self.has_pos:
            self.pos_encoder = PositionalEncoding(n_units, dropout)
        encoder_layers = TransformerEncoderLayer(n_units, n_heads, dim_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device), diagonal=-self.mask_delay) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.bias.data.zero_()
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, activation=None):

        ilens = [x.shape[0] for x in src]
        src = nn.utils.rnn.pad_sequence(src, padding_value=-1, batch_first=True)
        src = self.bn(src.transpose(1, 2)).transpose(1, 2)

        src_mask = None
        if self.has_mask:
            src_mask = self._generate_square_subsequent_mask(src.shape[1], src.device)

        # src: (B, T, E)
        src = self.encoder(src)
        src = self.encoder_norm(src)
        # src: (T, B, E)
        src = src.transpose(0, 1)
        if self.has_pos:
            # src: (T, B, E)
            src = self.pos_encoder(src)
        # output: (T, B, E)
        output = self.transformer_encoder(src, src_mask)
        # output: (B, T, E)
        output = output.transpose(0, 1)

        if activation:
            output = activation(output)

        return output

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
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor, max_nspks):
        # Add positional information to each time step of x
        pe = self.pe[:, :max_nspks, :]
        pe = pe.unsqueeze(dim=0).repeat(x.shape[0], x.shape[1], 1, 1) # (B, T, C, D)
        x = x.unsqueeze(dim=2).repeat(1, 1, max_nspks, 1)
        # x = x + pe
        return pe



class TransformerEncoderFusionLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectively. Otherwise it's done after. Default: ``False`` (after).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)

    Fast path:
        forward() will use a special optimized implementation if all of the following
        conditions are met:

        - Either autograd is disabled (using ``torch.inference_mode`` or ``torch.no_grad``) or no tensor
          argument ``requires_grad``
        - training is disabled (using ``.eval()``)
        - batch_first is ``True`` and the input is batched (i.e., ``src.dim() == 3``)
        - activation is one of: ``"relu"``, ``"gelu"``, ``torch.functional.relu``, or ``torch.functional.gelu``
        - at most one of ``src_mask`` and ``src_key_padding_mask`` is passed
        - if src is a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_, neither ``src_mask``
          nor ``src_key_padding_mask`` is passed
        - the two ``LayerNorm`` instances have a consistent ``eps`` value (this will naturally be the case
          unless the caller has manually modified one without modifying the other)

        If the optimized implementation is in use, a
        `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ can be
        passed for ``src`` to represent padding more efficiently than using a padding
        mask. In this case, a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ will be
        returned, and an additional speedup proportional to the fraction of the input that
        is padding can be expected.
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderFusionLayer, self).__init__()
        self.self_attn1 = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        self.self_attn2 = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm11 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm12 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm21 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm22 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout11 = Dropout(dropout)
        self.dropout21 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super(TransformerEncoderFusionLayer, self).__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu


    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        if src_key_padding_mask is not None:
            _skpm_dtype = src_key_padding_mask.dtype
            if _skpm_dtype != torch.bool and not torch.is_floating_point(src_key_padding_mask):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported")
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        why_not_sparsity_fast_path = ''
        if not src.dim() == 3:
            why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        elif self.training:
            why_not_sparsity_fast_path = "training is enabled"
        elif not self.self_attn1.batch_first :
            why_not_sparsity_fast_path = "self_attn.batch_first was not True"
        elif not self.self_attn1._qkv_same_embed_dim :
            why_not_sparsity_fast_path = "self_attn._qkv_same_embed_dim was not True"
        elif not self.activation_relu_or_gelu:
            why_not_sparsity_fast_path = "activation_relu_or_gelu was not True"
        elif not (self.norm11.eps == self.norm12.eps):
            why_not_sparsity_fast_path = "norm1.eps is not equal to norm2.eps"
        elif src_mask is not None:
            why_not_sparsity_fast_path = "src_mask is not supported for fastpath"
        elif src.is_nested and src_key_padding_mask is not None:
            why_not_sparsity_fast_path = "src_key_padding_mask is not supported with NestedTensor input for fastpath"
        elif self.self_attn1.num_heads % 2 == 1:
            why_not_sparsity_fast_path = "num_head is odd"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"

        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                self.self_attn1.in_proj_weight,
                self.self_attn1.in_proj_bias,
                self.self_attn1.out_proj.weight,
                self.self_attn1.out_proj.bias,
                self.self_attn2.in_proj_weight,
                self.self_attn2.in_proj_bias,
                self.self_attn2.out_proj.weight,
                self.self_attn2.out_proj.bias,
                self.norm11.weight,
                self.norm11.bias,
                self.norm12.weight,
                self.norm12.bias,
                self.norm21.weight,
                self.norm21.bias,
                self.norm22.weight,
                self.norm22.bias,
                self.linear1.weight,
                self.linear1.bias,
                self.linear2.weight,
                self.linear2.bias,
            )

            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif not all((x.is_cuda or 'cpu' in str(x.device)) for x in tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument is neither CUDA nor CPU"
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
                                              "input/output projection weights or biases requires_grad")

            if not why_not_sparsity_fast_path:
                return torch._transformer_encoder_layer_fwd(
                    src,
                    self.self_attn1.embed_dim,
                    self.self_attn1.num_heads,
                    self.self_attn1.in_proj_weight,
                    self.self_attn1.in_proj_bias,
                    self.self_attn1.out_proj.weight,
                    self.self_attn1.out_proj.bias,
                    self.self_attn2.embed_dim,
                    self.self_attn2.num_heads,
                    self.self_attn2.in_proj_weight,
                    self.self_attn2.in_proj_bias,
                    self.self_attn2.out_proj.weight,
                    self.self_attn2.out_proj.bias,
                    self.activation_relu_or_gelu == 2,
                    self.norm_first,
                    self.norm11.eps,
                    self.norm11.weight,
                    self.norm11.bias,
                    self.norm12.weight,
                    self.norm12.bias,
                    self.norm21.eps,
                    self.norm21.weight,
                    self.norm21.bias,
                    self.norm22.weight,
                    self.norm22.bias,
                    self.linear1.weight,
                    self.linear1.bias,
                    self.linear2.weight,
                    self.linear2.bias,
                    # TODO: if src_mask and src_key_padding_mask merge to single 4-dim mask
                    src_mask if src_mask is not None else src_key_padding_mask,
                    1 if src_key_padding_mask is not None else
                    0 if src_mask is not None else
                    None,
                )


        # Self-attn on time-frame dim
        B, T, C, D = src.shape
        x = src.transpose(1, 2).reshape(B*C, T, D)
        if self.norm_first:
            x = x + self._sa_block1(self.norm11(x), src_mask, src_key_padding_mask)
            # x = x + self._ff_block(self.norm12(x))
        else:
            x = self.norm11(x + self._sa_block1(x, src_mask, src_key_padding_mask))
            # x = self.norm12(x + self._ff_block(x))
        x = x.reshape(B, C, T, D).transpose(1, 2).reshape(B*T, C, D)
        
        # Self-attention on spk dim
        if self.norm_first:
            x = x + self._sa_block2(self.norm21(x), None, None)
            x = x + self._ff_block(self.norm22(x))
        else:
            x = self.norm21(x + self._sa_block2(x, None, None))
            x = self.norm22(x + self._ff_block(x))
        x = x.reshape(B, T, C, D)

        return x

    # self-attention block
    def _sa_block1(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn1(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout11(x)
    
    # self-attention2 block
    def _sa_block2(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn2(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout21(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
