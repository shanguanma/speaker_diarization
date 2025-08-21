# copy and modified from https://huggingface.co/BUT-FIT/DiCoW_v3_2/blob/main/SCBs.py
import torch
from torch import nn
from transformers import WhisperConfig
from transformers.activations import ACT2FN
from transformers.models.whisper.modeling_whisper import WHISPER_ATTENTION_CLASSES
import torch.nn.functional as F
from .layers import CustomLinear, CustomDiagonalLinear, Gate


class MultiHeadCoAttention(nn.Module):
    def __init__(self, multi_dim, single_dim, num_heads):
        assert multi_dim % num_heads == 0, "multi_dim must be divisible by num_heads"
        assert single_dim % num_heads == 0, "single_dim must be divisible by num_heads"
        super().__init__()
        self.q_proj = nn.Linear(single_dim, single_dim)
        self.k_proj = nn.Linear(single_dim, single_dim)
        self.multi_v_proj = nn.Linear(multi_dim, multi_dim)  # D'
        self.single_v_proj = nn.Linear(single_dim, single_dim)  # D

        self.multi_out_proj = nn.Linear(multi_dim, multi_dim)  # D'
        self.single_out_proj = nn.Linear(single_dim, single_dim)  # D

        self.multi_dim = multi_dim
        self.single_dim = single_dim
        self.num_heads = num_heads

    def forward(self, query, key, multi_value, single_value):
        # q, k, multi_v: (T,B,ch,D')
        # single_v: (T,B,1,D)
        query = torch.transpose(query, 0, 1)  # (B,T,ch,D')...[32, 150, 4, 64]
        key = torch.transpose(key, 0, 1)  # (B,T,ch,D')...[32, 150, 4, 64]
        multi_value = torch.permute(
            multi_value, (1, 2, 0, 3)
        )  # (B,ch,T,D')...[32, 4, 150, 64]
        single_value = torch.permute(
            single_value, (1, 2, 0, 3)
        )  # (B,1,T,D)...[32, 1, 150, 256]
        ###########

        q = torch.split(
            self.q_proj(query), self.single_dim // self.num_heads, dim=-1
        )  # seq: (B,T,ch,D'/h)
        q = torch.stack(q, dim=1)  # (B,h,T,ch,D'/h)...[32, 8, 150, 4, 8]

        k = torch.split(
            self.k_proj(key), self.single_dim // self.num_heads, dim=-1
        )  # seq: (B,T,ch,D'/h)
        k = torch.stack(k, dim=1)  # (B,h,T,ch,D'/h)...[32, 8, 150, 4, 8]

        multi_v = torch.split(
            self.multi_v_proj(multi_value), self.multi_dim // self.num_heads, dim=-1
        )  # seq: (B,ch,T,D'/h)
        multi_v = torch.stack(
            multi_v, dim=1
        )  # (B, h, ch, T, D'/h)...[32, 8, 4, 150, 8]

        single_v = torch.split(
            self.single_v_proj(single_value), self.single_dim // self.num_heads, dim=-1
        )  # seq: (B,1,T,D/h)
        single_v = torch.stack(
            single_v, dim=1
        )  # seq: (B,h,1,T,D/h)...[32, 32, 1, 150, 8]

        q = q.view(*q.shape[:-2], -1)  # (B, h, T, ch*D/h)
        k = k.view(*k.shape[:-2], -1)  # (B, h, T, ch*D/h)
        normalizer = torch.sqrt(torch.Tensor([float(q.shape[-1])]).to(q.device))

        sim_mat = (
            torch.matmul(q, torch.transpose(k, -2, -1)) / normalizer
        )  # (B, h, T, T)
        att_mat = torch.unsqueeze(
            nn.functional.softmax(sim_mat, dim=-1), 2
        )  # (B, h, 1, T, T)

        # co-attention
        multi_result = torch.matmul(att_mat, multi_v)  # (B, h, ch, T, D'/h)
        single_result = torch.matmul(att_mat, single_v)  # (B, h, 1, T, D/h)

        multi_result = torch.permute(
            multi_result, (3, 0, 2, 1, 4)
        )  # (T, B, ch, h, D'/h)
        single_result = torch.permute(
            single_result, (3, 0, 2, 1, 4)
        )  # (T, B, 1, h, D/h)
        multi_result = torch.reshape(
            multi_result, multi_result.shape[:-2] + (-1,)
        )  # (T, B, ch, D')
        single_result = torch.reshape(
            single_result, single_result.shape[:-2] + (-1,)
        )  # (T, B, 1, D)

        multi_result = self.multi_out_proj(multi_result)
        single_result = self.single_out_proj(single_result)
        return multi_result, single_result


class CoAttention(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        single_dim=256,
        multi_dim=64,
        n_heads=8,
        attn_dropout=0.0,
        init_mult=1e-2,
    ):  # , pre_norm=True):
        super().__init__()
        self.init_mult = init_mult

        self.in_single_proj = nn.Linear(embed_dim, single_dim)  # single_dim == D
        self.in_single_ln = nn.LayerNorm(single_dim)

        self.in_multi_proj = nn.Linear(embed_dim, multi_dim)  # multi_dim == D'
        self.in_multi_ln = nn.LayerNorm(multi_dim)

        self.mca = MultiHeadCoAttention(multi_dim, single_dim, n_heads)
        self.mca_multi_out_ln = nn.LayerNorm(multi_dim)
        self.mca_single_out_ln = nn.LayerNorm(single_dim)

        # default MHA input: (seq, batch, feature)
        self.cross_frame_mha = nn.MultiheadAttention(
            single_dim, n_heads, dropout=attn_dropout, bias=True, kdim=None, vdim=None
        )
        self.mha_ln = nn.LayerNorm(single_dim)

        self.cat_proj = nn.Linear(single_dim + multi_dim, embed_dim)

        self.miso = False

    def scale_weights(self):
        self.cat_proj.bias.data *= 0.0
        self.cat_proj.weight.data *= self.init_mult

    def forward(self, x):
        # x: (T,B,ch,F); (150, 32, 4, 768)
        frames, B, chans, feat_dim = x.shape

        single_x = torch.mean(x, dim=2)  # (T,B,F)
        single_x = self.in_single_ln(self.in_single_proj(single_x)).unsqueeze(
            dim=-2
        )  # (T,B,1,D)

        multi_x = self.in_multi_ln(self.in_multi_proj(x))  # (T,B,ch,D')

        # MCA
        multi_mca, single_mca = self.mca(
            single_x, single_x, multi_x, single_x
        )  # (T,B,ch,D'), (T,B,ch,D)
        single_x = single_x + single_mca
        multi_x = multi_x + multi_mca
        multi_x = self.mca_multi_out_ln(multi_x)  # (T,B,ch,D')
        single_x = torch.squeeze(self.mca_single_out_ln(single_x), -2)  # (T,B,D)

        # MHA
        single_mha, _ = self.cross_frame_mha(
            single_x, single_x, single_x, need_weights=False
        )  # (T, B, D)
        single_x = self.mha_ln(single_mha + single_x)

        # join representations
        single_x = single_x.unsqueeze(-2)  # (T,B,1,D)
        single_x_tile = torch.tile(single_x, (1, 1, chans, 1))  # (T,B,ch,D)
        cat_x = torch.cat([single_x_tile, multi_x], dim=-1)  # (T,B,ch,D+D')
        out = self.cat_proj(cat_x)  # (T,B,ch,F)

        return out


class LowRankApproxSelectFirst(nn.Module):
    def __init__(self, d_in, d_out, rank):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.rank = rank
        self.proj_in = nn.Linear(d_in, rank)
        self.proj_out = nn.Linear(rank, d_out)

    def forward(self, x):
        return self.proj_out(self.proj_in(x))

    def _init_weights(self):
        # Create low-rank approximation of the identity projection from first d_out of input
        eye = torch.eye(self.d_out, self.d_in)  # (d_out x d_in)

        # Low-rank SVD of eye matrix
        U, S, Vh = torch.linalg.svd(
            eye, full_matrices=False
        )  # U: (d_out x d_out), Vh: (d_in x d_in)

        U_k = U[:, : self.rank]  # (d_out x rank)
        S_k = S[: self.rank]  # (rank,)
        V_k = Vh[: self.rank, :]  # (rank x d_in)

        A = V_k  # (rank x d_in)
        B = U_k @ torch.diag(S_k)  # (d_out x rank)

        # Set weights
        self.proj_in.weight.data.copy_(A)
        self.proj_in.bias.data.zero_()
        self.proj_out.weight.data.copy_(B)
        self.proj_out.bias.data.zero_()


class TACBlock(nn.Module):
    def __init__(self, config: WhisperConfig, d_int_factor: float = 1, num_speakers=2):
        super().__init__()
        d = config.d_model
        d_prime = int(d * d_int_factor)
        self.num_speakers = num_speakers
        self.proj_in_1 = nn.Linear(d, d_prime, bias=True)
        self.proj_in_2 = nn.Linear(d, d_prime, bias=True)
        self.proj_int = nn.Linear(d_prime, d_prime, bias=True)
        self.proj_out_1 = nn.Linear(d + d_prime, d, bias=True)
        self.proj_out_2 = nn.Linear(d + d_prime, d, bias=True)
        self.activation_fn = ACT2FN[config.activation_function]
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(self.num_speakers)])
        self.gate = Gate(self.num_speakers, 0.01)

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        # hidden_states: (B, self.num_speakers, T, F)

        x_proj = torch.stack(
            [
                self.activation_fn(self.proj_in_1(hidden_states[:, 0])),
                self.activation_fn(self.proj_in_2(hidden_states[:, 1])),
            ],
            dim=1,
        )  # (B, 2, T, d')
        x_mean = x_proj.mean(dim=1, keepdim=True)  # (B, 1, T, d')
        z = self.activation_fn(self.proj_int(x_mean))  # (B, 1, T, d')

        z_expand = z.expand(
            -1, self.num_speakers, -1, -1
        )  # (B, self.num_speakers, T, d')
        x_cat = torch.cat(
            [hidden_states, z_expand], dim=-1
        )  # (B, self.num_speakers, T, d + d')
        x_out = torch.stack(
            [
                self.norms[0](self.proj_out_1(x_cat[:, 0])),
                self.norms[1](self.proj_out_2(x_cat[:, 1])),
            ],
            dim=1,
        )  # (B, self.num_speakers, T, d)
        return hidden_states + self.gate(x_out, dim=1)


class CrossAttentionBlock(nn.Module):
    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.num_speakers = getattr(config, "mt_num_speakers", 2)
        if self.num_speakers != 2:
            raise ValueError("CrossAttentionBlock supports only 2 speakers.")

        # Separate attention block per speaker
        self.attn_blocks = nn.ModuleList(
            [
                WHISPER_ATTENTION_CLASSES[config._attn_implementation](
                    embed_dim=self.embed_dim,
                    num_heads=config.encoder_attention_heads,
                    dropout=config.attention_dropout,
                    config=config,
                )
                for _ in range(self.num_speakers)
            ]
        )

        self.norms = nn.ModuleList(
            [nn.LayerNorm(self.embed_dim) for _ in range(self.num_speakers)]
        )
        self.gate = Gate(self.num_speakers, 0.01)

    def forward(self, hidden_states):
        # hidden_states: (B, 2, T, F)
        outputs = []
        for s in range(self.num_speakers):
            q = hidden_states[:, s]  # (B, T, F)
            other_s = 1 - s
            kv = hidden_states[:, other_s]  # (B, T, F)

            attn_out, _, _ = self.attn_blocks[s](
                hidden_states=q, key_value_states=kv
            )  # (B, T, F)
            outputs.append(self.norms[s](attn_out[:, None, :, :]))
        outputs = torch.concat(outputs, dim=1)
        outputs_modulated = self.gate(outputs, dim=1) + hidden_states
        return outputs_modulated


class CompetitiveCrossAttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        assert (
            self.head_dim * self.num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.num_speakers = getattr(config, "mt_num_speakers", 2)
        if self.num_speakers != 2:
            raise ValueError("CompetitiveCrossAttentionBlock supports only 2 speakers.")

        # Separate projections for Q, K, V
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.norms = nn.ModuleList(
            [nn.LayerNorm(self.embed_dim) for _ in range(self.num_speakers)]
        )
        self.eps = 1e-6
        self.gate = Gate(self.num_speakers, 0.01)

    def _shape(self, tensor, seq_len, batch_size):
        # reshape into (B, num_heads, T, head_dim)
        return tensor.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def forward(self, hidden_states):
        # hidden_states: (B, 2, T, F)
        B, _, T, _ = hidden_states.shape

        h1, h2 = hidden_states[:, 0], hidden_states[:, 1]  # (B, T, F)

        # Project Q,K,V
        Q1 = self.q_proj(h1)  # (B, T, F)
        K2 = self.k_proj(h2)
        V2 = self.v_proj(h2)

        Q2 = self.q_proj(h2)
        K1 = self.k_proj(h1)
        V1 = self.v_proj(h1)

        # Reshape for multi-head attention
        Q1 = self._shape(Q1, T, B)  # (B, heads, T, head_dim)
        K2 = self._shape(K2, T, B)
        V2 = self._shape(V2, T, B)

        Q2 = self._shape(Q2, T, B)
        K1 = self._shape(K1, T, B)
        V1 = self._shape(V1, T, B)

        # Scaled dot-product attention logits
        scale = 1 / (self.head_dim**0.5)
        L_1to2 = torch.matmul(Q1, K2.transpose(-1, -2)) * scale  # (B, heads, T, T)
        L_2to1 = torch.matmul(Q2, K1.transpose(-1, -2)) * scale  # (B, heads, T, T)

        # Softmax over last dim (keys)
        S_1to2 = F.softmax(L_1to2, dim=-1)
        S_2to1 = F.softmax(L_2to1, dim=-1)

        # Competitive normalization (soft exclusivity)
        M_joint = S_1to2 + S_2to1 + self.eps
        A_1to2 = S_1to2 / M_joint
        A_2to1 = S_2to1 / M_joint

        # Weighted sum of values
        H1_attn = torch.matmul(A_1to2, V2)  # (B, heads, T, head_dim)
        H2_attn = torch.matmul(A_2to1, V1)

        # Concatenate heads back
        H1_attn = (
            H1_attn.transpose(1, 2).contiguous().view(B, T, self.embed_dim)
        )  # (B, T, F)
        H2_attn = H2_attn.transpose(1, 2).contiguous().view(B, T, self.embed_dim)

        # Output projection
        H1_attn = self.norms[0](self.out_proj(H1_attn))
        H2_attn = self.norms[1](self.out_proj(H2_attn))

        # Residuals
        out = hidden_states + self.gate(
            torch.concat([H1_attn[:, None, :, :], H2_attn[:, None, :, :]], dim=1), dim=1
        )

        return out  # (B, 2, T, F)


class CoAttentionWrapper(nn.Module):
    def __init__(self, config, num_speakers=2):
        super().__init__()
        self.coa = CoAttention(
            embed_dim=config.d_model,
            single_dim=config.d_model // 2,
            multi_dim=config.d_model // 4,
            n_heads=config.encoder_attention_heads,
            attn_dropout=config.attention_dropout,
        )
        self.gate = Gate(num_speakers, 0.01)

    def forward(self, coa_input: torch.Tensor) -> torch.Tensor:
        # hidden_states: (B, 2, T, F)
        hidden_states = coa_input.permute(-2, 0, 1, -1)
        hidden_states = self.coa(hidden_states)
        out = coa_input + self.gate(hidden_states.permute(1, 2, 0, -1), dim=1)
        return out


class SpeakerCommunicationBlock(nn.Module):
    def __init__(self, config, scb_method):
        super().__init__()
        self.num_speakers = getattr(config, "mt_num_speakers", 2)
        self.embed_dim = config.d_model
        self.scb_method = scb_method
        self.config = config

        if self.scb_method == "tac":
            self.method = TACBlock(config)
        elif self.scb_method == "cross_attention":
            self.method = CrossAttentionBlock(config)
        elif self.scb_method == "competitive_cross_attention":
            self.method = CompetitiveCrossAttentionBlock(config)
        elif self.scb_method == "co_attention":
            self.method = CoAttentionWrapper(config)
        elif self.scb_method == "identity":
            self.method = (
                nn.Parameter(torch.zeros(self.embed_dim))
                if config.fddt_bias_only
                else (
                    CustomDiagonalLinear(self.embed_dim, bias=True, init_eye_val=1.0)
                    if config.fddt_is_diagonal
                    else CustomLinear(
                        self.embed_dim, self.embed_dim, bias=True, init_eye_val=1.0
                    )
                )
            )
        else:
            raise ValueError(f"Unsupported scb_method: {self.scb_method}")

    def forward(self, x):
        # x: (B, T, F)
        B, T, F = x.shape
        S = self.num_speakers

        # Reshape to (B//S, S, T, F)
        x_reshaped = x.view(B // S, S, T, F)

        # Call the selected method
        out = self.method(x_reshaped)

        # Reshape back (B, T, F)
        out_merged = out.view(B, T, F)
        return out_merged
