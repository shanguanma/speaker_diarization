import logging
import torch
import torch.nn as nn
from torchaudio.models import Conformer
import torch.nn.functional as F
# from eend.eend.pytorch_backend.eend_m2f.net_utils import make_pad_mask
from typing import Tuple
from  eend.eend.pytorch_backend.eend_m2f.net_utils import PositionalEncoding

class Conv2dSubsampling8(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/8 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling8 object."""
        super(Conv2dSubsampling8, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * ((((idim - 1) // 2 - 1) // 2 - 1) // 2), odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 8.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 8.

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2][:, :, :-2:2]




class Conv2dSubsampling4(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling object."""
        super(Conv2dSubsampling4, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2]


class Conv2dSubsampling6(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/6 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling6 object."""
        super(Conv2dSubsampling6, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 5, 3),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 2) // 3), odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 6.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 6.

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-4:3]





class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout_prob):
        super(Downsample, self).__init__()

        self.stride = stride
        self.kernel_size = kernel_size

        # Depthwise convolution
        self.depthwise_conv = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size//2,
            groups=in_channels
        )

        # Pointwise convolution
        self.pointwise_conv = nn.Conv1d(
            in_channels,
            out_channels,
            1
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(out_channels)

        # Dropout
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, features, sequence_lengths):
        # Apply depthwise separable convolution
        x = self.depthwise_conv(features)
        x = F.relu(x)
        x = self.pointwise_conv(x)

        # Apply layer normalization
        x = self.layer_norm(x.transpose(1, 2)).transpose(1, 2)

        # Apply dropout
        x = self.dropout(x)

        # Compute new sequence lengths after downsampling
        new_sequence_lengths = self.compute_output_lengths(sequence_lengths)

        return x, new_sequence_lengths

    def compute_output_lengths(self, sequence_lengths):
        # Compute the length of the sequence after convolution
        return ((sequence_lengths + 2 * (self.kernel_size // 2) - self.kernel_size) // self.stride) + 1



class Backbone(nn.Module):
    """
    Construct a speech encoder that functions similarly to the speech encoder in `EEND-EDA`

    """

    def __init__(
        self,
        encoder_type: str = "conformer",
        encoder_n_layers: int = 6,
        ffn_dim: int = 2048,
        conformer_depthwise_conv_kernel_size: int = 49,
        n_heads: int = 4,
        tranformer_dropout=0.1,
        downsample_type: str = "depthwise_pointwise_conv_downsample10",
        input_feat_dim: int = 23,
        output_feat_dim: int = 256,
    ):
        super(Backbone, self).__init__()
        self.encoder_type = encoder_type
        self.downsample_type = downsample_type

        ## feature downsample part
        if self.downsample_type == "depthwise_pointwise_conv_downsample10":
            self.downsample = Downsample(
                input_feat_dim, output_feat_dim, kernel_size=15, stride=10, dropout_prob=0.1
            )
        else:
            raise NotImplementedError(
                f"downsample_type not support {self.downsample_type}!!!"
            )

        ## feature encoder part

        ## why add batch_first=True, it can solve the below warning:
        # UserWarning: enable_nested_tensor is True,
        # but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first
        # was not True(use batch_first for better inference performance)
        # warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
        # if self.encoder_type=="transformer":
        #    encoder_layers = TransformerEncoderLayer(output_feat_dim, n_heads, ffn_dim,tranformer_dropout,batch_first=True)
        #    self.encoder = TransformerEncoder(encoder_layers, encoder_n_layers)

        ## note(Duo Ma), because both input of torch transformer encoder and torchaudio conformer are different. I will unify the interface later
        if self.encoder_type == "conformer":
            ## it expect two inputs first it is shape (B, T, input_dim),
            # second it is shape (B,) and i-th element representing number of valid frames for i-th batch element in input.
            ## its output is same to its input, it does not downsample.
            self.encoder = Conformer(
                input_dim=output_feat_dim,
                num_heads=n_heads,
                ffn_dim=ffn_dim,
                num_layers=encoder_n_layers,
                depthwise_conv_kernel_size=conformer_depthwise_conv_kernel_size,
            )
        else:
            raise NotImplementedError(
                f"encoder_type not support {self.encoder_type}!!!"
            )
        self._logger = logging.getLogger(__name__) 
        
    def forward(self, x: torch.Tensor, x_len: torch.Tensor):
        # x: (BxTxD); x_len: (B,),and i-th element representing number of valid frames for i-th batch element in input.

        ## downsample part
        #self._logger.warn(f"in the backbone, input x shape: {x.shape}, x_len : {x_len}, x_len shape: {x_len.shape}")
        x = x.permute(0,2,1) # (B,D,T)
        x, x_sub_len = self.downsample(x, x_len)

        #self._logger.warn(f"in the backbone, after downsample, x: {x}, x shape: {x.shape}, x_sub_len: {x_sub_len}, x_sub_len shape: {x_sub_len.shape}")

        ## encoder part
        x = x.permute(0,2,1) #(B,D,T)->(B,T,D)
        x_down, _ = self.encoder(x, x_sub_len)

        return x_down


def lengths_to_padding_mask(lengths: torch.Tensor) -> torch.Tensor:
    r"""
    convert lengths to padding mask format
    Args:
        lengths (torch.Tensor): with shape`(B,)`,and i-th element representing
               number of valid frames for i-th batch element in ``input``.
    Returns:
        torch.Tensor
             padding_mask, with shape `(B,T)`
    Example:
         >>> lengths = torch.LongTensor([3,4,5])
         >>> batch_size = lengths.shape[0]
         >>> max_length = int(max(lengths))
         >>> padding_mask = torch.arange(max_length, device=lengths.device, dtype=lengths.dtype).expand(batch_size, max_length)>= lengths.unsqueeze(1)
         >>> padding_mask
         tensor([[False, False, False,  True,  True],
               [False, False, False, False,  True],
               [False, False, False, False, False]])

    it is used to as key_padding_mask of conformer of torchaudio, it shape: (B,T)
    """
    batch_size = lengths.shape[0]
    max_length = int(torch.max(lengths).item())
    padding_mask = torch.arange(
        max_length, device=lengths.device, dtype=lengths.dtype
    ).expand(batch_size, max_length) >= lengths.unsqueeze(1)
    return padding_mask


def padding_mask_to_lengths(padding_mask: torch.Tensor):
    r"""
    convert padding mask format to lengths:

    Args:
        encoder_padding_mask: with shape `(B,T)`
    Returns:
        torch.Tensor
          lengths (torch.Tensor): with shape`(B,)`,and i-th element representing
               number of valid frames for i-th batch element in ``input``.
        it is the inverse version of _lengths_to_padding_mask()
    Example:
        >>> padding_mask
        tensor([[False, False, False,  True,  True],
                [False, False, False, False,  True],
                [False, False, False, False, False]])
        >>> padding_mask.eq(0)
        tensor([[ True,  True,  True, False, False],
                [ True,  True,  True,  True, False],
                [ True,  True,  True,  True,  True]])
        >>> padding_mask.eq(0).long()
        tensor([[1, 1, 1, 0, 0],
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1]])
        >>> a = padding_mask.eq(0).long()
        >>> a_pad=torch.nn.functional.pad(a,(1,1),"constant", 0)
        >>> a_pad
        tensor([[0, 1, 1, 1, 0, 0, 0],
                [0, 1, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 1, 0]])
        >>> torch.diff(a_pad)
        tensor([[ 1,  0,  0, -1,  0,  0],
                [ 1,  0,  0,  0, -1,  0],
                [ 1,  0,  0,  0,  0, -1]])
        >>> torch.where(torch.diff(a_pad))
        (tensor([0, 0, 1, 1, 2, 2]), tensor([0, 3, 0, 4, 0, 5]))
        >>> torch.where(torch.diff(a_pad))[1][1::2]
            tensor([3, 4, 5])
    """
    encoder_padding_mask_int = padding_mask.eq(0).long()
    encoder_padding_mask_int_pad = torch.nn.functional.pad(
        encoder_padding_mask_int, (1, 1), "constant", 0
    )
    lengths = torch.where(torch.diff(encoder_padding_mask_int_pad))[1][1::2]
    return lengths


if __name__ == "__main__":
    #x = torch.randn(3, 24, 50)
    #up = OneDimTransposedConvolutionUpsampleLayer(50)
    #y = up(x)
    #print(f"y shape: {y.shape}")
    """
    x1 = torch.randn(3, 100, 23)
    down = DepthwiseSeparableConv1dSubsampling10(23, 256)
    y1, _ = down(x1, None)
    print(f"y1 shape: {y1.shape}")

    x2 = torch.randn(3, 100,23)
    x2_ilen = [100, 95, 80]
    x2_ilen = torch.LongTensor(x2_ilen)
    x2_mask = lengths_to_padding_mask(x2_ilen)[:,None,:]
    print(f"x2_mask shape: {x2_mask.shape}, x2_mask: {x2_mask}")
    down = DepthwiseSeparableConv1dSubsampling10(23, 256)
    y2, x2_sub_mask = down(x2, x2_mask)

    print(
        f"y2 shape: {y2.shape}, x2_sub_mask shape: {x2_sub_mask.shape}, x2_sub_mask: {x2_sub_mask}"
    )

    x3 = torch.randn(64, 500, 345)
    x3_ilen = [500, 500, 500, 500, 264, 500, 500, 500, 388, 500, 500, 500, 500, 500,
        201, 500, 500, 500, 500, 500, 116, 500, 500, 500, 500, 500, 500, 500,
        500, 269, 488, 500, 500, 246, 500, 500, 425, 500, 500, 500, 500, 120,
        363,  87, 500, 500, 500, 500, 215, 500, 500, 500, 500, 500, 500, 500,
        411, 500, 500, 500, 500, 500, 500, 500]
    x3_ilen = torch.LongTensor(x3_ilen)
    print(f"x3 shape: {x3.shape}, x3_ilen: {x3_ilen}, x3_ilen shape: {x3_ilen.shape}")
    x3_ilen_mask = lengths_to_padding_mask(x3_ilen)[:,None,:]
    print(f"x3_ilen_mask: {x3_ilen_mask}, x3_ilen_mask shape: {x3_ilen_mask.shape}")
    down = DepthwiseSeparableConv1dSubsampling10(345, 256)
    y3, x3_sub_mask = down(x3, x3_ilen_mask)
    print(
        f"y3 shape: {y3.shape}, x3_sub_mask shape: {x3_sub_mask.shape}, x3_sub_mask: {x3_sub_mask}"
    )

    down_= Conv2dSubsampling10(345,256,0.1)
    y4, x4_sub_mask = down_(x3, x3_ilen_mask)
    print(
        f"y4 shape: {y4.shape}, x4_sub_mask shape: {x4_sub_mask.shape}, x4_sub_mask: {x4_sub_mask}"
    )
    
    down_2= Conv2dSubsampling10()
    y4, x4_sub_mask = down_2(x3, x3_ilen_mask)
    print(
        f"y4 shape: {y4.shape}, x4_sub_mask shape: {x4_sub_mask.shape}, x4_sub_mask: {x4_sub_mask}"
    )
    """
 
    features = torch.randn(64, 345,500)
    features_ilen = [500, 500, 500, 500, 264, 500, 500, 500, 388, 500, 500, 500, 500, 500,
            201, 500, 500, 500, 500, 500, 116, 500, 500, 500, 500, 500, 500, 500,
            500, 269, 488, 500, 500, 246, 500, 500, 425, 500, 500, 500, 500, 120,
            363,  87, 500, 500, 500, 500, 215, 500, 500, 500, 500, 500, 500, 500,
            411, 500, 500, 500, 500, 500, 500, 500]
    sequence_lengths = torch.LongTensor(features_ilen)

    downsample = Downsample(in_channels=345, out_channels=256, kernel_size=15, stride=10, dropout_prob=0.1)
    downsampled_features, new_sequence_lengths = downsample(features, sequence_lengths)

    print("Original sequence lengths:", sequence_lengths)
    print("Downsampled sequence lengths:", new_sequence_lengths)
    print(f"downsampled_features shape: {downsampled_features.shape}")


    features = torch.randn(64, 345,161)
    sequence_length = 120  # Example sequence length
    sequence_lengths = torch.randint(low=1, high=sequence_length, size=(64,))
    #sequence_lengths = torch.LongTensor(features_ilen)

    downsample = Downsample(in_channels=345, out_channels=256, kernel_size=15, stride=10, dropout_prob=0.1)
    downsampled_features, new_sequence_lengths = downsample(features, sequence_lengths)

    print("Original sequence lengths:", sequence_lengths)
    print("Downsampled sequence lengths:", new_sequence_lengths)
    print(f"downsampled_features shape: {downsampled_features.shape}")

