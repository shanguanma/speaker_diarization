import torch
import torch.nn as nn
from torchaudio.models import Conformer
#from eend.eend.pytorch_backend.eend_m2f.net_utils import make_pad_mask 
from typing import Tuple

class DepthwiseSeparableConv1dSubsampling10(nn.Module):
    """
    it is modified from https://github.com/seungjunlee96/Depthwise-Separable-Convolution_Pytorch
    Convolutional 1D subsampling (to 1/10 length).
    """
    def __init__(self,nin, nout, kernel_size = 15, padding = 3, stride = 10,bias=False):
        super(DepthwiseSeparableConv1dSubsampling10, self).__init__()
        #self.depthwise = nn.Conv1d(nin, nin, kernel_size=kernel_size,stride=stride, padding=padding, groups=nin, bias=bias)
        self.depthwise = nn.Conv1d(nin, nin, kernel_size=kernel_size,stride=stride,padding=3, groups=nin, bias=bias)
        self.pointwise = nn.Conv1d(nin, nout, kernel_size=1,stride=1, bias=bias)
        self.layer_norm = nn.LayerNorm(nout)
        self.drop = nn.Dropout(p=0.1)
        self.act = nn.ReLU()

    def forward(self, x,x_mask):
        """
         Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 10.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 10.
        """
        x = x.permute(0,2,1) 
        x = self.depthwise(x)
        x = self.act(x)
        x = self.pointwise(x)
        x = self.act(x)
        x = x.permute(0,2,1)
        x = self.layer_norm(x)
        x = self.drop(x)
        if x_mask is None:
            return x, None
        #for k, s in zip(self.kernels, self.strides):
        #    x_mask = x_mask[:, :, : -k + 1 : s]
        #return x, x_mask
        x_mask = x_mask[:,:,:-14:9] 
        return x,x_mask


class OneDimTransposedConvolutionUpsampleLayer(nn.Module):
    """
    it is proposed by `EEND-M2F: Masked-attention mask transformers for speaker diarization`
    """
    def __init__(self,feat_dim:int=256,):
        super(OneDimTransposedConvolutionUpsampleLayer,self).__init__()
        self.transposed_conv1d1=nn.ConvTranspose1d(feat_dim, feat_dim, kernel_size=3, stride=2,padding=1,output_padding=1)
        self.transposed_conv1d2=nn.ConvTranspose1d(feat_dim, feat_dim, kernel_size=5, stride=5,padding=0,)
        self.linear_norm = nn.LayerNorm(feat_dim)
        self.act = nn.GELU()
    def forward(self,x):
        # x shape should be BxTxD
        # BxTxD -> BxDxT
        x = x.permute(0,2,1)
        x = self.transposed_conv1d1(x)
        x = self.act(x)
        x = self.transposed_conv1d2(x)
        x = self.act(x)
        x = x.permute(0,2,1) # BxDxT' -> BxT'xD
        x = self.linear_norm(x)
        x = self.act(x)
        return x #BxT'xD



class Backbone(nn.Module):
    """
    Construct a speech encoder that functions similarly to the speech encoder in `EEND-EDA`

    """
    def __init__(self,encoder_type:str="conformer",encoder_n_layers:int=6,ffn_dim:int=2048,conformer_depthwise_conv_kernel_size:int=49,n_heads:int=4,tranformer_dropout=0.1,downsample_type:str="DepthwiseDownsample10",upsample_type:str="TransposedConv1dUpsample10", input_feat_dim:int=23, output_feat_dim:int=256):

        self.encoder_type= encoder_type
        self.downsample_type=downsample_type
        self.upsample_type=upsample_type
        


        ## feature downsample part
        if self.downsample_type=="DepthwiseDownsample10":
            self.downsample=DepthwiseSeparableConv1dSubsampling10(input_feat_dim,output_feat_dim)
        else:
            raise NotImplementedError(f"downsample_type not support {self.downsample_type}!!!")

        
        ## feature encoder part

        ## why add batch_first=True, it can solve the below warning:
        # UserWarning: enable_nested_tensor is True,
        # but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first
        # was not True(use batch_first for better inference performance)
        # warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
        #if self.encoder_type=="transformer":
        #    encoder_layers = TransformerEncoderLayer(output_feat_dim, n_heads, ffn_dim,tranformer_dropout,batch_first=True)
        #    self.encoder = TransformerEncoder(encoder_layers, encoder_n_layers)

        ## note(Duo Ma), because both input of torch transformer encoder and torchaudio conformer are different. I will unify the interface later
        if self.encoder_type=="conformer":
            ## it expect two inputs first it is shape (B, T, input_dim),
            # second it is shape (B,) and i-th element representing number of valid frames for i-th batch element in input.
            ## its output is same to its input, it does not downsample.
            self.encoder = Conformer(input_dim=output_feat_dim,num_heads=n_heads,ffn_dim=ffn_dim,num_layers=encoder_n_layers,depthwise_conv_kernel_size=conformer_depthwise_conv_kernel_size)
        else:
            raise NotImplementedError(f"encoder_type not support {self.encoder_type}!!!")
        
        # feature upsample part
        if self.upsample_type=="TransposedConv1dUpsample10":
            self.upsample=OneDimTransposedConvolutionUpsampleLayer(output_feat_dim,output_feat_dim)
        else:
            raise NotImplementedError(f"upsample_type not support {self.upsample_type}!!!")
        

    def forward(self, x:torch.Tensor, x_len: torch.Tensor):
        # x: (BxTxD); x_len: (B,),and i-th element representing number of valid frames for i-th batch element in input.
       
        ## downsample part
        x_padding_mask = lengths_to_padding_mask(x_len)[:,None,:].to(x.device)
        x,x_padding_mask = self.downsample(x,x_padding_mask)
 
        ## encoder part
        x_downsample_len = padding_mask_to_lengths(x_padding_mask)
        x_down,_ = self.encoder(x,x_downsample_len)
        
        ## upsample part
        x_up = self.upsample(x)
        return x_down, x_up



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
    padding_mask = torch.arange(max_length, device=lengths.device, dtype=lengths.dtype).expand(
        batch_size, max_length
    ) >= lengths.unsqueeze(1)
    return padding_mask

def padding_mask_to_lengths(encoder_padding_mask: torch.Tensor):
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
    encoder_padding_mask_int_pad = torch.nn.functional.pad(encoder_padding_mask_int,(1,1),"constant", 0)
    lengths = torch.where(torch.diff(encoder_padding_mask_int_pad))[1][1::2]
    return lengths


if __name__ == "__main__":
    x = torch.randn(3,24,50)
    up = OneDimTransposedConvolutionUpsampleLayer(50)
    y = up(x)
    print(f"y shape: {y.shape}")

    x1 = torch.randn(3,100,50)
    down = DepthwiseSeparableConv1dSubsampling10(50,200)
    y1,_ = down(x1,None)
    print(f"y1 shape: {y1.shape}")

    
    x2 = torch.randn(3,100,50)
    x2_ilen = [100,95,80]
    x2_ilen = torch.LongTensor(x2_ilen)
    x2_mask = lengths_to_padding_mask(x2_ilen)[:,None,:]
    print(f"x2_mask shape: {x2_mask.shape}, x2_mask: {x2_mask}")
    down = DepthwiseSeparableConv1dSubsampling10(50,200)
    y2,x2_sub_mask = down(x2,x2_mask)

    print(f"y2 shape: {y2.shape}, x2_sub_mask shape: {x2_sub_mask.shape}, x2_sub_mask: {x2_sub_mask}")
    
