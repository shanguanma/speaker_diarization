import torch.nn as nn
from .net_utils import c2_xavier_fill
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
        # use 1x1 conv instead
        self.mask_features = nn.Conv1d(
            feat_dim,
            feat_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        c2_xavier_fill(self.mask_features)
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
        x = self.act(x)# 
        x = x.permute(0,2,1) # BxT'xD -> B,D,T'
        mask_feat = self.mask_features(x)# B,D,T'
        mask_feat = mask_feat.unsqueeze(-1) # (B,D,T',1)
        feat = [x.unsqueeze(-1)] # (B,D,T',1)
        return mask_feat, feat

