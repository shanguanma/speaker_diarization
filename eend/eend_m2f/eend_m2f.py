from typing import List
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.models import Conformer
from scipy.optimize import linear_sum_assignment


class Mask(nn.Module):
    def __init__(
        self,
    ):
        super(Mask, self).__init__()
        pass

    def forward(self, x):
        pass


class Query(nn.Module):
    def __init__(
        self,
    ):
        super(Mask, self).__init__()
        pass

    def forward(self, x):
        pass


class DownsampleLayer(nn.Module):
    """
    it is modified from https://github.com/seungjunlee96/Depthwise-Separable-Convolution_Pytorch
    """
    def __init__(self,nin, nout, kernel_size = 15, padding = 3, stride = 10,bias=False):
        super(DownsampleLayer, self).__init__()
        self.depthwise = nn.Conv1d(nin, nin, kernel_size=kernel_size,stride=stride, padding=padding, groups=nin, bias=bias)
        self.pointwise = nn.Conv1d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class EENDM2F(nn.Module):
    def __init__(
        self,
    ):
        super(EENDM2F, self).__init__()
        # self.downsample=
        # self.conformer=
        # self.upsample=
        pass

    def forward(self, src: List[Tensor],):
        """
        src: List[Tensor], it is log-mel feature. dimension is 23. frame shift is 10ms frame_win is 25ms

        """
        pass
    
    def compute_DER(sys, ref):
        """
        Args:
            sys(boolean, torch.Tensor): predict of network, shape (BxTxS^)
            ref(boolean, torch.Tensor): dataset offers speaker label. shape(BxTxS)
            it is from `EEND-M2F: Masked-attention mask transformers for speaker diarization`
        """
        nsys = sys.sum(1)
        nref = ref.sum(1)

        correct = torch.logical_and(sys[:, :, None], ref[:, None, :])
        matching = linear_sum_assignment(correct, maximize=True)
        ncor = correct[matching].sum()

        ms = torch.clamp(nref - nsys, 0).sum()
        fa = torch.clamp(nsys - nref, 0).sum()
        se = torch.min(nsys, nref).sum() - ncor
        der = torch.max(nsys, nref).sum() - ncor  # == ms + fa + se

        z = nref.sum()
        return ms / z, fa / z, se / z, der / z

if __name__ == "__main__":
   x = torch.randn(2,23,200)
   down = DownsampleLayer(23,256,15)
   output = down(x)
   print(f"output.shape: {output.shape}")
