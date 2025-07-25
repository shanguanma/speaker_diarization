# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp


def get_nonlinear(config_str, channels):
    nonlinear = nn.Sequential()
    for name in config_str.split("-"):
        if name == "relu":
            nonlinear.add_module("relu", nn.ReLU(inplace=True))
        elif name == "prelu":
            nonlinear.add_module("prelu", nn.PReLU(channels))
        elif name == "batchnorm":
            nonlinear.add_module("batchnorm", nn.BatchNorm1d(channels))
        elif name == "batchnorm_":
            nonlinear.add_module("batchnorm", nn.BatchNorm1d(channels, affine=False))
        else:
            raise ValueError("Unexpected module ({}).".format(name))
    return nonlinear


def statistics_pooling(x, dim=-1, keepdim=False, unbiased=True, eps=1e-2):
    mean = x.mean(dim=dim)
    std = x.std(dim=dim, unbiased=unbiased)
    stats = torch.cat([mean, std], dim=-1)
    if keepdim:
        stats = stats.unsqueeze(dim=dim)
    return stats


class StatsPool(nn.Module):
    def forward(self, x):
        return statistics_pooling(x)


class TDNNLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        config_str="batchnorm-relu",
    ):
        super(TDNNLayer, self).__init__()
        if padding < 0:
            assert (
                kernel_size % 2 == 1
            ), "Expect equal paddings, but got even kernel size ({})".format(
                kernel_size
            )
            padding = (kernel_size - 1) // 2 * dilation
        self.linear = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        x = self.linear(x)
        x = self.nonlinear(x)
        return x


class CAMLayer(nn.Module):
    def __init__(
        self,
        bn_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        bias,
        reduction=2,
    ):
        super(CAMLayer, self).__init__()
        self.linear_local = nn.Conv1d(
            bn_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.linear1 = nn.Conv1d(bn_channels, bn_channels // reduction, 1)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Conv1d(bn_channels // reduction, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.linear_local(x)
        context = x.mean(-1, keepdim=True) + self.seg_pooling(x)
        context = self.relu(self.linear1(context))
        m = self.sigmoid(self.linear2(context))
        return y * m

    def seg_pooling(self, x, seg_len=100, stype="avg"):
        if stype == "avg":
            seg = F.avg_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        elif stype == "max":
            seg = F.max_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        else:
            raise ValueError("Wrong segment pooling type.")
        shape = seg.shape
        seg = seg.unsqueeze(-1).expand(*shape, seg_len).reshape(*shape[:-1], -1)
        seg = seg[..., : x.shape[-1]]
        return seg


class CAMDenseTDNNLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        bn_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=False,
        config_str="batchnorm-relu",
        memory_efficient=False,
    ):
        super(CAMDenseTDNNLayer, self).__init__()
        assert (
            kernel_size % 2 == 1
        ), "Expect equal paddings, but got even kernel size ({})".format(kernel_size)
        padding = (kernel_size - 1) // 2 * dilation
        self.memory_efficient = memory_efficient
        self.nonlinear1 = get_nonlinear(config_str, in_channels)
        self.linear1 = nn.Conv1d(in_channels, bn_channels, 1, bias=False)
        self.nonlinear2 = get_nonlinear(config_str, bn_channels)
        self.cam_layer = CAMLayer(
            bn_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def bn_function(self, x):
        return self.linear1(self.nonlinear1(x))

    def forward(self, x):
        if self.training and self.memory_efficient:
            x = cp.checkpoint(self.bn_function, x)
        else:
            x = self.bn_function(x)
        x = self.cam_layer(self.nonlinear2(x))
        return x


class CAMDenseTDNNBlock(nn.ModuleList):
    def __init__(
        self,
        num_layers,
        in_channels,
        out_channels,
        bn_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=False,
        config_str="batchnorm-relu",
        memory_efficient=False,
    ):
        super(CAMDenseTDNNBlock, self).__init__()
        for i in range(num_layers):
            layer = CAMDenseTDNNLayer(
                in_channels=in_channels + i * out_channels,
                out_channels=out_channels,
                bn_channels=bn_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                bias=bias,
                config_str=config_str,
                memory_efficient=memory_efficient,
            )
            self.add_module("tdnnd%d" % (i + 1), layer)

    def forward(self, x):
        for layer in self:
            x = torch.cat([x, layer(x)], dim=1)
        return x


class TransitLayer(nn.Module):
    def __init__(
        self, in_channels, out_channels, bias=True, config_str="batchnorm-relu"
    ):
        super(TransitLayer, self).__init__()
        self.nonlinear = get_nonlinear(config_str, in_channels)
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        x = self.nonlinear(x)
        x = self.linear(x)
        return x


class DenseLayer(nn.Module):
    def __init__(
        self, in_channels, out_channels, bias=False, config_str="batchnorm-relu"
    ):
        super(DenseLayer, self).__init__()
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        if len(x.shape) == 2:
            x = self.linear(x.unsqueeze(dim=-1)).squeeze(dim=-1)
        else:
            x = self.linear(x)
        x = self.nonlinear(x)
        return x


class BasicResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=(stride, 1), padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=(stride, 1),
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FCM(nn.Module):
    def __init__(
        self, block=BasicResBlock, num_blocks=[2, 2], m_channels=32, feat_dim=80
    ):
        super(FCM, self).__init__()
        self.in_planes = m_channels
        self.conv1 = nn.Conv2d(
            1, m_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(m_channels)

        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, m_channels, num_blocks[0], stride=2)

        self.conv2 = nn.Conv2d(
            m_channels, m_channels, kernel_size=3, stride=(2, 1), padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(m_channels)
        self.out_channels = m_channels * (feat_dim // 8)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.relu(self.bn2(self.conv2(out)))

        shape = out.shape
        out = out.reshape(shape[0], shape[1] * shape[2], shape[3])
        return out


class CAMPPlus(nn.Module):
    def __init__(
        self,
        feat_dim=80,
        #embedding_size=512,
        embedding_size=192,
        growth_rate=32,
        bn_size=4,
        init_channels=128,
        config_str="batchnorm-relu",
        memory_efficient=True,
        #speech_encoder=False,
        #speech_encoder=True,
    ):
        super(CAMPPlus, self).__init__()

        self.head = FCM(feat_dim=feat_dim)
        channels = self.head.out_channels

        self.xvector = nn.Sequential(
            OrderedDict(
                [
                    (
                        "tdnn",
                        TDNNLayer(
                            channels,
                            init_channels,
                            5,
                            stride=2,
                            dilation=1,
                            padding=-1,
                            config_str=config_str,
                        ),
                    ),
                ]
            )
        )
        channels = init_channels
        for i, (num_layers, kernel_size, dilation) in enumerate(
            zip((12, 24, 16), (3, 3, 3), (1, 2, 2))
        ):
            block = CAMDenseTDNNBlock(
                num_layers=num_layers,
                in_channels=channels,
                out_channels=growth_rate,
                bn_channels=bn_size * growth_rate,
                kernel_size=kernel_size,
                dilation=dilation,
                config_str=config_str,
                memory_efficient=memory_efficient,
            )
            self.xvector.add_module("block%d" % (i + 1), block)
            channels = channels + num_layers * growth_rate
            self.xvector.add_module(
                "transit%d" % (i + 1),
                TransitLayer(
                    channels, channels // 2, bias=False, config_str=config_str
                ),
            )
            channels //= 2

        self.xvector.add_module("out_nonlinear", get_nonlinear(config_str, channels))

        self.xvector.add_module("stats", StatsPool())
        #if not speech_encoder:
        #    self.xvector.add_module(
        #       "dense", DenseLayer(channels * 2, embedding_size, config_str="batchnorm_")
        #    )
        self.xvector.add_module(
            "dense", DenseLayer(channels * 2, embedding_size, config_str="batchnorm_")
        )
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, get_time_out=False):
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        x = self.head(x)
        if get_time_out:
            #print(f"self.xvector[:-2]: {self.xvector[:-2]}")
            #print(f"self.xvector[-1]: {self.xvector[-1]}")
            x = self.xvector[:-2](x) # (B,F,T)
            #x = self.xvector[:-1](x)
            #x = self.xvector(x)
        else:
            x = self.xvector(x)
        return x

class CAMPPlusWithGSP(nn.Module):
    def __init__(
        self,
        feat_dim=80,
        #embedding_size=512,
        embedding_size=192,
        growth_rate=32,
        bn_size=4,
        init_channels=128,
        config_str="batchnorm-relu",
        memory_efficient=True,
        #speech_encoder=False,
        #speech_encoder=True,
        segment_length=16,
        #segment_shift=8,
        out_dim=256,
        use_gsp=False,
    ):
        super(CAMPPlusWithGSP, self).__init__()

        self.head = FCM(feat_dim=feat_dim)
        channels = self.head.out_channels

        self.xvector = nn.Sequential(
            OrderedDict(
                [
                    (
                        "tdnn",
                        TDNNLayer(
                            channels,
                            init_channels,
                            5,
                            stride=2,
                            dilation=1,
                            padding=-1,
                            config_str=config_str,
                        ),
                    ),
                ]
            )
        )
        channels = init_channels
        for i, (num_layers, kernel_size, dilation) in enumerate(
            zip((12, 24, 16), (3, 3, 3), (1, 2, 2))
        ):
            block = CAMDenseTDNNBlock(
                num_layers=num_layers,
                in_channels=channels,
                out_channels=growth_rate,
                bn_channels=bn_size * growth_rate,
                kernel_size=kernel_size,
                dilation=dilation,
                config_str=config_str,
                memory_efficient=memory_efficient,
            )
            self.xvector.add_module("block%d" % (i + 1), block)
            channels = channels + num_layers * growth_rate
            self.xvector.add_module(
                "transit%d" % (i + 1),
                TransitLayer(
                    channels, channels // 2, bias=False, config_str=config_str
                ),
            )
            channels //= 2

        self.xvector.add_module("out_nonlinear", get_nonlinear(config_str, channels))

        self.xvector.add_module("stats", StatsPool())
        #if not speech_encoder:
        #    self.xvector.add_module(
        #       "dense", DenseLayer(channels * 2, embedding_size, config_str="batchnorm_")
        #    )
        self.xvector.add_module(
            "dense", DenseLayer(channels * 2, embedding_size, config_str="batchnorm_")
        )
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.use_gsp = use_gsp
        self.output_proj=nn.Linear(512, out_dim)
        if self.use_gsp:
            self.out_dim = out_dim
            self.segment_length=segment_length
            self.output_gsp_proj = nn.Linear(out_dim*2, out_dim) #because cat two 'F1'

    def segmental_stat_pooling_time_aligned(self, feat, segment_length):
        feat = feat.permute(0,2,1) # (B,T,F1) -> (B,F1,T)
        # feat: [B, F1, T]
        
        B, F1, T = feat.shape
        #print(f"feat shape: {feat.shape}")
        pooled_seq = []
        for t in range(T):
            # 保证每个t都能取到一个segment
            start = max(0, t - segment_length // 2)
            end = min(T, start + segment_length)
            if end - start < segment_length:
                # 边界补齐
                start = end - segment_length
                if start < 0:
                    start = 0
                    end = segment_length
            seg = feat[..., start:end]  # [B, F1, segment_length]
            seg = seg.reshape(B, F1, -1)  # [B, F1, segment_length]
            mu = seg.mean(dim=-1)        # [B, F1]
            sigma = seg.std(dim=-1)      # [B, F1]
            pooled = torch.cat([mu, sigma], dim=-1)  # [B, 2F1]
            pooled_seq.append(pooled)
        # [T, B, 2F1] -> [B, T, 2F1]
        return torch.stack(pooled_seq, dim=1)
    
    def forward(self, x,):
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        x = self.head(x)
     
        #print(f"self.xvector[:-2]: {self.xvector[:-2]}")
        #print(f"self.xvector[-1]: {self.xvector[-1]}")
        x = self.xvector[:-2](x) # (B,F,T) # frame level before pool layer
        x = x.permute(0,2,1)#(B,F,T) -> (B,T,F)
        #print(f"x: shape: {x.shape}")
        x = self.output_proj(x)# (B,T,out_dim)
        if self.use_gsp:
            x = self.segmental_stat_pooling_time_aligned(
                x, self.segment_length)  # [B, T, 2F1]
            x = self.output_gsp_proj(x) # [B, T, out_dim]
        return x



if __name__ == '__main__':

    x = torch.zeros(10, 200, 80) # B,T,F
    #x = torch.zeros(10, 398, 80) # B,T,F
    #x = torch.zeros(10,300)
    #model = CAMPPlus(feat_dim=80,embedding_size=192,speech_encoder=True, )
    model = CAMPPlus(feat_dim=80,embedding_size=192)
    model.eval()
    out_1 = model(x,get_time_out=True) # torch.Size([10, 512, 100]) (B,F,T)  downsample is 200/100=2,
    out_2 = model(x,get_time_out=False) # torch.Size([10, 192])
    print(f"out_1 shape: {out_1.shape}") #out_1 shape: torch.Size([10, 512, 100]) #(B,F,T)
    print(f"out_2 shape: {out_2.shape}") #out_2 shape: torch.Size([10, 192]) #(B,F)
    #print(f"model: {str(model)}")
    #print(out_1.shape) # torch.Size([10, 512])
    model2 = CAMPPlusWithGSP(feat_dim=80, use_gsp=False)
    model2.eval()
    out2 = model2(x)
    print(f"use_gsp=false, out2 shape: {out2.shape}")
    model3 = CAMPPlusWithGSP(feat_dim=80, use_gsp=True)
    model3.eval()
    out3 = model3(x)
    print(f"use_gsp=True out3 shape: {out3.shape}")

    num_params = sum(param.numel() for param in model.parameters())
    #for name, v in model.state_dict():
    for name, v in model.named_parameters():
        print(f"name: {name}, v: {v.shape}")
    print("{} M".format(num_params / 1e6)) # 6.61M

