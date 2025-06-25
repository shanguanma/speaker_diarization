# Copyright (c) 2021 Shuai Wang (wsstriving@gmail.com)
#               2022 Zhengyang Chen (chenzhengyang117@gmail.com)
#               2023 Bing Han (hanbing97@sjtu.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ResNet in PyTorch.

Some modifications from the original architecture:
1. Smaller kernel size for the input layer
2. Smaller number of Channels
3. No max_pooling involved

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pooling_layers_wespeaker as pooling_layers


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
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
                    stride=stride,
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        m_channels=32,
        feat_dim=40,
        embed_dim=128,
        pooling_func="TSTP",
        two_emb_layer=True,
        speech_encoder=False,
    ):
        super(ResNet, self).__init__()
        self.in_planes = m_channels
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.stats_dim = int(feat_dim / 8) * m_channels * 8
        self.two_emb_layer = two_emb_layer

        self.conv1 = nn.Conv2d(
            1, m_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, m_channels * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, m_channels * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, m_channels * 8, num_blocks[3], stride=2)
        if not speech_encoder:
            self.pool = getattr(pooling_layers, pooling_func)(
                in_dim=self.stats_dim * block.expansion
            )
            self.pool_out_dim = self.pool.get_out_dim()
            self.seg_1 = nn.Linear(self.pool_out_dim, embed_dim)
            if self.two_emb_layer:
                self.seg_bn_1 = nn.BatchNorm1d(embed_dim, affine=False)
                self.seg_2 = nn.Linear(embed_dim, embed_dim)
            else:
                self.seg_bn_1 = nn.Identity()
                self.seg_2 = nn.Identity()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x,get_time_out=False):
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)

        x = x.unsqueeze_(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        print(f"befor time out out shape {out.shape}")
        if get_time_out:
            #out(B,C,F,T) it is from https://github.com/wenet-e2e/wespeaker/blob/master/wespeaker/models/pooling_layers.py#L79
                                     # https://github.com/wenet-e2e/wespeaker/blob/master/wespeaker/models/pooling_layers.py#L122
            out = out.reshape(out.shape[0], out.shape[1] * out.shape[2], out.shape[3])
            assert len(out.shape) == 3 #(B,D,T)
            #out = out.permute(0,2,1) # (B,D,T) -> (B,T,D)
            return out           # D is feature dimension, T is time-dimension (frame-dimension)
                                 # D is 2560 for ResNet34 and ResNet293
        else:
            stats = self.pool(out)

            embed_a = self.seg_1(stats)
            if self.two_emb_layer:
                out = F.relu(embed_a)
                out = self.seg_bn_1(out)
                embed_b = self.seg_2(out)
                return embed_a, embed_b
            else:
                return torch.tensor(0.0), embed_a
    
class ResNetWithGSP(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        m_channels=32,
        feat_dim=40,
        embed_dim=128,
        segment_length=16,
        segment_shift=8,
        out_dim=256,
    ):
        super(ResNetWithGSP, self).__init__()
        self.in_planes = m_channels
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.segment_length = segment_length
        self.segment_shift = segment_shift
        self.conv1 = nn.Conv2d(
            1, m_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, m_channels * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, m_channels * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, m_channels * 8, num_blocks[3], stride=2)
        # 输出通道数
        self.out_channels = m_channels * 8 * block.expansion
        self.output_proj = nn.Linear(self.out_channels*2, out_dim) # because cat two 'c',

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def segmental_stat_pooling(self, feat, segment_length, segment_shift):
        # feat: [B, C, H, T]
        B, C, H, T = feat.shape
        #print(f"feat shape: {feat.shape}")
        segments = []
        for start in range(0, T - segment_length + 1, segment_shift):
            seg = feat[..., start:start+segment_length]  # [B, C, H, segment_length]
            seg = seg.reshape(B, C, -1)  # [B, C, H*segment_length]
            mu = seg.mean(dim=-1)        # [B, C]
            sigma = seg.std(dim=-1)      # [B, C]
            pooled = torch.cat([mu, sigma], dim=-1)  # [B, 2C]
            segments.append(pooled)
        if len(segments) == 0:
            # 若输入太短，补一个全0
            pooled = torch.zeros(B, 2*C, device=feat.device, dtype=feat.dtype)
            segments.append(pooled)
        return torch.stack(segments, dim=1)  # [B, num_segments, 2C]

    def segmental_stat_pooling_time_aligned(self, feat, segment_length):
        # feat: [B, C, H, T]
        B, C, H, T = feat.shape
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
            seg = feat[..., start:end]  # [B, C, H, segment_length]
            seg = seg.reshape(B, C, -1)  # [B, C, H*segment_length]
            mu = seg.mean(dim=-1)        # [B, C]
            sigma = seg.std(dim=-1)      # [B, C]
            pooled = torch.cat([mu, sigma], dim=-1)  # [B, 2C]
            pooled_seq.append(pooled)
        # [T, B, 2C] -> [B, T, 2C]
        return torch.stack(pooled_seq, dim=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        #print(f"x shape {x.shape}")
        x = x.unsqueeze_(1)
        #print(f"after x.unsqueeze(1) x shape: {x.shape} ")
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out: [B, C, H, T]
        #print(f"out shape: {out.shape}")
        pooled_seq = self.segmental_stat_pooling_time_aligned(
            out, self.segment_length
        )  # [B, T, 2C]
        #pooled_seq = self.segmental_stat_pooling(out, self.segment_length,self.segment_shift)
        pooled_seq = self.output_proj(pooled_seq) # [B, T, out_dim]
        return pooled_seq


def ResNet18(feat_dim, embed_dim, pooling_func="TSTP", two_emb_layer=True, speech_encoder=False):
    return ResNet(
        BasicBlock,
        [2, 2, 2, 2],
        feat_dim=feat_dim,
        embed_dim=embed_dim,
        pooling_func=pooling_func,
        two_emb_layer=two_emb_layer,
        speech_encoder=speech_encoder,
    )


def ResNet34(feat_dim, embed_dim, pooling_func="TSTP", two_emb_layer=True,speech_encoder=False):
    return ResNet(
        BasicBlock,
        [3, 4, 6, 3],
        m_channels=32,
        feat_dim=feat_dim,
        embed_dim=embed_dim,
        pooling_func=pooling_func,
        two_emb_layer=two_emb_layer,
        speech_encoder=speech_encoder,

    )
# reference from the paper `Sequence-to-Sequence Neural Diarization with Automatic Speaker Detection and Representation`
def ResNet34_64ch(feat_dim, embed_dim, pooling_func="TSTP", two_emb_layer=True,speech_encoder=False):
    return ResNet(
        BasicBlock,
        [3, 4, 6, 3],
        m_channels=64,
        feat_dim=feat_dim,
        embed_dim=embed_dim,
        pooling_func=pooling_func,
        two_emb_layer=two_emb_layer,
        speech_encoder=speech_encoder,

    )


def ResNet50(feat_dim, embed_dim, pooling_func="TSTP", two_emb_layer=True,speech_encoder=False):
    return ResNet(
        Bottleneck,
        [3, 4, 6, 3],
        feat_dim=feat_dim,
        embed_dim=embed_dim,
        pooling_func=pooling_func,
        two_emb_layer=two_emb_layer,
        speech_encoder=speech_encoder,
    )


def ResNet101(feat_dim, embed_dim, pooling_func="TSTP", two_emb_layer=True,speech_encoder=False):
    return ResNet(
        Bottleneck,
        [3, 4, 23, 3],
        feat_dim=feat_dim,
        embed_dim=embed_dim,
        pooling_func=pooling_func,
        two_emb_layer=two_emb_layer,
        speech_encoder=speech_encoder,
    )


def ResNet152(feat_dim, embed_dim, pooling_func="TSTP", two_emb_layer=True,speech_encoder=False):
    return ResNet(
        Bottleneck,
        [3, 8, 36, 3],
        feat_dim=feat_dim,
        embed_dim=embed_dim,
        pooling_func=pooling_func,
        two_emb_layer=two_emb_layer,
        speech_encoder=speech_encoder,
    )


def ResNet221(feat_dim, embed_dim, pooling_func="TSTP", two_emb_layer=True,speech_encoder=False):
    return ResNet(
        Bottleneck,
        [6, 16, 48, 3],
        feat_dim=feat_dim,
        embed_dim=embed_dim,
        pooling_func=pooling_func,
        two_emb_layer=two_emb_layer,
        speech_encoder=speech_encoder,
    )


def ResNet293(feat_dim, embed_dim, pooling_func="TSTP", two_emb_layer=True,speech_encoder=False):
    return ResNet(
        Bottleneck,
        [10, 20, 64, 3],
        feat_dim=feat_dim,
        embed_dim=embed_dim,
        pooling_func=pooling_func,
        two_emb_layer=two_emb_layer,
        speech_encoder=speech_encoder,
    )


if __name__ == "__main__":
    x = torch.zeros(10, 200, 80)# (B,T,F)
    model = ResNet34(feat_dim=80, embed_dim=256, pooling_func="TSTP")
    #model = ResNet293(feat_dim=80, embed_dim=256, pooling_func="TSTP")
    model.eval()
    out = model(x)
    # [10, 256, 10, 25]
    out_1 = model(x,get_time_out=True) # torch.Size([10, 2560, 25]) (B,F,T) 200/25=8  downsample is 8
    print(f"out_1 shape: {out_1.shape}")
    #print(f"model: {model}")
    print(out[-1].size())
    
    # resnet-32ch
    model1 = ResNetWithGSP(BasicBlock,[3, 4, 6, 3],m_channels=32, feat_dim=80,embed_dim=256,out_dim=256)
    model1.eval()
    out1 = model1(x)
    print(f"out1 shape : {out1.shape}") #  torch.Size([10, 25, 256]) [B,T,D] downsample is 8
    num_params = sum(p.numel() for p in model1.parameters())
    print("{} M".format(num_params / 1e6))
    
    # resnet34-64ch
    model2 = ResNetWithGSP(BasicBlock,[3, 4, 6, 3],m_channels=64, feat_dim=80,embed_dim=256,out_dim=256)
    model2.eval()
    out2 = model2(x)
    print(f"out2 shape : {out2.shape}") #  torch.Size([10, 25, 256]) [B,T,D] downsample is 8
    num_params = sum(p.numel() for p in model2.parameters()) # 
    print("{} M".format(num_params / 1e6))

    # resnet152
    # it is same as the version from `Sequence-to-Sequence Neural Diarization with Automatic Speaker Detection and Representation`
    model3 = ResNetWithGSP(BasicBlock,[3, 8, 36, 3],m_channels=64, feat_dim=80,embed_dim=256,out_dim=256)
    model3.eval()
    out3 = model3(x)
    print(f"out3 shape : {out3.shape}")# out3 shape : torch.Size([10, 25, 256]) [B,T,D]downsample is 8
    num_params = sum(p.numel() for p in model3.parameters())
    print("{} M".format(num_params / 1e6))

    #num_params = sum(p.numel() for p in model.parameters())
    #print("{} M".format(num_params / 1e6))

    # from thop import profile
    # x_np = torch.randn(1, 200, 80)
    # flops, params = profile(model, inputs=(x_np, ))
    # print("FLOPS: {} G, Params: {} M".format(flops / 1e9, params / 1e6))
