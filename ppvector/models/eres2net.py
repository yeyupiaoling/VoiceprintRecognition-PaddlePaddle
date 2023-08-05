import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppvector.models.pooling import AttentiveStatisticsPooling, TemporalAveragePooling, TemporalStatsPool
from ppvector.models.pooling import SelfAttentivePooling, TemporalStatisticsPooling


class ReLU(nn.Hardtanh):

    def __init__(self, inplace=False):
        super(ReLU, self).__init__(0, 20, inplace)

    def __repr__(self):
        inplace_str = 'inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' + inplace_str + ')'


def conv1x1(in_planes, out_planes, stride=1):
    "1x1 convolution without padding"
    return nn.Conv2D(in_planes, out_planes, kernel_size=1, stride=stride, padding=0)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2D(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)


class AFF(nn.Layer):

    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2D(channels * 2, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(inter_channels),
            nn.SiLU(inplace=True),
            nn.Conv2D(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(channels),
        )

    def forward(self, x, ds_y):
        xa = torch.cat((x, ds_y), dim=1)
        x_att = self.local_att(xa)
        x_att = 1.0 + torch.tanh(x_att)
        xo = torch.mul(x, x_att) + torch.mul(ds_y, 2.0 - x_att)

        return xo


class BasicBlockERes2Net(nn.Layer):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, baseWidth=24, scale=3):
        super(BasicBlockERes2Net, self).__init__()
        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = conv1x1(in_planes, width * scale, stride)
        self.bn1 = nn.BatchNorm2D(width * scale)
        self.nums = scale

        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(conv3x3(width, width))
            bns.append(nn.BatchNorm2D(width))
        self.convs = nn.LayerList(convs)
        self.bns = nn.LayerList(bns)
        self.relu = ReLU(inplace=True)

        self.conv3 = conv1x1(width * scale, planes * self.expansion)
        self.bn3 = nn.BatchNorm2D(planes * self.expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2D(in_planes, self.expansion * planes, kernel_size=1, stride=stride),
                nn.BatchNorm2D(self.expansion * planes))
        self.stride = stride
        self.width = width
        self.scale = scale

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        spx = paddle.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = paddle.concat((out, sp), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.shortcut(x)
        out += residual
        out = self.relu(out)

        return out


class BasicBlockERes2Net_diff_AFF(nn.Layer):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, baseWidth=24, scale=3):
        super(BasicBlockERes2Net_diff_AFF, self).__init__()
        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = conv1x1(in_planes, width * scale, stride)
        self.bn1 = nn.BatchNorm2D(width * scale)

        self.nums = scale

        convs = []
        fuse_models = []
        bns = []
        for i in range(self.nums):
            convs.append(conv3x3(width, width))
            bns.append(nn.BatchNorm2D(width))
        for j in range(self.nums - 1):
            fuse_models.append(AFF(channels=width))

        self.convs = nn.LayerList(convs)
        self.bns = nn.LayerList(bns)
        self.fuse_models = nn.LayerList(fuse_models)
        self.relu = ReLU(inplace=True)

        self.conv3 = conv1x1(width * scale, planes * self.expansion)
        self.bn3 = nn.BatchNorm2D(planes * self.expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2D(in_planes, self.expansion * planes, kernel_size=1, stride=stride),
                nn.BatchNorm2D(self.expansion * planes))
        self.stride = stride
        self.width = width
        self.scale = scale

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        spx = paddle.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = self.fuse_models[i - 1](sp, spx[i])

            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = paddle.concat((out, sp), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.shortcut(x)
        out += residual
        out = self.relu(out)

        return out


class ERes2Net(nn.Layer):
    def __init__(self,
                 input_size,
                 block=BasicBlockERes2Net,
                 block_fuse=BasicBlockERes2Net_diff_AFF,
                 num_blocks=[3, 4, 6, 3],
                 m_channels=64,
                 embd_dim=192,
                 pooling_type='TSTP',
                 two_emb_layer=False):
        super(ERes2Net, self).__init__()
        self.in_planes = m_channels
        self.feat_dim = input_size
        self.emb_size = embd_dim
        self.stats_dim = int(input_size / 8) * m_channels * 8
        self.two_emb_layer = two_emb_layer

        self.conv1 = nn.Conv2D(1,
                               m_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.bn1 = nn.BatchNorm2D(m_channels)
        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, m_channels * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block_fuse, m_channels * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block_fuse, m_channels * 8, num_blocks[3], stride=2)

        self.layer1_downsample = nn.Conv2D(m_channels * 4, m_channels * 8, kernel_size=3, padding=1, stride=2)
        self.layer2_downsample = nn.Conv2D(m_channels * 8, m_channels * 16, kernel_size=3, padding=1, stride=2)
        self.layer3_downsample = nn.Conv2D(m_channels * 16, m_channels * 32, kernel_size=3, padding=1, stride=2)
        self.fuse_mode12 = AFF(channels=m_channels * 8)
        self.fuse_mode123 = AFF(channels=m_channels * 16)
        self.fuse_mode1234 = AFF(channels=m_channels * 32)

        self.n_stats = 1 if pooling_type == 'TAP' else 2
        if pooling_type == "ASP":
            self.pooling = AttentiveStatisticsPooling(m_channels, 128)
        elif pooling_type == "SAP":
            self.pooling = SelfAttentivePooling(m_channels, 128)
        elif pooling_type == "TAP":
            self.pooling = TemporalAveragePooling()
        elif pooling_type == "TSP":
            self.pooling = TemporalStatisticsPooling()
        elif pooling_type == "TSTP":
            self.pooling = TemporalStatsPool()
        else:
            raise Exception(f'没有{pooling_type}池化层！')

        self.seg_1 = nn.Linear(self.stats_dim * block.expansion * self.n_stats, embd_dim)
        if self.two_emb_layer:
            self.seg_bn_1 = nn.BatchNorm1D(embd_dim)
            self.seg_2 = nn.Linear(embd_dim, embd_dim)
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

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)

        x = x.unsqueeze_(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out1_downsample = self.layer1_downsample(out1)
        fuse_out12 = self.fuse_mode12(out2, out1_downsample)
        out3 = self.layer3(out2)
        fuse_out12_downsample = self.layer2_downsample(fuse_out12)
        fuse_out123 = self.fuse_mode123(out3, fuse_out12_downsample)
        out4 = self.layer4(out3)
        fuse_out123_downsample = self.layer3_downsample(fuse_out123)
        fuse_out1234 = self.fuse_mode1234(out4, fuse_out123_downsample)
        stats = self.pooling(fuse_out1234)

        embed_a = self.seg_1(stats)
        if self.two_emb_layer:
            out = F.relu(embed_a)
            out = self.seg_bn_1(out)
            embed_b = self.seg_2(out)
            return embed_b
        else:
            return embed_a
