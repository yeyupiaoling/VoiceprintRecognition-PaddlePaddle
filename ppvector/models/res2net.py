import math

import paddle
import paddle.nn as nn

from ppvector.models.pooling import AttentiveStatisticsPooling, TemporalAveragePooling
from ppvector.models.pooling import SelfAttentivePooling, TemporalStatisticsPooling
from ppvector.models.utils import BatchNorm1d, Conv1d


class Bottle2neck(nn.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2D(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm2D(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2D(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2D(width, width, kernel_size=3, stride=stride, padding=1))
            bns.append(nn.BatchNorm2D(width))
        self.convs = nn.LayerList(convs)
        self.bns = nn.LayerList(bns)

        self.conv3 = nn.Conv2D(width * scale, planes * self.expansion, kernel_size=1)
        self.bn3 = nn.BatchNorm2D(planes * self.expansion)

        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = paddle.split(out, self.scale, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = paddle.concat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = paddle.concat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = paddle.concat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Res2Net(nn.Layer):

    def __init__(self, input_size=80, layers=[3, 4, 6, 3], base_width=26, scale=4, embd_dim=192, pooling_type="ASP"):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.base_width = base_width
        self.scale = scale
        self.emb_size = embd_dim
        self.conv1 = nn.Conv2D(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2D(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottle2neck, 64, layers[0])
        self.layer2 = self._make_layer(Bottle2neck, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(Bottle2neck, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(Bottle2neck, 512, layers[3], stride=2)

        cat_channels = 512 * Bottle2neck.expansion * (input_size // 32)
        if pooling_type == "ASP":
            self.pooling = AttentiveStatisticsPooling(cat_channels, 128)
            self.pooling_bn = BatchNorm1d(input_size=cat_channels * 2)
            # Final linear transformation
            self.fc = Conv1d(in_channels=cat_channels * 2,
                             out_channels=self.emb_size,
                             kernel_size=1)
        elif pooling_type == "SAP":
            self.asp = SelfAttentivePooling(cat_channels, 128)
            self.asp_bn = nn.BatchNorm1D(cat_channels)
            # Final linear transformation
            self.fc = Conv1d(in_channels=cat_channels,
                             out_channels=self.emb_size,
                             kernel_size=1)
        elif pooling_type == "TAP":
            self.asp = TemporalAveragePooling()
            self.asp_bn = nn.BatchNorm1D(cat_channels)
            # Final linear transformation
            self.fc = Conv1d(in_channels=cat_channels,
                             out_channels=self.emb_size,
                             kernel_size=1)
        elif pooling_type == "TSP":
            self.asp = TemporalStatisticsPooling()
            self.asp_bn = nn.BatchNorm1D(cat_channels * 2)
            # Final linear transformation
            self.fc = Conv1d(in_channels=cat_channels * 2,
                             out_channels=self.emb_size,
                             kernel_size=1)
        else:
            raise Exception(f'没有{pooling_type}池化层！')

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2D(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample=downsample,
                        stype='stage', baseWidth=self.base_width, scale=self.scale)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.base_width, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x, lengths=None):
        x = x.transpose([0, 2, 1])
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.reshape([x.shape[0], -1, x.shape[-1]])

        out = self.pooling(x)
        out = self.pooling_bn(out)
        # Final linear transformation
        out = self.fc(out).squeeze(-1)  # (N, emb_size, 1) -> (N, emb_size)
        return out
