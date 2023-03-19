import paddle.nn as nn

from ppvector.models.pooling import AttentiveStatisticsPooling, TemporalAveragePooling
from ppvector.models.pooling import SelfAttentivePooling, TemporalStatisticsPooling
from ppvector.models.utils import Conv1d, BatchNorm1d


class SEBasicBlock(nn.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=8):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2D(inplanes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2D(planes)
        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2D(planes)
        self.relu = nn.ReLU()
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class SEBottleneck(nn.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=8):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2D(inplanes, planes, kernel_size=1)
        self.bn1 = nn.BatchNorm2D(planes)
        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2D(planes)
        self.conv3 = nn.Conv2D(planes, planes * 4, kernel_size=1)
        self.bn3 = nn.BatchNorm2D(planes * 4)
        self.relu = nn.ReLU()
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SELayer(nn.Layer):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).reshape([b, c])
        y = self.fc(y).reshape([b, c, 1, 1])
        return x * y


class ResNetSE(nn.Layer):
    def __init__(self, input_size=80, layers=[3, 4, 6, 3], num_filters=[32, 64, 128, 256], embd_dim=192,
                 pooling_type="ASP"):
        super(ResNetSE, self).__init__()
        self.inplanes = num_filters[0]
        self.emb_size = embd_dim
        self.conv1 = nn.Conv2D(1, num_filters[0], kernel_size=3, stride=(1, 1), padding=1)
        self.bn1 = nn.BatchNorm2D(num_filters[0])
        self.relu = nn.ReLU()

        self.layer1 = self._make_layer(SEBottleneck, num_filters[0], layers[0])
        self.layer2 = self._make_layer(SEBottleneck, num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(SEBottleneck, num_filters[2], layers[2], stride=(2, 2))
        self.layer4 = self._make_layer(SEBottleneck, num_filters[3], layers[3], stride=(2, 2))

        cat_channels = num_filters[3] * SEBottleneck.expansion * (input_size // 8)
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
                nn.Conv2D(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride),
                nn.BatchNorm2D(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, lengths=None):
        x = x.transpose([0, 2, 1])
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

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
