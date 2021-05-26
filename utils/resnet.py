import paddle.nn as nn


class IRBlock(nn.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True):
        super(IRBlock, self).__init__()
        self.bn0 = nn.BatchNorm2D(inplanes)
        self.conv1 = nn.Conv2D(inplanes, inplanes, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2D(inplanes)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2D(inplanes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2D(planes)
        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(planes)

    def forward(self, x):
        residual = x
        out = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.prelu(out)

        return out


class SEBlock(nn.Layer):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.PReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).reshape((b, c))
        y = self.fc(y).reshape((b, c, 1, 1))
        return x * y


class ResNet(nn.Layer):
    def __init__(self, block, layers, use_se=True):
        self.inplanes = 64
        self.use_se = use_se
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2D(1, 64, kernel_size=3)
        self.bn1 = nn.BatchNorm2D(64)
        self.prelu = nn.PReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=3)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.pool = nn.AdaptiveMaxPool2D((1, 1))
        self.bn4 = nn.BatchNorm2D(512)
        self.dropout = nn.Dropout()
        self.flatten = nn.Flatten()
        self.fc5 = nn.Linear(512, 512)
        self.bn5 = nn.BatchNorm1D(512)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2D(planes * block.expansion),)
        layers = [block(self.inplanes, planes, stride, downsample, use_se=self.use_se)]
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_se=self.use_se))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = self.bn4(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc5(x)
        x = self.bn5(x)

        return x


def resnet34(use_se=True):
    model = ResNet(IRBlock, [3, 4, 6, 3], use_se=use_se)
    return model
