import paddle
import paddle.nn as nn

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D

        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = nn.Conv2D(inplanes, planes, 3, padding=1, stride=stride, bias_attr=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2D(planes, planes, 3, padding=1, bias_attr=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleneckBlock(nn.Layer):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None):
        super(BottleneckBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = nn.Conv2D(inplanes, width, 1, bias_attr=False)
        self.bn1 = norm_layer(width)

        self.conv2 = nn.Conv2D(
            width,
            width,
            3,
            padding=dilation,
            stride=stride,
            groups=groups,
            dilation=dilation,
            bias_attr=False)
        self.bn2 = norm_layer(width)

        self.conv3 = nn.Conv2D(width, planes * self.expansion, 1, bias_attr=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Layer):
    """ResNet model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        Block (BasicBlock|BottleneckBlock): block module of model.
        depth (int): layers of resnet, default: 50.
        num_classes (int): output dim of last fc layer. If num_classes <=0, last fc layer
                            will not be defined. Default: 1000.
        with_pool (bool): use pool before the last fc layer or not. Default: True.

    Examples:
        .. code-block:: python

            from paddle.vision.models import ResNet
            from paddle.vision.models.resnet import BottleneckBlock, BasicBlock

            resnet50 = ResNet(BottleneckBlock, 50)

            resnet18 = ResNet(BasicBlock, 18)

    """

    def __init__(self, block, depth):
        super(ResNet, self).__init__()
        layer_cfg = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3]
        }
        layers = layer_cfg[depth]
        self._norm_layer = nn.BatchNorm2D

        self.inplanes = 64
        self.dilation = 1

        self.conv1 = nn.Conv2D(
            1,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias_attr=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.pool = nn.AdaptiveMaxPool2D((1, 1))

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(
                    self.inplanes,
                    planes * block.expansion,
                    1,
                    stride=stride,
                    bias_attr=False),
                norm_layer(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, 1, 64, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        feature = paddle.flatten(x, 1)
        return feature


def _resnet(Block, depth, **kwargs):
    model = ResNet(Block, depth, **kwargs)
    return model


def resnet18(**kwargs):
    """ResNet 18-layer model

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    Examples:
        .. code-block:: python

            from paddle.vision.models import resnet18

            # build model
            model = resnet18()

            # build model and load imagenet pretrained weight
            # model = resnet18(pretrained=True)
    """
    return _resnet(BasicBlock, 18, **kwargs)


def resnet34(**kwargs):
    """ResNet 34-layer model

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    Examples:
        .. code-block:: python

            from paddle.vision.models import resnet34

            # build model
            model = resnet34()

            # build model and load imagenet pretrained weight
            # model = resnet34(pretrained=True)
    """
    return _resnet(BasicBlock, 34, **kwargs)


def resnet50(**kwargs):
    """ResNet 50-layer model

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    Examples:
        .. code-block:: python

            from paddle.vision.models import resnet50

            # build model
            model = resnet50()

            # build model and load imagenet pretrained weight
            # model = resnet50(pretrained=True)
    """
    return _resnet(BottleneckBlock, 50, **kwargs)


def resnet101(**kwargs):
    """ResNet 101-layer model

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    Examples:
        .. code-block:: python

            from paddle.vision.models import resnet101

            # build model
            model = resnet101()

            # build model and load imagenet pretrained weight
            # model = resnet101(pretrained=True)
    """
    return _resnet(BottleneckBlock, 101, **kwargs)


def resnet152(**kwargs):
    """ResNet 152-layer model

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    Examples:
        .. code-block:: python

            from paddle.vision.models import resnet152

            # build model
            model = resnet152()

            # build model and load imagenet pretrained weight
            # model = resnet152(pretrained=True)
    """
    return _resnet(BottleneckBlock, 152, **kwargs)
