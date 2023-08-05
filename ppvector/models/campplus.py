from collections import OrderedDict

import paddle
import paddle.nn.functional as F
from paddle import nn


def get_nonlinear(config_str, channels):
    nonlinear = nn.Sequential()
    for name in config_str.split('-'):
        if name == 'relu':
            nonlinear.add_sublayer('relu', nn.ReLU())
        elif name == 'prelu':
            nonlinear.add_sublayer('prelu', nn.PReLU(channels))
        elif name == 'batchnorm':
            nonlinear.add_sublayer('batchnorm', nn.BatchNorm1D(channels))
        elif name == 'batchnorm_':
            nonlinear.add_sublayer('batchnorm', nn.BatchNorm1D(channels))
        else:
            raise ValueError('Unexpected module ({}).'.format(name))
    return nonlinear


def statistics_pooling(x, axis=-1, keepdim=False, unbiased=True, eps=1e-2):
    mean = x.mean(axis=axis)
    std = x.std(axis=axis, unbiased=unbiased)
    stats = paddle.concat([mean, std], axis=-1)
    if keepdim:
        stats = stats.unsqueeze(axis=axis)
    return stats


class StatsPool(nn.Layer):
    def forward(self, x):
        return statistics_pooling(x)


class TDNNLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=False,
                 config_str='batchnorm-relu'):
        super(TDNNLayer, self).__init__()
        if padding < 0:
            assert kernel_size % 2 == 1, 'Expect equal paddings, but got even kernel size ({})'.format(
                kernel_size)
            padding = (kernel_size - 1) // 2 * dilation
        self.linear = nn.Conv1D(in_channels,
                                out_channels,
                                kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        x = self.linear(x)
        x = self.nonlinear(x)
        return x


class CAMLayer(nn.Layer):
    def __init__(self,
                 bn_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 bias,
                 reduction=2):
        super(CAMLayer, self).__init__()
        self.linear_local = nn.Conv1D(bn_channels,
                                      out_channels,
                                      kernel_size,
                                      stride=stride,
                                      padding=padding,
                                      dilation=dilation)
        self.linear1 = nn.Conv1D(bn_channels, bn_channels // reduction, 1)
        self.relu = nn.ReLU()
        self.linear2 = nn.Conv1D(bn_channels // reduction, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.linear_local(x)
        context = x.mean(-1, keepdim=True) + self.seg_pooling(x)
        context = self.relu(self.linear1(context))
        m = self.sigmoid(self.linear2(context))
        return y * m

    def seg_pooling(self, x, seg_len=100, stype='avg'):
        if stype == 'avg':
            seg = F.avg_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        elif stype == 'max':
            seg = F.max_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        else:
            raise ValueError('Wrong segment pooling type.')
        shape = seg.shape
        seg = seg.unsqueeze(-1).expand((*shape, seg_len)).reshape((*shape[:-1], -1))
        seg = seg[..., :x.shape[-1]]
        return seg


class CAMDenseTDNNLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 bias=False,
                 config_str='batchnorm-relu',
                 memory_efficient=False):
        super(CAMDenseTDNNLayer, self).__init__()
        assert kernel_size % 2 == 1, 'Expect equal paddings, but got even kernel size ({})'.format(
            kernel_size)
        padding = (kernel_size - 1) // 2 * dilation
        self.memory_efficient = memory_efficient
        self.nonlinear1 = get_nonlinear(config_str, in_channels)
        self.linear1 = nn.Conv1D(in_channels, bn_channels, 1)
        self.nonlinear2 = get_nonlinear(config_str, bn_channels)
        self.cam_layer = CAMLayer(bn_channels,
                                  out_channels,
                                  kernel_size,
                                  stride=stride,
                                  padding=padding,
                                  dilation=dilation,
                                  bias=bias)

    def bn_function(self, x):
        return self.linear1(self.nonlinear1(x))

    def forward(self, x):
        x = self.bn_function(x)
        x = self.cam_layer(self.nonlinear2(x))
        return x


class CAMDenseTDNNBlock(nn.LayerList):
    def __init__(self,
                 num_layers,
                 in_channels,
                 out_channels,
                 bn_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 bias=False,
                 config_str='batchnorm-relu',
                 memory_efficient=False):
        super(CAMDenseTDNNBlock, self).__init__()
        for i in range(num_layers):
            layer = CAMDenseTDNNLayer(in_channels=in_channels + i * out_channels,
                                      out_channels=out_channels,
                                      bn_channels=bn_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      dilation=dilation,
                                      bias=bias,
                                      config_str=config_str,
                                      memory_efficient=memory_efficient)
            self.add_sublayer('tdnnd%d' % (i + 1), layer)

    def forward(self, x):
        for layer in self:
            x = paddle.concat([x, layer(x)], axis=1)
        return x


class TransitLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True,
                 config_str='batchnorm-relu'):
        super(TransitLayer, self).__init__()
        self.nonlinear = get_nonlinear(config_str, in_channels)
        self.linear = nn.Conv1D(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.nonlinear(x)
        x = self.linear(x)
        return x


class DenseLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=False,
                 config_str='batchnorm-relu'):
        super(DenseLayer, self).__init__()
        self.linear = nn.Conv1D(in_channels, out_channels, 1)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        if len(x.shape) == 2:
            x = self.linear(x.unsqueeze(axis=-1)).squeeze(axis=-1)
        else:
            x = self.linear(x)
        x = self.nonlinear(x)
        return x


class BasicResBlock(nn.Layer):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicResBlock, self).__init__()
        self.conv1 = nn.Conv2D(in_planes,
                               planes,
                               kernel_size=3,
                               stride=(stride, 1),
                               padding=1)
        self.bn1 = nn.BatchNorm2D(planes)
        self.conv2 = nn.Conv2D(planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.bn2 = nn.BatchNorm2D(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2D(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=(stride, 1)),
                nn.BatchNorm2D(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FCM(nn.Layer):
    def __init__(self,
                 block=BasicResBlock,
                 num_blocks=[2, 2],
                 m_channels=32,
                 feat_dim=80):
        super(FCM, self).__init__()
        self.in_planes = m_channels
        self.conv1 = nn.Conv2D(1, m_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2D(m_channels)

        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, m_channels, num_blocks[0], stride=2)

        self.conv2 = nn.Conv2D(m_channels, m_channels, kernel_size=3, stride=(2, 1), padding=1)
        self.bn2 = nn.BatchNorm2D(m_channels)
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
        out = out.reshape((shape[0], shape[1] * shape[2], shape[3]))
        return out


class CAMPPlus(nn.Layer):
    def __init__(self,
                 input_size,
                 embd_dim=512,
                 growth_rate=32,
                 bn_size=4,
                 init_channels=128,
                 config_str='batchnorm-relu',
                 memory_efficient=True):
        super(CAMPPlus, self).__init__()

        self.head = FCM(feat_dim=input_size)
        channels = self.head.out_channels
        self.embd_dim = embd_dim

        self.xvector = nn.Sequential(('tdnn', TDNNLayer(channels,
                                                        init_channels,
                                                        5,
                                                        stride=2,
                                                        dilation=1,
                                                        padding=-1,
                                                        config_str=config_str)))
        channels = init_channels
        for i, (num_layers, kernel_size,
                dilation) in enumerate(zip((12, 24, 16), (3, 3, 3), (1, 2, 2))):
            block = CAMDenseTDNNBlock(num_layers=num_layers,
                                      in_channels=channels,
                                      out_channels=growth_rate,
                                      bn_channels=bn_size * growth_rate,
                                      kernel_size=kernel_size,
                                      dilation=dilation,
                                      config_str=config_str,
                                      memory_efficient=memory_efficient)
            self.xvector.add_sublayer('block%d' % (i + 1), block)
            channels = channels + num_layers * growth_rate
            self.xvector.add_sublayer('transit%d' % (i + 1),
                                    TransitLayer(channels,
                                                 channels // 2,
                                                 bias=False,
                                                 config_str=config_str))
            channels //= 2

        self.xvector.add_sublayer('out_nonlinear', get_nonlinear(config_str, channels))

        self.xvector.add_sublayer('stats', StatsPool())
        self.xvector.add_sublayer('dense', DenseLayer(channels * 2, embd_dim, config_str='batchnorm_'))

    def forward(self, x):
        x = x.transpose((0, 2, 1))  # (B,T,F) => (B,F,T)
        x = self.head(x)
        x = self.xvector(x)
        return x
