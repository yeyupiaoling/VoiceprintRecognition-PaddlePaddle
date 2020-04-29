import math
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr


class ResNet50:
    def net(self, input, class_dim):
        depth = [3, 4, 6, 3]
        num_filters = [64, 128, 256, 512]

        conv = self.conv_bn_layer(input=input,
                                  num_filters=64,
                                  filter_size=7,
                                  stride=2,
                                  act='relu', )
        conv = fluid.layers.pool2d(input=conv,
                                   pool_size=3,
                                   pool_stride=2,
                                   pool_padding=1,
                                   pool_type='max')
        for block in range(len(depth)):
            for i in range(depth[block]):
                conv = self.bottleneck_block(input=conv,
                                             num_filters=num_filters[block],
                                             stride=2 if i == 0 and block != 0 else 1)

        pool = fluid.layers.pool2d(input=conv, pool_type='avg', global_pooling=True)
        flatten = fluid.layers.flatten(x=pool)
        stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
        out = fluid.layers.fc(input=pool,
                              size=class_dim,
                              param_attr=fluid.param_attr.ParamAttr(initializer=fluid.initializer.Uniform(-stdv, stdv)))
        return out, flatten

    def conv_bn_layer(self, input, num_filters, filter_size, stride=1, groups=1, act=None):
        conv = fluid.layers.conv2d(input=input,
                                   num_filters=num_filters,
                                   filter_size=filter_size,
                                   stride=stride,
                                   padding=(filter_size - 1) // 2,
                                   groups=groups,
                                   param_attr=ParamAttr(),
                                   bias_attr=False)

        return fluid.layers.batch_norm(input=conv,
                                       act=act,
                                       param_attr=ParamAttr(),
                                       bias_attr=ParamAttr())

    def shortcut(self, input, ch_out, stride, is_first):
        ch_in = input.shape[1]
        if ch_in != ch_out or stride != 1 or is_first == True:
            return self.conv_bn_layer(input, ch_out, 1, stride)
        else:
            return input

    def bottleneck_block(self, input, num_filters, stride):
        conv0 = self.conv_bn_layer(input=input,
                                   num_filters=num_filters,
                                   filter_size=1,
                                   act='relu')
        conv1 = self.conv_bn_layer(input=conv0,
                                   num_filters=num_filters,
                                   filter_size=3,
                                   stride=stride,
                                   act='relu')
        conv2 = self.conv_bn_layer(input=conv1,
                                   num_filters=num_filters * 4,
                                   filter_size=1,
                                   act=None)

        short = self.shortcut(input,
                              num_filters * 4,
                              stride,
                              is_first=False)

        return fluid.layers.elementwise_add(x=short, y=conv2, act='relu')

    def basic_block(self, input, num_filters, stride, is_first):
        conv0 = self.conv_bn_layer(input=input,
                                   num_filters=num_filters,
                                   filter_size=3,
                                   act='relu',
                                   stride=stride)
        conv1 = self.conv_bn_layer(input=conv0,
                                   num_filters=num_filters,
                                   filter_size=3,
                                   act=None)
        short = self.shortcut(input, num_filters, stride, is_first)
        return fluid.layers.elementwise_add(x=short, y=conv1, act='relu')


def resnet50(input, class_dim):
    resnet = ResNet50()
    net, feature = resnet.net(input=input, class_dim=class_dim)
    return net, feature
