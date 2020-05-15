import paddle
import paddle.fluid as fluid

__all__ = ["VGGNet", "VGG11", "VGG13", "VGG16", "VGG19"]


class VGGNet():
    def __init__(self, layers=16):
        self.layers = layers

    def net(self, input, class_dim=1000):
        layers = self.layers
        vgg_spec = {
            11: ([1, 1, 2, 2, 2]),
            13: ([2, 2, 2, 2, 2]),
            16: ([2, 2, 3, 3, 3]),
            19: ([2, 2, 4, 4, 4])
        }
        nums = vgg_spec[layers]
        conv1 = self.conv_block(input, 64, nums[0])
        conv2 = self.conv_block(conv1, 128, nums[1])
        conv3 = self.conv_block(conv2, 256, nums[2])
        conv4 = self.conv_block(conv3, 512, nums[3])
        conv5 = self.conv_block(conv4, 512, nums[4])

        fc1 = fluid.layers.fc(input=conv5, size=1024, act='relu')
        fc1 = fluid.layers.dropout(x=fc1, dropout_prob=0.5)
        fc2 = fluid.layers.fc(input=fc1, size=1024, act='relu')
        fc2 = fluid.layers.dropout(x=fc2, dropout_prob=0.5)
        out = fluid.layers.fc(input=fc2, size=class_dim, act='softmax')

        return out, fc2

    def conv_block(self, input, num_filter, groups):
        conv = input
        for i in range(groups):
            conv = fluid.layers.conv2d(input=conv,
                                       num_filters=num_filter,
                                       filter_size=3,
                                       stride=1,
                                       padding=1,
                                       act='relu',
                                       param_attr=fluid.param_attr.ParamAttr(),
                                       bias_attr=False)
        return fluid.layers.pool2d(input=conv, pool_size=2, pool_type='max', pool_stride=2)


def VGG11():
    model = VGGNet(layers=11)
    return model


def VGG13():
    model = VGGNet(layers=13)
    return model


def VGG16():
    model = VGGNet(layers=16)
    return model


def VGG19():
    model = VGGNet(layers=19)
    return model
