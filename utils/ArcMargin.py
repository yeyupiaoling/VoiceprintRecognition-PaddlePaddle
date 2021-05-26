import math

import paddle
from paddle.nn.initializer import XavierUniform
import paddle.nn.functional as F


class ArcNet(paddle.nn.Layer):
    """
    Args:
        feature_dim: size of each input sample
        class_dim: size of each output sample
        s: norm of input feature
        m: margin
    """

    def __init__(self, feature_dim, class_dim, s=64.0, m=0.50):
        super(ArcNet, self).__init__()
        self.weight = paddle.create_parameter([class_dim, feature_dim], dtype='float32', attr=XavierUniform())
        self.class_dim = class_dim
        self.m = m
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, feature, label):
        cos_theta = F.linear(F.normalize(feature), paddle.transpose(F.normalize(self.weight), perm=[1, 0]))
        sin_theta = paddle.sqrt(paddle.clip(1.0 - paddle.pow(cos_theta, 2), min=0, max=1))
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m
        cos_theta_m = paddle.where(cos_theta > self.threshold, cos_theta_m, cos_theta - self.mm)
        one_hot = paddle.nn.functional.one_hot(label, self.class_dim)
        output = (one_hot * cos_theta_m) + ((1.0 - one_hot) * cos_theta)
        output *= self.s
        return output
