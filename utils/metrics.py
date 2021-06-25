import paddle
from paddle.nn.initializer import XavierUniform
import math
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
        self.weight = paddle.create_parameter([feature_dim, class_dim], dtype='float32', attr=XavierUniform())
        self.class_dim = class_dim
        self.m = m
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = self.sin_m * m

    def forward(self, feature, label):
        cos_theta = paddle.mm(F.normalize(feature, axis=1), F.normalize(self.weight, axis=0))
        sin_theta = paddle.sqrt(paddle.clip(1.0 - paddle.pow(cos_theta, 2), min=0, max=1))
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m
        cos_theta_m = paddle.where(cos_theta > self.threshold, cos_theta_m, cos_theta - self.mm)
        one_hot = paddle.nn.functional.one_hot(label, self.class_dim)
        output = (one_hot * cos_theta_m) + (paddle.abs((1.0 - one_hot)) * cos_theta)
        output *= self.s
        # 简单的分类方法，学习率需要设置为0.1
        # cosine = self.cosine_sim(feature, self.weight)
        # one_hot = paddle.nn.functional.one_hot(label, self.class_dim)
        # output = self.s * (cosine - one_hot * self.m)
        return output

    @staticmethod
    def cosine_sim(feature, weight, eps=1e-8):
        ip = paddle.mm(feature, weight)
        w1 = paddle.norm(feature, 2, axis=1).unsqueeze(1)
        w2 = paddle.norm(weight, 2, axis=0).unsqueeze(0)
        outer = paddle.matmul(w1, w2)
        return ip / outer.clip(min=eps)
