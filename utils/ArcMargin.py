import math

import paddle
import paddle.nn.functional as F


class ArcMarginProduct(paddle.nn.Layer):
    r"""Implement of large margin arc distance: :
        Args:
            feature_dim: size of each input sample
            class_dim: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, feature_dim, class_dim, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.weight = paddle.to_tensor(paddle.randn((feature_dim, class_dim), dtype='float32'), stop_gradient=False)
        self.class_dim = class_dim
        self.s = s
        self.m = m

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = paddle.sqrt(paddle.clip(1.0 - paddle.pow(cosine, 2), min=0, max=1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = paddle.where(cosine > 0, phi, cosine)
        else:
            phi = paddle.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = paddle.nn.functional.one_hot(label, self.class_dim)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output
