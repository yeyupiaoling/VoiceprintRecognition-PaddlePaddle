import paddle
from paddle.nn.initializer import XavierUniform


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

    def forward(self, feature, label):
        cosine = self.cosine_sim(feature, self.weight)
        one_hot = paddle.nn.functional.one_hot(label, self.class_dim)
        output = self.s * (cosine - one_hot * self.m)
        return output

    @staticmethod
    def cosine_sim(feature, weight, eps=1e-8):
        ip = paddle.mm(feature, weight)
        w1 = paddle.norm(feature, 2, axis=1).unsqueeze(1)
        w2 = paddle.norm(weight, 2, axis=0).unsqueeze(0)
        outer = paddle.matmul(w1, w2)
        return ip / outer.clip(min=eps)
