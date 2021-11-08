import paddle
import paddle.nn as nn
import math


class ArcNet(nn.Layer):
    def __init__(self,
                 feature_dim,
                 class_dim,
                 margin=0.2,
                 scale=30.0,
                 easy_margin=False):
        super().__init__()
        self.feature_dim = feature_dim
        self.class_dim = class_dim
        self.margin = margin
        self.scale = scale
        self.easy_margin = easy_margin
        self.weight = self.create_parameter(
            shape=[self.feature_dim, self.class_dim],
            is_bias=False,
            default_initializer=paddle.nn.initializer.XavierNormal())

    def forward(self, input, label):
        input_norm = paddle.sqrt(paddle.sum(paddle.square(input), axis=1, keepdim=True))
        input = paddle.divide(input, input_norm)

        weight_norm = paddle.sqrt(paddle.sum(paddle.square(self.weight), axis=0, keepdim=True))
        weight = paddle.divide(self.weight, weight_norm)

        cos = paddle.matmul(input, weight)
        sin = paddle.sqrt(1.0 - paddle.square(cos) + 1e-6)
        cos_m = math.cos(self.margin)
        sin_m = math.sin(self.margin)
        phi = cos * cos_m - sin * sin_m

        th = math.cos(self.margin) * (-1)
        mm = math.sin(self.margin) * self.margin
        if self.easy_margin:
            phi = self._paddle_where_more_than(cos, 0, phi, cos)
        else:
            phi = self._paddle_where_more_than(cos, th, phi, cos - mm)
        one_hot = paddle.nn.functional.one_hot(label, self.class_dim)
        one_hot = paddle.squeeze(one_hot, axis=[1])
        output = paddle.multiply(one_hot, phi) + paddle.multiply((1.0 - one_hot), cos)
        output = output * self.scale
        return output

    def _paddle_where_more_than(self, target, limit, x, y):
        mask = paddle.cast(x=(target > limit), dtype='float32')
        output = paddle.multiply(mask, x) + paddle.multiply((1.0 - mask), y)
        return output
