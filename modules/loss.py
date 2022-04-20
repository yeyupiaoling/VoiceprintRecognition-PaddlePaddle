import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class AdditiveAngularMargin(nn.Layer):
    def __init__(self, margin=0.0, scale=1.0, easy_margin=False):
        """The Implementation of Additive Angular Margin (AAM) proposed
       in the following paper: '''Margin Matters: Towards More Discriminative Deep Neural Network Embeddings for Speaker Recognition'''
       (https://arxiv.org/abs/1906.07317)

        Args:
            margin (float, optional): margin factor. Defaults to 0.0.
            scale (float, optional): scale factor. Defaults to 1.0.
            easy_margin (bool, optional): easy_margin flag. Defaults to False.
        """
        super(AdditiveAngularMargin, self).__init__()
        self.margin = margin
        self.scale = scale
        self.easy_margin = easy_margin

        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.th = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin

    def forward(self, outputs, targets):
        cosine = outputs.astype('float32')
        sine = paddle.sqrt(1.0 - paddle.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = paddle.where(cosine > 0, phi, cosine)
        else:
            phi = paddle.where(cosine > self.th, phi, cosine - self.mm)
        outputs = (targets * phi) + ((1.0 - targets) * cosine)
        return self.scale * outputs


class AAMLoss(nn.Layer):
    def __init__(self, margin=0.2, scale=30, easy_margin=False):
        super(AAMLoss, self).__init__()
        self.loss_fn = AdditiveAngularMargin(margin=margin, scale=scale, easy_margin=easy_margin)
        self.criterion = paddle.nn.KLDivLoss(reduction="sum")

    def forward(self, outputs, targets):
        targets = F.one_hot(targets, outputs.shape[1])
        predictions = self.loss_fn(outputs, targets)
        predictions = F.log_softmax(predictions, axis=1)
        loss = self.criterion(predictions, targets) / targets.sum()
        return loss
