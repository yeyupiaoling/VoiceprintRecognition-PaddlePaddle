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
        cosine = outputs.astype(paddle.float32)
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


class AMLoss(nn.Layer):
    def __init__(self, margin=0.2, scale=30):
        super(AMLoss, self).__init__()
        self.m = margin
        self.s = scale
        self.criterion = paddle.nn.CrossEntropyLoss(reduction="sum")

    def forward(self, outputs, targets):
        delt_costh = paddle.zeros(outputs.shape)
        for i, index in enumerate(targets):
            delt_costh[i, index] = self.m
        costh_m = outputs - delt_costh
        predictions = self.s * costh_m
        loss = self.criterion(predictions, targets) / targets.shape[0]
        return loss


class ARMLoss(nn.Layer):
    def __init__(self, margin=0.2, scale=30):
        super(ARMLoss, self).__init__()
        self.m = margin
        self.s = scale
        self.criterion = paddle.nn.CrossEntropyLoss(reduction="sum")

    def forward(self, outputs, targets):
        delt_costh = paddle.zeros(outputs.shape)
        for i, index in enumerate(targets):
            delt_costh[i, index] = self.m
        costh_m = outputs - delt_costh
        costh_m_s = self.s * costh_m
        delt_costh_m_s = paddle.zeros([outputs.shape[0], 1], dtype=paddle.float32)
        for i, index in enumerate(targets):
            delt_costh_m_s[i] = costh_m_s[i, index]
        delt_costh_m_s = delt_costh_m_s.tile([1, costh_m_s.shape[1]])
        costh_m_s_reduct = costh_m_s - delt_costh_m_s
        predictions = paddle.where(costh_m_s_reduct < 0.0, paddle.zeros_like(costh_m_s), costh_m_s)
        loss = self.criterion(predictions, targets) / targets.shape[0]
        return loss


class CELoss(nn.Layer):
    def __init__(self):
        super(CELoss, self).__init__()
        self.criterion = paddle.nn.CrossEntropyLoss(reduction="sum")

    def forward(self, outputs, targets):
        loss = self.criterion(outputs, targets) / targets.shape[0]
        return loss
