import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class AAMLoss(nn.Layer):
    def __init__(self, margin=0.2, scale=32, easy_margin=False):
        """The Implementation of Additive Angular Margin (AAM) proposed
       in the following paper: '''Margin Matters: Towards More Discriminative Deep Neural Network Embeddings for Speaker Recognition'''
       (https://arxiv.org/abs/1906.07317)

        Args:
            margin (float, optional): margin factor. Defaults to 0.3.
            scale (float, optional): scale factor. Defaults to 32.0.
            easy_margin (bool, optional): easy_margin flag. Defaults to False.
        """
        super(AAMLoss, self).__init__()
        self.scale = scale
        self.easy_margin = easy_margin
        self.criterion = nn.CrossEntropyLoss()

        self.update(margin)

    def forward(self, cosine, label):
        """
        Args:
            cosine (torch.Tensor): cosine distance between the two tensors, shape [batch, num_classes].
            label (torch.Tensor): label of speaker id, shape [batch, ].

        Returns:
            torch.Tensor: loss value.
        """
        sine = paddle.sqrt(1.0 - paddle.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = paddle.where(cosine > 0, phi, cosine)
        else:
            phi = paddle.where(cosine > self.th, phi, cosine - self.mmm)

        one_hot = F.one_hot(label, cosine.shape[1])
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        loss = self.criterion(output, label)
        return loss

    def update(self, margin=0.2):
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.m = self.margin
        self.mmm = 1.0 + math.cos(math.pi - margin)


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

    def update(self, margin=0.2):
        self.m = margin


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

    def update(self, margin=0.2):
        self.m = margin


class CELoss(nn.Layer):
    def __init__(self, **kwargs):
        super(CELoss, self).__init__()
        self.criterion = paddle.nn.CrossEntropyLoss(reduction="sum")

    def forward(self, outputs, targets):
        loss = self.criterion(outputs, targets) / targets.shape[0]
        return loss

    def update(self, margin=0.2):
        pass


class SubCenterLoss(nn.Layer):
    r"""Implement of large margin arc distance with subcenter:
    Reference:Sub-center ArcFace: Boosting Face Recognition byLarge-Scale Noisy
     Web Faces.https://ibug.doc.ic.ac.uk/media/uploads/documents/eccv_1445.pdf

     Args:
        margin (float, optional): margin factor. Defaults to 0.3.
        scale (float, optional): scale factor. Defaults to 32.0.
        easy_margin (bool, optional): easy_margin flag. Defaults to False.
        K: number of sub-centers, same classifier K.
    """

    def __init__(self, margin=0.2, scale=32, easy_margin=False, K=3):
        super(SubCenterLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        # subcenter
        self.K = K
        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.mmm = 1.0 + math.cos(math.pi - margin)
        self.m = self.margin
        self.criterion = nn.CrossEntropyLoss()

        self.update(margin)

    def update(self, margin=0.2):
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.m = self.margin
        self.mmm = 1.0 + math.cos(math.pi - margin)

    def forward(self, input, label):
        # (batch, out_dim, k)
        cosine = paddle.reshape(input, (-1, input.shape[1] // self.K, self.K))
        # (batch, out_dim)
        cosine = paddle.max(cosine, 2)
        sine = paddle.sqrt(1.0 - paddle.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = paddle.where(cosine > 0, phi, cosine)
        else:
            phi = paddle.where(cosine > self.th, phi, cosine - self.mmm)

        one_hot = F.one_hot(label, cosine.shape[1])
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        loss = self.criterion(output, label)
        return loss
