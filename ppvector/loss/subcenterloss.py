import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


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

    def __init__(self, margin=0.2, scale=32, easy_margin=False, K=3, label_smoothing=0.0):
        super(SubCenterLoss, self).__init__()
        self.scale = scale
        # subcenter
        self.K = K
        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mmm = 1.0 + math.cos(math.pi - margin)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, inputs, labels):
        """
        Args:
            inputs(dict): 模型输出的特征向量 (batch_size, feat_dim) 和分类层输出的logits(batch_size, class_num)
            labels(paddle.Tensor): 类别标签 (batch_size)
        """
        features, logits = inputs['features'], inputs['logits']
        # (batch, out_dim, k)
        cosine = paddle.reshape(logits, (-1, logits.shape[1] // self.K, self.K))
        # (batch, out_dim)
        cosine = paddle.max(cosine, 2)
        sine = paddle.sqrt(1.0 - paddle.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = paddle.where(cosine > 0, phi, cosine)
        else:
            phi = paddle.where(cosine > self.th, phi, cosine - self.mmm)

        one_hot = F.one_hot(labels, cosine.shape[1])
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        loss = self.criterion(output, labels)
        return loss

    def update(self, margin=0.2):
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mmm = 1.0 + math.cos(math.pi - margin)
