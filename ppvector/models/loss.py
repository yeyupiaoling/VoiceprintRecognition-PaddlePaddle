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

    def forward(self, inputs, labels):
        """
        Args:
            inputs(dict): 模型输出的特征向量 (batch_size, feat_dim) 和分类层输出的logits(batch_size, class_num)
            labels(paddle.Tensor): 类别标签 (batch_size)
        """
        features, logits = inputs['features'], inputs['logits']
        sine = paddle.sqrt(1.0 - paddle.pow(logits, 2))
        phi = logits * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = paddle.where(logits > 0, phi, logits)
        else:
            phi = paddle.where(logits > self.th, phi, logits - self.mmm)

        one_hot = F.one_hot(labels, logits.shape[1])
        output = (one_hot * phi) + ((1.0 - one_hot) * logits)
        output *= self.scale

        loss = self.criterion(output, labels)
        return loss

    def update(self, margin=0.2):
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.m = self.margin
        self.mmm = 1.0 + math.cos(math.pi - margin)


class SphereFace2(nn.Layer):
    def __init__(self, margin=0.2, scale=32.0, lanbuda=0.7, t=3, margin_type='C'):
        """Implement of sphereface2 for speaker verification:
            Reference:
                [1] Exploring Binary Classification Loss for Speaker Verification
                https://ieeexplore.ieee.org/abstract/document/10094954
                [2] Sphereface2: Binary classification is all you need for deep face recognition
                https://arxiv.org/pdf/2108.01513
            Args:
                scale: norm of logits feature
                margin: margin
                lanbuda: weight of positive and negative pairs
                t: parameter for adjust score distribution
                margin_type: A:cos(theta+margin) or C:cos(theta)-margin
            Recommend margin:
                training: 0.2 for C and 0.15 for A
                LMF: 0.3 for C and 0.25 for A
        """
        super(SphereFace2, self).__init__()
        self.scale = scale
        self.bias = paddle.create_parameter([1, 1], dtype=paddle.float32, is_bias=True)
        self.t = t
        self.lanbuda = lanbuda
        self.margin_type = margin_type

        self.update(margin)

    def fun_g(self, z, t: int):
        gz = 2 * paddle.pow((z + 1) / 2, t) - 1
        return gz

    def forward(self, inputs, labels):
        """
        Args:
            inputs(dict): 模型输出的特征向量 (batch_size, feat_dim) 和分类层输出的logits(batch_size, class_num)
            labels(paddle.Tensor): 类别标签 (batch_size)
        """
        features, logits = inputs['features'], inputs['logits']
        if self.margin_type == 'A':  # arcface type
            sin = paddle.sqrt(1.0 - paddle.pow(logits, 2))
            cos_m_theta_p = self.scale * self.fun_g(
                paddle.where(logits > self.th, logits * self.cos_m - sin * self.sin_m, logits - self.mmm), self.t) + \
                            self.bias[0][0]
            cos_m_theta_n = self.scale * self.fun_g(logits * self.cos_m + sin * self.sin_m, self.t) + self.bias[0][0]
            cos_p_theta = self.lanbuda * paddle.log(1 + paddle.exp(-1.0 * cos_m_theta_p))
            cos_n_theta = (1 - self.lanbuda) * paddle.log(1 + paddle.exp(cos_m_theta_n))
        else:
            # cosface type
            cos_m_theta_p = self.scale * (self.fun_g(logits, self.t) - self.margin) + self.bias[0][0]
            cos_m_theta_n = self.scale * (self.fun_g(logits, self.t) + self.margin) + self.bias[0][0]
            cos_p_theta = self.lanbuda * paddle.log(1 + paddle.exp(-1.0 * cos_m_theta_p))
            cos_n_theta = (1 - self.lanbuda) * paddle.log(1 + paddle.exp(cos_m_theta_n))

        target_mask = F.one_hot(labels, logits.shape[1])
        nontarget_mask = 1 - target_mask
        loss = (target_mask * cos_p_theta + nontarget_mask * cos_n_theta).sum(1).mean()
        return loss

    def update(self, margin=0.2):
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin)
        self.mmm = 1.0 + math.cos(math.pi - margin)


class AMLoss(nn.Layer):
    def __init__(self, margin=0.2, scale=30):
        super(AMLoss, self).__init__()
        self.m = margin
        self.s = scale
        self.criterion = paddle.nn.CrossEntropyLoss(reduction="sum")

    def forward(self, inputs, labels):
        """
        Args:
            inputs(dict): 模型输出的特征向量 (batch_size, feat_dim) 和分类层输出的logits(batch_size, class_num)
            labels(paddle.Tensor): 类别标签 (batch_size)
        """
        features, logits = inputs['features'], inputs['logits']
        delt_costh = paddle.zeros(logits.shape)
        for i, index in enumerate(labels):
            delt_costh[i, index] = self.m
        costh_m = logits - delt_costh
        predictions = self.s * costh_m
        loss = self.criterion(predictions, labels) / labels.shape[0]
        return loss

    def update(self, margin=0.2):
        self.m = margin


class ARMLoss(nn.Layer):
    def __init__(self, margin=0.2, scale=30):
        super(ARMLoss, self).__init__()
        self.m = margin
        self.s = scale
        self.criterion = paddle.nn.CrossEntropyLoss(reduction="sum")

    def forward(self, inputs, labels):
        """
        Args:
            inputs(dict): 模型输出的特征向量 (batch_size, feat_dim) 和分类层输出的logits(batch_size, class_num)
            labels(paddle.Tensor): 类别标签 (batch_size)
        """
        features, logits = inputs['features'], inputs['logits']
        delt_costh = paddle.zeros(logits.shape)
        for i, index in enumerate(labels):
            delt_costh[i, index] = self.m
        costh_m = logits - delt_costh
        costh_m_s = self.s * costh_m
        delt_costh_m_s = paddle.zeros([logits.shape[0], 1], dtype=paddle.float32)
        for i, index in enumerate(labels):
            delt_costh_m_s[i] = costh_m_s[i, index]
        delt_costh_m_s = delt_costh_m_s.tile([1, costh_m_s.shape[1]])
        costh_m_s_reduct = costh_m_s - delt_costh_m_s
        predictions = paddle.where(costh_m_s_reduct < 0.0, paddle.zeros_like(costh_m_s), costh_m_s)
        loss = self.criterion(predictions, labels) / labels.shape[0]
        return loss

    def update(self, margin=0.2):
        self.m = margin


class CELoss(nn.Layer):
    def __init__(self, **kwargs):
        super(CELoss, self).__init__()
        self.criterion = paddle.nn.CrossEntropyLoss(reduction="sum")

    def forward(self, inputs, labels):
        """
        Args:
            inputs(dict): 模型输出的特征向量 (batch_size, feat_dim) 和分类层输出的logits(batch_size, class_num)
            labels(paddle.Tensor): 类别标签 (batch_size)
        """
        features, logits = inputs['features'], inputs['logits']
        loss = self.criterion(logits, labels) / labels.shape[0]
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


class TripletAngularMarginLoss(nn.Layer):
    """A more robust triplet loss with hard positive/negative mining on angular margin instead of relative distance between d(a,p) and d(a,n).

    Args:
        margin (float, optional): angular margin. Defaults to 0.5.
        normalize_feature (bool, optional): whether to apply L2-norm in feature before computing distance(cos-similarity). Defaults to True.
        add_absolute (bool, optional): whether add absolute loss within d(a,p) or d(a,n). Defaults to False.
        absolute_loss_weight (float, optional): weight for absolute loss. Defaults to 1.0.
        ap_value (float, optional): weight for d(a, p). Defaults to 0.9.
        an_value (float, optional): weight for d(a, n). Defaults to 0.5.
    """

    def __init__(self,
                 margin=0.5,
                 normalize_feature=True,
                 add_absolute=True,
                 absolute_loss_weight=1.0,
                 ap_value=0.8,
                 an_value=0.4):
        super(TripletAngularMarginLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = paddle.nn.loss.MarginRankingLoss(margin=margin)
        self.normalize_feature = normalize_feature
        self.add_absolute = add_absolute
        self.ap_value = ap_value
        self.an_value = an_value
        self.absolute_loss_weight = absolute_loss_weight
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs, labels):
        """
        Args:
            inputs(dict): 模型输出的特征向量 (batch_size, feat_dim) 和分类层输出的logits(batch_size, class_num)
            labels(paddle.Tensor): 类别标签 (batch_size)
        """
        features, logits = inputs['features'], inputs['logits']
        loss_ce = self.criterion(logits, labels)

        if self.normalize_feature:
            features = paddle.divide(features, paddle.norm(features, p=2, axis=-1, keepdim=True))

        bs = features.shape[0]

        # compute distance(cos-similarity)
        dist = paddle.matmul(features, features.t())

        # hard negative mining
        is_pos = paddle.expand(labels, (bs, bs)).equal(paddle.expand(labels, (bs, bs)).t())
        is_neg = paddle.expand(labels, (bs, bs)).not_equal(paddle.expand(labels, (bs, bs)).t())

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        d1 = paddle.masked_select(dist, is_pos)
        d2 = paddle.reshape(d1, (bs, -1))
        dist_ap = paddle.min(d2, axis=1, keepdim=True)
        # `dist_an` means distance(anchor, negative)
        # both `dist_an` and `relative_n_inds` with shape [N, 1]
        dist_an = paddle.max(paddle.reshape(
            paddle.masked_select(dist, is_neg), (bs, -1)), axis=1, keepdim=True)
        # shape [N]
        dist_ap = paddle.squeeze(dist_ap, axis=1)
        dist_an = paddle.squeeze(dist_an, axis=1)

        # Compute ranking hinge loss
        y = paddle.ones_like(dist_an)
        loss = self.ranking_loss(dist_ap, dist_an, y)

        if self.add_absolute:
            absolut_loss_ap = self.ap_value - dist_ap
            absolut_loss_ap = paddle.where(absolut_loss_ap > 0, absolut_loss_ap, paddle.zeros_like(absolut_loss_ap))

            absolut_loss_an = dist_an - self.an_value
            absolut_loss_an = paddle.where(absolut_loss_an > 0, absolut_loss_an, paddle.ones_like(absolut_loss_an))

            loss = (absolut_loss_an.mean() + absolut_loss_ap.mean()) * self.absolute_loss_weight + loss.mean()
        loss = loss + loss_ce
        return loss

    def update(self, margin=0.5):
        self.margin = margin
