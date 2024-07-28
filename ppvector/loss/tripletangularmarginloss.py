import paddle
import paddle.nn as nn


class TripletAngularMarginLoss(nn.Layer):
    """A more robust triplet loss with hard positive/negative mining on angular margin instead of relative distance between d(a,p) and d(a,n).

    Args:
        margin (float, optional): angular margin. Defaults to 0.5.
        normalize_feature (bool, optional): whether to apply L2-norm in feature before computing distance(cos-similarity). Defaults to True.
        add_absolute (bool, optional): whether add absolute loss within d(a,p) or d(a,n). Defaults to True.
        absolute_loss_weight (float, optional): weight for absolute loss. Defaults to 1.0.
        ap_value (float, optional): weight for d(a, p). Defaults to 0.8.
        an_value (float, optional): weight for d(a, n). Defaults to 0.4.
    """

    def __init__(self,
                 margin=0.5,
                 normalize_feature=True,
                 add_absolute=True,
                 absolute_loss_weight=1.0,
                 ap_value=0.8,
                 an_value=0.4,
                 label_smoothing=0.0):
        super(TripletAngularMarginLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = paddle.nn.loss.MarginRankingLoss(margin=margin)
        self.normalize_feature = normalize_feature
        self.add_absolute = add_absolute
        self.ap_value = ap_value
        self.an_value = an_value
        self.absolute_loss_weight = absolute_loss_weight
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

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
        self.ranking_loss.margin = margin
