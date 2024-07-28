import paddle
import paddle.nn as nn


class ARMLoss(nn.Layer):
    def __init__(self, margin=0.2, scale=30, label_smoothing=0.0):
        super(ARMLoss, self).__init__()
        self.margin = margin
        self.scale = scale
        self.criterion = paddle.nn.CrossEntropyLoss(reduction="sum", label_smoothing=label_smoothing)

    def forward(self, inputs, labels):
        """
        Args:
            inputs(dict): 模型输出的特征向量 (batch_size, feat_dim) 和分类层输出的logits(batch_size, class_num)
            labels(paddle.Tensor): 类别标签 (batch_size)
        """
        features, logits = inputs['features'], inputs['logits']
        delt_costh = paddle.zeros(logits.shape)
        for i, index in enumerate(labels):
            delt_costh[i, index] = self.margin
        costh_m = logits - delt_costh
        costh_m_s = self.scale * costh_m
        delt_costh_m_s = paddle.zeros([logits.shape[0], 1], dtype=paddle.float32)
        for i, index in enumerate(labels):
            delt_costh_m_s[i] = costh_m_s[i, index]
        delt_costh_m_s = delt_costh_m_s.tile([1, costh_m_s.shape[1]])
        costh_m_s_reduct = costh_m_s - delt_costh_m_s
        predictions = paddle.where(costh_m_s_reduct < 0.0, paddle.zeros_like(costh_m_s), costh_m_s)
        loss = self.criterion(predictions, labels) / labels.shape[0]
        return loss

    def update(self, margin=0.2):
        self.margin = margin

