import paddle
import paddle.nn as nn


class AMLoss(nn.Layer):
    def __init__(self, margin=0.2, scale=30, label_smoothing=0.0):
        super(AMLoss, self).__init__()
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
        predictions = self.scale * costh_m
        loss = self.criterion(predictions, labels) / labels.shape[0]
        return loss

    def update(self, margin=0.2):
        self.margin = margin
