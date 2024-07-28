import paddle
import paddle.nn as nn


class CELoss(nn.Layer):
    def __init__(self, label_smoothing=0.0):
        super(CELoss, self).__init__()
        self.criterion = paddle.nn.CrossEntropyLoss(reduction="sum", label_smoothing=label_smoothing)

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

