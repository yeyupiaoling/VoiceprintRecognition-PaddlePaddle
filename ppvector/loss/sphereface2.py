import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F



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

        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mmm = 1.0 + math.cos(math.pi - margin)

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
        self.mmm = 1.0 + math.cos(math.pi - margin)

