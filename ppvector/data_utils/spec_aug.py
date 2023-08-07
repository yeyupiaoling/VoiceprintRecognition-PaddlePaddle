import paddle
import paddle.nn as nn


class SpecAug(nn.Layer):

    def __init__(self, freq_mask_width=(0, 8), time_mask_width=(0, 10)):
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        super().__init__()

    def freq_mask(self, x):
        batch, _, fea = x.shape
        mask_len = paddle.randint(self.freq_mask_width[0], self.freq_mask_width[1], (batch, 1)).unsqueeze(2)
        mask_pos = paddle.randint(0, max(1, fea - mask_len.max()), (batch, 1)).unsqueeze(2)
        arange = paddle.arange(fea).reshape((1, 1, -1))
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(axis=1)
        mask = mask.unsqueeze(1)
        x = self.masked_fill(x, mask, 0.0)
        return x

    def time_mask(self, x):
        batch, time, _ = x.shape
        mask_len = paddle.randint(self.time_mask_width[0], self.time_mask_width[1], (batch, 1)).unsqueeze(2)
        mask_pos = paddle.randint(0, max(1, time - mask_len.max()), (batch, 1)).unsqueeze(2)
        arange = paddle.arange(time).reshape((1, 1, -1))
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(axis=1)
        mask = mask.unsqueeze(2)
        x = self.masked_fill(x, mask, 0.0)
        return x

    @staticmethod
    def masked_fill(x, mask, value):
        y = paddle.full(x.shape, value, x.dtype)
        return paddle.where(mask, y, x)

    def forward(self, x):
        x = self.freq_mask(x)
        x = self.time_mask(x)
        return x
