import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppvector.models.utils import length_to_mask, Conv1d, TDNNBlock


class TemporalAveragePooling(nn.Layer):
    def __init__(self):
        """TAP
        Paper: Multi-Task Learning with High-Order Statistics for X-vector based Text-Independent Speaker Verification
        Link: https://arxiv.org/pdf/1903.12058.pdf
        """
        super(TemporalAveragePooling, self).__init__()

    def forward(self, x, lengths=None):
        """Computes Temporal Average Pooling Module
        Args:
            x (torch.Tensor): Input tensor (#batch, channels, frames).
        Returns:
            torch.Tensor: Output tensor (#batch, channels)
        """
        x = paddle.mean(x, axis=2)
        x = x.unsqueeze(2)
        return x


class TemporalStatisticsPooling(nn.Layer):
    def __init__(self):
        """TSP
        Paper: X-vectors: Robust DNN Embeddings for Speaker Recognition
        Link： http://www.danielpovey.com/files/2018_icassp_xvectors.pdf
        """
        super(TemporalStatisticsPooling, self).__init__()

    def forward(self, x, lengths=None):
        """Computes Temporal Statistics Pooling Module
        Args:
            x (torch.Tensor): Input tensor (#batch, channels, frames).
        Returns:
            torch.Tensor: Output tensor (#batch, channels*2)
        """
        mean = paddle.mean(x, axis=2)
        var = paddle.var(x, axis=2)
        x = paddle.concat((mean, var), axis=1)
        x = x.unsqueeze(2)
        return x


class SelfAttentivePooling(nn.Layer):
    """SAP"""

    def __init__(self, in_dim, bottleneck_dim=128):
        # Use Conv1d with stride == 1 rather than Linear, then we don't need to transpose inputs.
        # attention dim = 128
        super(SelfAttentivePooling, self).__init__()
        self.linear1 = nn.Conv1D(in_dim, bottleneck_dim, kernel_size=1)  # equals W and b in the paper
        self.linear2 = nn.Conv1D(bottleneck_dim, in_dim, kernel_size=1)  # equals V and k in the paper

    def forward(self, x, lengths=None):
        # DON'T use ReLU here! In experiments, I find ReLU hard to converge.
        alpha = paddle.tanh(self.linear1(x))
        alpha = paddle.nn.functional.softmax(self.linear2(alpha), axis=2)
        mean = paddle.sum(alpha * x, axis=2)
        mean = mean.unsqueeze(2)
        return mean


class AttentiveStatisticsPooling(nn.Layer):
    """TSP"""

    def __init__(self, channels, attention_channels=128, global_context=True):
        super().__init__()
        self.eps = 1e-12
        self.global_context = global_context
        if global_context:
            self.tdnn = TDNNBlock(channels * 3, attention_channels, 1, 1)
        else:
            self.tdnn = TDNNBlock(channels, attention_channels, 1, 1)
        self.tanh = nn.Tanh()
        self.conv = Conv1d(
            in_channels=attention_channels,
            out_channels=channels,
            kernel_size=1)

    def forward(self, x, lengths=None):
        C, L = x.shape[1], x.shape[2]  # KP: (N, C, L)

        def _compute_statistics(x, m, axis=2, eps=self.eps):
            mean = (m * x).sum(axis)
            std = paddle.sqrt((m * (x - mean.unsqueeze(axis)).pow(2)).sum(axis).clip(eps))
            return mean, std

        if lengths is None:
            lengths = paddle.ones([x.shape[0]])

        # Make binary mask of shape [N, 1, L]
        mask = length_to_mask(lengths * L, max_len=L)
        mask = mask.unsqueeze(1)

        # 通过允许自我注意观察话语的全局属性，扩展汇集层的时间上下文。
        if self.global_context:
            total = mask.sum(axis=2, keepdim=True).astype('float32')
            mean, std = _compute_statistics(x, mask / total)
            mean = mean.unsqueeze(2).tile((1, 1, L))
            std = std.unsqueeze(2).tile((1, 1, L))
            attn = paddle.concat([x, mean, std], axis=1)
        else:
            attn = x

        # Apply layers
        attn = self.conv(self.tanh(self.tdnn(attn)))

        # Filter out zero-paddings
        attn = paddle.where(
            mask.tile((1, C, 1)) == 0,
            paddle.ones_like(attn) * float("-inf"), attn)

        attn = F.softmax(attn, axis=2)
        mean, std = _compute_statistics(x, attn)

        # Append mean and std of the batch
        pooled_stats = paddle.concat((mean, std), axis=1)

        return pooled_stats


class TemporalStatsPool(nn.Layer):
    """TSTP
    Temporal statistics pooling, concatenate mean and std, which is used in
    x-vector
    Comment: simple concatenation can not make full use of both statistics
    """

    def __init__(self):
        super(TemporalStatsPool, self).__init__()

    def forward(self, x, lengths=None):
        # The last dimension is the temporal axis
        pooling_mean = x.mean(axis=-1)
        pooling_std = paddle.sqrt(paddle.var(x, axis=-1) + 1e-8)
        pooling_mean = pooling_mean.flatten(start_axis=1)
        pooling_std = pooling_std.flatten(start_axis=1)

        stats = paddle.concat((pooling_mean, pooling_std), 1)
        return stats
