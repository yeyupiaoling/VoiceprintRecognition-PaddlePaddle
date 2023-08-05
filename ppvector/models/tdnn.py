import paddle.nn as nn
import paddle.nn.functional as F

from ppvector.models.pooling import AttentiveStatisticsPooling, TemporalAveragePooling, TemporalStatsPool
from ppvector.models.pooling import SelfAttentivePooling, TemporalStatisticsPooling
from ppvector.models.utils import Conv1d, BatchNorm1d


class TDNN(nn.Layer):
    def __init__(self, input_size=80, channels=512, embd_dim=192, pooling_type="ASP"):
        super(TDNN, self).__init__()
        self.embd_dim = embd_dim
        self.td_layer1 = nn.Conv1D(in_channels=input_size, out_channels=512, dilation=1, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm1D(512)
        self.td_layer2 = nn.Conv1D(in_channels=512, out_channels=512, dilation=2, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm1D(512)
        self.td_layer3 = nn.Conv1D(in_channels=512, out_channels=512, dilation=3, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm1D(512)
        self.td_layer4 = nn.Conv1D(in_channels=512, out_channels=512, dilation=1, kernel_size=1, stride=1)
        self.bn4 = nn.BatchNorm1D(512)
        self.td_layer5 = nn.Conv1D(in_channels=512, out_channels=channels, dilation=1, kernel_size=1, stride=1)

        if pooling_type == "ASP":
            self.pooling = AttentiveStatisticsPooling(channels, 128)
            self.pooling_bn = BatchNorm1d(input_size=channels * 2)
            # Final linear transformation
            self.fc = Conv1d(in_channels=channels * 2,
                             out_channels=self.embd_dim,
                             kernel_size=1)
        elif pooling_type == "SAP":
            self.asp = SelfAttentivePooling(channels, 128)
            self.asp_bn = nn.BatchNorm1D(channels)
            # Final linear transformation
            self.fc = Conv1d(in_channels=channels,
                             out_channels=self.embd_dim,
                             kernel_size=1)
        elif pooling_type == "TAP":
            self.asp = TemporalAveragePooling()
            self.asp_bn = nn.BatchNorm1D(channels)
            # Final linear transformation
            self.fc = Conv1d(in_channels=channels,
                             out_channels=self.embd_dim,
                             kernel_size=1)
        elif pooling_type == "TSP":
            self.asp = TemporalStatisticsPooling()
            self.asp_bn = nn.BatchNorm1D(channels * 2)
            # Final linear transformation
            self.fc = Conv1d(in_channels=channels * 2,
                             out_channels=self.embd_dim,
                             kernel_size=1)
        elif pooling_type == "TSTP":
            self.asp = TemporalStatsPool()
            self.asp_bn = nn.BatchNorm1D(channels * 2)
            # Final linear transformation
            self.fc = Conv1d(in_channels=channels * 2,
                             out_channels=self.embd_dim,
                             kernel_size=1)
        else:
            raise Exception(f'没有{pooling_type}池化层！')

    def forward(self, x):
        """
        Compute embeddings.

        Args:
            x (torch.Tensor): Input data with shape (N, time, freq).

        Returns:
            torch.Tensor: Output embeddings with shape (N, self.emb_size, 1)
        """
        x = x.transpose([0, 2, 1])
        x = F.relu(self.td_layer1(x))
        x = self.bn1(x)
        x = F.relu(self.td_layer2(x))
        x = self.bn2(x)
        x = F.relu(self.td_layer3(x))
        x = self.bn3(x)
        x = F.relu(self.td_layer4(x))
        x = self.bn4(x)
        x = F.relu(self.td_layer5(x))
        out = self.pooling_bn(self.pooling(x))
        # Final linear transformation
        out = self.fc(out).squeeze(-1)  # (N, emb_size, 1) -> (N, emb_size)
        return out
