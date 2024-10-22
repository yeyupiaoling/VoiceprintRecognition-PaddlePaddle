import paddle
import paddle.nn as nn

from ppvector.models.pooling import AttentiveStatisticsPooling, SelfAttentivePooling
from ppvector.models.pooling import TemporalAveragePooling, TemporalStatisticsPooling
from ppvector.models.utils import BatchNorm1d, Conv1d, TDNNBlock, length_to_mask

__all__ = ['EcapaTdnn']


class Res2NetBlock(nn.Layer):
    def __init__(self, in_channels, out_channels, scale=8, dilation=1):
        """Implementation of Res2Net Block with dilation
           The paper is refered as "Res2Net: A New Multi-scale Backbone Architecture",
           whose url is https://arxiv.org/abs/1904.01169
        Args:
            in_channels (int): input channels or input dimensions
            out_channels (int): output channels or output dimensions
            scale (int, optional): scale in res2net bolck. Defaults to 8.
            dilation (int, optional): dilation of 1-d convolution in TDNN block. Defaults to 1.
        """
        super().__init__()
        assert in_channels % scale == 0
        assert out_channels % scale == 0

        in_channel = in_channels // scale
        hidden_channel = out_channels // scale

        self.blocks = nn.LayerList([
            TDNNBlock(
                in_channel, hidden_channel, kernel_size=3, dilation=dilation)
            for i in range(scale - 1)
        ])
        self.scale = scale

    def forward(self, x):
        y = []
        for i, x_i in enumerate(paddle.chunk(x, self.scale, axis=1)):
            if i == 0:
                y_i = x_i
            elif i == 1:
                y_i = self.blocks[i - 1](x_i)
            else:
                y_i = self.blocks[i - 1](x_i + y_i)
            y.append(y_i)
        y = paddle.concat(y, axis=1)
        return y


class SEBlock(nn.Layer):
    def __init__(self, in_channels, se_channels, out_channels):
        """Implementation of SEBlock
           The paper is refered as "Squeeze-and-Excitation Networks"
           whose url is https://arxiv.org/abs/1709.01507
        Args:
            in_channels (int): input channels or input data dimensions
            se_channels (_type_): _description_
            out_channels (int): output channels or output data dimensions
        """
        super().__init__()

        self.conv1 = Conv1d(
            in_channels=in_channels, out_channels=se_channels, kernel_size=1)
        self.relu = paddle.nn.ReLU()
        self.conv2 = Conv1d(
            in_channels=se_channels, out_channels=out_channels, kernel_size=1)
        self.sigmoid = paddle.nn.Sigmoid()

    def forward(self, x, lengths=None):
        L = x.shape[-1]
        if lengths is not None:
            mask = length_to_mask(lengths * L, max_len=L)
            mask = mask.unsqueeze(1)
            total = mask.sum(axis=2, keepdim=True)
            s = (x * mask).sum(axis=2, keepdim=True) / total
        else:
            s = x.mean(axis=2, keepdim=True)

        s = self.relu(self.conv1(s))
        s = self.sigmoid(self.conv2(s))

        return s * x


class SERes2NetBlock(nn.Layer):
    def __init__(
            self,
            in_channels,
            out_channels,
            res2net_scale=8,
            se_channels=128,
            kernel_size=1,
            dilation=1,
            activation=nn.ReLU, ):
        """Implementation of Squeeze-Extraction Res2Blocks in ECAPA-TDNN network model
           The paper is refered "Squeeze-and-Excitation Networks"
           whose url is: https://arxiv.org/pdf/1709.01507.pdf
        Args:
            in_channels (int): input channels or input data dimensions
            out_channels (int): output channels or output data dimensions
            res2net_scale (int, optional): scale in the res2net block. Defaults to 8.
            se_channels (int, optional): embedding dimensions of res2net block. Defaults to 128.
            kernel_size (int, optional): kernel size of 1-d convolution in TDNN block. Defaults to 1.
            dilation (int, optional): dilation of 1-d convolution in TDNN block. Defaults to 1.
            activation (paddle.nn.class, optional): activation function. Defaults to nn.ReLU.
        """
        super().__init__()
        self.out_channels = out_channels
        self.tdnn1 = TDNNBlock(
            in_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
            activation=activation, )
        self.res2net_block = Res2NetBlock(out_channels, out_channels,
                                          res2net_scale, dilation)
        self.tdnn2 = TDNNBlock(
            out_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
            activation=activation, )
        self.se_block = SEBlock(out_channels, se_channels, out_channels)

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1, )

    def forward(self, x, lengths=None):
        residual = x
        if self.shortcut:
            residual = self.shortcut(x)

        x = self.tdnn1(x)
        x = self.res2net_block(x)
        x = self.tdnn2(x)
        x = self.se_block(x, lengths)

        return x + residual


class EcapaTdnn(nn.Layer):
    def __init__(
            self,
            input_size,
            embd_dim=192,
            pooling_type="ASP",
            activation=nn.ReLU,
            channels=[512, 512, 512, 512, 1536],
            kernel_sizes=[5, 3, 3, 3, 1],
            dilations=[1, 2, 3, 4, 1],
            attention_channels=128,
            res2net_scale=8,
            se_channels=128,
            global_context=True, ):
        """Implementation of ECAPA-TDNN backbone model network
           The paper is refered as "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification"
           whose url is: https://arxiv.org/abs/2005.07143
        Args:
            input_size (_type_): input fature dimension
            embd_dim (int, optional): speaker embedding size. Defaults to 192.
            activation (paddle.nn.class, optional): activation function. Defaults to nn.ReLU.
            channels (list, optional): inter embedding dimension. Defaults to [512, 512, 512, 512, 1536].
            kernel_sizes (list, optional): kernel size of 1-d convolution in TDNN block . Defaults to [5, 3, 3, 3, 1].
            dilations (list, optional): dilations of 1-d convolution in TDNN block. Defaults to [1, 2, 3, 4, 1].
            attention_channels (int, optional): attention dimensions. Defaults to 128.
            res2net_scale (int, optional): scale value in res2net. Defaults to 8.
            se_channels (int, optional): dimensions of squeeze-excitation block. Defaults to 128.
            global_context (bool, optional): global context flag. Defaults to True.
        """
        super().__init__()
        assert len(channels) == len(kernel_sizes)
        assert len(channels) == len(dilations)
        self.input_size = input_size
        self.channels = channels
        self.blocks = nn.LayerList()
        self.embd_dim = embd_dim

        # The initial TDNN layer
        self.blocks.append(
            TDNNBlock(
                input_size,
                channels[0],
                kernel_sizes[0],
                dilations[0],
                activation, ))

        # SE-Res2Net layers
        for i in range(1, len(channels) - 1):
            self.blocks.append(
                SERes2NetBlock(
                    channels[i - 1],
                    channels[i],
                    res2net_scale=res2net_scale,
                    se_channels=se_channels,
                    kernel_size=kernel_sizes[i],
                    dilation=dilations[i],
                    activation=activation, ))

        # Multi-layer feature aggregation
        self.mfa = TDNNBlock(
            channels[-1],
            channels[-1],
            kernel_sizes[-1],
            dilations[-1],
            activation, )

        cat_channels = channels[-1]
        if pooling_type == "ASP":
            self.asp = AttentiveStatisticsPooling(channels[-1],
                                                  attention_channels=attention_channels,
                                                  global_context=global_context)
            self.asp_bn = BatchNorm1d(input_size=channels[-1] * 2)
            # Final linear transformation
            self.fc = Conv1d(in_channels=channels[-1] * 2,
                             out_channels=self.embd_dim,
                             kernel_size=1)
        elif pooling_type == "SAP":
            self.asp = SelfAttentivePooling(cat_channels, 128)
            self.asp_bn = nn.BatchNorm1D(cat_channels)
            # Final linear transformation
            self.fc = Conv1d(in_channels=cat_channels,
                             out_channels=self.embd_dim,
                             kernel_size=1)
        elif pooling_type == "TAP":
            self.asp = TemporalAveragePooling()
            self.asp_bn = nn.BatchNorm1D(cat_channels)
            # Final linear transformation
            self.fc = Conv1d(in_channels=cat_channels,
                             out_channels=self.embd_dim,
                             kernel_size=1)
        elif pooling_type == "TSP":
            self.asp = TemporalStatisticsPooling()
            self.asp_bn = nn.BatchNorm1D(cat_channels * 2)
            # Final linear transformation
            self.fc = Conv1d(in_channels=cat_channels * 2,
                             out_channels=self.embd_dim,
                             kernel_size=1)
        else:
            raise Exception(f'没有{pooling_type}池化层！')

    def forward(self, x, lengths=None):
        """
        Compute embeddings.

        Args:
            x (paddle.Tensor): Input data with shape (N, time, freq).
            lengths (paddle.Tensor, optional): Length proportions of batch length with shape (N). Defaults to None.

        Returns:
            paddle.Tensor: Output embeddings with shape (N, self.emb_size, 1)
        """
        x = x.transpose([0, 2, 1])
        xl = []
        for layer in self.blocks:
            try:
                x = layer(x, lengths=lengths)
            except TypeError:
                x = layer(x)
            xl.append(x)

        # Multi-layer feature aggregation
        x = paddle.concat(xl[1:], axis=1)
        x = self.mfa(x)

        # Attentive Statistical Pooling
        x = self.asp(x, lengths=lengths)
        x = self.asp_bn(x)
        x = x.unsqueeze(2)
        # Final linear transformation
        x = self.fc(x).squeeze(-1)  # (N, emb_size, 1) -> (N, emb_size)

        return x
