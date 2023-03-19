import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def length_to_mask(length, max_len=None, dtype=None):
    assert len(length.shape) == 1

    if max_len is None:
        max_len = length.max().astype('int').item()  # using arange to generate mask
    mask = paddle.arange(max_len, dtype=length.dtype).expand((len(length), max_len)) < length.unsqueeze(1)

    if dtype is None:
        dtype = length.dtype

    mask = paddle.to_tensor(mask, dtype=dtype)
    return mask


class Conv1d(nn.Layer):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding="same",
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="reflect", ):
        """_summary_

        Args:
            in_channels (int): intput channel or input data dimensions
            out_channels (int): output channel or output data dimensions
            kernel_size (int): kernel size of 1-d convolution
            stride (int, optional): strid in 1-d convolution . Defaults to 1.
            padding (str, optional): padding value. Defaults to "same".
            dilation (int, optional): dilation in 1-d convolution. Defaults to 1.
            groups (int, optional): groups in 1-d convolution. Defaults to 1.
            bias (bool, optional): bias in 1-d convolution . Defaults to True.
            padding_mode (str, optional): padding mode. Defaults to "reflect".
        """
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.padding_mode = padding_mode

        self.conv = nn.Conv1D(
            in_channels,
            out_channels,
            self.kernel_size,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
            groups=groups,
            bias_attr=bias, )

    def forward(self, x):
        if self.padding == "same":
            x = self._manage_padding(x, self.kernel_size, self.dilation, self.stride)
        else:
            raise ValueError("Padding must be 'same'. Got {self.padding}")

        return self.conv(x)

    def _manage_padding(self, x, kernel_size: int, dilation: int, stride: int):
        L_in = x.shape[-1]  # Detecting input shape
        padding = self._get_padding_elem(L_in, stride, kernel_size, dilation)  # Time padding
        x = F.pad(x, padding, mode=self.padding_mode, data_format="NCL")  # Applying padding
        return x

    def _get_padding_elem(self,
                          L_in: int,
                          stride: int,
                          kernel_size: int,
                          dilation: int):
        if stride > 1:
            n_steps = math.ceil(((L_in - kernel_size * dilation) / stride) + 1)
            L_out = stride * (n_steps - 1) + kernel_size * dilation
            padding = [kernel_size // 2, kernel_size // 2]
        else:
            L_out = (L_in - dilation * (kernel_size - 1) - 1) // stride + 1

            padding = [(L_in - L_out) // 2, (L_in - L_out) // 2]

        return padding


class BatchNorm1d(nn.Layer):
    def __init__(
            self,
            input_size,
            eps=1e-05,
            momentum=0.9,
            weight_attr=None,
            bias_attr=None,
            data_format='NCL',
            use_global_stats=None, ):
        super().__init__()

        self.norm = nn.BatchNorm1D(
            input_size,
            epsilon=eps,
            momentum=momentum,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            data_format=data_format,
            use_global_stats=use_global_stats, )

    def forward(self, x):
        x_n = self.norm(x)
        return x_n


class TDNNBlock(nn.Layer):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            dilation,
            activation=nn.ReLU, ):
        """Implementation of TDNN network

        Args:
            in_channels (int): input channels or input embedding dimensions
            out_channels (int): output channels or output embedding dimensions
            kernel_size (int): the kernel size of the TDNN network block
            dilation (int): the dilation of the TDNN network block
            activation (paddle class, optional): the activation layers. Defaults to nn.ReLU.
        """
        super().__init__()
        self.conv = Conv1d(in_channels=in_channels,
                           out_channels=out_channels,
                           kernel_size=kernel_size,
                           dilation=dilation, )
        self.activation = activation()
        self.norm = BatchNorm1d(input_size=out_channels)

    def forward(self, x):
        return self.norm(self.activation(self.conv(x)))
