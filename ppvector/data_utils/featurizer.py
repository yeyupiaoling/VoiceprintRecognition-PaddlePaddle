import paddle
from paddle import nn
from paddle.audio.features import LogMelSpectrogram, MelSpectrogram, Spectrogram, MFCC


class AudioFeaturizer(nn.Layer):
    """音频特征器

    :param feature_conf: 预处理方法的参数
    :type feature_conf: dict
    :param sample_rate: 用于训练的音频的采样率
    :type sample_rate: int
    """

    def __init__(self, feature_method='MelSpectrogram', feature_conf={}):
        super().__init__()
        self._feature_conf = feature_conf
        self._feature_method = feature_method
        if feature_method == 'LogMelSpectrogram':
            self.feat_fun = LogMelSpectrogram(**feature_conf)
        elif feature_method == 'MelSpectrogram':
            self.feat_fun = MelSpectrogram(**feature_conf)
        elif feature_method == 'Spectrogram':
            self.feat_fun = Spectrogram(**feature_conf)
        elif feature_method == 'MFCC':
            self.feat_fun = MFCC(**feature_conf)
        else:
            raise Exception(f'预处理方法 {self._feature_method} 不存在!')

    def forward(self, waveforms, input_lens_ratio):
        """从AudioSegment中提取音频特征

        :param waveforms: Audio segment to extract features from.
        :type waveforms: AudioSegment
        :param input_lens_ratio: input length ratio
        :type input_lens_ratio: tensor
        :return: Spectrogram audio feature in 2darray.
        :rtype: ndarray
        """
        feature = self.feat_fun(waveforms)
        feature = feature.transpose([0, 2, 1])
        # 归一化
        mean = paddle.mean(feature, 1, keepdim=True)
        std = paddle.std(feature, 1, keepdim=True)
        feature = (feature - mean) / (std + 1e-5)
        # 对掩码比例进行扩展
        input_lens = (input_lens_ratio * feature.shape[1]).astype(paddle.int32)
        mask_lens = input_lens.unsqueeze(1)
        # 生成掩码张量
        idxs = paddle.arange(feature.shape[1])
        idxs = idxs.tile([feature.shape[0], 1])
        mask = idxs < mask_lens
        mask = mask.unsqueeze(-1)
        # 对特征进行掩码操作
        feature_masked = paddle.where(mask, feature, paddle.zeros_like(feature))
        return feature_masked, input_lens

    @property
    def feature_dim(self):
        """返回特征大小

        :return: 特征大小
        :rtype: int
        """
        if self._feature_method == 'LogMelSpectrogram':
            return self._feature_conf.n_mels
        elif self._feature_method == 'MelSpectrogram':
            return self._feature_conf.n_mels
        elif self._feature_method == 'Spectrogram':
            return self._feature_conf.n_fft // 2 + 1
        elif self._feature_method == 'MFCC':
            return self._feature_conf.n_mfcc
        else:
            raise Exception('没有{}预处理方法'.format(self._feature_method))
