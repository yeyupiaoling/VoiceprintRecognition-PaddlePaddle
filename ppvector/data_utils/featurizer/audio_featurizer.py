import numpy as np
import paddle
from paddleaudio.compliance.kaldi import fbank, spectrogram

from ppvector.data_utils.utils import cmvn_floating_kaldi


class AudioFeaturizer(object):
    """音频特征器

    :param sample_rate: 用于训练的音频的采样率
    :type sample_rate: int
    :param use_dB_normalization: 是否对音频进行音量归一化
    :type use_dB_normalization: bool
    :param target_dB: 对音频进行音量归一化的音量分贝值
    :type target_dB: float
    """

    def __init__(self,
                 feature_method='spectrogram',
                 n_mels=80,
                 sample_rate=16000,
                 use_dB_normalization=True,
                 target_dB=-20,
                 frame_shift=10,
                 frame_length=25):
        self._feature_method = feature_method
        self._target_sample_rate = sample_rate
        self._n_mels = n_mels
        self._use_dB_normalization = use_dB_normalization
        self._target_dB = target_dB
        self._frame_shift = frame_shift
        self._frame_length = frame_length

    def featurize(self, audio_segment):
        """从AudioSegment中提取音频特征

        :param audio_segment: Audio segment to extract features from.
        :type audio_segment: AudioSegment
        :return: Spectrogram audio feature in 2darray.
        :rtype: ndarray
        """
        # upsampling or downsampling
        if audio_segment.sample_rate != self._target_sample_rate:
            audio_segment.resample(self._target_sample_rate)
        # decibel normalization
        if self._use_dB_normalization:
            audio_segment.normalize(target_db=self._target_dB)
        # 获取音频特征
        waveform = paddle.to_tensor(np.expand_dims(audio_segment.samples, 0), dtype=paddle.float32)
        if self._feature_method == 'spectrogram':
            # 计算声谱图
            feature = spectrogram(waveform=waveform,
                                  frame_length=self._frame_length,
                                  frame_shift=self._frame_shift,
                                  sr=audio_segment.sample_rate).numpy()
        elif self._feature_method == 'melspectrogram':
            # 计算梅尔频谱
            feature = fbank(waveform=waveform,
                            n_mels=self._n_mels,
                            frame_length=self._frame_length,
                            frame_shift=self._frame_shift,
                            sr=audio_segment.sample_rate).numpy()
        else:
            raise Exception(f'预处理方法 {self._feature_method} 不存在！')
        # 归一化
        feature = cmvn_floating_kaldi(feature, LC=150, RC=149, norm_vars=False).astype(np.float32)
        feature = feature.T
        return feature

    @property
    def feature_dim(self):
        """返回特征大小

        :return: 特征大小
        :rtype: int
        """
        if self._feature_method == 'melspectrogram':
            return 80
        elif self._feature_method == 'spectrogram':
            return 257
        else:
            raise Exception('没有{}预处理方法'.format(self._feature_method))
