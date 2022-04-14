import os
import random
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import librosa


class NoisePerturbAugmentor(object):
    """用于添加背景噪声的增强模型

    :param min_snr_dB: Minimal signal noise ratio, in decibels.
    :type min_snr_dB: float
    :param max_snr_dB: Maximal signal noise ratio, in decibels.
    :type max_snr_dB: float
    :param noise_path: Manifest path for noise audio data.
    :type noise_path: str
    """

    def __init__(self, min_snr_dB=10, max_snr_dB=30, noise_path="dataset/noise", sr=16000, prob=0.5):
        self.prob = prob
        self.sr = sr
        self._min_snr_dB = min_snr_dB
        self._max_snr_dB = max_snr_dB
        self._noise_files = self.get_noise_file(noise_path=noise_path)

    # 获取全部噪声数据
    @staticmethod
    def get_noise_file(noise_path):
        noise_files = []
        for file in os.listdir(noise_path):
            noise_files.append(os.path.join(noise_path, file))
        return noise_files

    @staticmethod
    def rms_db(wav):
        """返回以分贝为单位的音频均方根能量

        :return: Root mean square energy in decibels.
        :rtype: float
        """
        mean_square = np.mean(wav ** 2)
        return 10 * np.log10(mean_square)

    def __call__(self, wav):
        """Add background noise audio.

        Note that this is an in-place transformation.

        :param audio_segment: Audio segment to add effects to.
        :type audio_segment: AudioSegmenet|SpeechSegment
        """
        if random.random() > self.prob: return wav
        if len(self._noise_files) == 0: return wav
        snr_dB = random.uniform(self._min_snr_dB, self._max_snr_dB)
        noise, r = librosa.load(random.choice(self._noise_files), sr=self.sr)
        noise_gain_db = min(self.rms_db(wav) - self.rms_db(noise) - snr_dB, 300)
        noise *= 10. ** (noise_gain_db / 20.)
        noise_new = np.zeros(wav.shape, dtype=np.float32)
        if noise.shape[0] >= wav.shape[0]:
            start = random.randint(0, noise.shape[0] - wav.shape[0])
            noise_new[:wav.shape[0]] = noise[start: start + wav.shape[0]]
        else:
            start = random.randint(0, wav.shape[0] - noise.shape[0])
            noise_new[start:start + noise.shape[0]] = noise[:]
        wav += noise_new
        return wav
