import random


class VolumePerturbAugmentor(object):
    """添加随机体积扰动的增强模型
    
    This is used for multi-loudness training of PCEN. See

    https://arxiv.org/pdf/1607.05666v1.pdf

    for more details.

    :param min_gain_dBFS: Minimal gain in dBFS.
    :type min_gain_dBFS: float
    :param max_gain_dBFS: Maximal gain in dBFS.
    :type max_gain_dBFS: float
    """

    def __init__(self, min_gain_dBFS=-15, max_gain_dBFS=15, prob=0.5):
        self.prob = prob
        self._min_gain_dBFS = min_gain_dBFS
        self._max_gain_dBFS = max_gain_dBFS

    def __call__(self, wav):
        """Change audio loadness.

        Note that this is an in-place transformation.

        :param wav: Audio data
        :type wav: numpy.ndarray
        """
        if random.random() > self.prob:
            return wav
        gain = random.uniform(self._min_gain_dBFS, self._max_gain_dBFS)
        wav *= 10.**(gain / 20.)
        return wav
