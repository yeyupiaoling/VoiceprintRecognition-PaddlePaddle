import struct
import mmap
import numpy as np


class ReadData(object):
    def __init__(self, prefix_path):
        self.offset_dict = {}
        for line in open(prefix_path + '.header', 'rb'):
            key, val_pos, val_len = line.split('\t'.encode('ascii'))
            self.offset_dict[key] = (int(val_pos), int(val_len))
        self.fp = open(prefix_path + '.data', 'rb')
        self.m = mmap.mmap(self.fp.fileno(), 0, access=mmap.ACCESS_READ)
        print('loading label')
        # 获取label
        self.label = {}
        for line in open(prefix_path + '.label', 'rb'):
            key, label = line.split(b'\t')
            self.label[key] = [int(label.decode().replace('\n', ''))]
        print('finish loading data:', len(self.label))

    # 获取图像数据
    def get_data(self, key):
        p = self.offset_dict.get(key, None)
        if p is None:
            return None
        val_pos, val_len = p
        return self.m[val_pos:val_pos + val_len]

    # 获取图像标签
    def get_label(self, key):
        return self.label.get(key)

    # 获取所有keys
    def get_keys(self):
        return self.label.keys()


def mapper(sample):
    data, label = sample
    # [可能需要修改] 梅尔频谱的shape
    data = list(struct.unpack('%sd' % (128 * 128), data))
    data = np.array(data).reshape((1, 128, 128)).astype(np.float32)
    assert (data is not None), 'data is None'
    return data, label


def train_reader(data_path, batch_size):
    def reader():
        readData = ReadData(data_path)
        keys = readData.get_keys()
        keys = list(keys)
        np.random.shuffle(keys)

        batch_data, batch_label = [], []
        for key in keys:
            data = readData.get_data(key)
            assert (data is not None)
            label = readData.get_label(key)
            assert (label is not None)
            sample = (data, label)
            d, label = mapper(sample)
            batch_data.append([d])
            batch_label.append(label)
            if len(batch_data) == batch_size:
                yield np.vstack(batch_data), np.vstack(batch_label).astype(np.int64)
                batch_data, batch_label = [], []

    return reader


def test_reader(data_path, batch_size):
    def reader():
        readData = ReadData(data_path)
        keys = readData.get_keys()
        keys = list(keys)

        batch_data, batch_label = [], []
        for key in keys:
            data = readData.get_data(key)
            assert (data is not None)
            label = readData.get_label(key)
            assert (label is not None)
            sample = (data, label)
            d, label = mapper(sample)
            batch_data.append([d])
            batch_label.append(label)
            if len(batch_data) == batch_size:
                yield np.vstack(batch_data), np.vstack(batch_label).astype(np.int64)
                batch_data, batch_label = [], []

    return reader
