import os
import struct
import uuid
import librosa
import numpy as np
from tqdm import tqdm


class DataSetWriter(object):
    def __init__(self, prefix):
        # 创建对应的数据文件
        self.data_file = open(prefix + '.data', 'wb')
        self.header_file = open(prefix + '.header', 'wb')
        self.label_file = open(prefix + '.label', 'wb')
        self.offset = 0
        self.header = ''

    def add_data(self, key, data):
        # 写入图像数据
        self.data_file.write(struct.pack('I', len(key)))
        self.data_file.write(key.encode('ascii'))
        self.data_file.write(struct.pack('I', len(data)))
        self.data_file.write(data)
        self.offset += 4 + len(key) + 4
        self.header = key + '\t' + str(self.offset) + '\t' + str(len(data)) + '\n'
        self.header_file.write(self.header.encode('ascii'))
        self.offset += len(data)

    def add_label(self, label):
        # 写入标签数据
        self.label_file.write(label.encode('ascii') + '\n'.encode('ascii'))


# 格式二进制转换
def convert_data(data_list_path, output_prefix):
    # 读取列表
    data_list = open(data_list_path, "r").readlines()
    print("train_data size:", len(data_list))

    # 开始写入数据
    writer = DataSetWriter(output_prefix)
    for record in tqdm(data_list):
        try:
            path, label = record.replace('\n', '').split('\t')
            wav, sr = librosa.load(path)
            intervals = librosa.effects.split(wav, top_db=20)
            wav_output = []
            for sliced in intervals:
                wav_output.extend(wav[sliced[0]:sliced[1]])
            # 裁剪过长的音频，过短的补0
            if len(wav_output) > 65489:
                wav_output = wav_output[:65489]
            else:
                wav_output.extend(np.zeros(shape=[65489 - len(wav_output)], dtype=np.float32))
            wav_output = np.array(wav_output)
            # 转成梅尔频谱
            ps = librosa.feature.melspectrogram(y=wav_output, sr=sr).reshape(-1).tolist()
            if len(ps) != 128 * 128: continue
            data = struct.pack('%sd' % len(ps), *ps)
            # 写入对应的数据
            key = str(uuid.uuid1())
            writer.add_data(key, data)
            writer.add_label('\t'.join([key, label.replace('\n', '')]))
        except Exception as e:
            print(e)


# 生成数据列表
def get_data_list(audio_path, list_path):
    files = os.listdir(audio_path)

    f_train = open(os.path.join(list_path, 'train_list.txt'), 'w')
    f_test = open(os.path.join(list_path, 'test_list.txt'), 'w')

    sound_sum = 0
    s = set()
    for file in files:
        if '.wav' not in file:
            continue
        s.add(file[:15])
        sound_path = os.path.join(audio_path, file)
        if sound_sum % 100 == 0:
            f_test.write('%s\t%d\n' % (sound_path.replace('\\', '/'), len(s) - 1))
        else:
            f_train.write('%s\t%d\n' % (sound_path.replace('\\', '/'), len(s) - 1))
        sound_sum += 1

    f_test.close()
    f_train.close()


if __name__ == '__main__':
    get_data_list('dataset/ST-CMDS-20170001_1-OS', 'dataset')
    convert_data('dataset/train_list.txt', 'dataset/train')
    convert_data('dataset/test_list.txt', 'dataset/test')
