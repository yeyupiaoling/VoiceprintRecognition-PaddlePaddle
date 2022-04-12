import argparse
import functools

import numpy as np
import paddle

from utils.reader import load_audio
from utils.utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('audio_path1',      str,    'audio/a_1.wav',          '预测第一个音频')
add_arg('audio_path2',      str,    'audio/b_2.wav',          '预测第二个音频')
add_arg('threshold',        float,   0.7,                     '判断是否为同一个人的阈值')
add_arg('feature_method',   str,    'melspectrogram',         '音频特征提取方法')
add_arg('model_path',       str,    'models/infer/model',     '预测模型的路径')
args = parser.parse_args()

print_arguments(args)

# 加载模型
model = paddle.jit.load(args.model_path)
model.eval()


# 预测音频
def infer(audio_path):
    data = load_audio(audio_path, mode='infer', feature_method=args.feature_method)
    data = data[np.newaxis, :]
    data = paddle.to_tensor(data, dtype='float32')
    data_length = paddle.to_tensor(data.shape[-1], dtype='float32')
    # 执行预测
    feature = model(data, data_length)
    return feature.numpy()


if __name__ == '__main__':
    # 要预测的两个人的音频文件
    feature1 = infer(args.audio_path1)[0]
    feature2 = infer(args.audio_path2)[0]
    # 对角余弦值
    dist = np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
    if dist > args.threshold:
        print(f"{args.audio_path1} 和 {args.audio_path2} 为同一个人，相似度为：{dist}")
    else:
        print(f"{args.audio_path1} 和 {args.audio_path2} 不是同一个人，相似度为：{dist}")
