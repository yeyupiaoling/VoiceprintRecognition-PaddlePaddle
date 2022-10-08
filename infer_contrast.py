import argparse
import functools
import os

import numpy as np
import paddle

from modules.ecapa_tdnn import EcapaTdnn, SpeakerIdetification
from data_utils.reader import load_audio, CustomDataset
from utils.utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('use_model',        str,    'ecapa_tdnn',             '所使用的模型')
add_arg('audio_path1',      str,    'audio/a_1.wav',          '预测第一个音频')
add_arg('audio_path2',      str,    'audio/b_2.wav',          '预测第二个音频')
add_arg('threshold',        float,  0.6,                      '判断是否为同一个人的阈值')
add_arg('audio_duration',   float,  3,                        '预测的音频长度，单位秒')
add_arg('feature_method',   str,    'melspectrogram',         '音频特征提取方法', choices=['melspectrogram', 'spectrogram'])
add_arg('resume',           str,    'models/',                '模型文件夹路径')
args = parser.parse_args()
print_arguments(args)

dataset = CustomDataset(data_list_path=None, feature_method=args.feature_method)
# 获取模型
if args.use_model == 'ecapa_tdnn':
    ecapa_tdnn = EcapaTdnn(input_size=dataset.input_size)
    model = SpeakerIdetification(backbone=ecapa_tdnn)
else:
    raise Exception(f'{args.use_model} 模型不存在！')
# 加载模型
model_path = os.path.join(args.resume, args.use_model, 'model.pdparams')
model.set_state_dict(paddle.load(model_path))
print(f"成功加载模型参数和优化方法参数：{model_path}")
model.eval()


# 预测音频
def infer(audio_path):
    data = load_audio(audio_path, mode='infer', feature_method=args.feature_method, chunk_duration=args.audio_duration)
    data = data[np.newaxis, :]
    data = paddle.to_tensor(data, dtype='float32')
    data_length = paddle.to_tensor([1], dtype='float32')
    # 执行预测
    feature = model.backbone(data, data_length)
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
