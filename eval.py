import argparse
import functools
import os

import numpy as np
import paddle
from paddle.io import DataLoader
from paddle.metric import accuracy
from tqdm import tqdm

from modules.ecapa_tdnn import EcapaTdnn, SpeakerIdetification
from data_utils.reader import CustomDataset, collate_fn
from utils.utility import add_arguments, print_arguments, cal_accuracy_threshold

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('use_model',        str,    'ecapa_tdnn',             '所使用的模型')
add_arg('num_speakers',     int,    3242,                     '分类的类别数量')
add_arg('audio_duration',   float,  3,                        '评估的音频长度，单位秒')
add_arg('feature_method',   str,    'melspectrogram',         '音频特征提取方法', choices=['melspectrogram', 'spectrogram'])
add_arg('list_path',        str,    'dataset/test_list.txt',  '测试数据的数据列表路径')
add_arg('resume',           str,    'models/',                '模型文件夹路径')
args = parser.parse_args()
print_arguments(args)


dataset = CustomDataset(data_list_path=None, feature_method=args.feature_method)
# 获取模型
if args.use_model == 'ecapa_tdnn':
    ecapa_tdnn = EcapaTdnn(input_size=dataset.input_size)
    model = SpeakerIdetification(backbone=ecapa_tdnn, num_class=args.num_speakers)
else:
    raise Exception(f'{args.use_model} 模型不存在！')
# 加载模型
model_path = os.path.join(args.resume, args.use_model, 'model.pdparams')
model.set_state_dict(paddle.load(model_path))
print(f"成功加载模型参数和优化方法参数：{model_path}")
model.eval()


def get_all_audio_feature(list_path):
    print('开始提取全部的音频特征...')
    # 测试数据
    eval_dataset = CustomDataset(list_path,
                                 feature_method=args.feature_method,
                                 mode='eval',
                                 sr=16000,
                                 chunk_duration=args.audio_duration)
    eval_loader = DataLoader(dataset=eval_dataset,
                             batch_size=32,
                             collate_fn=collate_fn)
    accuracies = []
    features, labels = None, None
    for batch_id, (audio, label, audio_lens) in tqdm(enumerate(eval_loader())):
        output = model(audio, audio_lens)
        # 计算准确率
        acc = accuracy(input=paddle.nn.functional.softmax(output), label=paddle.reshape(label, shape=(-1, 1)))
        accuracies.append(acc.numpy()[0])
        # 获取特征
        feature = model.backbone(audio, audio_lens).numpy()
        features = np.concatenate((features, feature)) if features is not None else feature
        labels = np.concatenate((labels, label.numpy())) if labels is not None else label.numpy()
    labels = labels.astype(np.int32)
    print('分类准确率为：{:.4f}'.format(sum(accuracies) / len(accuracies)))
    return features, labels


def main():
    features, labels = get_all_audio_feature(args.list_path)
    scores = []
    y_true = []
    print('开始两两对比音频特征...')
    for i in tqdm(range(len(features))):
        feature_1 = features[i]
        feature_1 = np.expand_dims(feature_1, 0).repeat(len(features) - i, axis=0)
        feature_2 = features[i:]
        feature_1 = paddle.to_tensor(feature_1, dtype=paddle.float32)
        feature_2 = paddle.to_tensor(feature_2, dtype=paddle.float32)
        score = paddle.nn.functional.cosine_similarity(feature_1, feature_2, axis=-1).numpy().tolist()
        scores.extend(score)
        y_true.extend(np.array(labels[i] == labels[i:]).astype(np.int32))
    print('找出最优的阈值和对应的准确率...')
    accuracy, threshold = cal_accuracy_threshold(scores, y_true)
    print(f'当阈值为{threshold:.2f}, 两两对比准确率最大，准确率为：{accuracy:.5f}')


if __name__ == '__main__':
    main()
