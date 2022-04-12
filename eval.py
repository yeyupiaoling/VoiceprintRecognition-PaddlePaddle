import argparse
import functools

import numpy as np
import paddle
from paddle.fluid.reader import DataLoader
from tqdm import tqdm

from utils.reader import CustomDataset, collate_fn
from utils.utility import add_arguments, print_arguments, cal_accuracy_threshold

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('feature_method',   str,    'melspectrogram',         '音频特征提取方法')
add_arg('list_path',        str,    'dataset/test_list.txt',  '测试数据的数据列表路径')
add_arg('model_path',       str,    'models/infer/model',     '预测模型的路径')
args = parser.parse_args()

print_arguments(args)

# 加载模型
model = paddle.jit.load(args.model_path)
model.eval()


def get_all_audio_feature(list_path):
    print('开始提取全部的音频特征...')
    # 测试数据
    test_dataset = CustomDataset(list_path,
                                 feature_method=args.feature_method,
                                 mode='test',
                                 sr=16000,
                                 chunk_duration=3)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=32,
                             collate_fn=collate_fn)

    features, labels = None, None
    for batch_id, (audio, label, audio_lens) in enumerate(test_loader()):
        feature = model(audio, audio_lens).numpy()
        features = np.concatenate((features, feature)) if features is not None else feature
        labels = np.concatenate((labels, label.numpy())) if labels is not None else label.numpy()
    labels = labels.astype(np.int32)
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
    accuracy, threshold = cal_accuracy_threshold(scores, y_true)
    print(f'当阈值为{threshold}, 准确率最大，准确率为：{accuracy}')


if __name__ == '__main__':
    main()
