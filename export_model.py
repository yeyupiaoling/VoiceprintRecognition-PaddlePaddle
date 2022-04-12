import argparse
import functools
import os
from datetime import datetime

import paddle
from paddle.static import InputSpec

from modules.model import EcapaTdnn, SpeakerIdetification
from utils.reader import CustomDataset
from utils.utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('feature_method',   str,    'melspectrogram',         '音频特征提取方法')
add_arg('save_model',       str,    'models/',                '模型保存的路径')
add_arg('resume',           str,    'models/epoch_49',        '导出模型文件夹路径')
args = parser.parse_args()
print_arguments(args)

dataset = CustomDataset(data_list_path=None)
# 获取模型
ecapa_tdnn = EcapaTdnn(input_size=dataset.input_size)
model = SpeakerIdetification(backbone=ecapa_tdnn, num_class=1)
paddle.summary(model.backbone, input_size=(1, dataset.input_size, 98))

model.set_state_dict(paddle.load(os.path.join(args.resume, 'model.pdparams')))
print('[%s] 成功加载模型参数和优化方法参数' % datetime.now())


# 保存预测模型
if not os.path.exists(os.path.join(args.save_model, 'infer')):
    os.makedirs(os.path.join(args.save_model, 'infer'))
paddle.jit.save(layer=model.backbone,
                path=os.path.join(args.save_model, 'infer/model'),
                input_spec=[InputSpec(shape=[None, dataset.input_size, None], dtype=paddle.float32),
                            InputSpec(shape=[None], dtype=paddle.float32)])
print('[%s] 模型导出成功：%s' % (datetime.now(), os.path.join(args.save_model, 'infer/model')))
