import argparse
import functools

from ppvector.trainer import PPVectorTrainer
from ppvector.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,    'configs/cam++.yml',        '配置文件')
add_arg('save_dir',         str,    'dataset/features',         '保存特征的路径')
add_arg('max_duration',     int,    100,                        '提取特征的最大时长，单位秒')
args = parser.parse_args()
print_arguments(args=args)

# 获取训练器
trainer = PPVectorTrainer(configs=args.configs)

# 提取特征保存文件
trainer.extract_features(save_dir=args.save_dir, max_duration=args.max_duration)
