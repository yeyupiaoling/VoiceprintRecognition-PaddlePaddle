import argparse
import functools

import yaml

from ppvector.predict import PPVectorPredictor
from ppvector.utils.record import RecordAudio
from ppvector.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,    'configs/ecapa_tdnn.yml',   '配置文件')
add_arg('use_gpu',          bool,   True,                       '是否使用GPU预测')
add_arg('audio_db_path',    str,    'audio_db/',                '音频库的路径')
add_arg('record_seconds',   int,    3,                          '录音长度')
add_arg('threshold',        float,  0.6,                        '判断是否为同一个人的阈值')
add_arg('model_path',       str,    'models/{}_{}/best_model/', '导出的预测模型文件路径')
args = parser.parse_args()

# 读取配置文件
with open(args.configs, 'r', encoding='utf-8') as f:
    configs = yaml.load(f.read(), Loader=yaml.FullLoader)
print_arguments(args, configs)

# 获取识别器
predictor = PPVectorPredictor(configs=configs,
                              threshold=args.threshold,
                              audio_db_path=args.audio_db_path,
                              model_path=args.model_path.format(configs['use_model'], configs['preprocess_conf']['feature_method']),
                              use_gpu=args.use_gpu)

record_audio = RecordAudio()

while True:
    select_fun = int(input("请选择功能，0为注册音频到声纹库，1为执行声纹识别："))
    if select_fun == 0:
        audio_data = record_audio.record(record_seconds=args.record_seconds)
        name = input("请输入该音频用户的名称：")
        if name == '': continue
        predictor.register(user_name=name, audio_data=audio_data)
    elif select_fun == 1:
        audio_data = record_audio.record(record_seconds=args.record_seconds)
        name = predictor.recognition(audio_data)
        if name:
            print(f"识别说话的为：{name}")
        else:
            print(f"没有识别到说话人，可能是没注册。")
    else:
        print('请正确选择功能')
