import librosa
import numpy as np
import paddle.fluid as fluid

# 创建执行器
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

# 保存预测模型路径
save_path = 'models/infer'


[infer_program,
 feeded_var_names,
 target_var] = fluid.io.load_inference_model(dirname=save_path, executor=exe)


# 读取音频数据
def load_data(data_path):
    wav, sr = librosa.load(data_path, sr=16000)
    intervals = librosa.effects.split(wav, top_db=20)
    wav_output = []
    for sliced in intervals:
        wav_output.extend(wav[sliced[0]:sliced[1]])
    # [可能需要修改] 裁剪的音频长度：16000 * 秒数
    wav_len = int(16000 * 2.04)
    # 裁剪过长的音频，过短的补0
    if len(wav_output) > wav_len:
        wav_output = wav_output[:wav_len]
    else:
        wav_output.extend(np.zeros(shape=[wav_len - len(wav_output)], dtype=np.float32))
    wav_output = np.array(wav_output)
    # 获取梅尔频谱
    ps = librosa.feature.melspectrogram(y=wav_output, sr=sr, hop_length=256).astype(np.float32)
    ps = ps[np.newaxis, np.newaxis, ...]
    return ps


def infer(audio_path):
    data = load_data(audio_path)
    # 执行预测
    feature = exe.run(program=infer_program,
                      feed={feeded_var_names[0]: data},
                      fetch_list=target_var)[0]
    return feature[0]


if __name__ == '__main__':
    # 要预测的两个人的音频文件
    person1 = 'dataset/ST-CMDS-20170001_1-OS/20170001P00001A0101.wav'
    person2 = 'dataset/ST-CMDS-20170001_1-OS/20170001P00001A0001.wav'
    feature1 = infer(person1)
    feature2 = infer(person2)
    # 对角余弦值
    dist = np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
    if dist > 0.9:
        print("%s 和 %s 为同一个人，相似度为：%f" % (person1, person2, dist))
    else:
        print("%s 和 %s 不是同一个人，相似度为：%f" % (person1, person2, dist))
