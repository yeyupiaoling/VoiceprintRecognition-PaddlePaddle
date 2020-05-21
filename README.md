# 前言
本章介绍如何使用PaddlePaddle实现简单的声纹识别模型，首先你需要熟悉音频分类，没有了解的可以查看这篇文章[《基于PaddlePaddle实现声音分类》](https://blog.doiduoyi.com/articles/1587999549174.html)
。基于这个知识基础之上，我们训练一个声纹识别模型，通过这个模型我们可以识别说话的人是谁，可以应用在一些需要音频验证的项目。

# 环境准备
主要介绍libsora，PyAudio，pydub的安装，其他的依赖包根据需要自行安装。
 - Python 3.7
 - PaddlePaddle 1.7

## 安装libsora
最简单的方式就是使用pip命令安装，如下：
```shell
pip install pytest-runner
pip install librosa
```

如果pip命令安装不成功，那就使用源码安装，下载源码：[https://github.com/librosa/librosa/releases/](https://github.com/librosa/librosa/releases/)， windows的可以下载zip压缩包，方便解压。
```shell
pip install pytest-runner
tar xzf librosa-<版本号>.tar.gz 或者 unzip librosa-<版本号>.tar.gz
cd librosa-<版本号>/
python setup.py install
```

如果出现`libsndfile64bit.dll': error 0x7e`错误，请指定安装版本0.6.3，如`pip install librosa==0.6.3`

安装ffmpeg， 下载地址：[http://blog.gregzaal.com/how-to-install-ffmpeg-on-windows/](http://blog.gregzaal.com/how-to-install-ffmpeg-on-windows/)，笔者下载的是64位，static版。
然后到C盘，笔者解压，修改文件名为`ffmpeg`，存放在`C:\Program Files\`目录下，并添加环境变量`C:\Program Files\ffmpeg\bin`

最后修改源码，路径为`C:\Python3.7\Lib\site-packages\audioread\ffdec.py`，修改32行代码，如下：
```python
COMMANDS = ('C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe', 'avconv')
```

## 安装PyAudio
使用pip安装命令，如下：
```shell
pip install pyaudio
```
 在安装的时候需要使用到C++库进行编译，如果读者的系统是windows，Python是3.7，可以在这里下载whl安装包，下载地址：[https://github.com/intxcc/pyaudio_portaudio/releases](https://github.com/intxcc/pyaudio_portaudio/releases)

## 安装pydub
使用pip命令安装，如下：
```shell
pip install pydub
```

# 创建数据
本教程笔者使用的是[Free ST Chinese Mandarin Corpus数据集](http://www.openslr.org/38)，这个数据集一共有855个人的语音数据，有102600条语音数据。如果读者有其他更好的数据集，可以混合在一起使用。

如何已经读过笔者[《基于PaddlePaddle实现声音分类》](https://blog.doiduoyi.com/articles/1587999549174.html)这篇文章，应该知道语音数据小而多，最好的方法就是把这些音频文件生成二进制文件，加快训练速度。所以创建`create_data.py`用于生成二进制文件。

首先是创建一个数据列表，数据列表的格式为`<语音文件路径\t语音分类标签>`，创建这个列表主要是方便之后的读取，也是方便读取使用其他的语音数据集，不同的语音数据集，可以通过编写对应的生成数据列表的函数，把这些数据集都写在同一个数据列表中，这样就可以在下一步直接生成二进制文件了。
```python
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
```

有了上面创建的数据列表，就可以把语音数据转换成训练数据了，主要是把语音数据转换成梅尔频谱（Mel Spectrogram），使用librosa可以很方便得到音频的梅尔频谱，使用的API为`librosa.feature.melspectrogram()`，输出的是numpy值，可以直接用tensorflow训练和预测。关于梅尔频谱具体信息读者可以自行了解，跟梅尔频谱同样很重要的梅尔倒谱（MFCCs）更多用于语音识别中，对应的API为`librosa.feature.mfcc()`。在转换过程中，笔者还使用了`librosa.effects.split`裁剪掉静音部分的音频，这样可以减少训练数据的噪声，提供训练准确率。笔者目前默认每条语音的长度为2.04秒，这个读者可以根据自己的情况修改语音的长度，如果要修改训练语音的长度，需要根据注释的提示修改相应的数据值。如果语音长度比较长的，程序会随机裁剪20次，以达到数据增强的效果。
```python
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
            wav, sr = librosa.load(path, sr=16000)
            intervals = librosa.effects.split(wav, top_db=20)
            wav_output = []
            # [可能需要修改] 裁剪的音频长度：16000 * 秒数
            wav_len = int(16000 * 2.04)
            for sliced in intervals:
                wav_output.extend(wav[sliced[0]:sliced[1]])
            for i in range(20):
                # 裁剪过长的音频，过短的补0
                if len(wav_output) > wav_len:
                    l = len(wav_output) - wav_len
                    r = random.randint(0, l)
                    wav_output = wav_output[r:wav_len + r]
                else:
                    wav_output.extend(np.zeros(shape=[wav_len - len(wav_output)], dtype=np.float32))
                wav_output = np.array(wav_output)
                # 转成梅尔频谱
                ps = librosa.feature.melspectrogram(y=wav_output, sr=sr, hop_length=256).reshape(-1).tolist()
                # [可能需要修改] 梅尔频谱的shape，librosa.feature.melspectrogram(y=wav_output, sr=sr, hop_length=256).shape
                if len(ps) != 128 * 128: continue
                data = struct.pack('%sd' % len(ps), *ps)
                # 写入对应的数据
                key = str(uuid.uuid1())
                writer.add_data(key, data)
                writer.add_label('\t'.join([key, label.replace('\n', '')]))
                if len(wav_output) <= wav_len:
                    break
        except Exception as e:
            print(e)
            
if __name__ == '__main__':
    convert_data('dataset/train_list.txt', 'dataset/train')
    convert_data('dataset/test_list.txt', 'dataset/test')
```

创建`reader.py`用于在训练时读取数据。编写一个`ReadData`类，用读取上一步生成的二进制文件，通过`.header`中的`key`和每条数据的偏移量，将`.data`的数据读取出来，并通过`key`来绑定`data`和`label`的对应关系。
```python
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
```

使用上面的工具，创建`train_reader`和`est_reader`函数，用于在训练读取训练数据和测试数据，`train_reader`多了`np.random.shuffle(keys)`操作，作用是为了每一轮的训练，数据都是打乱的，使得每次一轮的输入数据顺序都不一样。如果读取修改了输入语音的长度，需要相应修改`mapper()`函数中的值。
```python
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
```

# 训练模型
创建`train.py`开始训练模型，搭建一个CNN分类模型，数据输入层设置为`[None, 1, 128, 128]`，这个大小就是梅尔频谱的shape，如果读者使用了其他的语音长度，也需要修改这个值。`save_path`是所有模型的保存路径，`init_model`是初始化模型的路径，`CLASS_DIM`为分类的总数，Free ST Chinese Mandarin Corpus数据集一共有855个人的语音数据，所以这里分类总数为855。
```python
# 保存模型路径
save_path = 'models/'
# 初始化模型路径
init_model = None
# 类别总数
CLASS_DIM = 855

# [可能需要修改] 梅尔频谱的shape
audio = fluid.data(name='audio', shape=[None, 1, 128, 128], dtype='float32')
label = fluid.data(name='label', shape=[None, 1], dtype='int64')

# 卷积神经网络
def cnn(input, class_dim):
    conv1 = fluid.layers.conv2d(input=input,
                                num_filters=20,
                                filter_size=5,
                                act='relu')
    conv2 = fluid.layers.conv2d(input=conv1,
                                num_filters=50,
                                filter_size=5,
                                act='relu')
    pool1 = fluid.layers.pool2d(input=conv2, pool_type='avg', global_pooling=True)
    drop = fluid.layers.dropout(x=pool1, dropout_prob=0.5)
    f1 = fluid.layers.fc(input=drop, size=128, act='relu')
    bn = fluid.layers.batch_norm(f1)
    f2 = fluid.layers.fc(input=bn, size=128, act='relu')
    f3 = fluid.layers.fc(input=f2, size=class_dim, act='softmax')
    return f3

# 获取网络模型
model = cnn(audio, CLASS_DIM)

# 获取损失函数和准确率函数
cost = fluid.layers.cross_entropy(input=model, label=label)
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input=model, label=label)
# 获取训练和测试程序
test_program = fluid.default_main_program().clone(for_test=True)

# 定义优化方法
optimizer = fluid.optimizer.AdamOptimizer(learning_rate=1e-3,
                                          regularization=fluid.regularizer.L2Decay(
                                              regularization_coeff=0.001))
opts = optimizer.minimize(avg_cost)

# 获取自定义数据
train_reader = reader.train_reader('dataset/train', batch_size=32)
test_reader = reader.test_reader('dataset/test', batch_size=32)
```

如果读者之前已经训练过，可以使用初始化模型恢复训练。通过修改`place`可以选择使用CPU训练还是GPU训练。
```python
place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

# 加载初始化模型
if init_model:
    fluid.load(program=fluid.default_main_program(),
               model_path=init_model,
               executor=exe,
               var_list=fluid.io.get_program_parameter(fluid.default_main_program()))
    print("Init model from: %s." % init_model)
```

开始执行训练，目前是训练500轮，在训练过程中是从打包的二进制文件中读取训练数据的。每训练00个batch打印一次训练日志，每一轮训练结束，执行一次测试和保存模型。在保存预测模型时，保存的是最后分类层的上一层，这样在执行预测时，就可以输出语音的特征值，通过使用这些特征值就可以实现声纹识别了。
```python
for pass_id in range(500):
    # 进行训练
    for batch_id, data in enumerate(train_reader()):
        train_cost, train_acc = exe.run(program=fluid.default_main_program(),
                                        feed={audio.name: data[0], label.name: data[1]},
                                        fetch_list=[avg_cost, acc])

        # 每100个batch打印一次信息
        if batch_id % 100 == 0:
            print('Pass:%d, Batch:%d, Cost:%0.5f, Accuracy:%0.5f' %
                  (pass_id, batch_id, train_cost[0], train_acc[0]))

    # 进行测试
    test_accs = []
    test_costs = []
    for batch_id, data in enumerate(test_reader()):
        test_cost, test_acc = exe.run(program=test_program,
                                      feed={audio.name: data[0], label.name: data[1]},
                                      fetch_list=[avg_cost, acc])
        test_accs.append(test_acc[0])
        test_costs.append(test_cost[0])
    # 求测试结果的平均值
    test_cost = (sum(test_costs) / len(test_costs))
    test_acc = (sum(test_accs) / len(test_accs))
    print('Test:%d, Cost:%0.5f, Accuracy:%0.5f' % (pass_id, test_cost, test_acc))

    # 保存参数
    if not os.path.exists(os.path.join(save_path, 'params')):
        os.makedirs(os.path.join(save_path, 'params'))
    fluid.save(program=fluid.default_main_program(),
               model_path=os.path.join(os.path.join(save_path, 'params'), "model"))
    print("Saved model to: %s" % os.path.join(save_path, 'params'))

    # 保存预测模型
    if not os.path.exists(os.path.join(save_path, 'infer')):
        os.makedirs(os.path.join(save_path, 'infer'))
    fluid.io.save_inference_model(dirname=os.path.join(save_path, 'infer'), feeded_var_names=[audio.name], target_vars=[feature], executor=exe)
    print("Saved model to: %s" % os.path.join(save_path, 'infer'))
```

# 声纹对比
下面开始实现声纹对比，创建`infer_contrast.py`程序，编写两个函数，分类是加载数据和执行预测的函数，在这个加载数据函数中裁剪数据的长度必须要跟训练时的输入长度一样。而在执行预测之后得到数据的是语音的特征值。
```python
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

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
```

有了上面两个函数，就可以做声纹识别了。我们输入两个语音，通过预测函数获取他们的特征数据，使用这个特征数据可以求他们的对角余弦值，得到的结果可以作为他们相识度。对于这个相识度的阈值，读者可以根据自己项目的准确度要求进行修改。
```python
if __name__ == '__main__':
    # 要预测的两个人的音频文件
    person1 = 'dataset/ST-CMDS-20170001_1-OS/20170001P00001A0101.wav'
    person2 = 'dataset/ST-CMDS-20170001_1-OS/20170001P00001A0001.wav'
    feature1 = infer(person1)
    feature2 = infer(person2)
    # 对角余弦值
    dist = np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
    if dist > 0.7:
        print("%s 和 %s 为同一个人，相似度为：%f" % (person1, person2, dist))
    else:
        print("%s 和 %s 不是同一个人，相似度为：%f" % (person1, person2, dist))
```


# 声纹识别
在上面的声纹对比的基础上，我们创建`infer_recognition.py`实现声纹识别。同样是使用上面声纹对比的数据加载函数和预测函数，通过这两个同样获取语音的特征数据。
```python
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

save_path = 'models/infer'
person_feature = []
person_name = []

[infer_program,
 feeded_var_names,
 target_var] = fluid.io.load_inference_model(dirname=save_path, executor=exe)

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
    feature = exe.run(program=infer_program,
                      feed={feeded_var_names[0]: data},
                      fetch_list=target_var)[0]
    return feature[0]
```

不同的是笔者增加了`load_audio_db()`和`recognition()`，第一个函数是加载语音库中的语音数据，这些音频就是相当于已经注册的用户，他们注册的语音数据会存放在这里，如果有用户需要通过声纹登录，就需要拿到用户的语音和语音库中的语音进行声纹对比，如果对比成功，那就相当于登录成功并且获取用户注册时的信息数据。完成识别的主要在`recognition()`函数中，这个函数就是将输入的语音和语音库中的语音一一对比。
```python
def load_audio_db(audio_db_path):
    audios = os.listdir(audio_db_path)
    for audio in audios:
        path = os.path.join(audio_db_path, audio)
        name = audio[:-4]
        feature = infer(path)
        person_name.append(name)
        person_feature.append(feature)

def recognition(path):
    name = ''
    pro = 0
    feature = infer(path)
    for i, person_f in enumerate(person_feature):
        dist = np.dot(feature, person_f) / (np.linalg.norm(feature) * np.linalg.norm(person_f))
        if dist > pro:
            pro = dist
            name = person_name[i]
    return name, pro
```


有了上面的声纹识别的函数，读者可以根据自己项目的需求完成声纹识别的方式，例如笔者下面提供的是通过录音来完成声纹识别。首先必须要加载语音库中的语音，语音库文件夹为`audio_db`，然后用户回车后录音3秒钟，然后程序会自动录音，并使用录音到的音频进行声纹识别，去匹配语音库中的语音，获取用户的信息。通过这样方式，读者也可以修改成通过服务请求的方式完成声纹识别，例如提供一个API供APP调用，用户在APP上通过声纹登录时，把录音到的语音发送到后端完成声纹识别，再把结果返回给APP，前提是用户已经使用语音注册，并成功把语音数据存放在`audio_db`文件夹中。
```python
if __name__ == '__main__':
    load_audio_db('audio_db')
    # 录音参数
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = 3
    WAVE_OUTPUT_FILENAME = "infer_audio.wav"

    # 打开录音
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    while True:
        try:
            i = input("按下回车键开机录音，录音3秒中：")
            print("开始录音......")
            frames = []
            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(data)

            print("录音已结束!")

            wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            # 识别对比音频库的音频
            name, p = recognition(WAVE_OUTPUT_FILENAME)
            if p > 0.7:
                print("识别说话的为：%s，相似度为：%f" % (name, p))
            else:
                print("音频库没有该用户的语音")
        except:
            pass
```
