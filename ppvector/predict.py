import os
import pickle

import numpy as np
import paddle
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from ppvector import SUPPORT_MODEL
from ppvector.data_utils.audio import AudioSegment
from ppvector.data_utils.featurizer.audio_featurizer import AudioFeaturizer
from ppvector.models.ecapa_tdnn import EcapaTdnn, SpeakerIdetification
from ppvector.utils.logger import setup_logger
from ppvector.utils.utils import dict_to_object

logger = setup_logger(__name__)


class PPVectorPredictor:
    def __init__(self,
                 configs,
                 threshold=0.6,
                 audio_db_path=None,
                 model_path='models/ecapa_tdnn_spectrogram/best_model/',
                 use_gpu=True):
        """
        语音识别预测工具
        :param configs: 配置参数
        :param threshold: 判断是否为同一个人的阈值
        :param audio_db_path: 声纹库路径
        :param model_path: 导出的预测模型文件夹路径
        :param use_gpu: 是否使用GPU预测
        """
        if use_gpu:
            assert paddle.is_compiled_with_cuda(), 'GPU不可用'
            paddle.device.set_device("gpu")
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            paddle.device.set_device("cpu")
        # 索引候选数量
        self.cdd_num = 5
        self.threshold = threshold
        self.configs = dict_to_object(configs)
        assert self.configs.use_model in SUPPORT_MODEL, f'没有该模型：{self.configs.use_model}'
        self._audio_featurizer = AudioFeaturizer(**self.configs.preprocess_conf)
        # 创建模型
        if not os.path.exists(model_path):
            raise Exception("模型文件不存在，请检查{}是否存在！".format(model_path))
        # 获取模型
        if self.configs.use_model == 'ecapa_tdnn':
            ecapa_tdnn = EcapaTdnn(input_size=self._audio_featurizer.feature_dim, **self.configs.model_conf)
            model = SpeakerIdetification(backbone=ecapa_tdnn, num_class=self.configs.dataset_conf.num_speakers)
        else:
            raise Exception(f'{self.configs.use_model} 模型不存在！')
        # 加载模型
        if os.path.isdir(model_path):
            model_path = os.path.join(model_path, 'model.pdparams')
        assert os.path.exists(model_path), f"{model_path} 模型不存在！"
        model.set_state_dict(paddle.load(model_path))
        print(f"成功加载模型参数：{model_path}")
        model.eval()
        self.predictor = model.backbone

        # 声纹库的声纹特征
        self.audio_feature = None
        # 声纹特征对应的用户名
        self.users_name = []
        # 声纹特征对应的声纹文件路径
        self.users_audio_path = []
        # 加载声纹库
        self.audio_db_path = audio_db_path
        if self.audio_db_path is not None:
            self.audio_indexes_path = os.path.join(audio_db_path, "audio_indexes.bin")
            # 加载声纹库中的声纹
            self.__load_faces(self.audio_db_path)

    # 加载声纹特征索引
    def __load_face_indexes(self):
        # 如果存在声纹特征索引文件就加载
        if not os.path.exists(self.audio_indexes_path): return
        with open(self.audio_indexes_path, "rb") as f:
            indexes = pickle.load(f)
        self.users_name = indexes["users_name"]
        self.audio_feature = indexes["faces_feature"]
        self.users_audio_path = indexes["users_image_path"]

    # 保存声纹特征索引
    def __write_index(self):
        with open(self.audio_indexes_path, "wb") as f:
            pickle.dump({"users_name": self.users_name,
                         "faces_feature": self.audio_feature,
                         "users_image_path": self.users_audio_path}, f)

    # 加载声纹库中的声纹
    def __load_faces(self, audio_db_path):
        # 先加载声纹特征索引
        self.__load_face_indexes()
        os.makedirs(audio_db_path, exist_ok=True)
        audios_path = []
        for name in os.listdir(audio_db_path):
            audio_dir = os.path.join(audio_db_path, name)
            if not os.path.isdir(audio_dir):continue
            for file in os.listdir(audio_dir):
                audios_path.append(os.path.join(audio_dir, file).replace('\\', '/'))
        # 声纹库没数据就跳过
        if len(audios_path) == 0: return
        logger.info('正在加载声纹库数据...')
        input_audios = []
        for audio_path in tqdm(audios_path):
            # 如果声纹特征已经在索引就跳过
            if audio_path in self.users_audio_path: continue
            # 读取声纹库音频
            audio_data = AudioSegment.from_file(audio_path)
            audio_feature = self._audio_featurizer.featurize(audio_data)
            # 获取用户名
            user_name = os.path.basename(os.path.dirname(audio_path))
            self.users_name.append(user_name)
            self.users_audio_path.append(audio_path)
            input_audios.append(audio_feature)
            # 处理一批数据
            if len(input_audios) == self.configs.dataset_conf.batch_size:
                features = self.predict_batch(input_audios)
                if self.audio_feature is None:
                    self.audio_feature = features
                else:
                    self.audio_feature = np.vstack((self.audio_feature, features))
                input_audios = []
        # 处理不满一批的数据
        if len(input_audios) != 0:
            features = self.predict_batch(input_audios)
            if self.audio_feature is None:
                self.audio_feature = features
            else:
                self.audio_feature = np.vstack((self.audio_feature, features))
        assert len(self.audio_feature) == len(self.users_name) == len(self.users_audio_path), '加载的数量对不上！'
        # 将声纹特征保存到索引文件中
        self.__write_index()
        logger.info('声纹库数据加载完成！')

    # 声纹检索
    def __retrieval(self, np_feature):
        labels = []
        for feature in np_feature:
            similarity = cosine_similarity(self.audio_feature, feature.reshape(1, -1)).squeeze()
            abs_similarity = np.abs(similarity)
            # 获取候选索引
            if len(abs_similarity) < self.cdd_num:
                candidate_idx = np.argpartition(abs_similarity, -len(abs_similarity))[-len(abs_similarity):]
            else:
                candidate_idx = np.argpartition(abs_similarity, -self.cdd_num)[-self.cdd_num:]
            # 过滤低于阈值的索引
            remove_idx = np.where(abs_similarity[candidate_idx] < self.threshold)
            candidate_idx = np.delete(candidate_idx, remove_idx)
            # 获取标签最多的值
            candidate_label_list = list(np.array(self.users_name)[candidate_idx])
            if len(candidate_label_list) == 0:
                max_label = None
            else:
                max_label = max(candidate_label_list, key=candidate_label_list.count)
            labels.append(max_label)
        return labels

    def predict(self,
                audio_data,
                sample_rate=16000):
        """预测一个音频的特征

        :param audio_data: 需要识别的数据，支持文件路径，字节，numpy
        :param sample_rate: 如果传入的事numpy数据，需要指定采样率
        :return: 声纹特征向量
        """
        # 加载音频文件，并进行预处理
        if isinstance(audio_data, str):
            input_data = AudioSegment.from_file(audio_data)
        elif isinstance(audio_data, np.ndarray):
            input_data = AudioSegment.from_ndarray(audio_data, sample_rate)
        elif isinstance(audio_data, bytes):
            input_data = AudioSegment.from_wave_bytes(audio_data)
        else:
            raise Exception(f'不支持该数据类型，当前数据类型为：{type(audio_data)}')
        audio_feature = self._audio_featurizer.featurize(input_data)
        input_data = paddle.to_tensor(audio_feature, dtype=paddle.float32).unsqueeze(0)
        data_length = paddle.to_tensor([input_data.shape[0]], dtype=paddle.int64)
        # 执行预测
        feature = self.predictor(input_data, data_length).numpy()[0]
        return feature

    def predict_batch(self, audios_data):
        """预测一批音频的特征

        :param audios_data: 需要预测音频的路径
        :return: 声纹特征向量
        """
        # 找出音频长度最长的
        batch = sorted(audios_data, key=lambda a: a.shape[0], reverse=True)
        freq_size = batch[0].shape[1]
        max_audio_length = batch[0].shape[0]
        batch_size = len(batch)
        # 以最大的长度创建0张量
        inputs = np.zeros((batch_size, max_audio_length, freq_size), dtype=np.float32)
        for i, sample in enumerate(batch):
            seq_length = sample.shape[0]
            # 将数据插入都0张量中，实现了padding
            inputs[i, :seq_length, :] = sample[:, :]
        audios_data = paddle.to_tensor(inputs, dtype=paddle.float32)
        data_length = paddle.to_tensor([a.shape[0] for a in audios_data], dtype=paddle.int64)
        # 执行预测
        features = self.predictor(audios_data, data_length).numpy()
        return features

    # 声纹对比
    def contrast(self, audio_data1, audio_data2):
        feature1 = self.predict(audio_data1)
        feature2 = self.predict(audio_data2)
        # 对角余弦值
        dist = np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
        return dist

    # 声纹注册
    def register(self,
                 user_name,
                 audio_data,
                 sample_rate=16000):
        # 加载音频文件，并进行预处理
        if isinstance(audio_data, str):
            input_data = AudioSegment.from_file(audio_data)
        elif isinstance(audio_data, np.ndarray):
            input_data = AudioSegment.from_ndarray(audio_data, sample_rate)
        elif isinstance(audio_data, bytes):
            input_data = AudioSegment.from_wave_bytes(audio_data)
        else:
            raise Exception(f'不支持该数据类型，当前数据类型为：{type(audio_data)}')
        feature = self.predict(audio_data=input_data.samples, sample_rate=input_data.sample_rate)
        if self.audio_feature is None:
            self.audio_feature = feature
        else:
            self.audio_feature = np.vstack((self.audio_feature, feature))
        # 保存
        if not os.path.exists(os.path.join(self.audio_db_path, user_name)):
            audio_path = os.path.join(self.audio_db_path, user_name, '0.wav')
        else:
            audio_path = os.path.join(self.audio_db_path, user_name,
                                      f'{len(os.listdir(os.path.join(self.audio_db_path, user_name)))}.wav')
        os.makedirs(os.path.dirname(audio_path), exist_ok=True)
        input_data.to_wav_file(audio_path)
        self.users_audio_path.append(audio_path.replace('\\', '/'))
        self.users_name.append(user_name)
        self.__write_index()
        return True, "注册成功"

    # 声纹识别
    def recognition(self, audio_data):
        feature = self.predict(audio_data)
        name = self.__retrieval(np_feature=[feature])[0]
        return name
