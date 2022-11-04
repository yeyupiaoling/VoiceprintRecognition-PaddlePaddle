import os
import pickle
import shutil

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
                 audio_indexes_path=None,
                 model_path='models/ecapa_tdnn_spectrogram/best_model/',
                 use_gpu=True):
        """
        语音识别预测工具
        :param configs: 配置参数
        :param model_path: 导出的预测模型文件夹路径
        :param use_gpu: 是否使用GPU预测
        """
        # 索引候选数量
        self.cdd_num = 5
        self.threshold = threshold
        self.configs = dict_to_object(configs)
        assert self.configs.use_model in SUPPORT_MODEL, f'没有该模型：{self.configs.use_model}'
        self.use_gpu = use_gpu
        self._audio_featurizer = AudioFeaturizer(**self.configs.preprocess_conf)
        # 创建模型
        if not os.path.exists(model_path):
            raise Exception("模型文件不存在，请检查{}是否存在！".format(model_path))

        # 获取模型
        audio_featurizer = AudioFeaturizer(**self.configs.preprocess_conf)
        # 获取模型
        if self.configs.use_model == 'ecapa_tdnn':
            ecapa_tdnn = EcapaTdnn(input_size=audio_featurizer.feature_dim)
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
        self.audio_indexes_path = os.path.join(audio_db_path, "audio_indexes.bin")
        if self.audio_db_path is not None:
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
        for root, dirs, files in os.walk(audio_db_path):
            for file in files:
                if file == "audio_indexes.bin": continue
                audios_path.append(os.path.join(root, file).replace('\\', '/'))
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
            candidate_idx = np.argpartition(abs_similarity, -self.cdd_num)[-self.cdd_num:]
            # 过滤低于阈值的索引
            remove_idx = np.where(abs_similarity[candidate_idx] < self.threshold)
            candidate_idx = np.delete(candidate_idx, remove_idx)
            # 获取标签最多的值
            candidate_label_list = list(np.array(self.users_name)[candidate_idx])
            if len(candidate_label_list) == 0:
                max_label = "unknown"
            else:
                max_label = max(candidate_label_list, key=candidate_label_list.count)
            labels.append(max_label)
        return labels

    # 预测一个音频的特征
    def predict(self,
                audio_path=None,
                audio_bytes=None,
                audio_ndarray=None,
                sample_rate=16000):
        """
        预测函数，只预测完整的一句话。
        :param audio_path: 需要预测音频的路径
        :param audio_bytes: 需要预测的音频wave读取的字节流
        :param audio_ndarray: 需要预测的音频未预处理的numpy值
        :param sample_rate: 如果传入的事numpy数据，需要指定采样率
        :return: 声纹特征向量
        """
        assert audio_path is not None or audio_bytes is not None or audio_ndarray is not None, \
            'audio_path，audio_bytes和audio_ndarray至少有一个不为None！'
        # 加载音频文件，并进行预处理
        if audio_path is not None:
            audio_data = AudioSegment.from_file(audio_path)
        elif audio_ndarray is not None:
            audio_data = AudioSegment.from_ndarray(audio_ndarray, sample_rate)
        else:
            audio_data = AudioSegment.from_bytes(audio_bytes)
        audio_feature = self._audio_featurizer.featurize(audio_data)
        audio_data = paddle.to_tensor(audio_feature, dtype=paddle.float32).unsqueeze(0)
        data_length = paddle.to_tensor([audio_data.shape[1]], dtype=paddle.int64)
        # 执行预测
        feature = self.predictor(audio_data, data_length).numpy()[0]
        return feature

    # 预测一批音频的特征
    def predict_batch(self, audios_data):
        """
        预测函数，只预测完整的一句话。
        :param audios_data: 需要预测音频的路径
        :return: 声纹特征向量
        """
        audios_data = paddle.to_tensor(audios_data, dtype=paddle.float32)
        data_length = paddle.to_tensor([a.shape[1] for a in audios_data], dtype=paddle.int64)
        # 执行预测
        features = self.predictor(audios_data, data_length).numpy()
        return features

    # 声纹对比
    def contrast(self, audio_path1, audio_path2):
        feature1 = self.predict(audio_path1)
        feature2 = self.predict(audio_path2)
        # 对角余弦值
        dist = np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
        return dist

    # 声纹注册
    def register(self, audio_path, user_name):
        audio_data = AudioSegment.from_file(audio_path)
        feature = self.predict(audio_ndarray=audio_data, sample_rate=audio_data.sample_rate)
        if self.audio_feature is None:
            self.audio_feature = feature
        else:
            self.audio_feature = np.vstack((self.audio_feature, feature))
        # 保存
        audio_path = os.path.join(self.audio_db_path, user_name,
                                  f'{len(os.listdir(os.path.join(self.audio_db_path, user_name)))}.wav')
        audio_data.to_wav_file(audio_path)
        self.users_audio_path.append(audio_path.replace('\\', '/'))
        self.users_name.append(user_name)
        self.__write_index()
        return True, "注册成功"

    # 声纹识别
    def recognition(self, audio_path):
        feature = self.predict(audio_path)
        users = self.__retrieval(np_feature=[feature])
        return users
