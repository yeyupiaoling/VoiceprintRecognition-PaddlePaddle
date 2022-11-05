import io
import json
import os
import shutil
import time
from datetime import timedelta

import numpy as np
import paddle
from paddle.distributed import fleet
from paddle.io import DataLoader
from paddle.metric import accuracy
from tqdm import tqdm
from visualdl import LogWriter

from ppvector import SUPPORT_MODEL
from ppvector.data_utils.collate_fn import collate_fn
from ppvector.data_utils.featurizer.audio_featurizer import AudioFeaturizer
from ppvector.data_utils.reader import CustomDataset

from ppvector.models.ecapa_tdnn import EcapaTdnn, SpeakerIdetification
from ppvector.models.loss import AAMLoss
from ppvector.utils.logger import setup_logger
from ppvector.utils.scheduler import WarmupLR
from ppvector.utils.utils import dict_to_object, cal_accuracy_threshold

logger = setup_logger(__name__)


class PPVectorTrainer(object):
    def __init__(self, configs, use_gpu=True):
        """ ppvector集成工具类

        :param configs: 配置字典
        :param use_gpu: 是否使用GPU训练模型
        """
        if use_gpu:
            assert paddle.is_compiled_with_cuda(), 'GPU不可用'
            paddle.device.set_device("gpu")
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            paddle.device.set_device("cpu")
        self.use_gpu = use_gpu
        self.configs = dict_to_object(configs)
        assert self.configs.use_model in SUPPORT_MODEL, f'没有该模型：{self.configs.use_model}'
        self.model = None
        self.test_loader = None

    def __setup_dataloader(self, augment_conf_path=None, is_train=False):
        # 获取训练数据
        if augment_conf_path is not None and os.path.exists(augment_conf_path) and is_train:
            augmentation_config = io.open(augment_conf_path, mode='r', encoding='utf8').read()
        else:
            if augment_conf_path is not None and not os.path.exists(augment_conf_path):
                logger.info('数据增强配置文件{}不存在'.format(augment_conf_path))
            augmentation_config = '{}'
        if is_train:
            self.train_dataset = CustomDataset(preprocess_configs=self.configs.preprocess_conf,
                                               data_list_path=self.configs.dataset_conf.train_list,
                                               do_vad=self.configs.dataset_conf.chunk_duration,
                                               chunk_duration=self.configs.dataset_conf.chunk_duration,
                                               min_duration=self.configs.dataset_conf.min_duration,
                                               augmentation_config=augmentation_config,
                                               mode='train')
            # 设置支持多卡训练
            self.train_batch_sampler = paddle.io.DistributedBatchSampler(dataset=self.train_dataset,
                                                                         batch_size=self.configs.dataset_conf.batch_size,
                                                                         shuffle=True)
            self.train_loader = DataLoader(dataset=self.train_dataset,
                                           collate_fn=collate_fn,
                                           batch_sampler=self.train_batch_sampler,
                                           num_workers=self.configs.dataset_conf.num_workers)
        # 获取测试数据
        self.test_dataset = CustomDataset(preprocess_configs=self.configs.preprocess_conf,
                                          data_list_path=self.configs.dataset_conf.test_list,
                                          do_vad=self.configs.dataset_conf.chunk_duration,
                                          chunk_duration=self.configs.dataset_conf.chunk_duration,
                                          min_duration=self.configs.dataset_conf.min_duration,
                                          mode='train')
        self.test_loader = DataLoader(dataset=self.test_dataset,
                                      batch_size=self.configs.dataset_conf.batch_size,
                                      collate_fn=collate_fn,
                                      num_workers=self.configs.dataset_conf.num_workers)

    def __setup_model(self, input_size, is_train=False):
        # 获取模型
        if self.configs.use_model == 'ecapa_tdnn':
            self.ecapa_tdnn = EcapaTdnn(input_size=input_size, **self.configs.model_conf)
            self.model = SpeakerIdetification(backbone=self.ecapa_tdnn, num_class=self.configs.dataset_conf.num_speakers)
        else:
            raise Exception(f'{self.configs.use_model} 模型不存在！')
        # print(self.model)
        # 获取损失函数
        self.loss = AAMLoss()
        if is_train:
            # 设置优化方法
            self.scheduler = WarmupLR(warmup_steps=self.configs.optimizer_conf.warmup_steps,
                                      learning_rate=float(self.configs.optimizer_conf.learning_rate))
            self.optimizer = paddle.optimizer.Adam(parameters=self.model.parameters(),
                                                   learning_rate=self.scheduler,
                                                   weight_decay=float(self.configs.optimizer_conf.weight_decay))

    def __load_pretrained(self, pretrained_model):
        # 加载预训练模型
        if pretrained_model is not None:
            if os.path.isdir(pretrained_model):
                pretrained_model = os.path.join(pretrained_model, 'model.pdparams')
            assert os.path.exists(pretrained_model), f"{pretrained_model} 模型不存在！"
            model_dict = self.model.state_dict()
            model_state_dict = paddle.load(pretrained_model)
            # 过滤不存在的参数
            for name, weight in model_dict.items():
                if name in model_state_dict.keys():
                    if list(weight.shape) != list(model_state_dict[name].shape):
                        logger.warning('{} not used, shape {} unmatched with {} in model.'.
                                       format(name, list(model_state_dict[name].shape), list(weight.shape)))
                        model_state_dict.pop(name, None)
                else:
                    logger.warning('Lack weight: {}'.format(name))
            self.model.set_state_dict(model_state_dict)
            logger.info('成功加载预训练模型：{}'.format(pretrained_model))

    def __load_checkpoint(self, save_model_path, resume_model):
        last_epoch = -1
        best_loss = 1e4
        last_model_dir = os.path.join(save_model_path,
                                      f'{self.configs.use_model}_{self.configs.preprocess_conf.feature_method}',
                                      'last_model')
        if resume_model is not None or (os.path.exists(os.path.join(last_model_dir, 'model.pdparams'))
                                        and os.path.exists(os.path.join(last_model_dir, 'optimizer.pdopt'))):
            # 自动获取最新保存的模型
            if resume_model is None: resume_model = last_model_dir
            assert os.path.exists(os.path.join(resume_model, 'model.pdparams')), "模型参数文件不存在！"
            assert os.path.exists(os.path.join(resume_model, 'optimizer.pdopt')), "优化方法参数文件不存在！"
            self.model.set_state_dict(paddle.load(os.path.join(resume_model, 'model.pdparams')))
            self.optimizer.set_state_dict(paddle.load(os.path.join(resume_model, 'optimizer.pdopt')))
            with open(os.path.join(resume_model, 'model.state'), 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                last_epoch = json_data['last_epoch'] - 1
                best_loss = json_data['best_loss']
            logger.info('成功恢复模型参数和优化方法参数：{}'.format(resume_model))
        return last_epoch, best_loss

    # 保存模型
    def __save_checkpoint(self, save_model_path, epoch_id, best_loss=1e4, best_model=False):
        if best_model:
            model_path = os.path.join(save_model_path,
                                      f'{self.configs.use_model}_{self.configs.preprocess_conf.feature_method}',
                                      'best_model')
        else:
            model_path = os.path.join(save_model_path,
                                      f'{self.configs.use_model}_{self.configs.preprocess_conf.feature_method}',
                                      'epoch_{}'.format(epoch_id))
        os.makedirs(model_path, exist_ok=True)
        try:
            paddle.save(self.optimizer.state_dict(), os.path.join(model_path, 'optimizer.pdopt'))
            paddle.save(self.model.state_dict(), os.path.join(model_path, 'model.pdparams'))
        except Exception as e:
            logger.error(f'保存模型时出现错误，错误信息：{e}')
            return
        with open(os.path.join(model_path, 'model.state'), 'w', encoding='utf-8') as f:
            f.write('{"last_epoch": %d, "best_loss": %f}' % (epoch_id, best_loss))
        if not best_model:
            last_model_path = os.path.join(save_model_path,
                                           f'{self.configs.use_model}_{self.configs.preprocess_conf.feature_method}',
                                           'last_model')
            shutil.rmtree(last_model_path, ignore_errors=True)
            shutil.copytree(model_path, last_model_path)
            # 删除旧的模型
            old_model_path = os.path.join(save_model_path,
                                          f'{self.configs.use_model}_{self.configs.preprocess_conf.feature_method}',
                                          'epoch_{}'.format(epoch_id - 3))
            if os.path.exists(old_model_path):
                shutil.rmtree(old_model_path)
        logger.info('已保存模型：{}'.format(model_path))

    def __train_epoch(self, epoch_id, save_model_path, local_rank, writer):
        train_times, accuracies, loss_sum = [], [], []
        start = time.time()
        sum_batch = len(self.train_loader) * self.configs.train_conf.max_epoch
        for batch_id, (audio, label, audio_lens) in enumerate(self.train_loader()):
            output = self.model(audio, audio_lens)
            # 计算损失值
            los = self.loss(output, label)
            los.backward()
            self.optimizer.step()
            self.optimizer.clear_grad()
            # 计算准确率
            label = paddle.reshape(label, shape=(-1, 1))
            acc = accuracy(input=paddle.nn.functional.softmax(output), label=label)
            accuracies.append(acc.numpy()[0])
            loss_sum.append(los.numpy()[0])
            train_times.append((time.time() - start) * 1000)

            # 多卡训练只使用一个进程打印
            if batch_id % self.configs.train_conf.log_interval == 0 and local_rank == 0:
                # 计算每秒训练数据量
                train_speed = self.configs.dataset_conf.batch_size / (sum(train_times) / len(train_times) / 1000)
                # 计算剩余时间
                eta_sec = (sum(train_times) / len(train_times)) * (
                        sum_batch - (epoch_id - 1) * len(self.train_loader) - batch_id)
                eta_str = str(timedelta(seconds=int(eta_sec / 1000)))
                logger.info(f'Train epoch: [{epoch_id}/{self.configs.train_conf.max_epoch}], '
                            f'batch: [{batch_id}/{len(self.train_loader)}], '
                            f'loss: {sum(loss_sum) / len(loss_sum):.5f}, '
                            f'accuracy: {sum(accuracies) / len(accuracies):.5f}, '
                            f'learning rate: {self.scheduler.get_lr():>.8f}, '
                            f'speed: {train_speed:.2f} data/sec, eta: {eta_str}')
                writer.add_scalar('Train/Loss', sum(loss_sum) / len(loss_sum), self.train_step)
                writer.add_scalar('Train/Accuracy', (sum(accuracies) / len(accuracies)), self.train_step)
                train_times = []
                self.scheduler.step()
            # 固定步数也要保存一次模型
            if batch_id % 10000 == 0 and batch_id != 0 and local_rank == 0:
                self.__save_checkpoint(save_model_path=save_model_path, epoch_id=epoch_id)
            start = time.time()

    def train(self,
              save_model_path='models/',
              resume_model=None,
              pretrained_model=None,
              augment_conf_path='configs/augmentation.json'):
        """
        训练模型
        :param save_model_path: 模型保存的路径
        :param resume_model: 恢复训练，当为None则不使用预训练模型
        :param pretrained_model: 预训练模型的路径，当为None则不使用预训练模型
        :param augment_conf_path: 数据增强的配置文件，为json格式
        """
        paddle.seed(1000)
        # 获取有多少张显卡训练
        nranks = paddle.distributed.get_world_size()
        local_rank = paddle.distributed.get_rank()
        writer = None
        if local_rank == 0:
            # 日志记录器
            writer = LogWriter(logdir='log')

        if nranks > 1 and self.use_gpu:
            # 初始化Fleet环境
            strategy = fleet.DistributedStrategy()
            fleet.init(is_collective=True, strategy=strategy)

        # 获取数据
        self.__setup_dataloader(augment_conf_path=augment_conf_path, is_train=True)
        # 获取模型
        self.__setup_model(input_size=self.test_dataset.feature_dim, is_train=True)

        # 支持多卡训练
        if nranks > 1 and self.use_gpu:
            self.optimizer = fleet.distributed_optimizer(self.optimizer)
            self.model = fleet.distributed_model(self.model)
        logger.info('训练数据：{}'.format(len(self.train_dataset)))

        self.__load_pretrained(pretrained_model=pretrained_model)
        # 加载恢复模型
        last_epoch, best_loss = self.__load_checkpoint(save_model_path=save_model_path, resume_model=resume_model)

        test_step, self.train_step = 0, 0
        last_epoch += 1
        self.train_batch_sampler.epoch = last_epoch
        if local_rank == 0:
            writer.add_scalar('Train/lr', self.scheduler.get_lr(), last_epoch)
        # 开始训练
        for epoch_id in range(last_epoch, self.configs.train_conf.max_epoch):
            epoch_id += 1
            start_epoch = time.time()
            # 训练一个epoch
            self.__train_epoch(epoch_id=epoch_id, save_model_path=save_model_path, local_rank=local_rank,
                               writer=writer)
            # 多卡训练只使用一个进程执行评估和保存模型
            if local_rank == 0:
                logger.info('=' * 70)
                loss, acc = self.evaluate(resume_model=None)
                logger.info('Test epoch: {}, time/epoch: {}, loss: {:.5f}, accuracy: {:.5f}'.format(
                    epoch_id, str(timedelta(seconds=(time.time() - start_epoch))), loss, acc))
                logger.info('=' * 70)
                writer.add_scalar('Test/Accuracy', acc, test_step)
                writer.add_scalar('Test/Loss', loss, test_step)
                test_step += 1
                self.model.train()
                # 记录学习率
                writer.add_scalar('Train/lr', self.scheduler.last_lr, epoch_id)
                # # 保存最优模型
                if loss <= best_loss:
                    best_loss = loss
                    self.__save_checkpoint(save_model_path=save_model_path, epoch_id=epoch_id, best_loss=loss,
                                           best_model=True)
                # 保存模型
                self.__save_checkpoint(save_model_path=save_model_path, epoch_id=epoch_id, best_loss=loss)

    def evaluate(self, resume_model='models/ecapa_tdnn_spectrogram/best_model/', cal_threshold=False):
        """
        评估模型
        :param resume_model: 所使用的模型
        :param display_result: 是否打印识别结果
        :return: 评估结果
        """
        if self.test_loader is None:
            self.__setup_dataloader()
        if self.model is None:
            self.__setup_model(input_size=self.test_dataset.feature_dim)
        if resume_model is not None:
            if os.path.isdir(resume_model):
                resume_model = os.path.join(resume_model, 'model.pdparams')
            assert os.path.exists(resume_model), f"{resume_model} 模型不存在！"
            model_state_dict = paddle.load(resume_model)
            self.model.set_state_dict(model_state_dict)
            logger.info(f'成功加载模型：{resume_model}')
        self.model.eval()
        if isinstance(self.model, paddle.DataParallel):
            eval_model = self.model._layers
        else:
            eval_model = self.model

        accuracies, losses = [], []
        features, labels = None, None
        with paddle.no_grad():
            for batch_id, (audio, label, audio_lens) in enumerate(tqdm(self.test_loader())):
                output = eval_model(audio, audio_lens)
                feature = eval_model.backbone(audio).numpy()
                los = self.loss(output, label)
                # 计算准确率
                label = paddle.reshape(label, shape=(-1, 1))
                acc = accuracy(input=paddle.nn.functional.softmax(output), label=label)
                features = np.concatenate((features, feature)) if features is not None else feature
                labels = np.concatenate((labels, label)) if labels is not None else label
                accuracies.append(acc.numpy()[0])
                losses.append(los.numpy()[0])
        loss = float(sum(losses) / len(losses))
        acc = float(sum(accuracies) / len(accuracies))
        self.model.train()
        if cal_threshold:
            scores, y_true = [], []
            labels = labels.astype(np.int32)
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
            best_acc, threshold = cal_accuracy_threshold(scores, y_true)
            print(f'当阈值为{threshold:.2f}, 两两对比准确率最大，准确率为：{best_acc:.5f}')
        return loss, acc


    def export(self, save_model_path='models/', resume_model='models/ecapa_tdnn_spectrogram/best_model/'):
        """
        导出预测模型
        :param save_model_path: 模型保存的路径
        :param resume_model: 准备转换的模型路径
        :return:
        """
        # 获取模型
        audio_featurizer = AudioFeaturizer(**self.configs.preprocess_conf)
        self.__setup_model(input_size=audio_featurizer.feature_dim)
        # 加载预训练模型
        if os.path.isdir(resume_model):
            resume_model = os.path.join(resume_model, 'model.pdparams')
        assert os.path.exists(resume_model), f"{resume_model} 模型不存在！"
        model_state_dict = paddle.load(resume_model)
        self.model.set_state_dict(model_state_dict)
        logger.info('成功恢复模型参数和优化方法参数：{}'.format(resume_model))
        self.model.eval()
        # 获取静态模型
        infer_model = self.model.export()
        infer_model_dir = os.path.join(save_model_path,
                                       f'{self.configs.use_model}_{self.configs.preprocess_conf.feature_method}',
                                       'infer')
        os.makedirs(infer_model_dir, exist_ok=True)
        infer_model_path = os.path.join(infer_model_dir, 'model')
        paddle.jit.save(infer_model, infer_model_path)
        logger.info("预测模型已保存：{}".format(infer_model_path))
