import json
import os
import platform
import shutil
import time
from datetime import timedelta

import numpy as np
import paddle
import paddle.nn as nn
import yaml
from paddle import summary
from paddle.distributed import fleet
from paddle.io import DataLoader, DistributedBatchSampler
from paddle.metric import accuracy
from paddle.optimizer.lr import CosineAnnealingDecay
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from visualdl import LogWriter

from ppvector import SUPPORT_MODEL, __version__
from ppvector.data_utils.collate_fn import collate_fn
from ppvector.data_utils.featurizer import AudioFeaturizer
from ppvector.data_utils.reader import PPVectorDataset
from ppvector.data_utils.spec_aug import SpecAug
from ppvector.metric.metrics import compute_fnr_fpr, compute_eer, compute_dcf
from ppvector.models.campplus import CAMPPlus
from ppvector.models.ecapa_tdnn import EcapaTdnn
from ppvector.models.eres2net import ERes2Net, ERes2NetV2
from ppvector.models.fc import SpeakerIdentification
from ppvector.models.loss import AAMLoss, AMLoss, ARMLoss, CELoss, SubCenterLoss, SphereFace2
from ppvector.models.res2net import Res2Net
from ppvector.models.resnet_se import ResNetSE
from ppvector.models.tdnn import TDNN
from ppvector.utils.logger import setup_logger
from ppvector.utils.scheduler import cosine_decay_with_warmup, MarginScheduler
from ppvector.utils.utils import dict_to_object, print_arguments

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
        # 读取配置文件
        if isinstance(configs, str):
            with open(configs, 'r', encoding='utf-8') as f:
                configs = yaml.load(f.read(), Loader=yaml.FullLoader)
            print_arguments(configs=configs)
        self.configs = dict_to_object(configs)
        assert self.configs.use_model in SUPPORT_MODEL, f'没有该模型：{self.configs.use_model}'
        self.model = None
        self.backbone = None
        self.audio_featurizer = None
        self.train_dataset = None
        self.train_loader = None
        self.enroll_dataset = None
        self.enroll_loader = None
        self.trials_dataset = None
        self.trials_loader = None
        self.margin_scheduler = None
        self.amp_scaler = None
        self.spec_aug = SpecAug(**self.configs.dataset_conf.get('spec_aug_args', {}))
        if platform.system().lower() == 'windows':
            self.configs.dataset_conf.dataLoader.num_workers = 0
            logger.warning('Windows系统不支持多线程读取数据，已自动关闭！')
        self.max_step, self.train_step = None, None
        self.train_loss, self.train_acc = None, None
        self.train_eta_sec = None
        self.eval_eer, self.eval_min_dcf, self.eval_threshold = None, None, None
        self.test_log_step, self.train_log_step = 0, 0
        self.stop_train, self.stop_eval = False, False

    def __setup_dataloader(self, is_train=False):
        # 获取特征器
        self.audio_featurizer = AudioFeaturizer(feature_method=self.configs.preprocess_conf.feature_method,
                                                method_args=self.configs.preprocess_conf.get('method_args', {}))
        if is_train:
            self.train_dataset = PPVectorDataset(data_list_path=self.configs.dataset_conf.train_list,
                                                 audio_featurizer=self.audio_featurizer,
                                                 do_vad=self.configs.dataset_conf.do_vad,
                                                 max_duration=self.configs.dataset_conf.max_duration,
                                                 min_duration=self.configs.dataset_conf.min_duration,
                                                 sample_rate=self.configs.dataset_conf.sample_rate,
                                                 aug_conf=self.configs.dataset_conf.aug_conf,
                                                 num_speakers=self.configs.model_conf.classifier.num_speakers,
                                                 use_dB_normalization=self.configs.dataset_conf.use_dB_normalization,
                                                 target_dB=self.configs.dataset_conf.target_dB,
                                                 mode='train')
            # 设置支持多卡训练
            train_sampler = None
            if paddle.distributed.get_world_size() > 1:
                # 设置支持多卡训练
                train_sampler = DistributedBatchSampler(dataset=self.train_dataset,
                                                        batch_size=self.configs.dataset_conf.dataLoader.batch_size,
                                                        shuffle=True)
            self.train_loader = DataLoader(dataset=self.train_dataset,
                                           collate_fn=collate_fn,
                                           shuffle=(train_sampler is None),
                                           batch_sampler=train_sampler,
                                           **self.configs.dataset_conf.dataLoader)
        # 获取评估的注册数据和检验数据
        self.enroll_dataset = PPVectorDataset(data_list_path=self.configs.dataset_conf.enroll_list,
                                              audio_featurizer=self.audio_featurizer,
                                              do_vad=self.configs.dataset_conf.do_vad,
                                              max_duration=self.configs.dataset_conf.eval_conf.max_duration,
                                              min_duration=self.configs.dataset_conf.min_duration,
                                              sample_rate=self.configs.dataset_conf.sample_rate,
                                              use_dB_normalization=self.configs.dataset_conf.use_dB_normalization,
                                              target_dB=self.configs.dataset_conf.target_dB,
                                              mode='eval')
        self.enroll_loader = DataLoader(dataset=self.enroll_dataset,
                                        collate_fn=collate_fn,
                                        batch_size=self.configs.dataset_conf.eval_conf.batch_size,
                                        num_workers=self.configs.dataset_conf.dataLoader.num_workers)
        self.trials_dataset = PPVectorDataset(data_list_path=self.configs.dataset_conf.trials_list,
                                              audio_featurizer=self.audio_featurizer,
                                              do_vad=self.configs.dataset_conf.do_vad,
                                              max_duration=self.configs.dataset_conf.eval_conf.max_duration,
                                              min_duration=self.configs.dataset_conf.min_duration,
                                              sample_rate=self.configs.dataset_conf.sample_rate,
                                              use_dB_normalization=self.configs.dataset_conf.use_dB_normalization,
                                              target_dB=self.configs.dataset_conf.target_dB,
                                              mode='eval')
        self.trials_loader = DataLoader(dataset=self.trials_dataset,
                                        collate_fn=collate_fn,
                                        batch_size=self.configs.dataset_conf.eval_conf.batch_size,
                                        num_workers=self.configs.dataset_conf.dataLoader.num_workers)

    # 提取特征保存文件
    def extract_features(self, save_dir='dataset/features'):
        self.audio_featurizer = AudioFeaturizer(feature_method=self.configs.preprocess_conf.feature_method,
                                                method_args=self.configs.preprocess_conf.get('method_args', {}))
        for i, data_list in enumerate([self.configs.dataset_conf.train_list,
                                       self.configs.dataset_conf.enroll_list,
                                       self.configs.dataset_conf.trials_list]):
            # 获取测试数据
            test_dataset = PPVectorDataset(data_list_path=data_list,
                                           audio_featurizer=self.audio_featurizer,
                                           do_vad=self.configs.dataset_conf.do_vad,
                                           sample_rate=self.configs.dataset_conf.sample_rate,
                                           use_dB_normalization=self.configs.dataset_conf.use_dB_normalization,
                                           target_dB=self.configs.dataset_conf.target_dB,
                                           mode='extract_feature')
            save_data_list = data_list.replace('.txt', '_features.txt')
            with open(save_data_list, 'w', encoding='utf-8') as f:
                for i in tqdm(range(len(test_dataset))):
                    feature, label = test_dataset[i]
                    feature = feature.numpy()
                    label = int(label)
                    save_path = os.path.join(save_dir, str(label), f'{int(time.time() * 1000)}.npy').replace('\\', '/')
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    np.save(save_path, feature)
                    f.write(f'{save_path}\t{label}\n')
            logger.info(f'{data_list}列表中的数据已提取特征完成，新列表为：{save_data_list}')

    def __setup_model(self, input_size, is_train=False):
        # 获取模型
        if self.configs.use_model == 'ERes2Net':
            self.backbone = ERes2Net(input_size=input_size, **self.configs.model_conf.backbone)
        elif self.configs.use_model == 'ERes2NetV2':
            self.backbone = ERes2NetV2(input_size=input_size, **self.configs.model_conf.backbone)
        elif self.configs.use_model == 'CAMPPlus':
            self.backbone = CAMPPlus(input_size=input_size, **self.configs.model_conf.backbone)
        elif self.configs.use_model == 'EcapaTdnn':
            self.backbone = EcapaTdnn(input_size=input_size, **self.configs.model_conf.backbone)
        elif self.configs.use_model == 'Res2Net':
            self.backbone = Res2Net(input_size=input_size, **self.configs.model_conf.backbone)
        elif self.configs.use_model == 'ResNetSE':
            self.backbone = ResNetSE(input_size=input_size, **self.configs.model_conf.backbone)
        elif self.configs.use_model == 'TDNN':
            self.backbone = TDNN(input_size=input_size, **self.configs.model_conf.backbone)
        else:
            raise Exception(f'{self.configs.use_model} 模型不存在！')

        # 获取训练所需的函数
        if is_train:
            if self.configs.train_conf.enable_amp:
                # 自动混合精度训练，逻辑2，定义GradScaler
                self.amp_scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
            use_loss = self.configs.loss_conf.get('use_loss', 'AAMLoss')
            # 获取分类器
            num_class = self.configs.model_conf.classifier.num_speakers
            # 语速扰动要增加分类数量
            self.configs.model_conf.classifier.num_speakers = num_class * 3 \
                if self.configs.dataset_conf.aug_conf.speed_perturb else num_class
            classifier = SpeakerIdentification(input_dim=self.backbone.embd_dim,
                                               loss_type=use_loss,
                                               **self.configs.model_conf.classifier)
            # 合并模型
            self.model = nn.Sequential(self.backbone, classifier)

            # 获取损失函数
            loss_args = self.configs.loss_conf.get('args', {})
            loss_args = loss_args if loss_args is not None else {}
            if use_loss == 'AAMLoss':
                self.loss = AAMLoss(**loss_args)
            elif use_loss == 'SphereFace2':
                self.loss = SphereFace2(**loss_args)
            elif use_loss == 'SubCenterLoss':
                self.loss = SubCenterLoss(**loss_args)
            elif use_loss == 'AMLoss':
                self.loss = AMLoss(**loss_args)
            elif use_loss == 'ARMLoss':
                self.loss = ARMLoss(**loss_args)
            elif use_loss == 'CELoss':
                self.loss = CELoss(**loss_args)
            else:
                raise Exception(f'没有{use_loss}损失函数！')
            # 损失函数margin调度器
            if self.configs.loss_conf.get('use_margin_scheduler', False):
                margin_scheduler_args = dict(increase_start_epoch=int(self.configs.train_conf.max_epoch * 0.3),
                                             fix_epoch=int(self.configs.train_conf.max_epoch * 0.7),
                                             initial_margin=0.0,
                                             final_margin=0.3)
                if self.configs.loss_conf.margin_scheduler_args:
                    for k, v in self.configs.loss_conf.margin_scheduler_args.items():
                        margin_scheduler_args[k] = v
                self.margin_scheduler = MarginScheduler(criterion=self.loss,
                                                        step_per_epoch=len(self.train_loader),
                                                        **margin_scheduler_args)
            # 学习率衰减函数
            scheduler_args = self.configs.optimizer_conf.get('scheduler_args', {}) \
                if self.configs.optimizer_conf.get('scheduler_args', {}) is not None else {}
            if self.configs.optimizer_conf.scheduler == 'CosineAnnealingLR':
                max_step = int(self.configs.train_conf.max_epoch * 1.2) * len(self.train_loader)
                self.scheduler = CosineAnnealingDecay(T_max=max_step,
                                                      **scheduler_args)
            elif self.configs.optimizer_conf.scheduler == 'WarmupCosineSchedulerLR':
                self.scheduler = cosine_decay_with_warmup(step_per_epoch=len(self.train_loader),
                                                          **scheduler_args)
            else:
                raise Exception(f'不支持学习率衰减函数：{self.configs.optimizer_conf.scheduler}')
            # 获取优化方法
            optimizer = self.configs.optimizer_conf.optimizer
            if optimizer == 'Adam':
                self.optimizer = paddle.optimizer.Adam(parameters=self.model.parameters(),
                                                       learning_rate=self.scheduler,
                                                       weight_decay=self.configs.optimizer_conf.weight_decay)
            elif optimizer == 'AdamW':
                self.optimizer = paddle.optimizer.AdamW(parameters=self.model.parameters(),
                                                        learning_rate=self.scheduler,
                                                        weight_decay=self.configs.optimizer_conf.weight_decay)
            elif optimizer == 'Momentum':
                self.optimizer = paddle.optimizer.Momentum(parameters=self.model.parameters(),
                                                           momentum=self.configs.optimizer_conf.get('momentum', 0.9),
                                                           learning_rate=self.scheduler,
                                                           weight_decay=self.configs.optimizer_conf.weight_decay)
            else:
                raise Exception(f'不支持优化方法：{optimizer}')
        else:
            # 不训练模型不包含分类器
            self.model = nn.Sequential(self.backbone)
        summary(self.model, (1, 98, input_size))

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
        best_eer = 1
        last_model_dir = os.path.join(save_model_path,
                                      f'{self.configs.use_model}_{self.configs.preprocess_conf.feature_method}',
                                      'last_model')
        if resume_model is not None or (os.path.exists(os.path.join(last_model_dir, 'model.pdparams'))
                                        and os.path.exists(os.path.join(last_model_dir, 'optimizer.pdopt'))):
            # 自动获取最新保存的模型
            if resume_model is None: resume_model = last_model_dir
            assert os.path.exists(os.path.join(resume_model, 'model.pdparams')), "模型参数文件不存在！"
            assert os.path.exists(os.path.join(resume_model, 'optimizer.pdopt')), "优化方法参数文件不存在！"
            missing_keys, unexpected_keys = self.model.set_state_dict(
                paddle.load(os.path.join(resume_model, 'model.pdparams')))
            if len(missing_keys) != 0 or len(unexpected_keys) != 0:
                logger.warning(f'模型加载部分失败，请检查模型是否匹配，'
                               f'missing_keys: {missing_keys}, unexpected_keys: {unexpected_keys}')
            self.optimizer.set_state_dict(paddle.load(os.path.join(resume_model, 'optimizer.pdopt')))
            # 自动混合精度参数
            if self.amp_scaler is not None and os.path.exists(os.path.join(resume_model, 'scaler.pdparams')):
                self.amp_scaler.load_state_dict(paddle.load(os.path.join(resume_model, 'scaler.pdparams')))
            with open(os.path.join(resume_model, 'model.state'), 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                last_epoch = json_data['last_epoch'] - 1
                if 'eer' in json_data.keys():
                    best_eer = json_data['eer']
            if last_epoch >= 0:
                # 恢复学习率和margin
                self.scheduler.step((last_epoch + 1) * len(self.train_loader))
                if self.margin_scheduler:
                    self.margin_scheduler.step((last_epoch + 1) * len(self.train_loader))
            logger.info('成功恢复模型参数和优化方法参数：{}'.format(resume_model))
        return last_epoch, best_eer

    # 保存模型
    def __save_checkpoint(self, save_model_path, epoch_id, eer=None, min_dcf=None, threshold=None, best_model=False):
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
            # 自动混合精度参数
            if self.amp_scaler is not None:
                paddle.save(self.amp_scaler.state_dict(), os.path.join(model_path, 'scaler.pdparams'))
        except Exception as e:
            logger.error(f'保存模型时出现错误，错误信息：{e}')
            return
        with open(os.path.join(model_path, 'model.state'), 'w', encoding='utf-8') as f:
            data = {"last_epoch": epoch_id, "version": __version__}
            if eer is not None:
                data['threshold'] = threshold
                data['eer'] = eer
                data['min_dcf'] = min_dcf
            if self.margin_scheduler:
                data['margin'] = self.margin_scheduler.get_margin()
            f.write(json.dumps(data, ensure_ascii=False))
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
        use_loss = self.configs.loss_conf.get('use_loss', 'AAMLoss')
        for batch_id, (features, label, input_lens) in enumerate(self.train_loader()):
            if self.stop_train: break
            if self.configs.dataset_conf.use_spec_aug:
                features = self.spec_aug(features)
            # 执行模型计算，是否开启自动混合精度
            with paddle.amp.auto_cast(enable=self.configs.train_conf.enable_amp, level='O1'):
                output = self.model(features)
            # 计算损失值
            los = self.loss(output, label)
            # 是否开启自动混合精度
            if self.configs.train_conf.enable_amp:
                # loss缩放，乘以系数loss_scaling
                scaled = self.amp_scaler.scale(los)
                scaled.backward()
            else:
                los.backward()
            # 是否开启自动混合精度
            if self.configs.train_conf.enable_amp:
                # 更新参数（参数梯度先除系数loss_scaling再更新参数）
                self.amp_scaler.step(self.optimizer)
                # 基于动态loss_scaling策略更新loss_scaling系数
                self.amp_scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.clear_grad()
            # 计算准确率
            if use_loss == 'SubCenterLoss':
                loss_args = self.configs.loss_conf.get('args', {})
                cosine = paddle.reshape(output, (-1, output.shape[1] // loss_args.K, loss_args.K))
                output = paddle.max(cosine, 2)
            label = paddle.reshape(label, shape=(-1, 1))
            acc = accuracy(input=paddle.nn.functional.softmax(output), label=label)
            accuracies.append(float(acc))
            loss_sum.append(float(los))
            train_times.append((time.time() - start) * 1000)
            self.train_step += 1

            # 多卡训练只使用一个进程打印
            if batch_id % self.configs.train_conf.log_interval == 0 and local_rank == 0:
                # 计算每秒训练数据量
                train_speed = self.configs.dataset_conf.dataLoader.batch_size / (
                        sum(train_times) / len(train_times) / 1000)
                # 计算剩余时间
                self.train_eta_sec = (sum(train_times) / len(train_times)) * (self.max_step - self.train_step) / 1000
                eta_str = str(timedelta(seconds=int(self.train_eta_sec)))
                self.train_loss = sum(loss_sum) / len(loss_sum)
                self.train_acc = sum(accuracies) / len(accuracies)
                margin_str = f'margin: {self.margin_scheduler.get_margin()}' if self.margin_scheduler else ''
                logger.info(f'Train epoch: [{epoch_id}/{self.configs.train_conf.max_epoch}], '
                            f'batch: [{batch_id}/{len(self.train_loader)}], '
                            f'loss: {self.train_loss:.5f}, accuracy: {self.train_acc:.5f}, '
                            f'learning rate: {self.scheduler.get_lr():.8f}, {margin_str} '
                            f'speed: {train_speed:.2f} data/sec, eta: {eta_str}')
                writer.add_scalar('Train/Loss', self.train_loss, self.train_log_step)
                writer.add_scalar('Train/Accuracy', self.train_acc, self.train_log_step)
                # 记录学习率
                writer.add_scalar('Train/lr', self.scheduler.get_lr(), self.train_log_step)
                if self.margin_scheduler:
                    writer.add_scalar('Train/margin', self.margin_scheduler.get_margin(), self.train_log_step)
                self.train_log_step += 1
                train_times, accuracies, loss_sum = [], [], []
            # 固定步数也要保存一次模型
            if batch_id % 10000 == 0 and batch_id != 0 and local_rank == 0:
                self.__save_checkpoint(save_model_path=save_model_path, epoch_id=epoch_id)
            start = time.time()
            self.scheduler.step()
            if self.margin_scheduler:
                self.margin_scheduler.step()

    def train(self,
              save_model_path='models/',
              resume_model=None,
              pretrained_model=None,
              do_eval=True):
        """
        训练模型
        :param save_model_path: 模型保存的路径
        :param resume_model: 恢复训练，当为None则不使用预训练模型
        :param pretrained_model: 预训练模型的路径，当为None则不使用预训练模型
        :param do_eval: 训练时是否评估模型
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
        self.__setup_dataloader(is_train=True)
        # 获取模型
        self.__setup_model(input_size=self.audio_featurizer.feature_dim, is_train=True)

        # 支持多卡训练
        if nranks > 1 and self.use_gpu:
            self.optimizer = fleet.distributed_optimizer(self.optimizer)
            self.model = fleet.distributed_model(self.model)
        logger.info('训练数据：{}'.format(len(self.train_dataset)))

        self.__load_pretrained(pretrained_model=pretrained_model)
        # 加载恢复模型
        last_epoch, best_eer = self.__load_checkpoint(save_model_path=save_model_path, resume_model=resume_model)

        self.train_loss, self.train_acc = None, None
        self.test_log_step, self.train_log_step = 0, 0
        self.eval_eer, self.eval_min_dcf, self.eval_threshold = None, None, None
        last_epoch += 1
        if local_rank == 0:
            writer.add_scalar('Train/lr', self.scheduler.get_lr(), last_epoch)
        # 最大步数
        self.max_step = len(self.train_loader) * self.configs.train_conf.max_epoch
        self.train_step = max(last_epoch, 0) * len(self.train_loader)
        # 开始训练
        for epoch_id in range(last_epoch, self.configs.train_conf.max_epoch):
            if self.stop_train: break
            epoch_id += 1
            start_epoch = time.time()
            # 训练一个epoch
            self.__train_epoch(epoch_id=epoch_id, save_model_path=save_model_path, local_rank=local_rank,
                               writer=writer)
            # 多卡训练只使用一个进程执行评估和保存模型
            if local_rank == 0 and do_eval:
                if self.stop_eval: continue
                logger.info('=' * 70)
                self.eval_eer, self.eval_min_dcf, self.eval_threshold = self.evaluate()
                logger.info('Test epoch: {}, time/epoch: {}, threshold: {:.2f}, EER: {:.5f}, '
                            'MinDCF: {:.5f}'.format(epoch_id, str(timedelta(
                    seconds=(time.time() - start_epoch))), self.eval_threshold, self.eval_eer, self.eval_min_dcf))
                logger.info('=' * 70)
                writer.add_scalar('Test/threshold', self.eval_threshold, self.test_log_step)
                writer.add_scalar('Test/min_dcf', self.eval_min_dcf, self.test_log_step)
                writer.add_scalar('Test/eer', self.eval_eer, self.test_log_step)
                self.test_log_step += 1
                self.model.train()
                # # 保存最优模型
                if self.eval_eer <= best_eer:
                    best_eer = self.eval_eer
                    self.__save_checkpoint(save_model_path=save_model_path, epoch_id=epoch_id, eer=self.eval_eer,
                                           min_dcf=self.eval_min_dcf, threshold=self.eval_threshold, best_model=True)
            if local_rank == 0:
                # 保存模型
                self.__save_checkpoint(save_model_path=save_model_path, epoch_id=epoch_id, eer=self.eval_eer,
                                       min_dcf=self.eval_min_dcf, threshold=self.eval_threshold)

    def evaluate(self, resume_model=None, save_image_path=None):
        """
        评估模型
        :param resume_model: 所使用的模型
        :param save_image_path: 保存图片的路径
        :return: 评估结果
        """
        if self.enroll_loader is None or self.trials_loader is None:
            self.__setup_dataloader()
        if self.model is None:
            self.__setup_model(input_size=self.audio_featurizer.feature_dim)
        if resume_model is not None:
            if os.path.isdir(resume_model):
                resume_model = os.path.join(resume_model, 'model.pdparams')
            assert os.path.exists(resume_model), f"{resume_model} 模型不存在！"
            model_state_dict = paddle.load(resume_model)
            missing_keys, unexpected_keys = self.model.set_state_dict(model_state_dict)
            if len(missing_keys) != 0 or len(unexpected_keys) != 0:
                logger.warning(f'模型加载部分失败，请检查模型是否匹配，'
                               f'missing_keys: {missing_keys}, unexpected_keys: {unexpected_keys}')
            logger.info(f'成功加载模型：{resume_model}')
        self.model.eval()
        if isinstance(self.model, paddle.DataParallel):
            eval_model = self.model._layers if len(self.model._layers) == 1 else self.model._layers[0]
        else:
            eval_model = self.model if len(self.model) == 1 else self.model[0]

        # 获取注册的声纹特征和标签
        enroll_features, enroll_labels = None, None
        with paddle.no_grad():
            for batch_id, (audio_features, label, input_lens) in enumerate(
                    tqdm(self.enroll_loader, desc="注册音频声纹特征")):
                if self.stop_eval: break
                feature = eval_model(audio_features).numpy()
                label = label.numpy()
                # 存放特征
                enroll_features = np.concatenate((enroll_features, feature)) if enroll_features is not None else feature
                enroll_labels = np.concatenate((enroll_labels, label)) if enroll_labels is not None else label
        # 获取检验的声纹特征和标签
        trials_features, trials_labels = None, None
        with paddle.no_grad():
            for batch_id, (audio_features, label, input_lens) in enumerate(
                    tqdm(self.trials_loader, desc="验证音频声纹特征")):
                if self.stop_eval: break
                feature = eval_model(audio_features).numpy()
                label = label.numpy()
                # 存放特征
                trials_features = np.concatenate((trials_features, feature)) if trials_features is not None else feature
                trials_labels = np.concatenate((trials_labels, label)) if trials_labels is not None else label
        self.model.train()
        enroll_labels = enroll_labels.astype(np.int32)
        trials_labels = trials_labels.astype(np.int32)
        print('开始对比音频特征...')
        all_score, all_labels = [], []
        for i in tqdm(range(len(trials_features)), desc='特征对比'):
            if self.stop_eval: break
            trials_feature = np.expand_dims(trials_features[i], 0).repeat(len(enroll_features), axis=0)
            score = cosine_similarity(trials_feature, enroll_features).tolist()[0]
            trials_label = np.expand_dims(trials_labels[i], 0).repeat(len(enroll_features), axis=0)
            y_true = np.array(enroll_labels == trials_label).astype(np.int32).tolist()
            all_score.extend(score)
            all_labels.extend(y_true)
        if self.stop_eval: return -1, -1, -1,
        # 计算EER
        all_score = np.array(all_score)
        all_labels = np.array(all_labels)
        fnr, fpr, thresholds = compute_fnr_fpr(all_score, all_labels)
        eer, threshold = compute_eer(fnr, fpr, all_score)
        min_dcf = compute_dcf(fnr, fpr)

        if save_image_path:
            import matplotlib.pyplot as plt
            index = np.where(np.array(thresholds) == threshold)[0][0]
            plt.plot(thresholds, fnr, color='blue', linestyle='-', label='fnr')
            plt.plot(thresholds, fpr, color='red', linestyle='-', label='fpr')
            plt.plot(threshold, fpr[index], 'ro-')
            plt.text(threshold, fpr[index], (round(threshold, 3), round(fpr[index], 5)), color='red')
            plt.xlabel('threshold')
            plt.title('fnr and fpr')
            plt.grid(True)  # 显示网格线
            # 保存图像
            os.makedirs(save_image_path, exist_ok=True)
            plt.savefig(os.path.join(save_image_path, 'result.png'))
            logger.info(f"结果图以保存在：{os.path.join(save_image_path, 'result.png')}")
        return eer, min_dcf, threshold

    def export(self, save_model_path='models/', resume_model='models/CAMPPlus_Fbank/best_model/'):
        """
        导出预测模型
        :param save_model_path: 模型保存的路径
        :param resume_model: 准备转换的模型路径
        :return:
        """
        # 获取模型
        self.__setup_model(input_size=self.audio_featurizer.feature_dim)
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
