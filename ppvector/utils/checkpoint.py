import json
import os
import shutil

import paddle

from loguru import logger
from ppvector import __version__


def load_pretrained(model, pretrained_model):
    """加载预训练模型

    :param model: 使用的模型
    :param pretrained_model: 预训练模型路径
    """
    # 加载预训练模型
    if pretrained_model is None: return model
    if os.path.isdir(pretrained_model):
        pretrained_model = os.path.join(pretrained_model, 'model.pdparams')
    assert os.path.exists(pretrained_model), f"{pretrained_model} 模型不存在！"
    model_dict = model.state_dict()
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
    # 加载权重
    missing_keys, unexpected_keys = model.set_state_dict(model_state_dict)
    if len(unexpected_keys) > 0:
        logger.warning('Unexpected key(s) in state_dict: {}. '
                       .format(', '.join('"{}"'.format(k) for k in unexpected_keys)))
    if len(missing_keys) > 0:
        logger.warning('Missing key(s) in state_dict: {}. '
                       .format(', '.join('"{}"'.format(k) for k in missing_keys)))
    logger.info('成功加载预训练模型：{}'.format(pretrained_model))
    return model


def load_checkpoint(configs, model, optimizer, amp_scaler, scheduler,margin_scheduler,
                    step_epoch, save_model_path, resume_model):
    """加载模型

    :param configs: 配置信息
    :param model: 使用的模型
    :param optimizer: 使用的优化方法
    :param amp_scaler: 使用的自动混合精度
    :param scheduler: 使用的学习率调整策略
    :param margin_scheduler: margin调整策略
    :param step_epoch: 每个epoch的step数量
    :param save_model_path: 模型保存路径
    :param resume_model: 恢复训练的模型路径
    """
    last_epoch1 = 0
    best_eer1 = 1

    def load_model(model_path):
        assert os.path.exists(os.path.join(model_path, 'model.pdparams')), "模型参数文件不存在！"
        assert os.path.exists(os.path.join(model_path, 'optimizer.pdopt')), "优化方法参数文件不存在！"
        state_dict = paddle.load(os.path.join(model_path, 'model.pdparams'))
        missing_keys, unexpected_keys = model.set_state_dict(state_dict)
        assert len(missing_keys) == len(unexpected_keys) == 0, "模型参数加载失败，参数权重不匹配，请可以考虑当做预训练模型！"
        optimizer.set_state_dict(paddle.load(os.path.join(model_path, 'optimizer.pdopt')))
        # 自动混合精度参数
        if amp_scaler is not None and os.path.exists(os.path.join(model_path, 'scaler.pdparams')):
            amp_scaler.set_state_dict(paddle.load(os.path.join(model_path, 'scaler.pdparams')))
        with open(os.path.join(model_path, 'model.state'), 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            last_epoch = json_data['last_epoch']
            best_eer = 1
            if 'eer' in json_data.keys():
                best_eer = json_data['eer']
        logger.info('成功恢复模型参数和优化方法参数：{}'.format(model_path))
        optimizer.step()
        [scheduler.step() for _ in range(last_epoch * step_epoch)]
        if margin_scheduler is not None:
            margin_scheduler.step(current_step=last_epoch * step_epoch)
        return last_epoch, best_eer

    # 获取最后一个保存的模型
    save_feature_method = configs.preprocess_conf.feature_method
    last_model_dir = os.path.join(save_model_path,
                                  f'{configs.model_conf.model}_{save_feature_method}',
                                  'last_model')
    if resume_model is not None or (os.path.exists(os.path.join(last_model_dir, 'model.pdparams'))
                                    and os.path.exists(os.path.join(last_model_dir, 'optimizer.pdopt'))):
        if resume_model is not None:
            last_epoch1, best_eer1 = load_model(resume_model)
        else:
            try:
                # 自动获取最新保存的模型
                last_epoch1, best_eer1 = load_model(last_model_dir)
            except Exception as e:
                logger.warning(f'尝试自动恢复最新模型失败，错误信息：{e}')
    return model, optimizer, amp_scaler, scheduler, margin_scheduler, last_epoch1, best_eer1


# 保存模型
def save_checkpoint(configs, model, optimizer, amp_scaler, margin_scheduler, save_model_path, epoch_id,
                    eer=None, min_dcf=None, threshold=None, best_model=False):
    """保存模型

    :param configs: 配置信息
    :param model: 使用的模型
    :param optimizer: 使用的优化方法
    :param amp_scaler: 使用的自动混合精度
    :param margin_scheduler: margin调整策略
    :param save_model_path: 模型保存路径
    :param epoch_id: 当前epoch
    :param eer: 当前eer
    :param min_dcf: 当前min_dcf
    :param threshold: 当前threshold
    :param best_model: 是否为最佳模型
    """
    # 保存模型的路径
    save_feature_method = configs.preprocess_conf.feature_method
    if best_model:
        model_path = os.path.join(save_model_path,
                                  f'{configs.model_conf.model}_{save_feature_method}', 'best_model')
    else:
        model_path = os.path.join(save_model_path,
                                  f'{configs.model_conf.model}_{save_feature_method}', 'epoch_{}'.format(epoch_id))
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    os.makedirs(model_path, exist_ok=True)
    # 保存模型参数
    paddle.save(optimizer.state_dict(), os.path.join(model_path, 'optimizer.pdopt'))
    paddle.save(model.state_dict(), os.path.join(model_path, 'model.pdparams'))
    # 自动混合精度参数
    if amp_scaler is not None:
        paddle.save(amp_scaler.state_dict(), os.path.join(model_path, 'scaler.pdparams'))
    with open(os.path.join(model_path, 'model.state'), 'w', encoding='utf-8') as f:
        use_loss = configs.loss_conf.get('use_loss', 'AAMLoss')
        data = {"last_epoch": epoch_id, "version": __version__, "model_conf.model": configs.model_conf.model,
                "feature_method": save_feature_method, "loss": use_loss}
        if eer is not None:
            data['threshold'] = threshold
            data['eer'] = eer
            data['min_dcf'] = min_dcf
        if margin_scheduler:
            data['margin'] = margin_scheduler.get_margin()
        f.write(json.dumps(data, indent=4, ensure_ascii=False))
    if not best_model:
        last_model_path = os.path.join(save_model_path,
                                       f'{configs.model_conf.model}_{save_feature_method}', 'last_model')
        shutil.rmtree(last_model_path, ignore_errors=True)
        shutil.copytree(model_path, last_model_path)
        # 删除旧的模型
        old_model_path = os.path.join(save_model_path,
                                      f'{configs.model_conf.model}_{save_feature_method}',
                                      'epoch_{}'.format(epoch_id - 3))
        if os.path.exists(old_model_path):
            shutil.rmtree(old_model_path)
    logger.info('已保存模型：{}'.format(model_path))
