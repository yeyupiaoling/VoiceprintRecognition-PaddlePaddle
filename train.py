import argparse
import functools
import os
import re
import shutil
from datetime import datetime

import paddle
import paddle.distributed as dist
from paddle.io import DataLoader
from paddle.metric import accuracy
from paddle.static import InputSpec
from visualdl import LogWriter
from utils.resnet import resnet34
from utils.ArcMargin import ArcNet
from utils.reader import CustomDataset
from utils.utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('gpus',             str,    '0,1',                    '训练使用的GPU序号，使用英文逗号,隔开，如：0,1')
add_arg('batch_size',       int,    32,                       '训练的批量大小')
add_arg('num_workers',      int,    4,                        '读取数据的线程数量')
add_arg('num_epoch',        int,    120,                      '训练的轮数')
add_arg('num_classes',      int,    3242,                     '分类的类别数量')
add_arg('learning_rate',    float,  1e-1,                     '初始学习率的大小')
add_arg('input_shape',      str,    '(None, 1, 257, 257)',    '数据输入的形状')
add_arg('train_list_path',  str,    'dataset/train_list.txt', '训练数据的数据列表路径')
add_arg('test_list_path',   str,    'dataset/test_list.txt',  '测试数据的数据列表路径')
add_arg('save_model',       str,    'models/',                '模型保存的路径')
add_arg('resume',           str,    None,                     '恢复训练，当为None则不使用恢复模型')
add_arg('pretrained_model', str,    None,                     '预训练模型的路径，当为None则不使用预训练模型')
args = parser.parse_args()


# 评估模型
@paddle.no_grad()
def test(model, metric_fc, test_loader):
    model.eval()
    accuracies = []
    for batch_id, (spec_mag, label) in enumerate(test_loader()):
        feature = model(spec_mag)
        output = metric_fc(feature, label)
        label = paddle.reshape(label, shape=(-1, 1))
        acc = accuracy(input=output, label=label)
        accuracies.append(acc.numpy()[0])
    model.train()
    return float(sum(accuracies) / len(accuracies))


# 保存模型
def save_model(args, epoch, model, metric_fc, optimizer):
    input_shape = eval(args.input_shape)
    model_params_path = os.path.join(args.save_model, 'params', 'epoch_%d' % epoch)
    if not os.path.exists(model_params_path):
        os.makedirs(model_params_path)
    # 保存模型参数
    paddle.save(model.state_dict(), os.path.join(model_params_path, 'model.pdparams'))
    paddle.save(metric_fc.state_dict(), os.path.join(model_params_path, 'metric_fc.pdparams'))
    paddle.save(optimizer.state_dict(), os.path.join(model_params_path, 'optimizer.pdopt'))
    # 删除旧的模型
    old_model_path = os.path.join(args.save_model, 'params', 'epoch_%d' % (epoch - 3))
    if os.path.exists(old_model_path):
        shutil.rmtree(old_model_path)
    # 保存预测模型
    if not os.path.exists(os.path.join(args.save_model, 'infer')):
        os.makedirs(os.path.join(args.save_model, 'infer'))
    paddle.jit.save(layer=model,
                    path=os.path.join(args.save_model, 'infer/model'),
                    input_spec=[InputSpec(shape=[input_shape[0], input_shape[1], input_shape[2], input_shape[3]], dtype='float32')])


def train(args):
    # 设置支持多卡训练
    if len(args.gpus.split(',')) > 1:
        dist.init_parallel_env()
    if dist.get_rank() == 0:
        shutil.rmtree('log', ignore_errors=True)
        # 日志记录器
        writer = LogWriter(logdir='log')
    # 数据输入的形状
    input_shape = eval(args.input_shape)
    # 获取数据
    train_dataset = CustomDataset(args.train_list_path, model='train', spec_len=input_shape[3])
    # 设置支持多卡训练
    if len(args.gpus.split(',')) > 1:
        train_batch_sampler = paddle.io.DistributedBatchSampler(train_dataset, batch_size=args.batch_size, shuffle=True)
    else:
        train_batch_sampler = paddle.io.BatchSampler(train_dataset, batch_size=args.batch_size, shuffle=True)
    train_loader = DataLoader(dataset=train_dataset, batch_sampler=train_batch_sampler, num_workers=args.num_workers)

    test_dataset = CustomDataset(args.test_list_path, model='test', spec_len=input_shape[3])
    test_batch_sampler = paddle.io.BatchSampler(train_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(dataset=test_dataset, batch_sampler=test_batch_sampler, num_workers=args.num_workers)

    # 获取模型
    model = resnet34()
    metric_fc = ArcNet(feature_dim=512, class_dim=args.num_classes)
    if dist.get_rank() == 0:
        paddle.summary(model, input_size=input_shape)

    # 设置支持多卡训练
    if len(args.gpus.split(',')) > 1:
        model = paddle.DataParallel(model)
        metric_fc = paddle.DataParallel(metric_fc)

    # 获取预训练的epoch数
    last_epoch = int(re.findall(r'\d+', args.resume)[-1]) + 1 if args.resume is not None else 0
    # 学习率衰减
    scheduler = paddle.optimizer.lr.StepDecay(learning_rate=args.learning_rate, step_size=10, gamma=0.1, last_epoch=last_epoch, verbose=True)
    # 设置优化方法
    optimizer = paddle.optimizer.Momentum(parameters=model.parameters() + metric_fc.parameters(),
                                          learning_rate=scheduler,
                                          momentum=0.9,
                                          weight_decay=paddle.regularizer.L2Decay(5e-4))

    # 加载预训练模型
    if args.pretrained_model is not None:
        model_dict = model.state_dict()
        param_state_dict = paddle.load(os.path.join(args.pretrained_model, 'model.pdparams'))
        for name, weight in model_dict.items():
            if name in param_state_dict.keys():
                if weight.shape != list(param_state_dict[name].shape):
                    print('{} not used, shape {} unmatched with {} in model.'.
                          format(name, list(param_state_dict[name].shape), weight.shape))
                    param_state_dict.pop(name, None)
            else:
                print('Lack weight: {}'.format(name))
        model.set_dict(param_state_dict)
        print('成功加载预训练模型参数')

    # 恢复训练
    if args.resume is not None:
        model.set_state_dict(paddle.load(os.path.join(args.resume, 'model.pdparams')))
        metric_fc.set_state_dict(paddle.load(os.path.join(args.resume, 'metric_fc.pdparams')))
        optimizer.set_state_dict(paddle.load(os.path.join(args.resume, 'optimizer.pdopt')))
        print('成功加载模型参数和优化方法参数')

    # 获取损失函数
    loss = paddle.nn.CrossEntropyLoss()
    train_step = 0
    test_step = 0
    # 开始训练
    for epoch in range(args.num_epoch):
        loss_sum = []
        for batch_id, (spec_mag, label) in enumerate(train_loader()):
            feature = model(spec_mag)
            output = metric_fc(feature, label)
            # 计算损失值
            los = loss(output, label)
            loss_sum.append(los)
            los.backward()
            optimizer.step()
            optimizer.clear_grad()
            # 多卡训练只使用一个进程打印
            if batch_id % 100 == 0 and dist.get_rank() == 0:
                print('[%s] Train epoch %d, batch_id: %d, loss: %f' % (
                    datetime.now(), epoch, batch_id, sum(loss_sum) / len(loss_sum)))
                writer.add_scalar('Train loss', los, train_step)
                train_step += 1
                loss_sum = []
        # 多卡训练只使用一个进程执行评估和保存模型
        if dist.get_rank() == 0:
            acc = test(model, metric_fc, test_loader)
            print('='*70)
            print('[%s] Test %d, accuracy: %f' % (datetime.now(), epoch, acc))
            print('='*70)
            writer.add_scalar('Test acc', acc, test_step)
            # 记录学习率
            writer.add_scalar('Learning rate', scheduler.last_lr, epoch)
            test_step += 1
            save_model(args, epoch, model, metric_fc, optimizer)
        scheduler.step()


if __name__ == '__main__':
    print_arguments(args)
    if len(args.gpus.split(',')) > 1:
        dist.spawn(train, args=(args,), gpus=args.gpus)
    else:
        train(args)
