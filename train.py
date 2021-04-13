import os
from datetime import datetime

import paddle
import paddle.nn as nn
from paddle.io import DataLoader
from paddle.static import InputSpec
from paddle.metric import accuracy
from reader import CustomDataset
from model import resnet50

# 训练参数值
train_list_path = 'dataset/train_list.txt'
test_list_path = 'dataset/test_list.txt'
mean_std_path = 'dataset/mean_std.npy'
input_shape = (1, 257, 257)
read_data_num_workers = 8
batch_size = 32
learning_rate = 1e-3
num_classes = 3242
epoch_num = 1000
model_path = 'models/'


# 评估模型
def test(model, test_loader):
    model.eval()
    accuracies = []
    for batch_id, (spec_mag, label) in enumerate(test_loader()):
        out, _ = model(spec_mag)
        acc = accuracy(input=out, label=label)
        accuracies.append(acc)
    model.train()
    return sum(accuracies) / len(accuracies)


# 保存模型
def save_model(model, optimizer):
    if not os.path.exists(os.path.join(model_path, 'params')):
        os.makedirs(os.path.join(model_path, 'params'))
    if not os.path.exists(os.path.join(model_path, 'infer')):
        os.makedirs(os.path.join(model_path, 'infer'))
    # 保存模型参数
    paddle.save(model.state_dict(), os.path.join(model_path, 'params/model.pdparams'))
    paddle.save(optimizer.state_dict(), os.path.join(model_path, 'params/optimizer.pdopt'))
    # 保存预测模型
    paddle.jit.save(layer=model,
                    path=os.path.join(model_path, 'infer/model'),
                    input_spec=[InputSpec(shape=(None, ) + input_shape, dtype='float32')])


def train():
    # 获取数据
    train_dataset = CustomDataset(train_list_path, mean_std_path=mean_std_path, model='train', spec_len=input_shape[2])
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=read_data_num_workers)

    test_dataset = CustomDataset(test_list_path, mean_std_path=mean_std_path, model='test', spec_len=input_shape[2])
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=read_data_num_workers)

    # 获取模型
    model = resnet50(num_classes=num_classes)
    paddle.summary(model, input_size=[(None, ) + input_shape])

    # 设置优化方法
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(),
                                      learning_rate=learning_rate,
                                      weight_decay=paddle.regularizer.L2Decay(1e-4))

    # 获取损失函数
    loss = nn.CrossEntropyLoss()
    # 开始训练
    for epoch in range(epoch_num):
        loss_sum = []
        for batch_id, (spec_mag, label) in enumerate(train_loader()):
            out, feature = model(spec_mag)
            # 计算损失值
            los = loss(out, label)
            loss_sum.append(los)
            los.backward()
            optimizer.step()
            optimizer.clear_grad()
            if batch_id % 100 == 0:
                print('[%s] Train epoch %d, batch_id: %d, loss: %f' % (
                    datetime.now(), epoch, batch_id, sum(loss_sum) / len(loss_sum)))
                loss_sum = []
        acc = test(model, test_loader)
        print('[%s] Train epoch %d, accuracy: %d' % (datetime.now(), epoch, acc))
        save_model(model, optimizer)


if __name__ == '__main__':
    train()
