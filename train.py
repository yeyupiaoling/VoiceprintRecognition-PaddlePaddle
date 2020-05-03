import os
import reader
import paddle.fluid as fluid
from vgg import VGG11

# 保存模型路径
save_path = 'models/'
# 初始化模型路径
init_model = None
# 类别总数
CLASS_DIM = 855

# 定义输入层
audio = fluid.data(name='audio', shape=[None, 1, 128, 128], dtype='float32')
label = fluid.data(name='label', shape=[None, 1], dtype='int64')


# 获取网络模型
vgg = VGG11()
model, feature = vgg.net(audio, CLASS_DIM)

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

# 定义一个使用GPU的执行器
place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
# 进行参数初始化
exe.run(fluid.default_startup_program())

# 加载初始化模型
if init_model:
    fluid.load(program=fluid.default_main_program(),
               model_path=init_model,
               executor=exe,
               var_list=fluid.io.get_program_parameter(fluid.default_main_program()))
    print("Init model from: %s." % init_model)

# 训练
for pass_id in range(100):
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
