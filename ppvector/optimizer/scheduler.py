import math

import paddle


def cosine_decay_with_warmup(learning_rate, step_per_epoch, fix_epoch=1000, warmup_epoch=5, min_lr=0.0):
    """
    :param learning_rate: 学习率
    :param step_per_epoch: 每个epoch的步数
    :param fix_epoch: 最大epoch数
    :param warmup_epoch: 预热步数
    :param min_lr: 最小学习率
    :return:
    """
    # 预热步数
    boundary = []
    value = []
    warmup_steps = warmup_epoch * step_per_epoch
    # 初始化预热步数
    for i in range(warmup_steps + 1):
        if warmup_steps > 0:
            alpha = i / warmup_steps
            lr = learning_rate * alpha
            value.append(lr)
        if i > 0:
            boundary.append(i)

    max_iters = fix_epoch * int(step_per_epoch)
    warmup_iters = len(boundary)
    # 初始化最大步数
    for i in range(int(boundary[-1]), max_iters):
        boundary.append(i)
        # 如果当前步数小于最大步数，则将当前步数设置为最小学习率
        if i < max_iters:
            decayed_lr = min_lr + (learning_rate - min_lr) * 0.5 * (math.cos(
                (i - warmup_iters) * math.pi / (max_iters - warmup_iters)) + 1)
            value.append(decayed_lr)
        else:
            value.append(min_lr)
    return paddle.optimizer.lr.PiecewiseDecay(boundary, value)


class MarginScheduler:
    def __init__(
            self,
            criterion,
            increase_start_epoch,
            fix_epoch,
            step_per_epoch,
            initial_margin=0.0,
            final_margin=0.3,
            increase_type='exp',
    ):
        assert hasattr(criterion, 'update'), "Loss function not has 'update()' attributes."
        self.criterion = criterion
        self.increase_start_step = increase_start_epoch * step_per_epoch
        self.fix_step = fix_epoch * step_per_epoch
        self.initial_margin = initial_margin
        self.final_margin = final_margin
        self.increase_type = increase_type
        self.margin = initial_margin

        self.current_step = 0
        self.increase_step = self.fix_step - self.increase_start_step

        self.init_margin()

    def init_margin(self):
        self.criterion.update(margin=self.initial_margin)

    def step(self, current_step=None):
        if current_step is not None:
            self.current_step = current_step

        self.margin = self.iter_margin()
        self.criterion.update(margin=self.margin)
        self.current_step += 1

    def iter_margin(self):
        if self.current_step < self.increase_start_step:
            return self.initial_margin

        if self.current_step >= self.fix_step:
            return self.final_margin

        a = 1.0
        b = 1e-3

        current_step = self.current_step - self.increase_start_step
        if self.increase_type == 'exp':
            # exponentially increase the margin
            ratio = 1.0 - math.exp(
                (current_step / self.increase_step) *
                math.log(b / (a + 1e-6))) * a
        else:
            # linearly increase the margin
            ratio = 1.0 * current_step / self.increase_step
        return self.initial_margin + (self.final_margin -
                                      self.initial_margin) * ratio

    def get_margin(self):
        return self.margin
