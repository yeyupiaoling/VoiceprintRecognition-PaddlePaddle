import math

import paddle


def cosine_decay_with_warmup(learning_rate, step_per_epoch, max_epochs=1000, warmup_epochs=5, min_lr_ratio=0.0):
    # 预热步数
    boundary = []
    value = []
    warmup_steps = warmup_epochs * step_per_epoch
    for i in range(warmup_steps + 1):
        if warmup_steps > 0:
            alpha = i / warmup_steps
            lr = learning_rate * alpha
            value.append(lr)
        if i > 0:
            boundary.append(i)

    max_iters = max_epochs * int(step_per_epoch)
    min_lr = learning_rate * min_lr_ratio
    warmup_iters = len(boundary)
    for i in range(int(boundary[-1]), max_iters):
        boundary.append(i)
        if i < max_iters:
            decayed_lr = min_lr + (learning_rate - min_lr) * 0.5 * (math.cos(
                (i - warmup_iters) * math.pi / (max_iters - warmup_iters)) + 1)
            value.append(decayed_lr)
        else:
            value.append(min_lr)
    return paddle.optimizer.lr.PiecewiseDecay(boundary, value)
