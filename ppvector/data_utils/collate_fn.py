import paddle


# 对一个batch的数据处理
def collate_fn(batch):
    # 找出音频长度最长的
    batch_sorted = sorted(batch, key=lambda sample: sample[0].shape[0], reverse=True)
    freq_size = batch_sorted[0][0].shape[1]
    max_freq_length = batch_sorted[0][0].shape[0]
    batch_size = len(batch_sorted)
    # 以最大的长度创建0张量
    features = paddle.zeros((batch_size, max_freq_length, freq_size), dtype=paddle.float32)
    input_lens, labels = [], []
    for x in range(batch_size):
        tensor, label = batch[x]
        seq_length = tensor.shape[0]
        # 将数据插入都0张量中，实现了padding
        features[x, :seq_length, :] = tensor[:, :]
        labels.append(label)
        input_lens.append(seq_length)
    labels = paddle.to_tensor(labels, dtype=paddle.int64)
    input_lens = paddle.to_tensor(input_lens, dtype=paddle.int64)
    return features, labels, input_lens
