import numpy as np
from re import compile as _Re

# 拆分包含Unicode字符的文本
def split_unicode_chrs(text):
    _unicode_chr_splitter = _Re('(?s)((?:[\ud800-\udbff][\udc00-\udfff])|.)').split
    return [chr for chr in _unicode_chr_splitter(text) if chr]
    # 分割字符串，返回一个字符列表

class Dis_dataloader():
    def __init__(self):
        self.vocab_size = 5000
        self.sequence_length = None

    def load_data_and_labels(self, positive_examples, negative_examples):
        x_text = positive_examples + negative_examples
        # 合并正面和负面例子为一个数据集
        y = np.concatenate([self._label_data(len(positive_examples), [0, 1]),
                            self._label_data(len(negative_examples), [1, 0])], 0)
        return [np.array(x_text), np.array(y)]

    def load_train_data(self, positive_file, negative_file):
        sentences, labels = self.load_data_and_labels(positive_file, negative_file)
        return self._shuffle_data(sentences, labels, self.sequence_length)
        # 加载训练数据，随机打乱并返回

    def load_test_data(self, positive_file, test_file):
        test_examples, test_labels = self._read_and_label_files(test_file, [1, 0], positive_file, [0, 1])
        return self._shuffle_data(test_examples, test_labels)

    def batch_iter(self, data, batch_size, num_epochs):
        data = np.array(list(data))
        num_batches_per_epoch = int(len(data) / batch_size) + (len(data) % batch_size != 0)
        for epoch in range(num_epochs):
            shuffled_data = self._shuffle_data(data)
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, len(data))
                yield shuffled_data[start_index:end_index]

    def _label_data(self, length, label):
        return [label for _ in range(length)]

    def _read_and_label_files(self, test_file, test_label, positive_file, positive_label):
        test_examples, test_labels = self._read_file(test_file, test_label)
        positive_examples, positive_labels = self._read_file(positive_file, positive_label)
        return test_examples + positive_examples, test_labels + positive_labels
        # 读取文件并标记，返回合并后的数据和标签

    def _read_file(self, file_name, label):
        examples, labels = [], []
        with open(file_name) as fin:
            for line in fin:
                parse_line = [int(x) for x in line.strip().split()]
                examples.append(parse_line)
                labels.append(label)
        return examples, labels
        # 读取文件，按行解析并标记

    def _shuffle_data(self, data, labels=None, sequence_length=None):
        shuffle_indices = np.random.permutation(np.arange(len(data)))
        data = data[shuffle_indices]
        if labels is not None:
            labels = labels[shuffle_indices]
        return [data, labels] if labels is not None else data
        # 数据随机打乱，如果有标签也一并打乱
