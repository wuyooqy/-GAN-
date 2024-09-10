import numpy as np

class Gen_Data_loader():
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def create_batches(self, samples):
        # 从样本中创建批次的公开方法
        self.num_batch = self._calculate_num_batches(samples)
        self.sequence_batch = self._prepare_batches(samples)
        self._reset_pointer()

    def next_batch(self):
        #  获取下一个批次的公开方法
        ret = self.sequence_batch[self.pointer]
        self._advance_pointer()
        return ret

    def reset_pointer(self):
        # 重置批次指针的公开方法
        self._reset_pointer()

    def _calculate_num_batches(self, samples):
        # 计算批次数量的内部方法
        return int(len(samples) / self.batch_size)

    def _prepare_batches(self, samples):
        #  通过重塑样本数组准备批次的内部方法
        samples = samples[:self.num_batch * self.batch_size]
        return np.split(np.array(samples), self.num_batch, 0)

    def _reset_pointer(self):
        # 将指针重置到批次开始的内部方法
        self.pointer = 0

    def _advance_pointer(self):
        # 将指针前移至下一个批次的内部方法
        self.pointer = (self.pointer + 1) % self.num_batch
