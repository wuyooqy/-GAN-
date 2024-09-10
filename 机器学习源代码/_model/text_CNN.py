import tensorflow as tf
from builtins import range
from tensorflow.contrib.rnn.python.ops import core_rnn_cell


def apply_highway_layer(output, input_, idx, size, bias, f):
    # 应用单个高速网络层
    def linear_transform(output, name_scope, size):
        # 应用线性变换
        with tf.variable_scope(name_scope):
            return f(core_rnn_cell._linear(output, size, 0))

    def transformation_gate(input_, size, bias):
        # 创建转换门
        with tf.variable_scope('transform_lin_%d' % idx):
            transform_gate = tf.sigmoid(core_rnn_cell._linear(input_, size, 0) + bias)
            carry_gate = 1. - transform_gate
        return transform_gate, carry_gate

    output = linear_transform(output, 'output_lin_%d' % idx, size)
    transform_gate, carry_gate = transformation_gate(input_, size, bias)
    return transform_gate * output + carry_gate * input_


def highway(input_, size, layer_size=1, bias=-2, f=tf.nn.relu):
    # 实现高速网络层，允许信息直接通过，有助于解决训练深度模型时的梯度消失

    output = input_
    for idx in range(layer_size):
        output = apply_highway_layer(output, input_, idx, size, bias, f)
    return output


class TextCNN(object):
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        # 初始化 TextCNN 类，定义其属性
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.l2_reg_lambda = l2_reg_lambda

        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self._create_embedding_layer()
        self.pooled_outputs = self._create_conv_layers()
        self.h_pool_flat = self._flatten_pooled_outputs()
        self.h_highway = self._create_highway_layer()
        self.h_drop = self._create_dropout_layer()
        self.scores, self.l2_loss = self._create_output_layer()
        self.ypred_for_auc = tf.nn.softmax(self.scores)
        self.predictions = tf.argmax(self.scores, 1, name="predictions")
        self.loss = self._create_loss()
        self.accuracy = self._create_accuracy()

    def _create_embedding_layer(self):
        # 创建词嵌入层
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0), name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

    def _create_conv_layers(self):
        # 创建卷积层，包括多个不同尺寸的卷积核
        pooled_outputs = []
        for filter_size, num_filter in zip(self.filter_sizes, self.num_filters):
            with tf.name_scope(f"conv-maxpool-{filter_size}"):
                filter_shape = [filter_size, self.embedding_size, 1, num_filter]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filter]), name="b")
                conv = tf.nn.conv2d(self.embedded_chars_expanded, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                pooled = tf.nn.max_pool(h, ksize=[1, self.sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool")
                pooled_outputs.append(pooled)
        return pooled_outputs

    def _flatten_pooled_outputs(self):
        # 将池化层输出扁平化
        num_filters_total = sum(self.num_filters)
        h_pool = tf.concat(self.pooled_outputs, axis=3)
        return tf.reshape(h_pool, [-1, num_filters_total])

    def _create_highway_layer(self):
        # 创建高速网络层
        with tf.name_scope("highway"):
            return highway(self.h_pool_flat, self.h_pool_flat.get_shape()[1], 1, 0)

    def _create_dropout_layer(self):
        # 创建dropout层以减少过拟合
        with tf.name_scope("dropout"):
            return tf.nn.dropout(self.h_highway, self.dropout_keep_prob)

    def _create_output_layer(self):
        # 创建输出层，计算类别得分和L2损失
        with tf.name_scope("output"):
            # 确保维度被转换为整数
            num_features = int(self.h_pool_flat.get_shape()[1].value)
            # 使用正确的形状创建权重变量
            W = tf.Variable(tf.truncated_normal([num_features, self.num_classes], stddev=0.1), name="W")
            # 创建偏置变量
            b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b")
            # 计算 L2 损失
            l2_loss = tf.nn.l2_loss(W) + tf.nn.l2_loss(b)
            # 使用线性操作计算得分
            scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")

            return scores, l2_loss

    def _create_loss(self):
        # 计算损失函数，包括交叉熵损失和L2正则化损失
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            return tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss

    def _create_accuracy(self):
        # 计算模型准确率
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            return tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
