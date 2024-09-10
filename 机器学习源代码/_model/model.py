import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops

class LSTM(object):
    def __init__(self, num_emb, batch_size, emb_dim, hidden_dim,
                 sequence_length, start_token,
                 learning_rate=0.005, reward_gamma=0.95):

        # 初始化模型参数
        self.num_emb = num_emb
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.start_token = tf.constant([start_token] * self.batch_size, dtype=tf.int32)
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.reward_gamma = reward_gamma
        self.g_params = []
        self.d_params = []
        self.temperature = 1.0
        self.grad_clip = 5.0

        self.expected_reward = tf.Variable(tf.zeros([self.sequence_length]))

        # 定义生成器网络
        with tf.variable_scope('generator'):
            self.g_embeddings = tf.Variable(self.init_matrix([self.num_emb, self.emb_dim]))
            self.g_params.append(self.g_embeddings)
            # 创建循环单元和输出单元
            self.g_recurrent_unit = self.create_recurrent_unit(self.g_params)
            self.g_output_unit = self.create_output_unit(self.g_params)

        # 定义占位符
        self.x = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_length])

        self.rewards = tf.placeholder(tf.float32, shape=[self.batch_size, self.sequence_length])

        # 处理批次数据
        with tf.device("/cpu:0"):
            inputs = tf.split(axis=1, num_or_size_splits=self.sequence_length, value=tf.nn.embedding_lookup(self.g_embeddings, self.x))
            self.processed_x = tf.stack(
                [tf.squeeze(input_, [1]) for input_ in inputs])

        self.h0 = tf.zeros([self.batch_size, self.hidden_dim])
        self.h0 = tf.stack([self.h0, self.h0])

        # 初始化用于存储生成序列的tensor数组
        gen_o = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length,
                                             dynamic_size=False, infer_shape=True)
        gen_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.sequence_length,
                                             dynamic_size=False, infer_shape=True)

        # 定义生成过程的循环函数
        def _g_recurrence(i, x_t, h_tm1, gen_o, gen_x):
            h_t = self.g_recurrent_unit(x_t, h_tm1)  # 更新隐藏状态
            o_t = self.g_output_unit(h_t)  # 生成当前时间步的输出
            log_prob = tf.log(tf.nn.softmax(o_t))  # 计算log概率
            next_token = tf.cast(tf.reshape(tf.multinomial(log_prob, 1), [self.batch_size]), tf.int32)
            x_tp1 = tf.nn.embedding_lookup(self.g_embeddings, next_token)  # 获取下一个标记的嵌入表示
            gen_o = gen_o.write(i, tf.reduce_sum(tf.multiply(tf.one_hot(next_token, self.num_emb, 1.0, 0.0),
                                                        tf.nn.softmax(o_t)), 1))  # 存储概率
            gen_x = gen_x.write(i, next_token)  # 存储标记
            return i + 1, x_tp1, h_t, gen_o, gen_x

        # 执行生成过程的循环
        _, _, _, self.gen_o, self.gen_x = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4: i < self.sequence_length,
            body=_g_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.nn.embedding_lookup(self.g_embeddings, self.start_token), self.h0, gen_o, gen_x))

        self.gen_x = self.gen_x.stack()
        self.gen_x = tf.transpose(self.gen_x, perm=[1, 0])  # 调整tensor维度

        # 预训练生成器
        g_predictions = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length,
            dynamic_size=False, infer_shape=True)

        g_logits = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length,
            dynamic_size=False, infer_shape=True)

        ta_emb_x = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length)
        ta_emb_x = ta_emb_x.unstack(self.processed_x)

        # 定义预训练的循环函数
        def _pretrain_recurrence(i, x_t, h_tm1, g_predictions, g_logits):
            h_t = self.g_recurrent_unit(x_t, h_tm1)
            o_t = self.g_output_unit(h_t)
            g_predictions = g_predictions.write(i, tf.nn.softmax(o_t))  # 存储预测结果
            g_logits = g_logits.write(i, o_t)  # 存储logits
            x_tp1 = ta_emb_x.read(i)
            return i + 1, x_tp1, h_t, g_predictions, g_logits

        # 执行预训练的循环
        _, _, _, self.g_predictions, self.g_logits = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4: i < self.sequence_length,
            body=_pretrain_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.nn.embedding_lookup(self.g_embeddings, self.start_token),
                       self.h0, g_predictions, g_logits))

        self.g_predictions = tf.transpose(
            self.g_predictions.stack(), perm=[1, 0, 2])  # 调整预测结果的维度

        self.g_logits = tf.transpose(
            self.g_logits.stack(), perm=[1, 0, 2])  # 调整logits的维度

        # 计算预训练损失
        self.pretrain_loss = -tf.reduce_sum(
            tf.one_hot(tf.to_int32(tf.reshape(self.x, [-1])), self.num_emb, 1.0, 0.0) * tf.log(
                tf.clip_by_value(tf.reshape(self.g_predictions, [-1, self.num_emb]), 1e-20, 1.0)
            )
        ) / (self.sequence_length * self.batch_size)

        # 定义预训练的优化器和梯度裁剪
        pretrain_opt = self.g_optimizer(self.learning_rate)

        self.pretrain_grad, _ = tf.clip_by_global_norm(
            tf.gradients(self.pretrain_loss, self.g_params), self.grad_clip)
        self.pretrain_updates = pretrain_opt.apply_gradients(zip(self.pretrain_grad, self.g_params))


        # 无监督训练

        # 计算无监督训练的损失
        self.g_loss = -tf.reduce_sum(
            tf.reduce_sum(
                tf.one_hot(tf.to_int32(tf.reshape(self.x, [-1])), self.num_emb, 1.0, 0.0) * tf.log(
                    tf.clip_by_value(tf.reshape(self.g_predictions, [-1, self.num_emb]), 1e-20, 1.0)
                ), 1) * tf.reshape(self.rewards, [-1])
        )

        # 定义无监督训练的优化器
        g_opt = self.g_optimizer(self.learning_rate)

        # 应用梯度裁剪并更新参数
        self.g_grad, _ = tf.clip_by_global_norm(
            tf.gradients(self.g_loss, self.g_params), self.grad_clip)
        self.g_updates = g_opt.apply_gradients(zip(self.g_grad, self.g_params))

    def generate(self, session):
        # 生成函数，在给定的会话中运行模型，生成序列
        outputs = session.run([self.gen_x])
        return outputs[0]

    def pretrain_step(self, session, x):
        # 执行一个预训练步骤，包括参数更新和损失计算
        outputs = session.run([self.pretrain_updates, self.pretrain_loss, self.g_predictions],
                              feed_dict={self.x: x})
        return outputs

    def generator_step(self, sess, samples, rewards):
        # 执行一个生成器训练步骤。接收样本和奖励，更新参数
        feed = {self.x: samples, self.rewards: rewards}
        _, g_loss = sess.run([self.g_updates, self.g_loss], feed_dict=feed)
        return g_loss

    def init_matrix(self, shape):
        # 初始化矩阵参数
        return tf.random_normal(shape, stddev=0.1)

    def init_vector(self, shape):
        # 初始化向量参数
        return tf.zeros(shape)

    def create_recurrent_unit(self, params):
        # 定义循环单元的权重和偏置

        # 输入门
        self.Wi = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Ui = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.bi = tf.Variable(self.init_matrix([self.hidden_dim]))

        # 遗忘门
        self.Wf = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uf = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bf = tf.Variable(self.init_matrix([self.hidden_dim]))

        # 输出门
        self.Wog = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uog = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bog = tf.Variable(self.init_matrix([self.hidden_dim]))

        # 新记忆细胞
        self.Wc = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uc = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bc = tf.Variable(self.init_matrix([self.hidden_dim]))

        # 将参数添加到参数列表
        params.extend([
            self.Wi, self.Ui, self.bi,
            self.Wf, self.Uf, self.bf,
            self.Wog, self.Uog, self.bog,
            self.Wc, self.Uc, self.bc])

        # 定义单元操作
        def unit(x, hidden_memory_tm1):
            previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)

            # 输入门计算
            i = tf.sigmoid(
                tf.matmul(x, self.Wi) +
                tf.matmul(previous_hidden_state, self.Ui) + self.bi
            )

            # 遗忘门计算
            f = tf.sigmoid(
                tf.matmul(x, self.Wf) +
                tf.matmul(previous_hidden_state, self.Uf) + self.bf
            )

            # 输出门计算
            o = tf.sigmoid(
                tf.matmul(x, self.Wog) +
                tf.matmul(previous_hidden_state, self.Uog) + self.bog
            )

            # 新记忆细胞内容
            c_ = tf.nn.tanh(
                tf.matmul(x, self.Wc) +
                tf.matmul(previous_hidden_state, self.Uc) + self.bc
            )

            # 最终记忆细胞
            c = f * c_prev + i * c_

            # 当前隐藏状态
            current_hidden_state = o * tf.nn.tanh(c)

            return tf.stack([current_hidden_state, c])

        return unit

    def create_output_unit(self, params):
        # 定义输出单元的权重和偏置
        self.Wo = tf.Variable(self.init_matrix([self.hidden_dim, self.num_emb]))
        self.bo = tf.Variable(self.init_matrix([self.num_emb]))
        params.extend([self.Wo, self.bo])

        def unit(hidden_memory_tuple):
            hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
            logits = tf.matmul(hidden_state, self.Wo) + self.bo
            return logits

        return unit

    def g_optimizer(self, *args, **kwargs):
        # 定义生成器的优化器
        return tf.train.GradientDescentOptimizer(*args, **kwargs)
