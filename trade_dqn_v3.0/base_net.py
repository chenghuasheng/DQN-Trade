import math
import os

import tensorflow as tf

from trade_input_data import *

# 生成权重变量的随机种子
const.SEED = 66478
const.LOG_DIR = 'log'
const.TAU = 0.01


class BaseNet(object):
    def __init__(self, sess=None, trainable=False):
        if sess is None:
            sess = tf.Session()
        self.sess = sess
        self.trainable = trainable
        self.variables = []
        self._placeholder_inputs()
        self._inference()
        # 要保存的变量字典
        # saver_dict = {}
        # for var in self.variables:
        #     saver_dict[var.name] = var
        # if saver_dict:
        # self.saver = tf.train.Saver(saver_dict)
        self.saver = tf.train.Saver()
        # 当前文件所在目录
        self.path = os.path.dirname(os.path.realpath(__file__))

    def save(self):
        if hasattr(self, "saver"):
            checkpoint_file = os.path.join(self.path, const.LOG_DIR, 'modkel.cpt')
            self.saver.save(self.sess, checkpoint_file)

    def restore(self):
        if hasattr(self, "saver"):
            checkpoint_dir = os.path.join(self.path, const.LOG_DIR)
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print("Restore form Model {}".format(ckpt.model_checkpoint_path))
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def replace_to(self, target_net, target_vars=None, source_vars=None, soft_update=True):
        if target_vars is None:
            target_vars = target_net.variables
        if source_vars is None:
            source_vars = self.variables
        if soft_update:
            replace_op = [tf.assign(t, (1 - const.TAU) * t + const.TAU * s) for t, s in zip(target_vars, source_vars)]
        else:
            replace_op = [tf.assign(t, s) for t, s in zip(target_vars, source_vars)]
        self.sess.run(replace_op)

    def _loss(self):
        pass

    def prediction(self, packaged_data):
        pass

    def fill_feed_dict(self, packaged_data, keep_prob=1.0):
        feed_dict = {
            self.input_index_daily: packaged_data.index_daily,
            # self.input_index_min: packaged_data.index_min,
            self.input_stock_daily: packaged_data.stock_daily,
            self.input_stock_min: packaged_data.stock_min,
            self.keep_prob: keep_prob
        }
        return feed_dict

    def _placeholder_inputs(self):
        self.keep_prob = tf.placeholder(tf.float32)  # 用于防止过拟合的参数
        self.input_index_daily = tf.placeholder(tf.float32, shape=[None, const.INDEX_DAILY_PERIOD, const.KLINE_SIZE])
        # self.input_index_min = tf.placeholder(tf.float32, shape=[None, const.INDEX_MIN_PERIOD, const.KLINE_SIZE])
        self.input_stock_daily = tf.placeholder(tf.float32, shape=[None, const.STOCK_DAILY_PERIOD, const.KLINE_SIZE])
        self.input_stock_min = tf.placeholder(tf.float32, shape=[None, const.STOCK_MIN_PERIOD, const.KLINE_SIZE])

    def _inference(self):
        with tf.name_scope("input_region") as scope:
            index_daily_cnn = self._kline_cnn(self.input_index_daily, const.INDEX_DAILY_PERIOD, 4,
                                              name_scope="index_daily_cnn")
            # index_min_cnn = self._kline_cnn(self.input_index_min, const.INDEX_MIN_PERIOD, 8,
            #                                name_scope="index_min_cnn")
            stock_daily_cnn = self._kline_cnn(self.input_stock_daily, const.STOCK_DAILY_PERIOD, 24,
                                              name_scope="stock_daily_cnn")
            stock_min_cnn = self._kline_cnn(self.input_stock_min, const.STOCK_MIN_PERIOD, 8,
                                            name_scope="stock_min_cnn")
            # self.input_gfc1 = tf.concat(
            #     [index_daily_cnn, index_min_cnn, stock_daily_cnn, stock_min_cnn], 1)
            self.input_gfc1 = tf.concat(
            [index_daily_cnn, stock_daily_cnn, stock_min_cnn], 1)

    # 建立权重变量
    def _weight_variable(self, shape, name='weights'):
        initial = tf.truncated_normal(shape, stddev=0.1, seed=const.SEED)
        var = tf.Variable(initial, name=name, trainable=self.trainable)
        self.variables.append(var)
        return var

    # 建立偏置变量
    def _bias_variable(self, shape, name='biases'):
        initial = tf.constant(0.1, shape=shape)
        var = tf.Variable(initial, name=name, trainable=self.trainable)
        self.variables.append(var)
        return var

    # 建立卷积层
    def _conv2d(self, input, filters, strides, activation_function=tf.nn.relu, name_scope="conv2d"):
        with tf.name_scope(name_scope) as scope:
            weights = self._weight_variable(shape=filters)
            biases = self._bias_variable(shape=[filters[-1]])
            if activation_function is None:
                return tf.nn.conv2d(input, weights, strides=strides, padding='SAME') + biases
            else:
                return activation_function(
                    tf.nn.conv2d(input, weights, strides=strides, padding='SAME') + biases)

    # 建立池化层
    def _pool(self, input, ksize, strides, mode="MAX"):
        # stride [1, x_movement, y_movement, 1]
        if mode == "MAX":
            return tf.nn.max_pool(input, ksize=ksize, strides=strides, padding='SAME')
        else:
            return tf.nn.avg_pool(input, ksize=ksize, strides=strides, padding='SAME')

    # 建立全连接层
    def _full_connection(self, input, in_size, out_size, activation_function=tf.nn.relu,
                         name_scope="full_connection"):
        with tf.name_scope(name_scope) as scope:
            weights = self._weight_variable([in_size, out_size])
            biases = self._bias_variable([out_size])
            wx_plus_b = tf.matmul(input, weights) + biases
            if activation_function is None:
                outputs = wx_plus_b
            else:
                outputs = activation_function(wx_plus_b)
            return outputs

    # 建立残差块
    def _residual_block(self, x, x_size, fx, fx_size):
        if x_size == fx_size:
            return tf.nn.relu(tf.add(fx, x))
        else:
            weights = self._weight_variable([x_size, fx_size])
            return tf.nn.relu(tf.add(fx, tf.matmul(x, weights)))

    # 建立K线的卷积神经网络
    def _kline_cnn(self, input, input_xsize, output_xsize, name_scope="kline_cnn"):
        with tf.name_scope(name_scope) as scope:
            input = tf.reshape(input, [-1, input_xsize, const.KLINE_SIZE, 1])
            # 卷积层一
            h_conv_1 = self._conv2d(input, filters=[1, const.KLINE_SIZE, 1, 32],
                                    strides=[1, 1, const.KLINE_SIZE, 1],
                                    name_scope="conv2d_1")  # -1,k,1,16
            h_conv_1 = tf.reshape(h_conv_1, [-1, input_xsize, 32, 1])  # -1,k,16,1
            # 卷积层二
            h_conv_2 = self._conv2d(h_conv_1, filters=[3, 1, 1, 8], strides=[1, 1, 1, 1],
                                    name_scope="conv2d_2")  # -1,k,16,8
            # 卷积层三
            h_conv_3 = self._conv2d(h_conv_1, filters=[5, 1, 1, 8], strides=[1, 1, 1, 1],
                                    name_scope="conv2d_3")  # -1,k,16,8
            h_conv_all = tf.concat([h_conv_1, h_conv_2, h_conv_3], 3)
            ## 池化层一
            h_pool_1 = self._pool(h_conv_all, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1])  # -1,k/2,16,17

            ## 全连接层一
            fc1_x_size = math.ceil(input_xsize / 2) * 32 * 17
            fc1_y_size = 256 * int(input_xsize / 5)
            h_pool_1 = tf.reshape(h_pool_1, [-1, fc1_x_size])
            h_fc1 = self._full_connection(h_pool_1, fc1_x_size, fc1_y_size, name_scope="full_conn_1")
            h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
            ## 全连接层二 ##
            return self._full_connection(h_fc1_drop, fc1_y_size, output_xsize, name_scope="full_conn_2")
