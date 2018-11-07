import tensorflow as tf
import math
import os
from trade_input_data import *

# 生成权重变量的随机种子
const.SEED = 66478
const.LOG_DIR = 'log'


class TradeDqn(object):
    def __init__(self, sess=None):
        if sess is None:
            sess = tf.Session()
        self.sess = sess
        self.variables = []
        self.__placeholder_inputs()
        self.__inference()

        # 要保存的变量字典
        saver_dict = {}
        for v in self.variables:
            saver_dict[v.name] = v
        self.saver = tf.train.Saver(saver_dict)
        # 当前文件所在目录
        self.path = os.path.dirname(os.path.realpath(__file__))

    def loss(self):
        return tf.reduce_mean(tf.reduce_sum(tf.square(self.output - self.input_label), axis=1))

    def save(self):
        checkpoint_file = os.path.join(self.path, const.LOG_DIR, 'modkel.cpt')
        self.saver.save(self.sess, checkpoint_file)

    def restore(self):
        checkpoint_dir = os.path.join(self.path, const.LOG_DIR)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Restore form Model {}".format(ckpt.model_checkpoint_path))
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def replace_to(self, target_net):
        target_vars = target_net.variables
        source_vars = self.variables
        replace_op = [tf.assign(t, s) for t, s in zip(target_vars, source_vars)]
        self.sess.run(replace_op)

    def fill_feed_dict(self, packaged_data, label=None, keep_prob=1.0):
        feed_dict = {
            self.input_index_daily: packaged_data.index_daily,
            self.input_stock_daily: packaged_data.stock_daily,
            self.input_stock_min5: packaged_data.stock_min5,
            self.input_holding_period: packaged_data.holding_period,
            self.keep_prob: keep_prob
        }
        if not (label is None):
            label = label.reshape((-1, 2))
            feed_dict[self.input_label] = label
        return feed_dict

    def prediction(self, packaged_data):
        feed_dict = self.fill_feed_dict(packaged_data)
        result = self.sess.run(self.output, feed_dict=feed_dict)
        return result

    def __placeholder_inputs(self):
        self.keep_prob = tf.placeholder(tf.float32)  # 用于防止过拟合的参数
        self.input_index_daily = tf.placeholder(tf.float32, shape=[None, const.INDEX_DAILY_PERIOD, const.KLINE_SIZE])
        self.input_stock_daily = tf.placeholder(tf.float32, shape=[None, const.STOCK_DAILY_PERIOD, const.KLINE_SIZE])
        self.input_stock_min5 = tf.placeholder(tf.float32, shape=[None, const.STOCK_MIN5_PERIOD, const.KLINE_SIZE])
        self.input_holding_period = tf.placeholder(tf.float32, shape=[None, 1])
        self.input_label = tf.placeholder(tf.float32, shape=[None, 2])

    def __inference(self):
        with tf.name_scope("global_full_connection_1") as scope:
            index_daily_cnn = self.__kline_cnn(self.input_index_daily, const.INDEX_DAILY_PERIOD, 4,
                                               name_scope="index_daily_cnn")
            stock_daily_cnn = self.__kline_cnn(self.input_stock_daily, const.STOCK_DAILY_PERIOD, 16,
                                               name_scope="stock_daily_cnn")
            stock_min5_cnn = self.__kline_cnn(self.input_stock_min5, const.STOCK_MIN5_PERIOD, 6,
                                              name_scope="stock_min5_cnn")
            input_gfc1 = tf.concat(
                [index_daily_cnn, stock_daily_cnn, stock_min5_cnn, self.input_holding_period], 1)
            h_gfc1 = self.__full_connection(input_gfc1, 27, 32, name_scope="h_gfc1")
            h_gfc2 = self.__full_connection(h_gfc1, 32, 16, name_scope="h_gfc2")
            h_gfc3 = self.__full_connection(h_gfc2, 16, 8, name_scope="h_gfc2")
            h_gfc4 = self.__full_connection(h_gfc3, 8, 8, activation_function=None, name_scope="h_gfc3")
            h_rb = self.__residual_block(h_gfc2, 16, h_gfc4, 8)
            self.output = self.__full_connection(h_rb, 8, 2, activation_function=tf.nn.tanh, name_scope="output")

    # 建立权重变量
    def __weight_variable(self, shape, name='weights'):
        initial = tf.truncated_normal(shape, stddev=0.1, seed=const.SEED)
        var = tf.Variable(initial, name=name)
        self.variables.append(var)
        return var

    # 建立偏置变量
    def __bias_variable(self, shape, name='biases'):
        initial = tf.constant(0.1, shape=shape)
        var = tf.Variable(initial, name=name)
        self.variables.append(var)
        return var

    # 建立卷积层
    def __conv2d(self, input, filters, strides, activation_function=tf.nn.relu, name_scope="conv2d"):
        with tf.name_scope(name_scope) as scope:
            weights = self.__weight_variable(shape=filters)
            biases = self.__bias_variable(shape=[filters[-1]])
            if activation_function is None:
                return tf.nn.conv2d(input, weights, strides=strides, padding='SAME') + biases
            else:
                return activation_function(tf.nn.conv2d(input, weights, strides=strides, padding='SAME') + biases)

    # 建立池化层
    def __pool(self, input, ksize, strides, mode="MAX"):
        # stride [1, x_movement, y_movement, 1]
        if mode == "MAX":
            return tf.nn.max_pool(input, ksize=ksize, strides=strides, padding='SAME')
        else:
            return tf.nn.avg_pool(input, ksize=ksize, strides=strides, padding='SAME')

    # 建立全连接层
    def __full_connection(self, input, in_size, out_size, activation_function=tf.nn.relu, name_scope="full_connection"):
        with tf.name_scope(name_scope) as scope:
            weights = self.__weight_variable([in_size, out_size])
            biases = self.__bias_variable([out_size])
            wx_plus_b = tf.matmul(input, weights) + biases
            if activation_function is None:
                outputs = wx_plus_b
            else:
                outputs = activation_function(wx_plus_b)
            return outputs

    # 建立残差块
    def __residual_block(self, x, x_size, fx, fx_size):
        if x_size == fx_size:
            return tf.nn.relu(tf.add(fx, x))
        else:
            weights = self.__weight_variable([x_size, fx_size])
            return tf.nn.relu(tf.add(fx, tf.matmul(x, weights)))

    # 建立卷积残差块
    # def __conv2d_residual_block(self, x, fx, filters=None):
    #     if filters is None:  ##无需改变维度
    #         return tf.nn.relu(tf.add(fx, x))
    #     else:  ##要改变维度，再相加
    #         W = self.__weight_variable(shape=filters)
    #         return tf.nn.relu(tf.add(fx, tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')))

    # 建立K线的卷积神经网络
    def __kline_cnn(self, input, input_xsize, output_xsize, name_scope="kline_cnn"):
        with tf.name_scope(name_scope) as scope:
            input = tf.reshape(input, [-1, input_xsize, const.KLINE_SIZE, 1])
            # 卷积层一
            h_conv_1 = self.__conv2d(input, filters=[1, const.KLINE_SIZE, 1, 32], strides=[1, 1, const.KLINE_SIZE, 1],
                                     name_scope="conv2d_1")  # -1,k,1,32
            h_conv_1 = tf.reshape(h_conv_1, [-1, input_xsize, 32, 1])  # -1,k,32,1
            # 卷积层二
            h_conv_2 = self.__conv2d(h_conv_1, filters=[3, 1, 1, 8], strides=[1, 1, 1, 1],
                                     name_scope="conv2d_2")  # -1,k,32,8
            # 卷积层三
            h_conv_3 = self.__conv2d(h_conv_1, filters=[5, 1, 1, 8], strides=[1, 1, 1, 1],
                                     name_scope="conv2d_3")  # -1,k,32,8
            h_conv_all = tf.concat([h_conv_1, h_conv_2, h_conv_3], 3)
            ## 池化层一
            h_pool_1 = self.__pool(h_conv_all, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1])  # -1,k/2,32,17

            ## 全连接层一
            fc1_x_size = math.ceil(input_xsize / 2) * 32 * 17
            fc1_y_size = 256 * int(input_xsize / 5)
            h_pool_1 = tf.reshape(h_pool_1, [-1, fc1_x_size])
            h_fc1 = self.__full_connection(h_pool_1, fc1_x_size, fc1_y_size, name_scope="full_conn_1")
            h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
            ## 全连接层二 ##
            return self.__full_connection(h_fc1_drop, fc1_y_size, output_xsize, name_scope="full_conn_2")
