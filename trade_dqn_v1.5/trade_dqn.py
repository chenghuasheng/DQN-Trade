import tensorflow as tf
import const
import math

const.KLINE_SIZE = 5


class trade_dqn(object):
    def __init__(self):
        self.variables = []
        self.__placeholder_inputs()
        self.__inference()

    def loss(self):
        return tf.reduce_mean(tf.square(self.output - self.input_label))

    def evaluation(self):
        mse = tf.reduce_mean(tf.square(self.output - self.input_label))
        label_mean = tf.reduce_mean(self.input_label)
        var = tf.reduce_mean(tf.square(self.input_label - label_mean))
        return 1 - mse / var

    def __placeholder_inputs(self):
        self.keep_prob = tf.placeholder(tf.float32)  # 用于防止过拟合的参数
        self.input_index_daily = tf.placeholder(tf.float32, shape=[None, const.INDEX_DAILY_PERIOD, const.KLINE_SIZE])
        self.input_index_min5 = tf.placeholder(tf.float32, shape=[None, const.INDEX_MIN5_PERIOD, const.KLINE_SIZE])
        self.input_stock_daily = tf.placeholder(tf.float32, shape=[None, const.STOCK_DAILY_PERIOD, const.KLINE_SIZE])
        self.input_stock_min5 = tf.placeholder(tf.float32, shape=[None, const.STOCK_MIN5_PERIOD, const.KLINE_SIZE])
        self.input_inside = tf.placeholder(tf.float32, shape=[None, 1])
        self.input_holding_period = tf.placeholder(tf.float32, shape=[None, 1])
        self.input_label = tf.placeholder(tf.float32, shape=[None, 1])

    def __inference(self):
        with tf.name_scope("global_full_connection_1") as scope:
            index_daily_cnn, size1 = self.__kline_cnn(self.input_index_daily, const.INDEX_DAILY_PERIOD,
                                                      name_scope="index_daily_cnn")
            index_min5_cnn, size2 = self.__kline_cnn(self.input_index_min5, const.INDEX_MIN5_PERIOD,
                                                     name_scope="index_min5_cnn")
            stock_daily_cnn, size3 = self.__kline_cnn(self.input_stock_daily, const.STOCK_DAILY_PERIOD,
                                                      name_scope="stock_daily_cnn")
            stock_min5_cnn, size4 = self.__kline_cnn(self.input_stock_min5, const.STOCK_MIN5_PERIOD,
                                                     name_scope="stock_min5_cnn")
            w1_gfc1 = self.__weight_variable([size1, 16])
            w2_gfc1 = self.__weight_variable([size2, 16])
            w3_gfc1 = self.__weight_variable([size3, 16])
            w4_gfc1 = self.__weight_variable([size4, 16])
            w5_gfc1 = self.__weight_variable([1, 16])
            w6_gfc1 = self.__weight_variable([1, 16])
            b_gfc1 = self.__bias_variable([16])
            h_gfc1 = tf.nn.relu(tf.matmul(index_daily_cnn, w1_gfc1) + tf.matmul(index_min5_cnn, w2_gfc1) +
                                tf.matmul(stock_daily_cnn, w3_gfc1) + tf.matmul(stock_min5_cnn, w4_gfc1) +
                                tf.matmul(self.input_holding_period, w5_gfc1) + tf.matmul(self.input_inside,
                                                                                          w6_gfc1) + b_gfc1)
            h_gfc2 = self.__full_connection(h_gfc1, 16, 8, name_scope="h_gfc2")
            h_gfc3 = self.__full_connection(h_gfc2, 8, 4, activation_function=None, name_scope="h_gfc3")
            h_rb = self.__residual_block(h_gfc1, 16, h_gfc3, 4)
            self.output = self.__full_connection(h_rb, 4, 1, activation_function=tf.nn.tanh, name_scope="output")

    # 建立权重变量
    def __weight_variable(self, shape, name='weights'):
        initial = tf.truncated_normal(shape, stddev=0.1)
        # initial = tf.zeros(shape=shape)
        var = tf.Variable(initial, name=name)
        self.variables.append(var)
        return var

    # 建立偏置变量
    def __bias_variable(self, shape, name='biases'):
        initial = tf.constant(0.1, shape=shape)
        # initial = tf.constant(0.001, shape=shape)
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
            Wx_plus_b = tf.matmul(input, weights) + biases
            if activation_function is None:
                outputs = Wx_plus_b
            else:
                outputs = activation_function(Wx_plus_b)
            return outputs

    # 建立残差块
    def __residual_block(self, x, x_size, fx, fx_size):
        if x_size == fx_size:
            return tf.nn.relu(tf.add(fx, x))
        else:
            weights = self.__weight_variable([x_size, fx_size])
            return tf.nn.relu(tf.add(fx, tf.matmul(x, weights)))

    # # 建立卷积残差块
    # def __conv2d_residual_block(self, x, fx, filters=None):
    #     if filters is None:  ##无需改变维度
    #         return tf.nn.relu(tf.add(fx, x))
    #     else:  ##要改变维度，再相加
    #         W = self.__weight_variable(shape=filters)
    #         return tf.nn.relu(tf.add(fx, tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')))

    # 建立K线的卷积神经网络
    def __kline_cnn(self, input, k, name_scope="kline_cnn"):
        with tf.name_scope(name_scope) as scope:
            input = tf.reshape(input, [-1, k, const.KLINE_SIZE, 1])
            # 卷积层一
            h_conv_1 = self.__conv2d(input, filters=[1, const.KLINE_SIZE, 1, 32], strides=[1, 1, const.KLINE_SIZE, 1],
                                     name_scope="conv2d_1")  # -1,k,1,32
            h_conv_1 = tf.reshape(h_conv_1, [-1, k, 32, 1])
            # 卷积层二
            h_conv_2 = self.__conv2d(h_conv_1, filters=[3, 1, 1, 8], strides=[1, 1, 1, 1],
                                     name_scope="conv2d_2")  # -1,k,32,8
            ## 池化层一
            h_pool_1 = self.__pool(h_conv_2, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1])  # -1,k/2,32,8
            ## 全连接层一
            fc1_x_size = math.ceil(k / 2) * 32 * 8
            fc1_y_size = 256 * int(k / 5)
            h_pool_1 = tf.reshape(h_pool_1, [-1, fc1_x_size])
            h_fc1 = self.__full_connection(h_pool_1, fc1_x_size, fc1_y_size, name_scope="full_conn_1")
            h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
            ## 全连接层二 ##
            fc2_y_size = int(k / 5) * 2
            return self.__full_connection(h_fc1_drop, fc1_y_size, fc2_y_size, name_scope="full_conn_2"), fc2_y_size
