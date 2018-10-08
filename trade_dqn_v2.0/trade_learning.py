import tensorflow as tf
import math
import argparse
import os
import sys
import time
from trade_input_data import *
from trade_dqn import TradeDqn
from tensorflow.python import debug as tfdbg

const.EVAL_BATCH_SIZE = 2000  # 评估时每批的数据量
const.DISCOUNT = 0.9  # 回报折扣率

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FLAGS = None


class TradeLearning(object):
    def __init__(self, use_double=False):
        self.use_double = use_double  # 是否使用双网络
        # 运行的会话
        self.sess = tf.Session()
        self.eval_net = TradeDqn(self.sess)
        if self.use_double:
            self.target_net = TradeDqn(self.sess)

        self.global_step = tf.Variable(0, trainable=False, name="global_step")

    def run_training(self, begin_date_string, end_date_string, num_epochs=100, batch_size=64, increment=False):
        # 准备回放的数据
        print("Prepare Training Data ...")
        TradeInputData.prepare_train_data(begin_date_string, end_date_string)
        train_size = len(TradeInputData.all_train_data)
        if train_size <= 0:
            print("Has no data.")
            return
        else:
            print("Data readied.")
        # 定义损失函数、学习率和训练者
        loss = self.eval_net.loss()
        learning_rate = tf.train.exponential_decay(0.001, self.global_step * batch_size, train_size, 0.99)
        trainer = self.__trainer(loss, learning_rate)

        # self.sess = tfdbg.LocalCLIDebugWrapperSession(self.sess)  # 被调试器封装的会话
        # self.sess.add_tensor_filter("has_inf_or_nan", tfdbg.has_inf_or_nan)
        # 初始化变量
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # 增量训练，会读出之前训练保存的变量
        if increment:
            self.eval_net.restore()

        # 自动确定训练步数
        max_steps = math.ceil(train_size * num_epochs / batch_size)

        # 开始训练
        for step in range(max_steps):
            # 双网络下用eval网络变量更新target网络
            if self.use_double and (step * batch_size % train_size < batch_size):
                # print('Copy eval net variables to target net')
                self.eval_net.replace_to(self.target_net)
                # print('Copy finished.')

            start_time = time.time()
            # 获取一批特征数据，并生成标签数据
            batch_data = TradeInputData.next_batch_data(batch_size)
            batch_packaged_data = TradeInputData.package_data(batch_data)
            batch_label = self.__get_qvalues(batch_data)
            # 填充数据，并开始一步次训练
            feed_dict = self.eval_net.fill_feed_dict(batch_packaged_data, batch_label, 1.0)
            _, loss_value = self.sess.run([trainer, loss], feed_dict=feed_dict)
            duration = time.time() - start_time
            # 每100步训练输出一次损失值和时间
            if step % 100 == 0:
                print('Step %d: loss = %.6f (%.3f sec)' % (step, loss_value, duration))

            # 每1000步训练进行一次评估
            if ((step + 1) % 1000 == 0) or (step + 1) == max_steps:
                print('Training Data Eval:')
                self.__do_eval(TradeInputData.all_train_data)

        # 训练结束后保存变量
        self.eval_net.save()

    # 计算当前状态的Q值
    def __get_qvalues(self, data):
        next_states, state_flags = TradeInputData.pick_up_next_states(data)
        next_max_qvalues = self.__get_next_max_qvalues(next_states)
        next_max_qvalues = next_max_qvalues * state_flags
        add = (next_max_qvalues * const.DISCOUNT).reshape((-1, 1))
        zeros = np.zeros(len(data))
        rewards = TradeInputData.pick_up_rewards(data)
        rewards = rewards + np.column_stack((add, zeros))
        return rewards

    # 所有后续状态的最大Q值,返回[ val ]
    def __get_next_max_qvalues(self, next_states):
        packaged_data = TradeInputData.package_data(next_states)
        qvalues_eval = self.eval_net.prediction(packaged_data)
        if self.use_double:
            argmax = np.argmax(qvalues_eval, axis=1)
            qvalues_target = self.target_net.prediction(packaged_data)
            return np.array([qvalues_target[i, j] for i, j in enumerate(argmax)])
        else:
            return np.max(qvalues_eval, axis=1)

    def __trainer(self, loss, learning_rate):
        # optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step=self.global_step)
        return train_op

    def __do_eval(self, data):
        # 由于一次性计算评估数据，内存消耗太大，故而改分批处理
        num = len(data)
        data_prediction = np.array([]).reshape((-1, 2))
        data_label = np.array([]).reshape((-1, 2))
        for i in range(0, math.ceil(num // const.EVAL_BATCH_SIZE)):
            offset = int(i * const.EVAL_BATCH_SIZE)
            batch_data = data[offset:(offset + const.EVAL_BATCH_SIZE)]
            batch_packaged_data = TradeInputData.package_data(batch_data)
            batch_prediction = self.eval_net.prediction(batch_packaged_data)
            data_prediction = np.concatenate((data_prediction, batch_prediction), axis=0)
            batch_label = self.__get_qvalues(batch_data)
            data_label = np.concatenate((data_label, batch_label), axis=0)

        mse = np.mean(np.sum(np.square(data_prediction - data_label), axis=1), axis=0)
        mean_label = np.mean(data_label, axis=0)
        var = np.mean(np.sum(np.square(data_label - mean_label), axis=1), axis=0)
        precision = 1 - mse / var
        print('Num examples: %d Precision @ 1: %0.06f Loss @ 2: %0.06f' % (num, precision, mse))


def main(_):
    tlg = TradeLearning(FLAGS.double_net)
    tlg.run_training(FLAGS.begin_date, FLAGS.end_date, FLAGS.num_epochs, FLAGS.batch_size, FLAGS.increment)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('begin_date', type=str, default="0000-00-00", help="begin date of the data to training")
    parser.add_argument('end_date', type=str, default="9999-99-99", help="end date of the data to training")
    parser.add_argument('-n', '--num_epochs', type=int, default=100, help="number of training epochs")
    parser.add_argument('-b', '--batch_size', type=int, default=64, help="size of each batch data")
    parser.add_argument('-i', '--increment', nargs='?', type=bool, default=False, help="if use increment-training")
    parser.add_argument('-d', '--double_net', nargs='?', type=bool, default=True, help="if use two net")
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
