import argparse
import math
import os
import sys
import time

import tensorflow as tf

from dqn_net import DqnNet
from trade_input_data import *

const.EVAL_BATCH_SIZE = 2000  # 评估时每批的数据量
const.DISCOUNT = 0.9  # 回报折扣率

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FLAGS = None


class TradeLearning(object):
    def __init__(self):
        # 运行的会话
        self.sess = tf.Session()
        self.dqn_net = DqnNet(self.sess, True)

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

        train_step = tf.Variable(0, trainable=False, name="train_step")
        learning_rate_1 = tf.train.exponential_decay(0.001, train_step * batch_size, train_size, 0.98)
        tf.summary.scalar('learning_rate_1', learning_rate_1)
        profit_loss = self.dqn_net.profit_loss()
        tf.summary.scalar('profit_loss', profit_loss)
        optimizer_1 = tf.train.AdamOptimizer(learning_rate_1)
        profit_trainer = optimizer_1.minimize(profit_loss, global_step=train_step, var_list=None)

        learning_rate_2 = tf.train.exponential_decay(0.001, train_step * batch_size, train_size, 0.98)
        tf.summary.scalar('learning_rate_2', learning_rate_2)
        winlose_loss = self.dqn_net.winlose_loss()
        tf.summary.scalar('winlose_loss', winlose_loss)
        optimizer_2 = tf.train.AdamOptimizer(learning_rate_2)
        rain_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='winlose_region')
        winlose_trainer = optimizer_2.minimize(winlose_loss, var_list=rain_vars)

        summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(const.LOG_DIR, self.sess.graph)

        # 初始化变量
        init = tf.global_variables_initializer()
        self.sess.run(init)
        # 增量训练，会读出之前训练保存的变量
        if increment:
            self.dqn_net.restore()
        # 自动确定训练步数
        max_steps = math.ceil(train_size * num_epochs / batch_size)

        # 开始训练
        for step in range(max_steps):
            start_time = time.time()
            # 获取一批特征数据，并生成标签数据
            batch_data = TradeInputData.next_batch_data(batch_size)
            batch_packaged_data = TradeInputData.package_data(batch_data)
            batch_profit = TradeInputData.pick_up_reward(batch_data)
            batch_winlose_onehot = self.get_winlose_onehot(batch_profit)
            batch_profit_label = batch_profit.reshape((-1, 1))

            # 训练网络
            feed_dict = self.dqn_net.fill_feed_dict(batch_packaged_data, batch_profit_label, batch_winlose_onehot, 0.8)
            _, profit_loss_val = self.sess.run([profit_trainer, profit_loss], feed_dict=feed_dict)
            _, winlose_loss_val = self.sess.run([winlose_trainer, winlose_loss], feed_dict=feed_dict)

            duration = time.time() - start_time

            # 每100步训练输出一次损失值和时间

            if step % 100 == 0:
                print('Step %d: profit_loss = %.6f , winlose_loss = %.6f (%.3f sec)' % (
                    step, profit_loss_val, winlose_loss_val, duration))
                summary_str = self.sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()
            
            if (step % 1000 == 0 and step > 0) or (step + 1 == max_steps):
                self._do_eval(TradeInputData.all_train_data)

        # 训练结束后保存变量
        self.dqn_net.save()

    def get_winlose_onehot(self, batch_profit):
        batch_winlose = []
        for p in batch_profit:
            if p > 0:
                batch_winlose.append(0)
            else:
                batch_winlose.append(1)
        m = np.array(batch_winlose).shape[0]
        one_hot = np.zeros(shape=[m, 2])
        for i, j in enumerate(batch_winlose):
            one_hot[i, j] = 1
        return one_hot

    def _do_eval(self, data):
        # 由于一次性计算评估数据，内存消耗太大，故而改分批处理
        num = len(data)
        profit_prediction = np.array([]).reshape((-1, 1))
        profit_label = np.array([]).reshape((-1, 1))
        winlose_onehot = np.array([]).reshape((-1, 2))
        winlose_probs_prediction = np.array([]).reshape((-1, 2))

        for i in range(0, num // const.EVAL_BATCH_SIZE + 1):
            offset = int(i * const.EVAL_BATCH_SIZE)
            batch_data = data[offset:(offset + const.EVAL_BATCH_SIZE)]
            batch_packaged_data = TradeInputData.package_data(batch_data)
            profit_batch_prediction = self.dqn_net.predict_profit_value(batch_packaged_data)
            profit_prediction = np.concatenate((profit_prediction, profit_batch_prediction), axis=0)
            profit_batch = TradeInputData.pick_up_reward(batch_data)
            profit_batch_label = profit_batch.reshape((-1, 1))
            profit_label = np.concatenate((profit_label, profit_batch_label), axis=0)
            winlose_probs_batch_prediction = self.dqn_net.predic_winlose_probs(batch_packaged_data)
            winlose_probs_prediction = np.concatenate((winlose_probs_prediction, winlose_probs_batch_prediction),
                                                      axis=0)
            winlose_onehot_batch = self.get_winlose_onehot(profit_batch)
            winlose_onehot = np.concatenate((winlose_onehot, winlose_onehot_batch), axis=0)

        mse_profit = np.mean(np.sum(np.square(profit_prediction - profit_label), axis=1), axis=0)
        mean_profit_label = np.mean(profit_label, axis=0)
        var_profit = np.mean(np.sum(np.square(profit_label - mean_profit_label), axis=1), axis=0)
        precision_profit = 1 - mse_profit / var_profit
        print("The Astringency Of Profit :")
        print('Num examples: %d  Precision @ 1: %0.06f MSE @ 2: %0.06f' % (num, precision_profit, mse_profit))

        winlose_loss = -1 * np.mean(np.sum((winlose_probs_prediction * winlose_onehot), axis=1),
                                    axis=0)
        mean_winlose_probs = np.mean(winlose_probs_prediction, axis=0)
        var_winlose_probs = np.mean(np.sum(np.square(winlose_probs_prediction - mean_winlose_probs), axis=1), axis=0)
        print("the last winlose loss is %0.06f" % winlose_loss)
        print(mean_winlose_probs, var_winlose_probs)


def main(_):
    tlg = TradeLearning()
    tlg.run_training(FLAGS.begin_date, FLAGS.end_date, FLAGS.num_epochs, FLAGS.batch_size, FLAGS.increment)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('begin_date', type=str, default="0000-00-00", help="begin date of the data to training")
    parser.add_argument('end_date', type=str, default="9999-99-99", help="end date of the data to training")
    parser.add_argument('-n', '--num_epochs', type=int, default=100, help="number of training epochs")
    parser.add_argument('-b', '--batch_size', type=int, default=64, help="size of each batch data")
    parser.add_argument('-i', '--increment', nargs='?', type=bool, default=False, help="if use increment-training")
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
