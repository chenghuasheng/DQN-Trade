import tensorflow as tf
import math
import argparse
import os
import sys
import time
from trade_input_data import *
from actor_critic_net import ActorCriticNet

const.EVAL_BATCH_SIZE = 2000  # 评估时每批的数据量
const.DISCOUNT = 0.9  # 回报折扣率

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FLAGS = None


class TradeLearning(object):
    def __init__(self):
        # 运行的会话
        self.sess = tf.Session()
        self.actor_critic_eval = ActorCriticNet(self.sess, True)
        self.actor_critic_target = ActorCriticNet(self.sess)

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

        # 初始化变量
        init = tf.global_variables_initializer()
        self.sess.run(init)
        # 增量训练，会读出之前训练保存的变量
        if increment:
            self.actor_critic_eval.restore()
            self.actor_critic_eval.replace_to(self.actor_critic_target, False)
        # 自动确定训练步数
        max_steps = math.ceil(train_size * num_epochs / batch_size)

        # 开始训练
        for step in range(max_steps):
            start_time = time.time()
            # 获取一批特征数据，并生成标签数据
            batch_data = TradeInputData.next_batch_data(batch_size)
            batch_packaged_data = TradeInputData.package_data(batch_data)
            action_probs = self.actor_critic_eval.predict_action_probs(batch_packaged_data)
            action = self.choose_action(action_probs)
            optimal_action = self.get_optimal_action(action_probs)
            action_qvalue, optimal_action_qvalue = self._get_qvalues(batch_data, action, optimal_action)

            # 训练网络
            critic_loss, actor_loss = self.actor_critic_eval.learn(batch_packaged_data, optimal_action_qvalue,
                                                                   action_qvalue, action, 1.0)
            duration = time.time() - start_time

            # 进行软更新
            if step % 10 == 0 and step > 0:
                self.actor_critic_eval.replace_to(self.actor_critic_target)

            # 每100步训练输出一次损失值和时间
            if step % 100 == 0:
                print('Step %d: actor_loss = %.6f , critic_loss = %.6f (%.3f sec)' % (
                    step, actor_loss, critic_loss, duration))
            if (step % 1000 == 0 and step > 0) or (step + 1 == max_steps):
                self._do_eval(TradeInputData.all_train_data)

        # 训练结束后保存变量
        self.actor_critic_eval.save()

    def choose_action(self, action_probs):
        action = np.array([np.random.choice(2, 1, p=j) for i, j in enumerate(action_probs)]).reshape((-1))
        return self.action_to_onehot(action)

    def get_optimal_action(self, action_probs):
        action = np.argmax(action_probs, axis=1)
        return self.action_to_onehot(action)

    def action_to_onehot(self, action):
        m = action.shape[0]
        one_hot = np.zeros(shape=[m, 2])
        for i, j in enumerate(action):
            one_hot[i, j] = 1
        return one_hot

    # 获取当前动作下的期望回报
    def _get_qvalues(self, batch_data, action, optimal_action):
        # 当前状态下所有动作的奖励
        rewards = TradeInputData.pick_up_rewards(batch_data)
        # 根据状态在获取下一状态，由于有些状态是没有下一状态的，但为了计算统一插入空状态作为
        # 下一状态，同时返回flag标志
        next_state, flag = TradeInputData.pick_up_next_states(batch_data)
        next_state_packaged_data = TradeInputData.package_data(next_state)
        next_state_value = self.actor_critic_target.predict_value(next_state_packaged_data)
        next_state_value = next_state_value.reshape((-1))
        gamma_return = const.DISCOUNT * next_state_value * flag
        n = len(gamma_return)
        all_qvalues = (rewards + np.column_stack((gamma_return.reshape((-1, 1)), np.zeros(n))))
        action_qvalue = np.sum(all_qvalues * action, axis=1)
        optimal_action_qvalue = np.sum(all_qvalues * optimal_action, axis=1)
        return action_qvalue.reshape((-1, 1)), optimal_action_qvalue.reshape((-1, 1))

    def _do_eval(self, data):
        # 由于一次性计算评估数据，内存消耗太大，故而改分批处理
        num = len(data)
        actor_prediction = np.array([]).reshape((-1, 2))
        actor_label = np.array([]).reshape((-1, 2))
        critic_prediction = np.array([]).reshape((-1, 1))
        critic_label = np.array([]).reshape((-1, 1))

        for i in range(0, num // const.EVAL_BATCH_SIZE + 1):
            offset = int(i * const.EVAL_BATCH_SIZE)
            batch_data = data[offset:(offset + const.EVAL_BATCH_SIZE)]
            batch_packaged_data = TradeInputData.package_data(batch_data)
            actor_batch_prediction = self.actor_critic_eval.predict_action_probs(batch_packaged_data)
            actor_prediction = np.concatenate((actor_prediction, actor_batch_prediction), axis=0)
            actor_batch_label = self.actor_critic_target.predict_action_probs(batch_packaged_data)
            actor_label = np.concatenate((actor_label, actor_batch_label), axis=0)

            critic_batch_prediction = self.actor_critic_eval.predict_value(batch_packaged_data)
            critic_prediction = np.concatenate((critic_prediction, critic_batch_prediction), axis=0)
            critic_batch_label = self.actor_critic_target.predict_value(batch_packaged_data)
            critic_label = np.concatenate((critic_label, critic_batch_label), axis=0)

        mse_critic = np.mean(np.sum(np.square(critic_prediction - critic_label), axis=1), axis=0)
        mean_critic_label = np.mean(critic_label, axis=0)
        var_critic = np.mean(np.sum(np.square(critic_label - mean_critic_label), axis=1), axis=0)
        precision_critic = 1 - mse_critic / var_critic

        print("The Astringency Of Critic Eval Net :")
        print('Num examples: %d  Precision @ 1: %0.06f MSE @ 2: %0.06f' % (num, precision_critic, mse_critic))

        mse_actor = np.mean(np.sum(np.square(actor_prediction - actor_label), axis=1), axis=0)
        mean_actor_label = np.mean(actor_label, axis=0)

        var_actor = np.mean(np.sum(np.square(actor_label - mean_actor_label), axis=1), axis=0)
        print(mse_actor, mean_actor_label, var_actor)
        precision_actor = 1 - mse_actor / var_actor
        print("The Astringency Of Actor Eval Net :")
        print('Num examples: %d  Precision @ 1: %0.06f MSE @ 2: %0.06f' % (num, precision_actor, mse_actor))


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
