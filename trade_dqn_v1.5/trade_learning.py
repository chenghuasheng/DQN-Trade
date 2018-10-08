import tensorflow as tf
import math
import os
import time
import const
from trade_input_data import *
from trade_dqn import trade_dqn

const.BATCH_SIZE = 64
const.MAX_STEPS = 15000
const.STEPS_FOR_COPYNET=50
const.DISCOUNT = 0.9
const.LOG_DIR = 'log'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class trade_learning(object):
    def __init__(self, use_double=False):
        self.use_double = use_double  ## 是否使用双网络
        self.eval_net = trade_dqn()
        if self.use_double:
            self.target_net = trade_dqn()
        ##要保存的变量字典
        saver_dict = {}
        for v in self.eval_net.variables:
            saver_dict[v.name] = v
        self.saver = tf.train.Saver(saver_dict)
        ## 运行的会话
        self.sess = tf.Session()

    def get_trade_return(self, data):
        qval = self.__prediction(data)
        return math.exp(qval)

    def run_training(self, begin_date_string, end_date_string, increment=False, auto_steps=False):
        ## 定义损失函数、学习率和训练者
        loss = self.eval_net.loss()
        self.global_step = tf.Variable(0, name='global_step', trainable=False)  # 返回训练步数，此变量不参与训练
        learning_rate = tf.train.exponential_decay(0.01, self.global_step, 100, 0.98, staircase=True,
                                                   name="learning_rate")
        trainer = self.__trainer(loss, learning_rate)
        ## 初始化变量
        init = tf.global_variables_initializer()
        self.sess.run(init)
        ## 准备回放的数据
        trade_input_data.prepare_data(begin_date_string, end_date_string)
        ## 增量训练，会读出之前训练保存的变量
        if increment:
            checkpoint_dir = os.path.join(const.LOG_DIR)
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print("Continue training form model {}".format(ckpt.model_checkpoint_path))
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        ## 自动确定训练步数
        if auto_steps:
            sample_number = len(trade_input_data.all_data_list)
            steps = math.ceil(sample_number * 20 / const.BATCH_SIZE)
        else:
            steps = const.MAX_STEPS
        ## 双网络下定义拷贝变量的操作
        if self.use_double:
            copy_op = self.__replace_target_op()
        ## 开始训练
        for step in range(steps):
            ## 双网络下用eval网络变量更新target网络
            if self.use_double and (step % const.STEPS_FOR_COPYNET ==0):
                print('Copy eval net variables to target net')
                self.sess.run(copy_op)
                print('Copy finished.')

            start_time = time.time()
            ## 获取一批特征数据，并生成标签数据
            batch_data = trade_input_data.next_batch_data(const.BATCH_SIZE)
            batch_label = []
            for data in batch_data:
                batch_label.append(self.__get_qvalue(data))
            ## 填充字典，并开始一次训练
            feed_dict = self.__fill_dict(batch_data, batch_label, 1.0)
            _, loss_value = self.sess.run([trainer, loss], feed_dict=feed_dict)
            duration = time.time() - start_time

            if step % 100 == 0:
                print('Step %d: loss = %.6f (%.3f sec)' % (step, loss_value, duration))

            ## 进行一次评估
            if (step + 1) % 5000 == 0 or (step + 1) == steps:
                print('Training Data Eval:')
                self.__do_eval(trade_input_data.all_data_list)

        ## 训练结束后保存变量
        checkpoint_file = os.path.join(const.LOG_DIR, 'modkel.cpt')
        self.saver.save(self.sess, checkpoint_file)

    def __fill_dict(self, batch_data, batch_label=None,keep_prob=1.0,dqn_net=None):
        index_daily, index_min5, stock_daily, stock_min5, inside, holding_period = trade_input_data.pick_up(batch_data)
        if dqn_net is None:
            dqn_net=self.eval_net
        feed_dict = {
            dqn_net.input_index_daily: index_daily,
            dqn_net.input_index_min5: index_min5,
            dqn_net.input_stock_daily: stock_daily,
            dqn_net.input_stock_min5: stock_min5,
            dqn_net.input_inside: inside,
            dqn_net.input_holding_period: holding_period,
            dqn_net.keep_prob: keep_prob
        }
        if not (batch_label is None):
            batch_label = np.array(batch_label).reshape((-1, 1))
            feed_dict[dqn_net.input_label] = batch_label
        return feed_dict

    ## 计算当前状态的Q值
    def __get_qvalue(self, data):
        if not data['NextStates'] is None:
            next_max_qvalue = self.__get_next_max_qvalue(data['NextStates'])
            return data['Reward'] + const.DISCOUNT * next_max_qvalue
        else:
            return data['Reward']

    ## 所有后续状态的最大Q值,返回[ val ]
    def __get_next_max_qvalue(self, next_state_data):
        qvalue_eval = self.__prediction(next_state_data)
        if self.use_double:
            # 用eval网络的Q值来选择动作，然后用target网络计算这动作的Q值
            argmax = np.argmax(qvalue_eval,axis=0)
            qvalue_target = self.__prediction(next_state_data, self.target_net)
            return qvalue_target[argmax[0]]
        else:
            return np.max(qvalue_eval,0)

    def __trainer(self, loss, learning_rate):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step=self.global_step)
        return train_op

    def __replace_target_op(self):
        target_vars = self.target_net.variables
        eval_vars = self.eval_net.variables
        return [tf.assign(t, e) for t, e in zip(target_vars, eval_vars)]

    def __do_eval(self, batch_data):
        num = len(batch_data)
        batch_label = []
        for data in batch_data:
            batch_label.append(self.__get_qvalue(data))
        feed_dict = self.__fill_dict(batch_data, batch_label)
        eval_correct = self.eval_net.evaluation()
        eval_loss = self.eval_net.loss()
        precision, loss = self.sess.run([eval_correct, eval_loss], feed_dict=feed_dict)
        print('Num examples: %d Precision @ 1: %0.06f Loss @ 2: %0.06f' % (num, precision, loss))

    def __prediction(self, data, dqn_net=None):
        if dqn_net is None :
            dqn_net = self.eval_net
        feed_dict = self.__fill_dict(data,dqn_net=dqn_net)
        return self.sess.run(dqn_net.output, feed_dict=feed_dict)


tl = trade_learning(use_double=True)
tl.run_training("2018-08-01", "2018-08-17",increment=False,auto_steps=False)
