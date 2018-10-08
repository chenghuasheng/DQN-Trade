import tensorflow as tf
import const
from base_net import BaseNet


class ActorCriticNet(BaseNet):
    def __init__(self, sess, trainable=False):
        BaseNet.__init__(self, sess, trainable)
        if self.trainable:
            self.actor_loss = self._actor_loss()
            actor_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_region')
            self.actor_trainer = self._trainer(self.actor_loss, 0.001, var_list=actor_train_vars)
            self.critic_loss = self._critic_loss()
            self.critic_trainer = self._trainer(self.critic_loss, 0.001)

    def learn(self, packaged_data, input_value, input_action_value, input_action, keep_prob=1.0):
        feed_dict = self.fill_feed_dict(packaged_data, input_value, input_action_value, input_action, keep_prob)
        _, critic_loss, _, actor_loss = self.sess.run(
            [self.critic_trainer, self.critic_loss, self.actor_trainer, self.actor_loss], feed_dict=feed_dict)
        return critic_loss, actor_loss

    def _actor_loss(self):
        return -1 * tf.reduce_mean(
            tf.reduce_sum(self.action_probs * (self.input_action_value - self.value) * self.input_action, axis=1),
            axis=0)

    def _critic_loss(self):
        return tf.reduce_mean(tf.square(self.input_value - self.value), axis=0)

    def predict_value(self, packaged_data):
        feed_dict = self.fill_feed_dict(packaged_data)
        value = self.sess.run(self.value, feed_dict=feed_dict)
        return value

    def predict_action_probs(self, packaged_data):
        feed_dict = self.fill_feed_dict(packaged_data)
        result = self.sess.run(self.action_probs, feed_dict=feed_dict)
        return result

    def fill_feed_dict(self, packaged_data, input_value=None, input_action_value=None, input_action=None,
                       keep_prob=1.0):
        feed_dict = BaseNet.fill_feed_dict(self, packaged_data, keep_prob)
        feed_dict[self.input_holding_period] = packaged_data.holding_period
        if not (input_value is None):
            feed_dict[self.input_value] = input_value
        if not (input_action_value is None):
            feed_dict[self.input_action_value] = input_action_value
        if not (input_action is None):
            feed_dict[self.input_action] = input_action
        return feed_dict

    def _placeholder_inputs(self):
        BaseNet._placeholder_inputs(self)
        self.input_holding_period = tf.placeholder(tf.float32, shape=[None, 1])
        self.input_action_value = tf.placeholder(tf.float32, shape=[None, 1])
        self.input_action = tf.placeholder(tf.float32, shape=[None, 2])
        self.input_value = tf.placeholder(tf.float32, shape=[None, 1])

    def _inference(self):
        with tf.name_scope("input_region") as scope:
            index_daily_cnn = self._kline_cnn(self.input_index_daily, const.INDEX_DAILY_PERIOD, 8,
                                              name_scope="index_daily_cnn")
            index_min5_cnn = self._kline_cnn(self.input_index_min5, const.INDEX_MIN5_PERIOD, 8,
                                             name_scope="index_min5_cnn")
            stock_daily_cnn = self._kline_cnn(self.input_stock_daily, const.STOCK_DAILY_PERIOD, 32,
                                              name_scope="stock_daily_cnn")
            stock_min5_cnn = self._kline_cnn(self.input_stock_min5, const.STOCK_MIN5_PERIOD, 12,
                                             name_scope="stock_min5_cnn")
            input_gfc1 = tf.concat(
                [index_daily_cnn, index_min5_cnn, stock_daily_cnn, stock_min5_cnn, self.input_holding_period], 1)

        with tf.name_scope("actor_region") as scope:
            actor_h_gfc1 = self._full_connection(input_gfc1, 61, 64, name_scope="actor_h_gfc1")
            actor_h_gfc2 = self._full_connection(actor_h_gfc1, 64, 32, name_scope="actor_h_gfc2")
            actor_h_gfc3 = self._full_connection(actor_h_gfc2, 32, 16, name_scope="actor_h_gfc2")
            self.action_probs = self._full_connection(actor_h_gfc3, 16, 2, activation_function=tf.nn.softmax,
                                                      name_scope="actor_output")
        with tf.name_scope("critic_region") as scope:
            critic_h_gfc1 = self._full_connection(input_gfc1, 61, 64, name_scope="critic_h_gfc1")
            critic_h_gfc2 = self._full_connection(critic_h_gfc1, 64, 32, name_scope="critic_h_gfc2")
            critic_h_gfc3 = self._full_connection(critic_h_gfc2, 32, 16, name_scope="critic_h_gfc2")
            self.value = self._full_connection(critic_h_gfc3, 16, 1, activation_function=tf.nn.tanh,
                                               name_scope="critic_output")
