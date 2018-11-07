import tensorflow as tf

from base_net import BaseNet


class DqnNet(BaseNet):
    def predict_profit_value(self, packaged_data):
        feed_dict = self.fill_feed_dict(packaged_data)
        profit_value = self.sess.run(self.profit_value, feed_dict=feed_dict)
        return profit_value

    def predic_winlose_probs(self, packaged_data):
        feed_dict = self.fill_feed_dict(packaged_data)
        winlose_probs = self.sess.run(self.winlose_probs, feed_dict=feed_dict)
        return winlose_probs

    def fill_feed_dict(self, packaged_data, input_profit_value=None, input_winlose_onehot=None, keep_prob=1.0):
        feed_dict = BaseNet.fill_feed_dict(self, packaged_data, keep_prob)
        if not (input_profit_value is None):
            feed_dict[self.input_profit_value] = input_profit_value
        if not (input_winlose_onehot is None):
            feed_dict[self.input_winlose_onehot] = input_winlose_onehot
        return feed_dict

    def profit_loss(self):
        return tf.reduce_mean(tf.reduce_sum(tf.square(self.input_profit_value - self.profit_value), axis=1), axis=0)

    def winlose_loss(self):
        return -1 * tf.reduce_mean(
            tf.reduce_sum((self.winlose_probs * self.input_winlose_onehot), axis=1),
            axis=0)

    def _placeholder_inputs(self):
        BaseNet._placeholder_inputs(self)
        self.input_profit_value = tf.placeholder(tf.float32, shape=[None, 1])
        self.input_winlose_onehot = tf.placeholder(tf.float32, shape=[None, 2])

    def _inference(self):
        BaseNet._inference(self)
        with tf.name_scope("profit_region") as scope:
            profit_h_gfc1 = self._full_connection(self.input_gfc1, 36, 64, name_scope="profit_h_gfc1")
            profit_h_gfc2 = self._full_connection(profit_h_gfc1, 64, 32, name_scope="profit_h_gfc2")
            profit_h_gfc3 = self._full_connection(profit_h_gfc2, 32, 16, name_scope="profit_h_gfc2")
            self.profit_value = self._full_connection(profit_h_gfc3, 16, 1, activation_function=None,
                                                      name_scope="profit_output")
        with tf.name_scope("winlose_region") as scope:
            winlose_h_gfc1 = self._full_connection(self.input_gfc1, 36, 64, name_scope="winlose_h_gfc1")
            winlose_h_gfc2 = self._full_connection(winlose_h_gfc1, 64, 32, name_scope="winlose_h_gfc2")
            winlose_h_gfc3 = self._full_connection(winlose_h_gfc2, 32, 16, name_scope="winlose_h_gfc2")
            self.winlose_probs = self._full_connection(winlose_h_gfc3, 16, 2, activation_function=tf.nn.softmax,
                                                       name_scope="winlose_output")
