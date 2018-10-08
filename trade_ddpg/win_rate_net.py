import tensorflow as tf
import const
from base_net import BaseNet


class WinRateNet(BaseNet):
    def _placeholder_inputs(self):
        BaseNet._placeholder_inputs(self)
        self.input_profit = tf.placeholder(tf.float32, shape=[None, 1])

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
                [index_daily_cnn, index_min5_cnn, stock_daily_cnn, stock_min5_cnn], 1)
            with tf.name_scope("rate_region") as scope:
                actor_h_gfc1 = self._full_connection(input_gfc1, 60, 64, name_scope="actor_h_gfc1")
                actor_h_gfc2 = self._full_connection(actor_h_gfc1, 64, 32, name_scope="actor_h_gfc2")
                actor_h_gfc3 = self._full_connection(actor_h_gfc2, 32, 16, name_scope="actor_h_gfc2")
                self.action_probs = self._full_connection(actor_h_gfc3, 16, 2, activation_function=tf.nn.softmax,
                                                          name_scope="actor_output")
