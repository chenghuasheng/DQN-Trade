import tensorflow as tf
from trade_input_data import *
from trade_dqn import TradeDqn
import argparse
import sys
import os

FLAGS = None

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class TradePrediction(object):
    def __init__(self):
        self.sess = tf.Session()
        self.dqn_net = TradeDqn(self.sess)
        # 初始化变量
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.dqn_net.restore()

    def run_predicting(self, date_string):
        print("Prepare predicting data ...")
        TradeInputData.prepare_predicting_data(date_string)
        if len(TradeInputData.all_predicting_data) > 0:
            print("Begin predicting ...")
            packaged_data = TradeInputData.package_data(TradeInputData.all_predicting_data)
            qvalues = self.dqn_net.prediction(packaged_data)
            print("Prediction finished.")
            TradeInputData.update_predicted_qvalues(qvalues)
            print("Update QValues finished.")
        else:
            print("Has no data.")


def main(_):
    tpn = TradePrediction()
    tpn.run_predicting(FLAGS.date)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('date', type=str, default="0000-00-00", help="the date of data want to predicting")
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
