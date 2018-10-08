import pymongo
import numpy as np
import const
import win_unicode_console

const.INDEX_DAILY_PERIOD = 10
const.INDEX_MIN5_PERIOD = 10
const.STOCK_DAILY_PERIOD = 40
const.STOCK_MIN5_PERIOD = 10
const.KLINE_SIZE = 5

win_unicode_console.enable()


class PackagedData(object):
    pass


class TradeInputData(object):
    all_train_data = []
    all_predicting_data = []

    @classmethod
    def prepare_train_data(cls, begin_date_string, end_date_string):
        client = pymongo.MongoClient("localhost", 27017)
        db = client.FinanceLast
        col = db["RandomTradeExperiencePool"]
        data_dict = {}
        for r in col.find({'Date': {'$gte': begin_date_string, '$lte': end_date_string}}):
            if not cls.check_data(r):
                continue
            cls.all_train_data.append(r)
            key = r['Date'] + r['Symbol'] + str(r['HoldingPeriod'])
            data_dict[key] = r

        for r in cls.all_train_data:
            if r['NextTradeDate'] != '':
                key = r['NextTradeDate'] + r['Symbol'] + str(r['HoldingPeriod'] + 1)
                r['NextStates'] = data_dict.get(key)
            else:
                r['NextStates'] = None

    @classmethod
    def next_batch_data(cls, batch_size):
        result = []
        n = len(cls.all_train_data)
        if n > 0:
            random_nums = np.random.randint(0, n, batch_size)
            for k in random_nums:
                r = cls.all_train_data[k]
                result.append(r)
        return result

    @classmethod
    def package_data(cls, ori_data):
        pkd = PackagedData()
        pkd.index_daily = np.array([r["IndexDaily"] for r in ori_data])
        pkd.index_min5 = np.array([r["IndexMin5"] for r in ori_data])
        pkd.stock_daily = np.array([r["StockDaily"] for r in ori_data])
        pkd.stock_min5 = np.array([r["StockMin5"] for r in ori_data])
        pkd.holding_period = np.array([r["HoldingPeriod"] for r in ori_data]).reshape((-1, 1))
        return pkd

    @classmethod
    def pick_up_rewards(cls, ori_data):
        return np.array([r["Reward"] for r in ori_data])

    @classmethod
    def pick_up_next_states(cls, ori_data):
        empty_state = {
            'IndexDaily': np.zeros(shape=(const.INDEX_DAILY_PERIOD, const.KLINE_SIZE)),
            'IndexMin5': np.zeros(shape=(const.INDEX_MIN5_PERIOD, const.KLINE_SIZE)),
            'StockDaily': np.zeros(shape=(const.STOCK_DAILY_PERIOD, const.KLINE_SIZE)),
            'StockMin5': np.zeros(shape=(const.STOCK_MIN5_PERIOD, const.KLINE_SIZE)),
            'HoldingPeriod': 0
        }
        next_states = []
        state_flags = []
        for r in ori_data:
            if not (r['NextStates'] is None):
                next_states.append(r["NextStates"])
                state_flags.append(1)
            else:
                next_states.append(empty_state)
                state_flags.append(0)
        return np.array(next_states), np.array(state_flags)

    @classmethod
    def check_data(cls, data):
        flag = True
        index_daily = data["IndexDaily"]
        if len(index_daily) < const.INDEX_DAILY_PERIOD:
            flag = False
        index_min5 = data["IndexMin5"]
        if len(index_min5) < const.INDEX_MIN5_PERIOD:
            flag = False
        stock_daily = data["StockDaily"]
        if len(stock_daily) < const.STOCK_DAILY_PERIOD:
            flag = False
        stock_min5 = data["StockMin5"]
        if len(stock_min5) < const.STOCK_MIN5_PERIOD:
            flag = False
        if "Reward" in data.keys():
            for r in data["Reward"]:
                if np.isinf(r) or np.isnan(r):
                    flag = False
        return flag

    @classmethod
    def prepare_predicting_data(cls, date_string):
        client = pymongo.MongoClient("localhost", 27017)
        db = client.FinanceLast
        col = db["DqnTradeObservationPool"]
        for r in col.find({'Date': date_string}):
            if not cls.check_data(r):
                continue
            cls.all_predicting_data.append(r)

    @classmethod
    def update_predicted_qvalues(cls, qvalues):
        client = pymongo.MongoClient("localhost", 27017)
        db = client.FinanceLast
        col = db["DqnTradeObservationPool"]
        # qvalues中的值是numpy中的float32,这不是python中的数据类型，无法在pymongo中直接使用，故使用float进行了转换
        for i, r in enumerate(cls.all_predicting_data):
            col.update({'_id': r['_id']},
                       {'$set': {'QValueIn': float(qvalues[i, 0]), 'QValueOut': float(qvalues[i, 1])}})
