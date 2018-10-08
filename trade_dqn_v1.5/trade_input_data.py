import pymongo
import numpy as np
import const

const.INDEX_DAILY_PERIOD = 20
const.INDEX_MIN5_PERIOD = 10
const.STOCK_DAILY_PERIOD = 40
const.STOCK_MIN5_PERIOD = 10


class trade_input_data(object):
    all_data_list = []
    all_data_dict = {}

    @classmethod
    def prepare_data(cls, begin_date_string, end_date_string):
        client = pymongo.MongoClient("localhost", 27017)
        db = client.FinanceLast
        col = db["RandomTradeRecords"]
        for r in col.find({'Date': {'$gte': begin_date_string, '$lte': end_date_string}}):
            if not cls.check_data(r):
                continue
            cls.all_data_list.append(r)
            key = r['Date'] + r['Symbol'] + str(r['HoldingPeriod'])
            value = cls.all_data_dict.get(key)
            if value is None:
                new_value = [r]
                cls.all_data_dict[key] = new_value
            else:
                value.append(r)
        for r in cls.all_data_list:
            if r['Inside'] == 1 and r['NextTradeDate'] != '':
                key = r['NextTradeDate'] + r['Symbol'] + str(r['HoldingPeriod'] + 1)
                r['NextStates'] = cls.all_data_dict.get(key)
            else:
                r['NextStates'] = None

    @classmethod
    def next_batch_data(cls, batch_size):
        result = []
        n = len(cls.all_data_list)
        if n > 0:
            random_nums = np.random.randint(0, n, batch_size)
            for k in random_nums:
                r = cls.all_data_list[k]
                result.append(r)
        return result

    @classmethod
    def pick_up(cls, source_data):
        index_daily = [r["IndexDaily"] for r in source_data]
        index_min5 = [r["IndexMin5"] for r in source_data]
        stock_daily = [r["StockDaily"] for r in source_data]
        stock_min5 = [r["StockMin5"] for r in source_data]
        inside = [r["Inside"] for r in source_data]
        holding_period = [r["HoldingPeriod"] for r in source_data]
        return np.array(index_daily), np.array(index_min5), np.array(stock_daily), np.array(stock_min5), np.array(inside).reshape((-1, 1)), np.array(
            holding_period).reshape((-1, 1))

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
        return flag

## test
# trade_input_data.prepare_data("2018-08-01","2018-08-15")
# # num_none=0
# # for r in trade_input_data.all_data_list:
# #     if r["NextStates"] is None:
# #         num_none+=1
# # print("num of all data is %d,None has %d"%(len(trade_input_data.all_data_list),num_none))
# # print(len(trade_input_data.all_data_list))
# # res=trade_input_data.next_batch_data(10)
# # for r in res:
# #    print(r['NextStates'])
