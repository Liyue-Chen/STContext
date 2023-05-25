from abc import ABC, abstractmethod
from preprocessor import Normalizer, SplitData, MoveSample, ST_MoveSample
from time_utils import is_valid_date, is_work_day_america, is_work_day_china
import numpy as np
import datetime

class TemporalContextLoader():

    def __init__(self, traffic_dataloader=None, ef=None):

        self.weather_data = self.get_weather(external_feature=ef, N=1)
        self.holiday_data = self.get_holiday(arg=None)
        # 初始化这里还没写好
        self.temporal_position = self.get_temporal_position(dt=datetime.datetime(2023,5,24,16,43,30))
        pass

    @abstractmethod
    def infer_weather(self, arr, N):
        """
            Parameters
            ----------
            arr: weather_arr: [T, D_t]
            N: the number of stations

            Returns
            -------
            tiled_arr : Copy the array N times on the second dimension: [T, N, D_t]

        """
        # 在第二维度上插入一个新的维度
        expanded_arr = np.expand_dims(arr, axis=1)
        # 使用tile函数在第二维度上复制N次
        tiled_arr = np.tile(expanded_arr, reps=(1, N, 1))
        return tiled_arr

    @abstractmethod
    def get_weather(self, external_feature, N, C=6, P=7, T=1):
        '''
        Input:
        ----------
            - external_feature: (T, D_t)
            - N: number of the stations
            - C: the length of closeness
            - P: the length of period
            - T: the length of trend

        Return:
        ----------
            - train move context: (train_length, D_t, C+P+T, 1)
            - val move context: (val_length, D_t, C+P+T, 1)  
            - test move context: (test_length, D_t, C+P+T, 1)  
            - infer weather: (T, N, D_t)
        '''
        # func 1 推断天气，目前是直接复制N次
        infer_ef = self.infer_weather(external_feature, N)

        # func 2 构造历史context样本
        # move Context sample
        self.closeness_len = C
        self.period_len = P
        self.trend_len = T
        self.daily_slots = 24
        

        self.train_ef_closeness = None
        self.train_ef_period = None
        self.train_ef_trend = None
        # self.train_lstm_ef =  None
        self.val_ef_closeness = None
        self.val_ef_period = None
        self.val_ef_trend = None
        # self.val_lstm_ef =  None
        self.test_ef_closeness = None
        self.test_ef_period = None
        self.test_ef_trend = None
        # self.test_lstm_ef = None
        if len(external_feature) > 0:
            self.external_move_sample = ST_MoveSample(closeness_len=self.closeness_len,
                                            period_len=self.period_len,
                                            trend_len=self.trend_len, target_length=0, daily_slots=self.daily_slots)
            
            closeness, period, trend, _ = self.external_move_sample.move_sample(external_feature)
            print(len(closeness), len(period), len(trend))

            # train_last_index = int(self.length * 7 / 10)
            # val_last_index = int(self.length * 8 / 10)
            # test_last_index = int(self.length)
            # print(train_last_index, val_last_index, test_last_index)
            # self.train_ef = external_feature[:train_last_index,:]
            # self.val_ef = external_feature[train_last_index:val_last_index,:]
            # self.test_ef = external_feature[val_last_index:test_last_index,:]
            # print(len(self.train_ef), len(self.val_ef), len(self.test_ef))
            def divide(data):
                length = len(data)
                train_last_index = int(length * 7 / 10)
                val_last_index = int(length * 8 / 10)
                test_last_index = int(length)
                return data[:train_last_index], data[train_last_index:val_last_index], data[val_last_index:]
            self.train_ef_closeness, self.val_ef_closeness, self.test_ef_closeness = divide(closeness)
            self.train_ef_period, self.val_ef_period, self.test_ef_period = divide(period)
            self.train_ef_trend, self.val_ef_trend, self.test_ef_trend = divide(trend)

            # self.train_ef_closeness, self.train_ef_period, self.train_ef_trend, _ = self.external_move_sample.move_sample(self.train_ef)
            # self.val_ef_closeness, self.val_ef_period, self.val_ef_trend, _ = self.external_move_sample.move_sample(self.val_ef)
            # self.test_ef_closeness, self.test_ef_period, self.test_ef_trend, _ = self.external_move_sample.move_sample(self.test_ef)
            print(self.train_ef_closeness.shape, self.train_ef_period.shape, self.train_ef_trend.shape)
            print(self.train_ef_closeness[:3,5,:,0], self.train_ef_period[:3,5,:,0], self.train_ef_trend[:3,5,:,0])
            train_ef = np.dstack([self.train_ef_closeness, self.train_ef_period, self.train_ef_trend])
            val_ef = np.dstack([self.val_ef_closeness, self.val_ef_period, self.val_ef_trend])
            test_ef = np.dstack([self.test_ef_closeness, self.test_ef_period, self.test_ef_trend])
            print(train_ef.shape, val_ef.shape, test_ef.shape)
            return train_ef,val_ef,test_ef,infer_ef

            # if self.external_lstm_len is not None and self.external_lstm_len > 0:    
            #     self.external_move_sample = ST_MoveSample(closeness_len=self.external_lstm_len,period_len=0,trend_len=0, target_length=0, daily_slots=self.daily_slots)

            #     self.train_lstm_ef, _, _, _ = self.external_move_sample.move_sample(self.train_ef)
            #     self.val_lstm_ef, _, _, _ = self.external_move_sample.move_sample(self.val_ef)
            #     self.test_lstm_ef, _, _, _ = self.external_move_sample.move_sample(self.test_ef)

            # self.train_ef = self.train_ef[-self.train_sequence_len - target_length: -target_length]
            # self.test_ef = self.test_ef[-self.test_sequence_len - target_length: -target_length]
            
            # # weather
            # self.train_lstm_ef = self.train_lstm_ef[-self.train_sequence_len - target_length: -target_length]
            # self.test_lstm_ef = self.test_lstm_ef[-self.test_sequence_len - target_length: -target_length]
        # return self.
        # pass

    # @abstractmethod
    # def infer_weather(sellf, arg):
    #     pass
    
    @abstractmethod
    def get_holiday(sellf, arg=None):
        '''
        Input:
            parser function or csv file

        Function to be implemented:
            Func 1 (T//daily_slots, 1) -> temporal replicate ->  (T, 1)
        '''
        # parse by api

        # load by file
        pass
    
    # @staticmethod
    @abstractmethod
    def get_temporal_position(self, dt: datetime.datetime):
        """
            args:
            -----
            dt: datetime.datetime
                must include day and hour 

            return:
            -----
            the one_hot encoding of dt in a week(7days) and in a day(24hours).\n
            e.g.
            [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
            means thursday and 16:00, because 1 appears in positions 3 and 24 
        """
        def get_one_hot_encoding(value, total_classes):
            # 创建一个全零向量
            one_hot = np.zeros(total_classes)
            
            # 将对应位置设置为1
            one_hot[value] = 1
            
            return one_hot
        weekday = dt.weekday()
        # 获取小时
        hour = dt.hour
        
        # 获取一周内7天的one-hot编码
        weekday_encoding = get_one_hot_encoding(weekday, 7)
        # 获取一天内24小时的one-hot编码
        hour_encoding = get_one_hot_encoding(hour, 24)
        
        # 将两个编码连接在一起
        one_hot_vector = np.concatenate((weekday_encoding, hour_encoding))
        print(one_hot_vector)
        
        return one_hot_vector



class SpatialContextLoader(ABC):
     
    def __init__(self, traffic_dataloader):
        self.poi = self.get_poi()
        
    @abstractmethod
    def get_poi(sellf, arg):
        '''
        Input:
            parser function or csv file

        Function to be implemented:
            Func 1 (?, N, D_s) -> temporal replicate -> (T, N, D_s)
        '''

        # load by file
        pass

import pandas as pd
df = pd.read_csv('./scene/season_7_1_2/select/Chicago (2014092823 - 2015010623).csv')
array = np.array(df)
print(len(array))
a = TemporalContextLoader(ef=array)