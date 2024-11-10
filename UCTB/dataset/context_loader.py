import logging
import numpy as np
import os
import pickle
from abc import ABC, abstractmethod
from ..preprocess.time_utils import is_work_day_china, is_work_day_america, is_valid_date
from abc import ABC, abstractmethod
import pandas as pd
from copy import copy,deepcopy
import math
from datetime import datetime
from dateutil.parser import parse
from tqdm import tqdm
from ..preprocess import chooseNormalizer
import os
import pickle as pkl

def cache_result(filename_func):
    """
    Decorator that caches function results in a file.

    Args:
        filename_func (callable): Function to generate the filename.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            filename = filename_func(*args, **kwargs)
            if os.path.exists(filename):
                with open(filename, 'rb') as file:
                    return pickle.load(file)
            result = func(*args, **kwargs)
            with open(filename, 'wb') as file:
                pickle.dump(result, file)
            return result
        return wrapper
    return decorator

class ContextLoaderBase(ABC):
    def __init__(self, traffic_dataloader, data_type_list, data_file_list, **kwargs):
        '''
        Args:
            traffic_dataloader (NodeTrafficLoader): It contains the traffic data in the shape [time_slot_num, ``station_number``, ``closeness_len``, 1].
            data_type_list (list): List of data types to be loaded. Supported data types are ['temporal_position', 'holiday'].
            data_file_list (list): List of data files to be loaded. The length of data_file_list should be the same as data_type_list.
        '''
        assert len(data_type_list) == len(data_file_list), "[ContextLoaderBase] Mismatched data types and files."
        self.loader_id = traffic_dataloader.loader_id
        self.data_type_list = data_type_list
        self.data_file_list = data_file_list
        self.load_func_dict = self.get_load_func_dict()
        self.data = {}
        self.load_data(traffic_dataloader, **kwargs)
        self.context_dimension_dict = {}
        self.feature_cnt = 0

    @abstractmethod
    def get_load_func_dict(self):
        pass

    def load_data(self, traffic_dataloader, **kwargs):
        for data_type, data_file in zip(self.data_type_list, self.data_file_list):
            if data_type not in self.load_func_dict:
                logging.error(f"Unsupported data type: {data_type}. Skipping.")
                continue
            data = self.load_func_dict[data_type](data_file, traffic_dataloader, **kwargs)
            self.data[data_type] = data

class TemporalContextLoader(ContextLoaderBase):
    def __init__(self, traffic_dataloader, data_type_list=[], data_file_list=[], past_slots=0, future_slots=1):
        '''
        Args:
            past_slots (int): The number of past or historical slots to be loaded.
            future_slots (int): The number of future slots to be loaded. The first element indicates the predicted slot.
        
        Attributes:
            past_temporal_data (np.ndarray): with shape [time_slot_num, 1, num_past_slots, num_features]
            future_temporal_data (np.ndarray): with shape [time_slot_num, 1, num_future_slots, num_features]
            context_dimension_dict (dict): with key as the data type and the value including the start index and the number of features.
        '''
        assert past_slots > 0 or future_slots > 0, "[TemporalContextLoader] Invalid slots configuration."
        self.past_slots = past_slots
        self.future_slots = future_slots

        super().__init__(traffic_dataloader, data_type_list, data_file_list, self.past_slots, self.future_slots)

        self.train_past_data, self.train_future_data, self.test_past_data, self.test_future_data = self.parse_data(self.data)


    def parse_data(self, data):
        train_past_data_list = [data[data_type][0] for data_type in self.data.keys()]
        train_future_data_list = [data[data_type][1] for data_type in self.data.keys()]
        test_past_data_list = [data[data_type][2] for data_type in self.data.keys()]
        test_future_data_list = [data[data_type][3] for data_type in self.data.keys()]

        for data_type, past_data, future_data in zip(self.data.keys(), train_past_data_list, train_future_data_list):
            feat_dim = past_data.shape[-1] if self.past_slots > 0 else future_data.shape[-1]
            self.context_dimension_dict[data_type] = {"start": self.feature_cnt, "length": feat_dim}
            self.feature_cnt += feat_dim

        return (
            np.concatenate(train_past_data_list, axis=-1),
            np.concatenate(train_future_data_list, axis=-1),
            np.concatenate(test_past_data_list, axis=-1),
            np.concatenate(test_future_data_list, axis=-1)
        )

    
    def get_load_func_dict(self):
        return {
            'temporal_position': self.get_temporal_position,
            'holiday': self.get_holiday,
        }

    @cache_result(lambda self, *_: f'{self.loader_id}_holiday.pkl')
    def get_holiday(self, data_file, traffic_dataloader, past_slots, future_slots):
        '''
        Input:
            data_file (str): The path to the holiday data file.
            traffic_dataloader (NodeTrafficLoader): It contains ``train_data_timestamp`` and ``test_data_timestamp``, whose lengths are ``train_time_slot_num`` and ''test_time_slot_num'', respectively.
            past_slots (int): The number of past or historical slots to be loaded.
            future_slots (int): The number of future slots to be loaded. The first element indicates the predicted slot.

        Returns:
            pass_holiday_data (np.ndarray): with shape [time_slot_num, 1, num_past_slots, num_features]
            future_holiday_data (np.ndarray): with shape [time_slot_num, 1, num_future_slots, num_features]
        '''
        if data_file:
            logging.info(f"Loading holiday data from file: {data_file}")
        else:
            logging.info("Loading holiday data using calendar packages.")
        train_pass_holiday_data = None
        train_future_holiday_data = None
        test_pass_holiday_data = None
        test_future_holiday_data = None
        return [train_pass_holiday_data, train_future_holiday_data, test_pass_holiday_data, test_future_holiday_data]

    @cache_result(lambda self, *_: f'{self.loader_id}_TP.pkl')
    def get_temporal_position(self, data_file, traffic_dataloader, past_slots, future_slots):
        '''
        Input:
            data_file (None): Set as None.
            traffic_dataloader (NodeTrafficLoader): It contains ``train_data_timestamp`` and ``test_data_timestamp``, whose lengths are ``train_time_slot_num`` and ''test_time_slot_num'', respectively.
            past_slots (int): The number of past or historical slots to be loaded.
            future_slots (int): The number of future slots to be loaded. The first element indicates the predicted slot.
        Returns:
            pass_TP_data (np.ndarray): with shape [time_slot_num, 1, num_past_slots, num_features]
            future_TP_data (np.ndarray): with shape [time_slot_num, 1, num_future_slots, num_features]
        '''
        train_pass_TP_data = None
        train_future_TP_data = None
        test_pass_TP_data = None
        test_future_TP_data = None
        return [train_pass_TP_data, train_future_TP_data, test_pass_TP_data, test_future_TP_data]

class SpatialContextLoader(ContextLoaderBase):
    def __init__(self, traffic_dataloader, data_type_list=[], data_file_list=[]):
        '''
        Attributes:
            Attributes:
            spatial_data (np.ndarray): with shape [num_stations, num_spatial_features]
            context_dimension_dict (dict): with key as the data type and the value including the start index and the number of features.
        '''
        super().__init__(traffic_dataloader, data_type_list, data_file_list)

        self.spatial_data = self.parse_data(self.data)

    def parse_data(self, data):

        spatial_data_list = []
        for data_type in self.data.keys():
            spatial_data_list.append(data[data_type])
            feat_dim = spatial_data_list[-1].shape[-1]
            self.context_dimension_dict[data_type] = {"start": self.feature_cnt, "length": feat_dim}
            self.feature_cnt += feat_dim
        
        return np.concatenate(spatial_data_list, axis=-1)

    def get_load_func_dict(self):
        return {
            'poi': self.get_poi,
            'demographic': self.get_demographic,
            'road_network': self.get_road_network,
            'administrative_division': self.get_administrative_division,
        }

    @cache_result(lambda self, *_: f'{self.loader_id}_POI.pkl')
    def get_poi(self, data_file, traffic_dataloader):
        '''
        Args:
            data_file (str): The path to the POI data file.
            traffic_dataloader (NodeTrafficLoader): It contains the station information, whose length is ``station_number``.

        Returns:
            poi_data (np.ndarray): with shape [num_stations, num_features]
        '''
        # load by file
        poi_data = np.random.rand(traffic_dataloader.station_number, 10)
        return poi_data

    @cache_result(lambda self, *_: f'{self.loader_id}_DEMO.pkl')
    def get_demographic(self, data_file, traffic_dataloader):
        '''
        Args:
            data_file (str): The path to the POI data file.
            traffic_dataloader (NodeTrafficLoader): It contains the station information, whose length is ``station_number``.

        Returns:
            demographic_data (np.ndarray): with shape [num_stations, num_features]
        '''
        demographic_data = np.random.rand(traffic_dataloader.station_number, 10)
        return demographic_data

    @cache_result(lambda self, *_: f'{self.loader_id}_Road.pkl')
    def get_road_network(self, data_file, traffic_dataloader):
        '''
        Args:
            data_file (str): The path to the POI data file.
            traffic_dataloader (NodeTrafficLoader): It contains the station information, whose length is ``station_number``.

        Returns:
            road_data (np.ndarray): with shape [num_stations, num_features]
        '''
        road_data = np.random.rand(traffic_dataloader.station_number, 10)
        return road_data

    @cache_result(lambda self, *_: f'{self.loader_id}_AD.pkl')
    def get_administrative_division(self, data_file, traffic_dataloader):
        '''
        Args:
            data_file (str): The path to the POI data file.
            traffic_dataloader (NodeTrafficLoader): It contains the station information, whose length is ``station_number``.

        Returns:
            administrative_division_data (np.ndarray): with shape [num_stations, num_features]
        '''
        administrative_division_data = np.random.rand(traffic_dataloader.station_number, 10)
        return administrative_division_data


def haversine(lat1, lon1, lat2, lon2):
    # 将经纬度转换为弧度
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    # 经纬度差
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # 使用haversine公式计算球面距离
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    radius = 6371  # 地球半径，单位为千米

    distance = radius * c
    return distance
def load_nyc_metro(type:str,is_multicolumn:bool):
    if type=='Historical Weather':
        csv_path = '/home/pku/fjy/STContext/dataset/STContextData/NYC/Historical Weather/nyc_weather_fusion_nafilled_2022.csv'
        context_df = pd.read_csv(csv_path)
        columns = list(context_df.columns)
        columns.remove('lat')
        columns.remove('lon')
        columns.remove('datetime')
        columns.remove('station')
        if not is_multicolumn:
            columns = ['precipitation']
        keys = copy(columns)
        columns = ['datetime','lat','lon']+columns
        context_df = context_df[columns]
        
        latlon_list = list(context_df[['lat','lon']].value_counts().index)
        st_context = {}
        # pdb.set_trace()
        for lat,lon in latlon_list:
            selected_context_df = context_df.loc[(context_df['lat']==lat)&(context_df['lon']==lon),:]
            if (lat,lon) not in st_context.keys():
                st_context[(lat,lon)] = {}
            for index,row in selected_context_df.iterrows():
                st_context[(lat,lon)][pd.to_datetime(row['datetime'])] = dict(zip(keys,row[keys]))
        
        return st_context
    elif type=='Forecast Weather':
        if os.path.exists('/home/pku/fjy/STContext/dataset/STContextData/NYC/Forecast Weather/forecast_st_context_metro.pkl'):
            with open('/home/pku/fjy/STContext/dataset/STContextData/NYC/Forecast Weather/forecast_st_context_metro.pkl','rb') as fp:
                st_context = pkl.load(fp)
            return st_context
        csv_path = '/home/pku/fjy/STContext/dataset/STContextData/NYC/Forecast Weather/NYC_2022_localtime.csv'
        context_df = pd.read_csv(csv_path)
        columns = list(context_df.columns)
        columns.remove('lat')
        columns.remove('lon')
        columns.remove('time')
        if not is_multicolumn:
            columns = ['total_precipitation']
        keys = copy(columns)
        columns = ['time','lat','lon'] + columns
        context_df = context_df[columns]
        context_df['time'] = pd.to_datetime(context_df['time'])
        # pdb.set_trace()
        context_df_t_selected = context_df.loc[(context_df['time']>=parse('2022-02-01'))&(context_df['time']<parse('2022-08-01')),:]
        
        latlon_list = list(context_df_t_selected[['lat','lon']].value_counts().index)
        st_context = {}
        # pdb.set_trace()
        for lat,lon in tqdm(latlon_list):
            selected_context_df = context_df_t_selected.loc[(context_df_t_selected['lat']==lat)&(context_df_t_selected['lon']==lon),:]
            if (lat,lon) not in st_context.keys():
                st_context[(lat,lon)] = {}
            for index,row in selected_context_df.iterrows():
                st_context[(lat,lon)][row['time']] = dict(zip(keys,row[keys]))
        with open('/home/pku/fjy/STContext/dataset/STContextData/NYC/Forecast Weather/forecast_st_context_metro.pkl','wb') as fp:
            pkl.dump(st_context,fp)
        
        return st_context
    elif type=='AQI':
        co_csv_path = '/home/pku/fjy/STContext/dataset/STContextData/NYC/AQI/hourly_CO_2022_NY.csv'
        co_df = pd.read_csv(co_csv_path)
        co_latlon_list = list(co_df[['Latitude','Longitude']].value_counts().index)
        no2_csv_path = '/home/pku/fjy/STContext/dataset/STContextData/NYC/AQI/hourly_NO2_2022_NY.csv'
        no2_df = pd.read_csv(no2_csv_path)
        no2_latlon_list = list(no2_df[['Latitude','Longitude']].value_counts().index)
        ozone_csv_path = '/home/pku/fjy/STContext/dataset/STContextData/NYC/AQI/hourly_Ozone_2022_NY.csv'
        ozone_df = pd.read_csv(ozone_csv_path)
        ozone_latlon_list = list(ozone_df[['Latitude','Longitude']].value_counts().index)
        pm25_csv_path = '/home/pku/fjy/STContext/dataset/STContextData/NYC/AQI/hourly_PM25_2022_NY.csv'
        pm25_df = pd.read_csv(pm25_csv_path)
        pm25_latlon_list = list(pm25_df[['Latitude','Longitude']].value_counts().index)
        so2_csv_path = '/home/pku/fjy/STContext/dataset/STContextData/NYC/AQI/hourly_SO2_2022_NY.csv'
        so2_df = pd.read_csv(so2_csv_path)
        so2_latlon_list = list(so2_df[['Latitude','Longitude']].value_counts().index)
        co_df['datetime'] = pd.to_datetime(co_df['Date Local'] + ' ' + co_df['Time Local'])
        no2_df['datetime'] = pd.to_datetime(no2_df['Date Local'] + ' ' + no2_df['Time Local'])
        ozone_df['datetime'] = pd.to_datetime(ozone_df['Date Local'] + ' ' + ozone_df['Time Local'])
        pm25_df['datetime'] = pd.to_datetime(pm25_df['Date Local'] + ' ' + pm25_df['Time Local'])
        so2_df['datetime'] = pd.to_datetime(so2_df['Date Local'] + ' ' + so2_df['Time Local'])
        tmp_set = set(co_latlon_list).intersection(set(no2_latlon_list)).intersection(set(ozone_latlon_list)).intersection(set(pm25_latlon_list)).intersection(set(so2_latlon_list))
        st_context = {}
        dt_list = pd.date_range(start='2022-02-01',end='2022-08-01',freq='H')
        dt_list = dt_list[:-1]
        co_values = []
        no2_values = []
        ozone_values = []
        pm25_values = []
        so2_values = []
        lat_list = []
        lon_list = []
        if is_multicolumn:
            for lat,lon in tmp_set:
                selected_co_df = co_df.loc[(co_df['Latitude']==lat)&(co_df['Longitude']==lon),:]
                selected_no2_df = no2_df.loc[(no2_df['Latitude']==lat)&(no2_df['Longitude']==lon),:]
                selected_ozone_df = ozone_df.loc[(ozone_df['Latitude']==lat)&(ozone_df['Longitude']==lon),:]
                selected_pm25_df = pm25_df.loc[(pm25_df['Latitude']==lat)&(pm25_df['Longitude']==lon),:]
                selected_so2_df = so2_df.loc[(so2_df['Latitude']==lat)&(so2_df['Longitude']==lon),:]
                for dt in dt_list:
                    co_values.append(selected_co_df.loc[selected_co_df['datetime']==dt,'Sample Measurement'].iloc[0] if len(selected_co_df.loc[selected_co_df['datetime']==dt,'Sample Measurement']) > 0 else np.nan)
                    no2_values.append(selected_no2_df.loc[selected_no2_df['datetime']==dt,'Sample Measurement'].iloc[0] if len(selected_no2_df.loc[selected_no2_df['datetime']==dt,'Sample Measurement']) > 0 else np.nan)
                    ozone_values.append(selected_ozone_df.loc[selected_ozone_df['datetime']==dt,'Sample Measurement'].iloc[0] if len(selected_ozone_df.loc[selected_ozone_df['datetime']==dt,'Sample Measurement']) > 0 else np.nan)
                    pm25_values.append(selected_pm25_df.loc[selected_pm25_df['datetime']==dt,'Sample Measurement'].iloc[0] if len(selected_pm25_df.loc[selected_pm25_df['datetime']==dt,'Sample Measurement']) > 0 else np.nan)
                    so2_values.append(selected_so2_df.loc[selected_so2_df['datetime']==dt,'Sample Measurement'].iloc[0] if len(selected_so2_df.loc[selected_so2_df['datetime']==dt,'Sample Measurement']) > 0 else np.nan)
                    lat_list.append(lat)
                    lon_list.append(lon)
            df = pd.DataFrame(data={
                'datetime':dt_list,
                'lat':lat_list,
                'lon':lon_list,
                'co':co_values,
                'no2':no2_values,
                'ozone':ozone_values,
                'pm25':pm25_values,
                'so2':so2_values,
            })
            df.ffill(inplace=True)
            keys = ['co','no2','ozone','pm25','so2']
            for lat,lon in tmp_set:
                df_selected = df.loc[(df['lat']==lat)&(df['lon']==lon),:]
                if (lat,lon) not in st_context.keys():
                    st_context[(lat,lon)] = {}
                for index,row in df_selected.iterrows():
                    st_context[(lat,lon)][row['datetime']] = dict(zip(keys,row[keys]))
        return st_context

    else:
        raise TypeError('We don\'t have this type of ST Context')

def load_nyc_bikeandtaxi(type:str,is_multicolumn:bool):
    if type=='Historical Weather':
        csv_path = '/home/pku/fjy/STContext/dataset/STContextData/NYC/Historical Weather/nyc_weather_fusion_nafilled.csv'
        context_df = pd.read_csv(csv_path)
        columns = list(context_df.columns)
        columns.remove('lat')
        columns.remove('lon')
        columns.remove('datetime')
        columns.remove('station')
        if not is_multicolumn:
            columns = ['precipitation']
        keys = copy(columns)
        columns = ['datetime','lat','lon']+columns
        context_df = context_df[columns]
        
        latlon_list = list(context_df[['lat','lon']].value_counts().index)
        st_context = {}
        # pdb.set_trace()
        for lat,lon in latlon_list:
            selected_context_df = context_df.loc[(context_df['lat']==lat)&(context_df['lon']==lon),:]
            if (lat,lon) not in st_context.keys():
                st_context[(lat,lon)] = {}
            for index,row in selected_context_df.iterrows():
                st_context[(lat,lon)][pd.to_datetime(row['datetime'])] = dict(zip(keys,row[keys]))
        return st_context
    elif type=='Forecast Weather':
        if os.path.exists('/home/pku/fjy/STContext/dataset/STContextData/NYC/Forecast Weather/forecast_st_context.pkl'):
            with open('/home/pku/fjy/STContext/dataset/STContextData/NYC/Forecast Weather/forecast_st_context.pkl','rb') as fp:
                st_context = pkl.load(fp)
            return st_context
        csv_path = '/home/pku/fjy/STContext/dataset/STContextData/NYC/Forecast Weather/NYC_2017_localtime.csv'
        context_df = pd.read_csv(csv_path)
        columns = list(context_df.columns)
        columns.remove('lat')
        columns.remove('lon')
        columns.remove('time')
        if not is_multicolumn:
            columns = ['total_precipitation']
        keys = copy(columns)
        columns = ['time','lat','lon'] + columns
        context_df = context_df[columns]
        context_df['time'] = pd.to_datetime(context_df['time'])
        # pdb.set_trace()
        context_df_t_selected = context_df.loc[(context_df['time']>=parse('2017-01-01'))&(context_df['time']<parse('2017-07-01')),:]
        
        latlon_list = list(context_df_t_selected[['lat','lon']].value_counts().index)
        st_context = {}
        # pdb.set_trace()
        for lat,lon in tqdm(latlon_list):
            selected_context_df = context_df_t_selected.loc[(context_df_t_selected['lat']==lat)&(context_df_t_selected['lon']==lon),:]
            if (lat,lon) not in st_context.keys():
                st_context[(lat,lon)] = {}
            for index,row in selected_context_df.iterrows():
                st_context[(lat,lon)][row['time']] = dict(zip(keys,row[keys]))
        with open('/home/pku/fjy/STContext/dataset/STContextData/NYC/Forecast Weather/forecast_st_context.pkl','wb') as fp:
            pkl.dump(st_context,fp)
        
        return st_context
  
    elif type=='AQI':
        co_csv_path = '/home/pku/fjy/STContext/dataset/STContextData/NYC/AQI/hourly_CO_2017_NY.csv'
        co_df = pd.read_csv(co_csv_path)
        co_latlon_list = list(co_df[['Latitude','Longitude']].value_counts().index)
        no2_csv_path = '/home/pku/fjy/STContext/dataset/STContextData/NYC/AQI/hourly_NO2_2017_NY.csv'
        no2_df = pd.read_csv(no2_csv_path)
        no2_latlon_list = list(no2_df[['Latitude','Longitude']].value_counts().index)
        ozone_csv_path = '/home/pku/fjy/STContext/dataset/STContextData/NYC/AQI/hourly_Ozone_2017_NY.csv'
        ozone_df = pd.read_csv(ozone_csv_path)
        ozone_latlon_list = list(ozone_df[['Latitude','Longitude']].value_counts().index)
        pm25_csv_path = '/home/pku/fjy/STContext/dataset/STContextData/NYC/AQI/hourly_PM25_2017_NY.csv'
        pm25_df = pd.read_csv(pm25_csv_path)
        pm25_latlon_list = list(pm25_df[['Latitude','Longitude']].value_counts().index)
        so2_csv_path = '/home/pku/fjy/STContext/dataset/STContextData/NYC/AQI/hourly_SO2_2017_NY.csv'
        so2_df = pd.read_csv(so2_csv_path)
        so2_latlon_list = list(so2_df[['Latitude','Longitude']].value_counts().index)
        co_df['datetime'] = pd.to_datetime(co_df['Date Local'] + ' ' + co_df['Time Local'])
        no2_df['datetime'] = pd.to_datetime(no2_df['Date Local'] + ' ' + no2_df['Time Local'])
        ozone_df['datetime'] = pd.to_datetime(ozone_df['Date Local'] + ' ' + ozone_df['Time Local'])
        pm25_df['datetime'] = pd.to_datetime(pm25_df['Date Local'] + ' ' + pm25_df['Time Local'])
        so2_df['datetime'] = pd.to_datetime(so2_df['Date Local'] + ' ' + so2_df['Time Local'])
        tmp_set = set(co_latlon_list).intersection(set(no2_latlon_list)).intersection(set(ozone_latlon_list)).intersection(set(pm25_latlon_list)).intersection(set(so2_latlon_list))
        st_context = {}
        dt_list = pd.date_range(start='2017-01-01',end='2017-07-01',freq='H')
        dt_list = dt_list[:-1]
        co_values = []
        no2_values = []
        ozone_values = []
        pm25_values = []
        so2_values = []
        lat_list = []
        lon_list = []
        if is_multicolumn:
            for lat,lon in tmp_set:
                selected_co_df = co_df.loc[(co_df['Latitude']==lat)&(co_df['Longitude']==lon),:]
                selected_no2_df = no2_df.loc[(no2_df['Latitude']==lat)&(no2_df['Longitude']==lon),:]
                selected_ozone_df = ozone_df.loc[(ozone_df['Latitude']==lat)&(ozone_df['Longitude']==lon),:]
                selected_pm25_df = pm25_df.loc[(pm25_df['Latitude']==lat)&(pm25_df['Longitude']==lon),:]
                selected_so2_df = so2_df.loc[(so2_df['Latitude']==lat)&(so2_df['Longitude']==lon),:]
                for dt in dt_list:
                    co_values.append(selected_co_df.loc[selected_co_df['datetime']==dt,'Sample Measurement'].iloc[0] if len(selected_co_df.loc[selected_co_df['datetime']==dt,'Sample Measurement']) > 0 else np.nan)
                    no2_values.append(selected_no2_df.loc[selected_no2_df['datetime']==dt,'Sample Measurement'].iloc[0] if len(selected_no2_df.loc[selected_no2_df['datetime']==dt,'Sample Measurement']) > 0 else np.nan)
                    ozone_values.append(selected_ozone_df.loc[selected_ozone_df['datetime']==dt,'Sample Measurement'].iloc[0] if len(selected_ozone_df.loc[selected_ozone_df['datetime']==dt,'Sample Measurement']) > 0 else np.nan)
                    pm25_values.append(selected_pm25_df.loc[selected_pm25_df['datetime']==dt,'Sample Measurement'].iloc[0] if len(selected_pm25_df.loc[selected_pm25_df['datetime']==dt,'Sample Measurement']) > 0 else np.nan)
                    so2_values.append(selected_so2_df.loc[selected_so2_df['datetime']==dt,'Sample Measurement'].iloc[0] if len(selected_so2_df.loc[selected_so2_df['datetime']==dt,'Sample Measurement']) > 0 else np.nan)
                    lat_list.append(lat)
                    lon_list.append(lon)
            df = pd.DataFrame(data={
                'datetime':dt_list,
                'lat':lat_list,
                'lon':lon_list,
                'co':co_values,
                'no2':no2_values,
                'ozone':ozone_values,
                'pm25':pm25_values,
                'so2':so2_values,
            })
            df.ffill(inplace=True)
            keys = ['co','no2','ozone','pm25','so2']
            for lat,lon in tmp_set:
                df_selected = df.loc[(df['lat']==lat)&(df['lon']==lon),:]
                if (lat,lon) not in st_context.keys():
                    st_context[(lat,lon)] = {}
                for index,row in df_selected.iterrows():
                    st_context[(lat,lon)][row['datetime']] = dict(zip(keys,row[keys]))
        return st_context
    else:
        raise TypeError('We don\'t have this type of ST Context')

def load_bay_speed(type:str,is_multicolumn:bool):
    if type=='Historical Weather':
        csv_path = '/home/pku/fjy/STContext/dataset/STContextData/BAY/bay_fusion_nafilled.csv'
        context_df = pd.read_csv(csv_path)
        columns = list(context_df.columns)
        columns.remove('lat')
        columns.remove('lon')
        columns.remove('datetime')
        columns.remove('station')
        if not is_multicolumn:
            columns = ['precipitation']
        keys = copy(columns)
        columns = ['datetime','lat','lon']+columns
        context_df = context_df[columns]
        
        latlon_list = list(context_df[['lat','lon']].value_counts().index)
        st_context = {}
        # pdb.set_trace()
        for lat,lon in latlon_list:
            selected_context_df = context_df.loc[(context_df['lat']==lat)&(context_df['lon']==lon),:]
            if (lat,lon) not in st_context.keys():
                st_context[(lat,lon)] = {}
            for index,row in selected_context_df.iterrows():
                st_context[(lat,lon)][pd.to_datetime(row['datetime'])] = dict(zip(keys,row[keys]))
        
        return st_context
    elif type=='Forecast Weather':
        if os.path.exists('/home/pku/fjy/STContext/dataset/STContextData/BAY/forecast_st_context.pkl'):
            with open('/home/pku/fjy/STContext/dataset/STContextData/BAY/forecast_st_context.pkl','rb') as fp:
                st_context = pkl.load(fp)
            return st_context
        csv_path = '/home/pku/fjy/STContext/dataset/STContextData/BAY/BAY_localtime.csv'
        context_df = pd.read_csv(csv_path)
        columns = list(context_df.columns)
        columns.remove('lat')
        columns.remove('lon')
        columns.remove('time')
        if not is_multicolumn:
            columns = ['total_precipitation']
        keys = copy(columns)
        columns = ['time','lat','lon'] + columns
        context_df = context_df[columns]
        context_df['time'] = pd.to_datetime(context_df['time'])
        # pdb.set_trace()
        context_df_t_selected = context_df.loc[(context_df['time']>=parse('2017-01-01'))&(context_df['time']<parse('2017-07-01')),:]
        
        latlon_list = list(context_df_t_selected[['lat','lon']].value_counts().index)
        st_context = {}
        # pdb.set_trace()
        for lat,lon in tqdm(latlon_list):
            selected_context_df = context_df_t_selected.loc[(context_df_t_selected['lat']==lat)&(context_df_t_selected['lon']==lon),:]
            if (lat,lon) not in st_context.keys():
                st_context[(lat,lon)] = {}
            for index,row in selected_context_df.iterrows():
                st_context[(lat,lon)][row['time']] = dict(zip(keys,row[keys]))
        with open('/home/pku/fjy/STContext/dataset/STContextData/BAY/forecast_st_context.pkl','wb') as fp:
            pkl.dump(st_context,fp)
        
        return st_context
    elif type=='AQI':
        pass
    else:
        raise TypeError('We don\'t have this type of ST Context')

def process_weather_state(x):
    value = 0
    if (x=='Light Rain Shower') or ('Drizzle' in x):
        value = 0.005
    elif x=='Rain Shower':
        value = 0.01
    elif x=='Light Rain':
        value = 0.02
    elif x=='Rain':
        value = 0.1
    elif 'Heavy' in x:
        value = 0.2
    return value

def load_mel_pedestrian(type:str,is_multicolumn:bool):
    if type=='Historical Weather':
        csv_path = '/home/pku/fjy/STContext/dataset/STContextData/MEL/mel_fusion_NAfilled.csv'
        context_df = pd.read_csv(csv_path)
        columns = list(context_df.columns)
        columns.remove('lat')
        columns.remove('lon')
        columns.remove('datetime')
        if 'station' in columns:
            columns.remove('station')
        if 'weatherstate' in columns:
            context_df['precipitation'] = context_df['weatherstate'].apply(process_weather_state)
            columns.remove('weatherstate')
        
        if not is_multicolumn:
            columns = ['precipitation']
        keys = copy(columns)
        columns = ['datetime','lat','lon']+columns
        context_df = context_df[columns]
        
        latlon_list = list(context_df[['lat','lon']].value_counts().index)
        st_context = {}
        # pdb.set_trace()
        for lat,lon in latlon_list:
            selected_context_df = context_df.loc[(context_df['lat']==lat)&(context_df['lon']==lon),:]
            if (lat,lon) not in st_context.keys():
                st_context[(lat,lon)] = {}
            for index,row in selected_context_df.iterrows():
                st_context[(lat,lon)][pd.to_datetime(row['datetime'])] = dict(zip(keys,row[keys]))
        
        return st_context
    elif type=='Forecast Weather':
        if os.path.exists('/home/pku/fjy/STContext/dataset/STContextData/MEL/forecast_st_context.pkl'):
            with open('/home/pku/fjy/STContext/dataset/STContextData/MEL/forecast_st_context.pkl','rb') as fp:
                st_context = pkl.load(fp)
            return st_context
        csv_path = '/home/pku/fjy/STContext/dataset/STContextData/MEL/MEL_localtime.csv'
        context_df = pd.read_csv(csv_path)
        columns = list(context_df.columns)
        columns.remove('lat')
        columns.remove('lon')
        columns.remove('time')
        if not is_multicolumn:
            columns = ['total_precipitation']
        keys = copy(columns)
        columns = ['time','lat','lon'] + columns
        context_df = context_df[columns]
        context_df['time'] = pd.to_datetime(context_df['time'])
        # pdb.set_trace()
        context_df_t_selected = context_df.loc[(context_df['time']>=parse('2022-02-01'))&(context_df['time']<parse('2022-08-01')),:]
        
        latlon_list = list(context_df_t_selected[['lat','lon']].value_counts().index)
        st_context = {}
        # pdb.set_trace()
        for lat,lon in tqdm(latlon_list):
            selected_context_df = context_df_t_selected.loc[(context_df_t_selected['lat']==lat)&(context_df_t_selected['lon']==lon),:]
            if (lat,lon) not in st_context.keys():
                st_context[(lat,lon)] = {}
            for index,row in selected_context_df.iterrows():
                st_context[(lat,lon)][row['time']] = dict(zip(keys,row[keys]))
        with open('/home/pku/fjy/STContext/dataset/STContextData/MEL/forecast_st_context.pkl','wb') as fp:
            pkl.dump(st_context,fp)
        
        return st_context
    else:
        pass

load_functions = {
    'nyc_metro':load_nyc_metro,
    'nyc_bikeandtaxi':load_nyc_bikeandtaxi,
    'bay_speed':load_bay_speed,
    'mel_pedestrian':load_mel_pedestrian
}


class STContextLoader():

    def __init__(self):
        pass

    def get_stcontext(self, args):
        '''
        Input:
            default weather features shape: (T, D_t)
            detailed weather features shape: (T, N, D_t)
            station info [[lat1, lng1], ....]

        Function to be implemented:
            Func 1 (T, ?, D_t) -> bind/replicate (optional) -> (T, N, D_t)
            Func 2 move Context sample
        '''
        
        self.args = args
        if args['type'] == 'Forecast Weather':
            load_function = load_functions[args['name']]
            self.st_context_current = load_function('Forecast Weather',args['is_multicolumn'])
            print('Forecast Finished')
            self.st_context = load_function('Historical Weather',args['is_multicolumn'])
            print('Historical Finished')
        else:
            load_function = load_functions[args['name']]
            self.st_context = load_function(args['type'],args['is_multicolumn'])
    def TTransformation(self,expected_time_range,expected_time_fitness):
        date_list = list(pd.date_range(expected_time_range[0],expected_time_range[1],freq='{}min'.format(expected_time_fitness)))
        date_list.pop(-1)
        # pdb.set_trace()


        station_info = list(self.st_context.keys())
        # 提前建立好 date_list 和 时间分辨率的关系
        latlon2tcontext = {}
        for sta_lat,sta_lon in station_info:
            context_date_list = list(self.st_context[(sta_lat,sta_lon)].keys())
            flow_date_list = list(date_list)
            # pdb.set_trace()
            t_context = self.st_context[(sta_lat,sta_lon)]
            t_context_transformed = np.array(t_transform(t_context=t_context,flow_date_list=flow_date_list,context_date_list=context_date_list))
            
            latlon2tcontext[(sta_lat,sta_lon)]=t_context_transformed
            
        self.latlon2context = latlon2tcontext
    
    def get_assignment_matrix(self,expected_station_info):
        if self.args['type'] == 'Forecast Weather':
            context_station_latlng_list = list(self.latlon2context.keys())
            num_context_stations = len(context_station_latlng_list)
            num_crowdflow_stations = len(expected_station_info)
            assignment_matrix = np.zeros([num_context_stations, num_crowdflow_stations])
            for i in range(num_crowdflow_stations):
                lat_crowdflow_station,lon_crowdflow_station = expected_station_info[i]
                distances = []
                for j in range(num_context_stations):
                    distances.append(haversine(context_station_latlng_list[j][0],context_station_latlng_list[j][1],lat_crowdflow_station,lon_crowdflow_station))

                distances = np.array(distances)
                assignment_matrix[np.argmin(distances),i] = 1
            context_station_latlng_list_current = list(self.latlon2context_current.keys())
            num_context_stations_current = len(context_station_latlng_list_current)
            num_crowdflow_stations = len(expected_station_info)
            assignment_matrix_current = np.zeros([num_context_stations_current, num_crowdflow_stations])
            for i in range(num_crowdflow_stations):
                lat_crowdflow_station,lon_crowdflow_station = expected_station_info[i]
                distances = []
                for j in range(num_context_stations):
                    distances.append(haversine(context_station_latlng_list[j][0],context_station_latlng_list[j][1],lat_crowdflow_station,lon_crowdflow_station))

                distances = np.array(distances)
                assignment_matrix_current[np.argmin(distances),i] = 1
                return assignment_matrix,assignment_matrix_current
        else:
            context_station_latlng_list = list(self.latlon2context.keys())
            num_context_stations = len(context_station_latlng_list)
            num_crowdflow_stations = len(expected_station_info)
            assignment_matrix = np.zeros([num_context_stations, num_crowdflow_stations])
            for i in range(num_crowdflow_stations):
                lat_crowdflow_station,lon_crowdflow_station = expected_station_info[i]
                distances = []
                for j in range(num_context_stations):
                    distances.append(haversine(context_station_latlng_list[j][0],context_station_latlng_list[j][1],lat_crowdflow_station,lon_crowdflow_station))

                distances = np.array(distances)
                assignment_matrix[np.argmin(distances),i] = 1
            print(assignment_matrix)
        # pdb.set_trace()
        # assignment_matrix = np.concatenate([np.ones((1,num_crowdflow_stations)),np.zeros((1,num_crowdflow_stations)),np.zeros((1,num_crowdflow_stations))],axis=0)
        # print(assignment_matrix.shape)
            return assignment_matrix
    
    def TTransformation_forecast(self,expected_time_range,expected_time_fitness):
        date_list = list(pd.date_range(expected_time_range[0],expected_time_range[1],freq='{}min'.format(expected_time_fitness)))
        date_list.pop(-1)
        # pdb.set_trace()


        station_info = list(self.st_context_current.keys())
        # 提前建立好 date_list 和 时间分辨率的关系
        latlon2tcontext_current = {}
        for sta_lat,sta_lon in station_info:
            context_date_list = list(self.st_context_current[(sta_lat,sta_lon)].keys())
            flow_date_list = list(date_list)
            # pdb.set_trace()
            t_context = self.st_context_current[(sta_lat,sta_lon)]
            t_context_transformed = np.array(t_transform(t_context=t_context,flow_date_list=flow_date_list,context_date_list=context_date_list))
            
            latlon2tcontext_current[(sta_lat,sta_lon)]=t_context_transformed
            
        self.latlon2context_current = latlon2tcontext_current
        
                
        
    
    def STTransformation(self,expected_time_range,expected_time_fitness,expected_station_info):
        '''
        Input:
            default STcontext with shape: {(T_1, D),(T_2, D),(T_3, D)...,(T_N,D)}, N is the number of station, including datetime information and spatial information.
            expected TimeRange and TimeFitness
            expected station_info
        Output:
            transformed_STcontext with shape: (T',N',D')
        '''
        date_list = list(pd.date_range(expected_time_range[0],expected_time_range[1],freq='{}min'.format(expected_time_fitness)))
        date_list.pop(-1)
        # pdb.set_trace()
        num_timeslots = len(date_list)
        num_stations = len(expected_station_info)

        station_info = list(self.st_context.keys())
        tmp_ind = list(self.st_context[station_info[0]].keys())[0]
        dimensions = len(self.st_context[station_info[0]][tmp_ind])
        st_context_transformed = np.zeros([num_timeslots,num_stations,dimensions])
        # 提前建立好 date_list 和 时间分辨率的关系
        latlon2tcontext = {}
        for sta_lat,sta_lon in station_info:
            context_date_list = list(self.st_context[(sta_lat,sta_lon)].keys())
            flow_date_list = list(date_list)
            t_context = self.st_context[(sta_lat,sta_lon)]
            t_context_transformed = np.array(t_transform(t_context=t_context,flow_date_list=flow_date_list,context_date_list=context_date_list,span_radius=5))
            latlon2tcontext[(sta_lat,sta_lon)]=t_context_transformed
        for s in tqdm(range(num_stations)):
            expected_lat,expected_lon = expected_station_info[s]
            coeffients = {}
            tmp_sum = 0
            tmp = np.zeros(t_context_transformed.shape)
            for sta_lat,sta_lon in station_info:
                coeffients[(sta_lat,sta_lon)] = haversine(expected_lat,expected_lon,sta_lat,sta_lon)
                tmp_sum += coeffients[(sta_lat,sta_lon)]
            
            for sta_lat,sta_lon in station_info:
                coeffients[(sta_lat,sta_lon)] /= tmp_sum
                tmp+=coeffients[(sta_lat,sta_lon)]*latlon2tcontext[(sta_lat,sta_lon)]
            st_context_transformed[:,s,:] = tmp
                # st_context_transformed[t,s,:] = 
        self.st_context_transformed = st_context_transformed
        # pass
        
    def get_feature(self, historical_window, train_data_index, test_data_index, future_window=0,normalize='MaxMin-column'):
        '''
        Input:
            stcontext_transformed: (T',N',D)
        Output:
            features with shape: (num_samples, P, N', D)
        '''
        if self.args['type'] == 'Forecast Weather':
            st_context_transformed = self.st_context_transformed
            st_context_transformed_forecast = self.st_context_forecast_transformed
            _,num_of_stations,dimensions = st_context_transformed.shape
            _,num_of_stations_forecast,dimensions_forecast = st_context_transformed_forecast.shape
            st_context_transformed_padding = st_context_transformed
            
            suffix = np.zeros([future_window,num_of_stations_forecast,dimensions_forecast])
            # pdb.set_trace()
            st_context_transformed_forecast_padding = np.concatenate([st_context_transformed_forecast,suffix])
            ind = min(test_data_index)
            self.normalizer = chooseNormalizer(normalize,st_context_transformed_padding[:ind])
            self.normalizer_forecast = chooseNormalizer(normalize,st_context_transformed_forecast_padding[:ind])
            st_context_transformed_padding = self.normalizer.transform(st_context_transformed_padding)
            st_context_transformed_forecast_padding = self.normalizer_forecast.transform(st_context_transformed_forecast_padding)
            train_st_context = []
            train_st_context_forecast = []
            test_st_context = []
            test_st_context_forecast = []
            # pdb.set_trace()
            # print(train_data_index)
            for ind in train_data_index:
                train_st_context.append(st_context_transformed_padding[ind-historical_window:ind,...])
                train_st_context_forecast.append(st_context_transformed_forecast_padding[ind:ind+future_window+1])
                
            self.train_st_context = np.array(train_st_context)
            self.train_st_context_forecast = np.array(train_st_context_forecast)
            
            for ind in test_data_index:
                test_st_context.append(st_context_transformed_padding[ind-historical_window:ind,...])
                test_st_context_forecast.append(st_context_transformed_forecast_padding[ind:ind+future_window+1])
            
            self.test_st_context = np.array(test_st_context)
            self.test_st_context_forecast = np.array(test_st_context_forecast)
            
        else:
            st_context_transformed = self.st_context_transformed
            _,num_of_stations,dimensions = st_context_transformed.shape
            if future_window==-1:
                st_context_transformed_padding = st_context_transformed
            else:
                suffix = np.zeros([future_window,num_of_stations,dimensions])
                st_context_transformed_padding = np.concatenate([st_context_transformed,suffix])
            ind = min(test_data_index)
            self.normalizer = chooseNormalizer(normalize,st_context_transformed_padding[:ind])
            st_context_transformed_padding = self.normalizer.transform(st_context_transformed_padding)
            train_st_context = []
            test_st_context = []
            # pdb.set_trace()
            # print(train_data_index)
            for ind in train_data_index:
                train_st_context.append(st_context_transformed_padding[ind-historical_window:ind+future_window+1,...])
            self.train_st_context = np.array(train_st_context)
            # pdb.set_trace()
            for ind in test_data_index:
                test_st_context.append(st_context_transformed_padding[ind-historical_window:ind+future_window+1,...])
            self.test_st_context = np.array(test_st_context)
        
        

def t_select_span(flow_date_list,context_date_list,span_radius=10):
    current_ind = 0
    window = []
    span_dict = {}
    for flow_date in flow_date_list:
        # if current_ind >= len(context_date_list):
        # pdb.set_trace()
        while (current_ind<len(context_date_list)) and (flow_date>=context_date_list[current_ind]):
            window.append(context_date_list[current_ind])
            current_ind+=1
        while len(window) > span_radius:
            window.pop(0)
        span_dict[flow_date] = deepcopy(window)
    # print(span_dict)
    return span_dict


def t_calculate(window,t_context,flow_date:pd.DatetimeIndex):
    value = np.zeros(len(t_context[list(t_context.keys())[0]]))
    if len(window) == 0:
        return value
    if len(window) == 1:
        # pdb.set_trace()
        value += np.array(list(t_context[window[0]].values()))
        return value
    weights = []
    for dt in window:
        weights.append((flow_date-dt).total_seconds())
    weights = np.array(weights)
    weights /= np.sum(weights)
    for ind,dt in enumerate(window):
        value+=weights[ind]*np.array(list(t_context[dt].values()))
    return value

def t_transform(t_context,flow_date_list,context_date_list,span_radius=1): 
    # pdb.set_trace()
    span_dict = t_select_span(flow_date_list,context_date_list,span_radius=span_radius)
    t_context_transformed = []
    
    for k,v in zip(span_dict.keys(),span_dict.values()):
        t_context_transformed.append(t_calculate(v,t_context,k))
    return t_context_transformed

def s_select_span(flow_date_list,context_date_list,span_radius=2):
    start_date = context_date_list[0]
    end_date = start_date
    for flow_date in flow_date_list:
        if flow_date>=start_date:
            pass
        pass
    pass

def s_calculate():
    pass

def s_transform():
    pass


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

if __name__ =='__main__':
    import pickle as pkl
    stcontextloader = STContextLoader()
    args = {'name':'nyc_bikeandtaxi','type':'Historical Weather'}
    stcontextloader.get_stcontext(args=args)
    with open('/home/pku/fjy/gwnet/data/Taxi_NYC.pkl','rb') as fp:
        dataset = pkl.load(fp)
    expected_station_info = []
    for s in dataset['Node']['StationInfo']:
        expected_station_info.append((float(s[2]),float(s[3])))
    expected_time_range = ['2017-01-01','2017-07-01']
    expected_time_fitness = 15
    stcontextloader.STTransformation(expected_time_range=expected_time_range,expected_station_info=expected_station_info,expected_time_fitness=expected_time_fitness)
    stcontextloader.get_feature(6,list(range(8,15000)),list(range(15000,17376)),future_window=0)
    print(stcontextloader.train_st_context.shape)
    print(stcontextloader.test_st_context.shape)