import logging
import numpy as np
import os
import pickle
from abc import ABC, abstractmethod
from ..preprocess.time_utils import is_work_day_china, is_work_day_america, is_valid_date

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