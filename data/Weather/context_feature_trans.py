import os
import pandas as pd
from tqdm import tqdm
import datetime
import calendar
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

col = ['key', 'class', 'expire_time_gmt', 'obs_id', 'obs_name', 'valid_time_gmt', 
         'day_ind', 'temp', 'wx_icon', 'icon_extd', 'wx_phrase', 'pressure_tend', 
         'pressure_desc', 'dewPt', 'heat_index', 'rh', 'pressure', 'vis', 'wc', 
         'wdir', 'wdir_cardinal', 'gust', 'wspd', 'max_temp', 'min_temp', 
         'precip_total', 'precip_hrly', 'snow_hrly', 'uv_desc', 'feels_like', 
         'uv_index', 'qualifier', 'qualifier_svrty', 'blunt_phrase', 'terse_phrase', 
         'clds', 'water_temp', 'primary_wave_period', 'primary_wave_height', 
         'primary_swell_period', 'primary_swell_height', 'primary_swell_direction', 
         'secondary_swell_period', 'secondary_swell_height', 'secondary_swell_direction']
all_type = ['Cloudy', 'Fair', 'Rainy', 'Foggy', 'Snowy',\
             'Windy', 'Overcast', 'Thunderstorm', 'Tornado', 'Hail']
cities_ls = ['NYC', 'BAY','Chicago','DC','LA','Melbourne']


def poor_or_good_weather(data: pd.DataFrame)->pd.DataFrame:
    """
    from 《An improved deep belief network for traffic prediction considering weather factors》
    1. bool: 1 for poor weather; 0 for good weather  
    2. poor weather refers to hail, fog, snowstorm, etc.  
    3. According to HCM 2000, the poor weather is defined as: rainfall > 0.1 in./h, snowfall > 0.05 in./h, or Fog Visibility < 1 miles.  
    
    args:
    -----
    data: [T, D_t]

    return:
    -----
    poor or good weather, bool(see 1.): [T, 1]

    """
    # poor weather refers to hail, fog, snowstorm, etc.
    num_rows = len(data)
    dict_new = {'valid_time_gmt':[], 'poor_weather':[]}
    for i in range(num_rows):
        dict_new['valid_time_gmt'].append(data.loc[i, 'valid_time_gmt'])
        state_wea = data.loc[i, 'wx_phrase']

        # poor weather refers to hail, fog, snowstorm, etc.
        # ['Cloudy', 'Fair', 'Rainy', 'Foggy', 'Snowy',
        #  'Windy', 'Overcast', 'Thunderstorm', 'Tornado', 'Hail']
        if state_wea[3] == '1' or state_wea[4] == '1' or state_wea[9] == '1' or\
           state_wea[6] == '1' or state_wea[7] == '1' or state_wea[8] == '1':
            dict_new['poor_weather'].append('1')
            continue

        # According to HCM 2000, the poor weather is defined as:
        # rainfall > 0.1 in./h, snowfall > 0.05 in./h, or Fog Visibility < 1 miles.
        rainfall = data.loc[i, 'precip_hrly']
        snowfall = data.loc[i, 'snow_hrly']
        vis      = data.loc[i, 'vis']
        # There are some differences in vis and snowfall, but Snowfall is scarce
        if rainfall>0.1 or snowfall>0.05 or vis<5:
            dict_new['poor_weather'].append('1')
            continue
        
        # good weather
        dict_new['poor_weather'].append('0')
    return pd.DataFrame(dict_new)

def get_wx_phrase(data: pd.DataFrame)->pd.DataFrame:
    """
    all_type = ['Cloudy', 'Fair', 'Rainy', 'Foggy', 'Snowy',\
             'Windy', 'Overcast', 'Thunderstorm', 'Tornado', 'Hail']
    
    args:
    -----
    data: [T, D_t]
             
    return:
    -----
    one-hot encoding: [T, 1] str  \n
    e.g.
        1000010000
        means Cloudy and Windy 
    """
    return data.loc[:, 'wx_phrase']

def comfort(data: pd.DataFrame)->pd.DataFrame:
    """
    args:
    -----
    data: [T, D_t]
             
    return:
    -----
    comfort value: [T, 1] float between [0, 1]  \n
    
    Sunshine_fine: 1  
    Cloudy: 0.8  
    Overcast_sky: 0.7  
    Sprinkle: 0.5  
    Middle_rain: 0.4  
    Drencher: 0.3  
    Cyclone: 0.1  
    """
    num_rows = len(data)
    dict_new = {'valid_time_gmt':[], 'comfort':[]}
    for i in range(num_rows):
        dict_new['valid_time_gmt'].append(data.loc[i, 'valid_time_gmt'])
        state_wea = data.loc[i, 'wx_phrase']
        cloud = data.loc[i, 'clds']
        rainfall = data.loc[i, 'precip_hrly']
        snowfall = data.loc[i, 'snow_hrly']
        vis      = data.loc[i, 'vis']

        if state_wea[7] == '1' or state_wea[8] == '1' or state_wea[9] == '1':
            dict_new['comfort'].append(0.1)
            continue
        if state_wea[2] == '1' or rainfall > 0.05 or snowfall>0:
            if rainfall>0.1: dict_new['comfort'].append(0.3)
            else: dict_new['comfort'].append(0.4)
            continue
        if state_wea[2] == '1' or rainfall > 0:
            dict_new['comfort'].append(0.5)
            continue
        if state_wea[6] == '1' or cloud == 'OVC' or vis < 5:
            dict_new['comfort'].append(0.7)
            continue
        if state_wea[0] == '1' or cloud in ['OVC', 'BKN', 'SCT']:
            dict_new['comfort'].append(0.8)
            continue
        
        dict_new['comfort'].append(1.0)
    return pd.DataFrame(dict_new)

def my_norm(data: pd.DataFrame)->pd.DataFrame:
    """
    normalize: Subtract the minimum value of each column of data,  \n
    and then divide it by the difference between the maximum and minimum values of that column

    args:
    -----
    data: [T, D_t]

    return:
    -----
    data_normalized: [T, D_t]

    """
    columns_to_normalize = ['temp', 'pressure_tend', 'dewPt', \
                'heat_index', 'rh', 'pressure', 'vis', 'wc',\
                'gust', 'wspd', 'max_temp',\
                'min_temp', 'precip_total', 'precip_hrly', 'snow_hrly',\
                'feels_like', 'uv_index', ]

    # 归一化处理
    data[columns_to_normalize] = (data[columns_to_normalize] - data[columns_to_normalize].min()) \
                                / (data[columns_to_normalize].max() - data[columns_to_normalize].min())

    return data

def differ(data: pd.DataFrame)->pd.DataFrame:
    """
    Correct the temperature value at each time point by subtracting the \n
    temperature value from the corresponding time point on the previous day

    args:
    -----
    data: [T, D_t]

    return:
    -----
    data: [T, D_t]

    """
    columns_to_diff = ['temp', 'dewPt', 'heat_index', 'wc', 'feels_like', ]
    num_rows = len(data)
    for i in range(24, num_rows):
        for col in columns_to_diff:
            data.loc[i, col] -= data.loc[i-24, col]
    return data 

def my_PCA(data: pd.DataFrame, n_components = 5, features = [], \
           fill_zero = [], del_features = True)->pd.DataFrame:
    """
    Perform principal component analysis(PCA) on the input data

    args:
    -----
    - data: [T, D_t]
    - n_components(int): Retain n_components maximum principal components, default: 5
    - features: A list containing multiple column names, indicating which columns to perform PCA on. 
    It should be noted that the column names inside must be column names that exist in the arg 'data', 
    and the data in these columns must be all numbers (because PCA can only analyze numbers).  \n
        default: Defined in this function, which can be viewed through the source code of this function
    - fill_zero: In PCA, if there is NaN, an error will be reported and the missing value needs to be filled in.
    A list containing multiple column names, which must appear in the arg'features',
    missing values in these columns will be filled with 0.
    Missing values in other columns will be filled with the mean of the data in this column.  \n
        default: Also defined in this function
    - del_features: Whether remove the columns in the arg 'features' from the arg 'data', default: True

    return:
    -----
    - data:  \n
    if del_features == True: [T, D_t - len(features) + n_components]  \n
    else: [T, D_t + n_components]

    """
    new_data = data.copy()
    if features == []:
        # Select the features to perform PCA on
        features = ['temp', 'pressure_tend', 'dewPt', \
                    'heat_index', 'rh', 'pressure', 'vis', 'wc',\
                    'gust', 'wspd', 'max_temp',\
                    'min_temp', 'precip_total', 'precip_hrly', 'snow_hrly',\
                    'feels_like', 'uv_index', ]
    if fill_zero == []:
        # Missing values in these columns will be filled with 0.
        fill_zero = ['pressure_tend', 'precip_total', 'precip_hrly', 'snow_hrly', 'uv_index']

    # Fill missing value
    for col in features:
        if col not in fill_zero and not new_data[col].isna().all():
            mean = new_data[col].mean()
            new_data[col] = new_data[col].fillna(mean)
        else: new_data[col] = new_data[col].fillna(0)

    # Extract the feature data for PCA
    X = new_data[features]

    # Create a PCA object and fit the data
    pca = PCA(n_components=n_components)  # Specify the number of principal components to retain
    X_pca = pca.fit_transform(X)

    # Create new column names
    new_columns = []
    for i in range(n_components):
        new_columns.append(f'PCA Feature{i+1}')

    # Add the PCA transformed result as a new column to the DataFrame
    new_data[new_columns] = pd.DataFrame(X_pca)

    if del_features:
        # Drop the original feature columns
        new_data.drop(columns=features, inplace=True)

    # Print the updated DataFrame
    # print(new_data)

    return new_data
