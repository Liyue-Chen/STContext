from abc import ABC, abstractmethod
from math import radians, cos, sin, asin, sqrt
import pandas as pd
import pickle
import  numpy as np


City='Pedestrian_Melbourne'
#Dataset_name,eg:Bike_DC,Bike_NYC.

Dataset_path="C:\\Users\\tf20\\Desktop\\DC\\新建文件夹\\code\\processed_code"
#The folder where the "City" dataset pkl file is located.eg: Bike_NYC.pkl is located in Dataset_path.

RAWPOI_path="C:\\Users\\tf20\\Desktop\\code\\processed_code\\RAWPOI\\Pedestrian_Melbourne"
#The folder where RawPOI files are located.eg: Pedestrian_Melbourne-2022-01-01.xls is located in Dataset_path.

POI_type_file_path='test_store.pkl'
#Important POI type is  stored in POI_type_file_path.

Coverage=130
#"Coverage" meters around the Node is considered as POIs around the site.

Time_begin=2021
Time_end=2022
#Here is the time span, e.g. 2013 to 2017,then Time_begin=2013,Time_end=2017

def compute_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    distance = c * r * 1000
    if (distance > Coverage):
        return False
    return True



def processed(city_type,Time):

    str1="{}\\{}.pkl".format(Dataset_path,city_type)
    Poi_exc="{}\\{}-{}.xls".format(RAWPOI_path,city_type,Time)
    with open(str1, "rb") as fp:
        Bikedata = pickle.load(fp)
    data = pd.read_excel(Poi_exc)  # 读取excel数据
    dict={Important_poi_list[i]:i for i in range(len(Important_poi_list)) }
    # dict is a dictionary mapping {poi name: serial number}

    NodeGeoLa = [item[2] for item in Bikedata['Node']['StationInfo']]
    NodeGeoIn = [item[3] for item in Bikedata['Node']['StationInfo']]
    # NodeGeoLa,NodeGeoIn are latitude and longitude coordinates of the dataset site

    # Bikedata读取的是站点信息。
    #data stores  the relative POI data, while Bikedata stores node information of the dataset.
    poi_num=len(Important_poi_list)
    poi_set= set(Important_poi_list)
    poi=np.zeros((len(Bikedata['Node']['TrafficNode'][0]), poi_num))

    for _ in range (len(NodeGeoIn)):
        for j in range(len(data)):
            if (data.loc[j]['fclass'] not in poi_set ):
                continue
            In, la = data.loc[j]['lon'],data.loc[j]['lat']
            if(compute_distance(float(la),float(In),float(NodeGeoLa[_]),float(NodeGeoIn[_]))):
                poi[_][dict[data.loc[j]['fclass']]]=poi[_][dict[data.loc[j]['fclass']]]+1
    return poi





#Load list of the important POI type.
with open(POI_type_file_path, 'rb') as f:
    Important_poi_list=pickle.load(f)


for i in range(int(Time_begin),int(Time_end)+1):
    Time_='{}-01-01'.format(i)
    a=processed(City,Time_)
    with open('{}_{}_Coverage={}.pkl'.format(City, Time_,Coverage), 'wb') as f:
        pickle.dump(a, f)
