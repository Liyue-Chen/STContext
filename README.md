# STContext

## Structure Overview

```bash
├── README.md
├── AnalysisCode
├── Data
│   ├── AQI
│   ├── Administrative_Division
│   ├── Demographics
│   ├── Historical_Weather
│   ├── Holiday
│   ├── POI
│   ├── Road_Network
│   ├── Traffic_Data
│   └── Weather_Forecast
├── Experiments
├── img
├── UCTB
└── DataProcessCode
```

- The `Data` directory is designated for storing spatiotemporal data and context data, including subfolders such as `Holiday`, `POI`, and `Weather`. The files within these subfolders should be named using the format `City+TimeSpan+DataType.csv`, for example, `NYC_20160101_20170101_Weather.csv`. Among these, `Traffic_Data` is our dataset containing spatiotemporal traffic data.

- The `AnalysisCode` directory is intended for storing the code or notebook files used for data analysis. The file names should reflect the purpose of the file. For instance, the code for analyzing the correlation between POI and traffic flow can be named `POI_correlation_analysis.py`.

- The `DataProcessCode` directory is used for storing data preprocessing code. The files can be named according to the type of data being processed. For example, the code for processing POI data can be named `POI_processing.py`.

- The `Experiments` directory is used to store the code for benchmark experiments.

- The `img` directory is for storing images required by documents such as `README.md`.

- `UCTB`: This is the version 0.3.5 of the code. All merged code needs to be compatible with this version of UCTB.



## Traffic Dataset

| **Application**  | **City**  | **Interval** |     **Time Span**     |
| :--------------: | :-------: | :----------: | :-------------------: |
|   Bike-sharing   |    NYC    |    5 mins    | 2013/07/01-2017/09/30 |
|   Bike-sharing   |  Chicago  |    5 mins    | 2013/07/01-2017/09/30 |
|   Bike-sharing   |    DC     |    5 mins    | 2013/07/01-2017/09/30 |
| Pedestrian count | Melbourne |   60 mins    | 2021/01/01-2022/11/01 |
|  Vehicle speed   |    LA     |    5 mins    | 2012/03/01-2012/06/28 |
|  Vehicle speed   |    BAY    |    5 mins    | 2017/01/01-2017/07/01 |
|   Ride-sharing   |  Chicago  |   60 mins    | 2013/01/01-2022/03/31 |
|   Ride-sharing   |    NYC    |   60 mins    | 2016/01/01-2023/06/01 |
|      Metro       |    NYC    |   60 mins    | 2022/02/01-2023/12/21 |

## Dataset Overview

|                      | Historical Wea. | Wea. Forecast | AQI | Holiday |  TP   |  POI  | Demo  | Road  |  AD   |
| :------------------: | :-------------: | :-----------: | --- | :-----: | :---: | :---: | :---: | :---: | :---: |
|       Bike NYC       |        ✅        |       ✅       | ✅   |  ✅       |  ✅     |  ✅     |       |   ✅    |   ✅    |
|     Bike Chicago     |        ✅        |       ✅       | ✅   |  ✅       |  ✅     |   ✅    |       |  ✅     |  ✅     |
|       Bike DC        |        ✅        |       ✅       | ✅   |  ✅       |  ✅     |  ✅     |       |  ✅     |  ✅     |
| Pedestrian Melbourne |        ✅        |       ✅       |     |   ✅      |   ✅    |  ✅     |      |  ✅     |   ✅    |
|       METR-LA        |        ✅        |       ✅       | ✅   |  ✅       |   ✅    |  ✅     |      | ✅      |  ✅     |
|       PEMS-BAY       |        ✅        |       ✅       | ✅   |  ✅       |  ✅     |  ✅     |       |  ✅     |  ✅     |
|       Taxi NYC       |        ✅        |       ✅       | ✅   |  ✅       |  ✅     |  ✅     |      | ✅      |   ✅    |
|     Ride Chicago     |        ✅        |       ✅       | ✅   |  ✅       |  ✅     |       |      |   ✅    |   ✅    |
|      Metro NYC       |        ✅        |       ✅       | ✅   |  ✅       |  ✅     |  ✅     |       |  ✅     |  ✅     |



##  Descirption Example

> First, we provide overall information about this contextual data, including its collection source, metadata description, and instructions for loading and using the data. Please use a few brief sentences.

### Meta Data

The metadata table is displayed as a table. You can use the following two examples for guidance: attribute and description are mandatory fields, while the third column may include additional information or offer a specific example.

| Attribute | Description                                   | Possible Range of Values              |
| --------- | --------------------------------------------- | ------------------------------------- |
| ID        | The identifier of a sensor in PeMS            | 6 to 9 digits number                  |
| Lat       | The latitude of a sensor                      | Real number                           |
| Lng       | The longitude of a sensor                     | Real number                           |
| District  | The district of a sensor in PeMS              | 3, 4, 5, 6, 7, 8, 10, 11, 12          |
| County    | The county of a sensor in California          | String                                |
| Fwy       | The highway where a sensor is located         | String starts with 'I', 'US', or 'SR' |
| Lane      | The number of lanes where a sensor is located | 1, 2, 3, 4, 5, 6, 7, 8                |
| Type      | The type of a sensor                          | Mainline                              |
| Direction | The direction of the highway                  | N, S, E, W                            |

| Attribute      | Description                                                                          | Example                                                                |
| -------------- | ------------------------------------------------------------------------------------ | ---------------------------------------------------------------------- |
| Order ID       | Unique identifier for each sales order                                               | 10013, 10014                                                           |
| Date           | Date of the sales transaction                                                        | 2024-01-04                                                             |
| Category       | Broad category of the product sold                                                   | Electronics, Home Appliances, Clothing, Books, Beauty Products, Sports |
| Product Name   | Specific name or model of the product sold                                           | iPhone 14 Pro, Dyson V11 Vacuum                                        |
| Quantity       | Number of units of the product sold in the transaction                               | 1, 5                                                                   |
| Unit Price     | Price of one unit of the product                                                     | 999 dollars                                                            |
| Total Price    | Total revenue generated from the sales transaction (Quantity * Unit Price).          | 63.96                                                                  |
| Region         | Geographic region where the transaction occurred (e.g., North America, Europe, Asia) | North America, Europe, Asia                                            |
| Payment Method | Method used for payment                                                              | Credit Card, PayPal, Debit Card                                        |



### Load and Use

This section provides an overview of the steps for preparing and organizing contextual datasets. The following processes are included:

1. **Data Preprocessing**: The data is preprocessed to align and filter the traffic data using a feature transformation script.
2. **Storage of Processed Data**: The processed data is saved in a directory organized by area, road, and point of interest, so on, with files named `City+TimeSpan+DataType.csv` (e.g., `NYC_2016_2017_POI_transformed.npy`).
3. **File Types**: Description of file types used for storing different kinds of information:
   - `xxx.geo` for geographic entity attributes
   - `xxx.rel` for relationships between entities
   - `xxx.dyna` for traffic condition information.
   - `config.json` for supplementary table descriptions.
4. **Customization**: Users can create their own function for data preparation (i.e., transformation function) and run construction scripts to adapt to various datasets.



## Historical Weather

### Meta Data

| Attribute       | Description                                                      | Example                                           |
| --------------- | ---------------------------------------------------------------- | ------------------------------------------------- |
| key             | Unique identifier of the observing site                          | KSJC,KLGA                                         |
| class           | Type of observation                                              | observation                                       |
| expire_time_gmt | Expiration time of the observation                               | 2017-01-01 03:53:00                               |
| obs_id          | Observation ID                                                   | KSJC                                              |
| obs_name        | Name of the observing site                                       | San Jose                                          |
| valid_time_gmt  | Observation validity period                                      | 2017-01-01 00:53:00                               |
| day_ind         | Time of day (day or night)                                       | D,N                                               |
| temp            | Temperature (Fahrenheit)                                         | -14-118                                           |
| wx_icon         | Weather Icon Code                                                | 0-33                                              |
| icon_extd       | Weather Icon Extension Code                                      | 3300                                              |
| wx_phrase       | Weather phrase description                                       | Fair, Mostly Cloudy,Light Rain                    |
| pressure_tend   | Air pressure trend                                               | 0(stable),1(increase),2(decrease)                 |
| pressure_desc   | Discription of air pressure trend                                | Rising Rapidly                                    |
| dewPt           | Dew Point Temperature (Fahrenheit)                               | -24-80                                            |
| heat_index      | Heat Index                                                       | -14-114                                           |
| rh              | Relative humidity(%)                                             | 0-100                                             |
| pressure        | pressure                                                         | 28-31                                             |
| vis             | Visibility (miles)                                               | 0-10                                              |
| wc              | Wind Chill Index                                                 | -41~118                                           |
| wdir            | Wind direction (angle)                                           | 10-360                                            |
| wdir_cardinal   | Description of wind direction                                    | NE,S,N                                            |
| gust            | Wind speed instantaneous value (miles per hour)                  | 16-74                                             |
| wspd            | Wind Speed ​​(mph)                                               | 0-60                                              |
| max_temp        | Maximum Temperature (Fahrenheit)                                 | -12-118                                           |
| min_temp        | Minimum Temperature (Fahrenheit)                                 | -15-94                                            |
| precip_total    | Total precipitation (inches)                                     | 0-4.88                                            |
| precip_hrly     | Hourly precipitation (inches)                                    | 0-1.98                                            |
| snow_hrly       | Hourly precipitation (inches)                                    | 1.0-3.0                                           |
| uv_desc         | UV Index Description                                             | 'Low', 'Moderate', 'High', 'Very High', 'Extreme' |
| feels_like      | Feeling temperature (Fahrenheit)                                 | -41-114                                           |
| uv_index        | [UV Index](https://www.epa.gov/sunsafety/calculating-uv-index-0) | -618-12                                           |

### Load and Use

```python
import pandas as pd
import os
# dataset path configuration
historical_weather_data_dir = '{your_dir_path}'
city='NYC'
start_date = '20130101'
end_date = '20140101'
context_type = 'Historical_Weather'
dataset_name = '{}_{}_{}_{}.csv'.format(city, start_date, end_date, context_type)
dataset_path = os.path.join(historical_weather_data_dir, dataset_name)

# read csv
df = pd.read_csv(dataset_path)

# fill missing values with the previous value to prevent data leakage

df.ffill(inplace=True)

# more process steps(optional)



```
You can obtain raw historical weather without missing values with code snippets above, however if you want to make traffic prediction with them, you need to preprocess the data to align with the traffic data. We implement a `context_dataloader` to align context data with crowd flow data. More details please refer to `UCTB/dataset/context_dataloader.py` 

## Weather Forecast

### Meta Data

| Attribute | Description                                                     | Example        |
| --------- | --------------------------------------------------------------- | -------------- |
| u10       | 10m u component of wind (metre per second)                      | -3.2, 5.1      |
| v10       | 10m v component of wind (metre per second)                      | 4.8, -1.6      |
| fg10      | 10m wind gust since previous post processing (metre per second) | 7.3, 12.1      |
| d2m       | 2m dewpoint temperature (kelvin)                                | 280.15, 290.55 |
| t2m       | 2m temperature (kelvin)                                         | 285.65, 300.25 |
| i10fg     | instantaneous 10m wind gust (metre per second)                  | 10.5, 15.3     |
| mx2t      | maximum 2m temperature since previous post processing (kelvin)  | 295.75, 310.85 |
| mn2t      | minimum 2m temperature since previous post processing (kelvin)  | 275.35, 283.15 |
| sd        | snow depth (metres of water equivalent)                         | 0.12, 0.25     |
| sf        | snowfall (metres of water equivalent)                           | 0.05, 0.10     |
| tcc       | total cloud cover (Dimensionless)                               | 0.7, 1.0       |
| tp        | total precipitation (metres)                                    | 0.02, 0.08     |

### Load and Use

```python
import pandas as pd
import os
# dataset path configuration
historical_weather_data_dir = '{your_dir_path}'
city='NYC'
start_date = '20130101'
end_date = '20140101'
context_type = 'Weather_Forecast'
dataset_name = '{}_{}_{}_{}.csv'.format(city, start_date, end_date, context_type)
dataset_path = os.path.join(historical_weather_data_dir, dataset_name)

# read csv
df = pd.read_csv(dataset_path)

# fill missing values with the previous value to prevent data leakage

df.ffill(inplace=True)

# more process steps(optional)

```
You can obtain raw weather forecast without missing values with code snippets above, however if you want to make traffic prediction with them, you need to preprocess the data to align with the traffic data. We implement a `context_dataloader` to align context data with crowd flow data. More details please refer to `UCTB/dataset/context_dataloader.py` 

## AQI

### Meta Data

| Attribute           | Description                                                                                                                         | Example                                           |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| State Code          | The FIPS code of the state in which the monitor resides.                                                                            | 36(New York)                                      |
| County Code         | The FIPS code of the county in which the monitor resides.                                                                           | 5(Brox)                                           |
| Site Num            | A unique number within the county identifying the site.                                                                             | 110                                               |
| Parameter Code      | The AQS code corresponding to the parameter measured by the monitor.                                                                | 42401(S02)                                        |
| POC                 | This is the “Parameter Occurrence Code” used to distinguish different instruments that measure the same parameter at the same site. | 1,2...                                            |
| Latitude            | The monitoring site’s angular distance north of the equator measured in decimal degrees.                                            | 40.816                                            |
| Longitude           | The monitoring site’s angular distance east of the prime meridian measured in decimal degrees.                                      | -73.902                                           |
| Datum               | The Datum associated with the Latitude and Longitude measures.                                                                      | WGS84                                             |
| Parameter Name      | The name or description assigned in AQS to the parameter measured by the monitor. Parameters may be pollutants or non-pollutants    | Sulfur dioxide                                    |
| Date Local          | The calendar date of the sample in Local Standard Time at the monitor.                                                              | 2022-02-01                                        |
| Time Local          | The time of day that sampling began on a 24-hour clock in Local Standard Time.                                                      | 00:00                                             |
| Sample Measurement  | The measured value in the standard units of measure for the parameter.                                                              | 0.9                                               |
| Units of Measure    | The unit of measure for the parameter.                                                                                              | Parts per billion                                 |
| MDL                 | The Method Detection Limit.                                                                                                         | 0.2                                               |
| Uncertainty         | The total measurement uncertainty associated with a reported measurement as indicated by the reporting agency.                      | N/A                                               |
| Qualifier           | Sample values may have qualifiers that indicate why they are missing or that they are out of the ordinary.                          | N/A                                               |
| Method Type         | An indication of whether the method used to collect the data is a federal reference method (FRM)                                    | FEM                                               |
| Method Code         | An internal system code indicating the method (processes, equipment, and protocols) used in gathering and measuring the sample.     | 560                                               |
| Method Name         | A short description of the processes, equipment, and protocols used in gathering and measuring the sample.                          | INSTRUMENTAL - Pulsed Fluorescent 43C-TLE/43i-TLE |
| State Name          | The name of the state where the monitoring site is located.                                                                         | New York                                          |
| County Name         | The name of the county where the monitoring site is located.                                                                        | Brox                                              |
| Date of Last Change | The date the last time any numeric values in this record were updated in the AQS data system.                                       | 2022-04-21                                        |

### Load and Use

```python
import pandas as pd
import os
# dataset path configuration
historical_weather_data_dir = '{your_dir_path}'
city='NYC'
start_date = '20130101'
end_date = '20140101'
context_type = 'AQI'
dataset_name = '{}_{}_{}_{}.csv'.format(city, start_date, end_date, context_type)
dataset_path = os.path.join(historical_weather_data_dir, dataset_name)

# read csv
df = pd.read_csv(dataset_path)

# fill missing values with the previous value to prevent data leakage

df.ffill(inplace=True)

# more process steps(optional)

```
You can obtain raw AQI without missing values with code snippets above, however if you want to make traffic prediction with them, you need to preprocess the data to align with the traffic data. We implement a `context_dataloader` to align context data with crowd flow data. More details please refer to `UCTB/dataset/context_dataloader.py`

## Holiday




| Attribute       | Description                                                      | Example                                           |
| --------------- | ---------------------------------------------------------------- | ------------------------------------------------- |
| Date          | The date of timestamp |    2013/7/1                          |
| Holiday           | Is it a holiday or not?                                  | 0 means no, 1 means yes.                         |
| Holiday_Type | Category of the holiday                              | Thanksgiving Day                             |

### Load and Use

```python


# dataset path configuration
user_data_dir = '{your_dir_path}'
City='Bike_DC'
Year='2013'
import pickle

# Specify the file path.
file_path = '{}__Holiday_{}.pkl'.format(City,Year)

dataset_path = os.path.join(user_data_dir, file_path)

# Load POI data.
with open(dataset_path, 'rb') as f:
    data = pickle.load(f)
```
You can use the following code to load data of Holiday for different datasets. However, if you want to make traffic prediction with them, you need to preprocess the data to align with the traffic data. We implement a `context_dataloader` to align context data with crowd flow data. More details please refer to `UCTB/dataset/context_dataloader.py`



## POI


| Attribute       | Description                                                      | Example                                           |
| --------------- | ---------------------------------------------------------------- | ------------------------------------------------- |
| Osm_id          | Unique identifier used to identify map elements in OpenStreetMap |    42538083                          |
| Name           | Name of the POI                                              | U.S.Bank                               |
| Fclass | Category of the POI                              | Restaurant                               |
| Other_tags          | Other information of the POI                                                   | "addr:city"=>"Santa Clara"                                              |
| Lat        | Latitude of the POI                                      |  40.81632                                       |
| Lng  |  Longitude of the POI                       | -73.90182                             |


The  folders  contain POI data for four cities and relative code.

**Folder:RAWPOI is the raw acquisition of POI data.**

  Each folder for the current city's POI file: file naming: "city - year - month - day.xls" (representing the current number and type of POI summary for this day under this city.eg:Bike_Chicago-2013-01-01.xls).

 - osmid&nbsp;&nbsp;&nbsp;(OSM Number)
 - name &nbsp;&nbsp;(POI Name)
 - fclass&nbsp;&nbsp;&nbsp;(POI Category)
 - lon  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(Longitude)
 - lat  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(Latitude)
 - FID  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(Order Number)
<img src="https://github.com/Liyue-Chen/HeteContext/blob/main/data/POIS/RAWPOI/Melbourne_POI_img.png" width = "950" height = "600" alt="Chicago_heat" align=center />

The above is the distribution map of POI in the Pedestrian Melbourne dataset. 



**Folder:PROCESSED POI is the pre-processed POI data.(The format is pkl files)**

When using POI data, for ease of use, we process the Excel file into a pickle file for easy loading. The pickle file stores a two-dimensional numpy array with the dimension of (Numnode, POI_type), where Numnode represents the number of sites and POI_Type is the number of POI types we are considering. The value of a site indicates the number of POIs of that kind within a radius of "R" meters.

"R" represents the site coverage.

Since the size of the area covered by the dataset is different, the value of "R" is also different,The following table shows the values of "R" in different datasets.
"R" as a parameter can be set to different values to generate different pickle files.

|       | Bike_NYC | Bike_DC | Bike_Chicago | Pedestrian_Melbourne |
| :---: | :------: | :-----: | :----------: | :------------------: |
| R(m)  |   500    |   350   |     350      |         130          |




From the graph, it can be seen that there are many types of POIs, but we only consider some significant POIs when using them specifically.For each dataset, we selecte some significant POIs. And we take the union of these POIs to get the final significant POIs.

**Final significant POIs：**


There are two steps to calculating significant POIs for each year.

**Step 1: Calculate the  scores of the POIs and select the POIs with the top 25 values.**

The score for calculating the POI can be formulated as follows:

$$Score_j=\sum_{i}^n N_{ij} \times f_i $$

Where n denotes the number of nodes, 
$i$
denotes the region number, 
$j$
denotes the 
$j$
th POI,
$N_{ij}$
denotes the number of 
$j$
th POI around the site 
$i$
, and 
$f_i$
denotes the average of the 
$i$
th site's traffic flow of  in the dataset.

Then the top 25 POIs with the highest score are selected.



**Step 2: Remove some strongly correlated POIs based on correlation and prior knowledge.**

First, for any two POIs (X,Y),the distribution of them at n sites can be summarized separately by the sequences 
\{
$x_1,x_2,x_3.... .x_i$
\}
,
\{
$y_1,y_2,y_3.... .y_i$
\}
.If  X and  Y have strong correlation (Pearson's correlation coefficient is greater than 0.8), we consider that the  correlations of them are strong, and get the combination of POI with strong correlation.

Next, for the POI combination with strong correlation, we further judge by a priori knowledge whether they are similar POI types and need to be removed.There are some examples as follows:
|       | (x1,y1) |    (x2,y2)     | (x3,y3) |   (x4,y4)   |
| :---: | :-----: | :------------: | :-----: | :---------: |
|   X   |  bank   |      bar       | theatre | car_sharing |
|   Y   |   atm   | drinking water | cinema  | car_rental  |


Some combinations like the above, we just need to retain one of them and remove the POI with smaller score.

Through the above two steps of POIs, we get significant POIs in a year of the dataset.  Then ,We count a union about the significant POIs in each year of the dataset as the dataset's significant POIs. Finally, we count a union about the significant POIs in all datasets to get the final significant POIs.

<img src="https://github.com/Liyue-Chen/HeteContext/blob/main/data/POIS/Preprocess_code/comparepicture.png" width = "320" height = "240" alt="Chicago_heat" align=center />

The figure above shows a comparison of the number of significant POI types and the number of raw POI types.



The final significant POIs are listed below:





<table>
  <tr>
    <th colspan="7" style="text-align:center;">significant POI List</th>
  </tr>
  <tr>
    <td align="center">fast_food</td>
    <td align="center">taxi</td>
    <td align="center">school</td>
    <td align="center">restaurant</td>
    <td align="center">theatre</td>
    <td align="center">pub</td>
    <td align="center">doctors</td>
  </tr>
  <tr>
    <td align="center">kindergarten</td>
    <td align="center">bar</td>
    <td align="center">ice_cream</td>
    <td align="center">telephone</td>
    <td align="center">bench</td>
    <td align="center">bank</td>
    <td align="center">place_of_worship</td>
  </tr>
  <tr>
    <td align="center">bicycle_parking</td>
    <td align="center">nightclub</td>
    <td align="center">car_sharing</td>
    <td align="center">post_box</td>
    <td align="center">embassy</td>
    <td align="center">cafe</td>
    <td align="center">post_office</td>
  </tr>
   <tr>
    <td align="center">vending_machine</td>
    <td align="center">hospital</td>
    <td align="center">police</td>
    <td align="center">food_court</td>
    <td align="center">grave_yard</td>
    <td align="center">toilets</td>
    <td align="center">pharmacy</td>
  <tr>
    <td align="center">ferry_terminal</td>
    <td align="center">fountain</td>
    <td align="center">clock</td>
    <td align="center">bicycle_rental</td>
    <td align="center">library</td>
    <td align="center">parking</td>
    <td align="center">fire_station</td>
  </tr>
</table>

As can be seen in the above table, we consider a total of 35 POIs.So the shape of numpy in the pickle file is (Numnode,35).

**Processing data from raw POI to pickle files.**

After setting "City", "Dataset_path", "N", "RAWPOI_path","Time_begin" and "Time_end" and "POI_type_file_path" parameters, run the following code directly.

```Python
from abc import ABC, abstractmethod
from math import radians, cos, sin, asin, sqrt
import pandas as pd
import pickle
import numpy as np

#Dataset_name,eg:Bike_DC,Bike_NYC.
City='Pedestrian_Melbourne'

#The folder where the "City" dataset pkl file is located.eg: Bike_NYC.pkl is located in Dataset_path.
Dataset_path="C:\\Users\\tf20\\Desktop\\DC\\新建文件夹\\code\\processed_code"

#The folder where RawPOI files are located.eg: Pedestrian_Melbourne-2022-01-01.xls is located in Dataset_path.
RAWPOI_path="C:\\Users\\tf20\\Desktop\\code\\processed_code\\RAWPOI\\Pedestrian_Melbourne"

#significant POI type is  stored in POI_type_file_path.
POI_type_file_path='test_store.pkl'

#"R" meters around the Node is considered as POIs around the site.
R=130

#Here is the time span, e.g. 2013 to 2017,then Time_begin=2013,Time_end=2017
Time_begin=2021
Time_end=2022

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
    if (distance > R):
        return False
    return True
def processed(city_type,Time):

    str1="{}\\{}.pkl".format(Dataset_path,city_type)
    Poi_exc="{}\\{}-{}.xls".format(RAWPOI_path,city_type,Time)
    with open(str1, "rb") as fp:
        Bikedata = pickle.load(fp)
    data = pd.read_excel(Poi_exc)  
    dict={Important_poi_list[i]:i for i in range(len(Important_poi_list)) }
    # dict is a dictionary mapping {poi name: serial number}

    NodeGeoLa = [item[2] for item in Bikedata['Node']['StationInfo']]
    NodeGeoIn = [item[3] for item in Bikedata['Node']['StationInfo']]
    # NodeGeoLa,NodeGeoIn are latitude and longitude coordinates of the dataset site

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
    with open('{}_{}_R=={}.pkl'.format(City, Time_,R), 'wb') as f:
        pickle.dump(a, f)
```

**How to use POI pickle file.**

If we use python:
```Python
#Specify the data set and the corresponding time.
# dataset path configuration
user_data_dir = '{your_dir_path}'
Time='2013-01-01'
City='Bike_DC'
R='130'
import pickle

# Specify the file path.
file_path = '{}_{}_R=={}.pkl'.format(City,Time,R)

dataset_path = os.path.join(user_data_dir, file_path)

# Load POI data.
with open(dataset_path, 'rb') as f:
    data = pickle.load(f)  
```
You can use the following code to load raw context data of POI. However, if you want to make traffic prediction with them, you need to preprocess the data to align with the traffic data. We implement a `context_dataloader` to align context data with crowd flow data. More details please refer to `UCTB/dataset/context_dataloader.py`
**Scene divide Dataset：**


In order to judge the traffic prediction within the different regional areas of a city,we divided each city into the following categories: Central Business District (CBD), Central District, and Non-Central District,three types of areas are mutually exclusive, and the prosperity of the area in decreasing order. A total of 4 cities were divided: NYC_Bike, Chicago_Bike, DC_bike, and Melbourne.

We store the polygonal areas of each region in Scenedevide in shape file.

The different areas are divided in detail as follows:

1. represents the CBD &nbsp;  
2. represents the central area of the city 
3. represents the non-central area of the city

- Chicago_Bike:Scene:1 2 3 （overlap all nodes）
  
  Chicago is divided into 3 scenes, Here we devide the CBD of Chicago as Scene 1, the area around the CBD as Scene 2, and the rest of the city of Chicago as Scene 3.
  Ps:[Link here to download the source of the administrative division.](https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_USA_shp.zip)

  <img src="https://github.com/Liyue-Chen/HeteContext/blob/main/data/POIS/Scene_divide/Chicago_heat.png" width = "600" height = "400" alt="Chicago_heat" align=center />
  
- NYC_Bike:Scene: 1 2 3（overlap all nodes）
  
  NYC is divided into 3 scenes, the region under New York City is divided into Manhattan, Queens and Brooklyn, according to the degree of regional economic prosperity, we devide Manhattan as Scene 1, Queens as Scene 2, Brooklyn and the areas around New York City  as Scene 3.Ps:[Link here to download the source of the administrative division.](https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_USA_shp.zip)

  <img src="https://github.com/Liyue-Chen/HeteContext/blob/main/data/POIS/Scene_divide/NYC_Heat.png" width = "600" height = "400" alt="Chicago_heat" align=center />
  
- DC_bike:Scene: 1 2 3（overlap all nodes）
  
   DC is divided into 3 scenes.Washington is divided into 4 districts in the administrative division, including the southeast, northwest, southwest and northeast.Among them, the northwest is the most economically active area of Washington DC, so as Scene 1 , the remaining three districts (southeast, southwest, northeast) as the Scene 2, there are also some external  bike nodes  outside Washington DC, we set the area distributed with external nodes  outside the administrative  as Scene 3.The green heat spots represent  the bike nodes located outside of Washington, DC.Ps:[Link here to download the source of the administrative division.](https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_USA_shp.zip)

   <img src="https://github.com/Liyue-Chen/HeteContext/blob/main/data/POIS/Scene_divide/DC_heat.png" width = "600" height = "400" alt="Chicago_heat" align=center />

- Melbourne:Scene: 1 2（overlap all nodes）
  
  Due to the  area occupied by the Melbourne dataset is small, the Melbourne dataset was divided into only two zones: the CBD of Melbourne as scene 1 and the non-CBD area of Melbourne as scene 2.Ps:[Link here to download the source of the administrative division.](https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_AUS_shp.zip)

  <img src="https://github.com/Liyue-Chen/HeteContext/blob/main/data/POIS/Scene_divide/Melbourne_heat.png" width = "600" height = "400" alt="Chicago_heat" align=center />

**Instructions for using the Scenedivide dataset**

The  file is stored in shape format, the file name: "Datasetname_SceneNumber.shp".Note that the shx file is a necessary adjunct to the shape file, thus shx files with the same file name must be downloaded in the same folder.

The shape file contains a polygon surrounded area, the area is the divided area.
- Datasetname indicates the name of the dataset.
- SceneNumber indicates the scene division number.

We can use python or arcgis to open the shape file.

If we use python:

```Python
pip install shapely
pip install geopandas
```
Load the shape file and get a polygon. If the location of a point (latitude and longitude) is within a polygon, we can determine whether the point belongs to that area.



## Demographics

| Attribute       | Description                                                      | Example                                           |
| --------------- | ---------------------------------------------------------------- | ------------------------------------------------- |
| FIPSCode          | Unique identifier used to identify map elements in OpenStreetMap |    42538083                   |
| Year           | Name of the POI                                              | U.S.Bank                               |
| GeoID | Category of the POI                              | Restaurant                               |
| x_center          | Other information of the POI                |      40.81632              |
| y_center        | Latitude of the POI                                      |       -73.2562                   |
| tract_bloc        |    The number of the census block and census tract where it is located                                  |    9801001010                       |
| polygon  |  The latitude and longitude coordinates of polygon vertices                       | ((-73.2521,40.2598),(-73.3451,42.9821),(-73,1120,42.3696))                     |

You can use the following code to load raw context data of demographics. However, if you want to make traffic prediction with them, you need to preprocess the data to align with the traffic data. We implement a `context_dataloader` to align context data with crowd flow data. More details please refer to `UCTB/dataset/context_dataloader.py`

### Load and Use

```python

# dataset path configuration
user_data_dir = '{your_dir_path}'
City='Bike_DC'
Year='2013'
R='500'
import pickle
import geopandas as gpd
from shapely.geometry import Point

# Specify the file path.
file_path = '{}__demographic_{}_{}.pkl'.format(City,R,Year)
dataset_path = os.path.join(user_data_dir, file_path)

# Load AD data.
gdf = gpd.read_file(dataset_path)
```


## Road Network
| Attribute       | Description                                                      | Example                                           |
| --------------- | ---------------------------------------------------------------- | ------------------------------------------------- |
| Name           |  Name of the road             | 55th Street                           |
| Highway |     Category of the road                         |       motorway                         |
| Other_tags          | Other information of the road                                                |   0.00035               |
| Z_order        | Z_order is a field in osm2pgsql datamodel                                   |   6            |
| polygon  |  The latitude and longitude coordinates of lines                       | ((-73.2521,40.2598),(-73.3451,42.9821),(-73,1120,42.3696))                     |

### Load and Use

```python

# dataset path configuration
user_data_dir = '{your_dir_path}'
City='Bike_DC'
Year='2013'
R='500'
import pickle
import geopandas as gpd
from shapely.geometry import Point

# Specify the file path.
file_path = '{}__road_{}_{}.pkl'.format(City,R,Year)
dataset_path = os.path.join(user_data_dir, file_path)

# Load road data.
gdf = gpd.read_file(dataset_path)

```
You can use the following code to load raw context data of road network for different datasets. However, if you want to make traffic prediction with them, you need to preprocess the data to align with the traffic data. We implement a `context_dataloader` to align context data with crowd flow data. More details please refer to `UCTB/dataset/context_dataloader.py`


## Administrative Division
| Attribute       | Description                                                      | Example                                           |
| --------------- | ---------------------------------------------------------------- | ------------------------------------------------- |
| ntaname          | Unique identifier used to identify map elements in OpenStreetMap |    42538083                   |
| ntaabbrev           | Name of the POI                                              | U.S.Bank                           |
| shape_leng | Category of the POI                              | Restaurant                               |
| Shape_Area          | Other information of the POI                                                   |     0.00035             |
| y_center        | Latitude of the POI                                      |  40.81632                          |
| x_center  |  Longitude of the POI                       | -73.90182                             |
| polygon  |  The latitude and longitude coordinates of polygon vertices                       | ((-73.2521,40.2598),(-73.3451,42.9821),(-73,1120,42.3696))                     |


### Load and Use

```python

# dataset path configuration
user_data_dir = '{your_dir_path}'
City='Bike_DC'
Year='2013'
R='500'
import pickle
import geopandas as gpd
from shapely.geometry import Point

# Specify the file path.
file_path = '{}__district_{}_{}.pkl'.format(City,R,Year)
dataset_path = os.path.join(user_data_dir, file_path)

# Load AD data.
gdf = gpd.read_file(dataset_path)

```

You can use the following code to load raw context data of administrative division for different datasets. However, if you want to make traffic prediction with them, you need to preprocess the data to align with the traffic data. We implement a `context_dataloader` to align context data with crowd flow data. More details please refer to `UCTB/dataset/context_dataloader.py`



