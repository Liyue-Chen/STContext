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

|                      | Historical Wea. | Wea. Forecast | AQI | Holiday |  TP   |  POI  | Demo  | Road Network  |  AD   |
| :------------------: | :-------------: | :-----------: | --- | :-----: | :---: | :---: | :---: | :---: | :---: |
|       Bike NYC       |        ✅        |       ✅       | ✅   |  ✅       |  ✅     |  ✅     |   ✅    |   ✅    |   ✅    |
|     Bike Chicago     |        ✅        |       ✅       | ✅   |  ✅       |  ✅     |   ✅    |  ✅     |  ✅     |  ✅     |
|       Bike DC        |        ✅        |       ✅       | ✅   |  ✅       |  ✅     |  ✅     |   ✅    |  ✅     |  ✅     |
| Pedestrian Melbourne |        ✅        |       ✅       |     |   ✅      |   ✅    |  ✅     |    ✅  |  ✅     |   ✅    |
|       METR-LA        |        ✅        |       ✅       | ✅   |  ✅       |   ✅    |  ✅     |   ✅   | ✅      |  ✅     |
|       PEMS-BAY       |        ✅        |       ✅       | ✅   |  ✅       |  ✅     |  ✅     |    ✅   |  ✅     |  ✅     |
|       Taxi NYC       |        ✅        |       ✅       | ✅   |  ✅       |  ✅     |    ✅   |  ✅    | ✅     |   ✅    |
|     Ride Chicago     |        ✅        |       ✅       | ✅   |  ✅       |  ✅     |    ✅   |    ✅  | ✅     |   ✅    |
|      Metro NYC       |        ✅        |       ✅       | ✅   |  ✅       |  ✅     |  ✅     |   ✅    |  ✅     |  ✅     |



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

### Meta Data


| Attribute       | Description                                                      | Example                                           |
| --------------- | ---------------------------------------------------------------- | ------------------------------------------------- |
| Date          | The date of timestamp |    2013/7/1                          |
| Holiday           | Is it a holiday or not?                                  | 0 means no, 1 means yes.                         |
| Holiday_Type | Category of the holiday                              | Thanksgiving Day                             |

### Load and Use

```python


# dataset path configuration
user_data_dir = '{your_dir_path}'
City='DC'
Year='2013'
import pickle
import pandas as pd

# Specify the file path.
file_path = 'Holiday_{}_{}.csv'.format(City,Year)

dataset_path = os.path.join(user_data_dir, file_path)

# Load Holiday data.
df = pd.read_csv('{}'.format(dataset_path))
print(df.head())

```
You can use the following code to load data of Holiday for different datasets. However, if you want to make traffic prediction with them, you need to preprocess the data to align with the traffic data. We implement a `context_dataloader` to align context data with crowd flow data. More details please refer to `UCTB/dataset/context_dataloader.py`



## POI
### Meta Data

| Attribute       | Description                                                      | Example                                           |
| --------------- | ---------------------------------------------------------------- | ------------------------------------------------- |
| Osm_id          | Unique identifier used to identify map elements in OpenStreetMap |    42538083                          |
| Name           | Name of the POI                                              | U.S.Bank                               |
| Fclass | Category of the POI                              | Bank                               |
| Other_tags          | Other information of the POI                                                   | "addr:city"=>"Santa Clara"                                              |
| Lat        | Latitude of the POI                                      |  40.81632                                       |
| Lng  |  Longitude of the POI                       | -73.90182                             |


The  folders  contain POI data for four cities and relative code.

**Processing data from raw POI to pickle files.**

After setting "City", "Dataset_path", "N", "RAWPOI_path","Time_begin" and "Time_end" and "POI_type_file_path" parameters, run the following code directly.

```Python

# dataset path configuration
user_data_dir = '{your_dir_path}'
City='DC'
Year='2013'
import pickle
import pandas as pd
# Specify the file path.
file_path = 'POI_{}-{}.xls'.format(City,Year)

dataset_path = os.path.join(user_data_dir, file_path)
df = pd.read_excel(dataset_path)  

# Load POI data.
print(df.head())

```



## Demographics
### Meta Data

| Attribute       | Description                                                      | Example                                           |
| --------------- | ---------------------------------------------------------------- | ------------------------------------------------- |
| FIPSCode          | Unique identifier used to identify map elements in OpenStreetMap |    42538083                   |
| Year           | Name of the POI                                              | 2013                               |
| GeoID | Category of the POI                              | G170319801001010                               |
| x_center          | Other information of the POI                |      40.81632              |
| y_center        | Latitude of the POI                                      |       -73.2562                   |
| tract_bloc        |    The number of the census block and census tract where it is located                                  |    9801001010                       |
| polygon  |  The latitude and longitude coordinates of polygon vertices                       | ((-73.2521,40.2598),(-73.3451,42.9821),(-73,1120,42.3696))                     |

You can use the following code to load raw context data of demographics. However, if you want to make traffic prediction with them, you need to preprocess the data to align with the traffic data. We implement a `context_dataloader` to align context data with crowd flow data. More details please refer to `UCTB/dataset/context_dataloader.py`

### Load and Use

```python

# dataset path configuration
user_data_dir = '{your_dir_path}'
City='DC'
Year='2013'
import pickle
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Specify the file path.
CSV_file_path = '{}_{}_Census.csv'.format(City,Year)
Shp_file_path = '{}_{}_Census.shp'.format(City,Year)
CSV_dataset_path = os.path.join(user_data_dir, CSV_file_path)
Shp_file_path = os.path.join(user_data_dir, Shp_file_path)

# Load Demographic data.
gdf = gpd.read_file(Shp_file_path)
df = pd.read_csv('{}'.format(CSV_dataset_path))
```


## Road Network
### Meta Data
| Attribute       | Description                                                      | Example                                           |
| --------------- | ---------------------------------------------------------------- | ------------------------------------------------- |
| Name           |  Name of the road             | 55th Street                           |
| Highway |     Category of the road                         |       motorway                         |
| Other_tags          | Other information of the road                                                |   "oneway"=>"yes"               |
| Z_order        | Z_order is a field in osm2pgsql datamodel                                   |   6            |
| polygon  |  The latitude and longitude coordinates of lines                       | ((-73.2521,40.2598),(-73.3451,42.9821),(-73,1120,42.3696))                     |

### Load and Use

```python

# dataset path configuration
user_data_dir = '{your_dir_path}'
City='DC'
Year='2013'
import pickle
import geopandas as gpd
from shapely.geometry import Point

# Specify the file path.
file_path = 'Road{}{}.shp'.format(City,Year)
dataset_path = os.path.join(user_data_dir, file_path)

# Load road data.
gdf = gpd.read_file(dataset_path)

```
You can use the following code to load raw context data of road network for different datasets. However, if you want to make traffic prediction with them, you need to preprocess the data to align with the traffic data. We implement a `context_dataloader` to align context data with crowd flow data. More details please refer to `UCTB/dataset/context_dataloader.py`


## Administrative Division
### Meta Data
| Attribute       | Description                                                      | Example                                           |
| --------------- | ---------------------------------------------------------------- | ------------------------------------------------- |
| ntaname          | Name of this Neighborhood |    Greenpoint                   |
| ntaabbrev           |  An abbreviation of Neighborhood area name             | Grnpt                           |
| shape_leng | The length of region                              | 28912.56                               |
| Shape_Area          | The Area of region                                                |   0.00035               |
| y_center        | Longitude of the Neighborhood's center                                      |  40.81632                          |
| x_center  |  Latitude of the Neighborhood's center                       | -73.90182                             |
| polygon  |  The latitude and longitude coordinates of polygon vertices             | ((-73.2521,40.2598),(-73.3451,42.9821),(-73,1120,42.3696))                     |



### Load and Use

```python

# dataset path configuration
user_data_dir = '{your_dir_path}'
City='DC'
import pickle
import geopandas as gpd
from shapely.geometry import Point

# Specify the file path.
file_path = 'AdministrativeDivision_{}.pkl'.format(City)
dataset_path = os.path.join(user_data_dir, file_path)

# Load AD data.
gdf = gpd.read_file(dataset_path)

```

You can use the following code to load raw context data of administrative division for different datasets. However, if you want to make traffic prediction with them, you need to preprocess the data to align with the traffic data. We implement a `context_dataloader` to align context data with crowd flow data. More details please refer to `UCTB/dataset/context_dataloader.py`



