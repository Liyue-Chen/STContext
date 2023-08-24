# Document Description：
The  folders  contain POI data for four cities and relative code.



# Folder:RAWPOI is the raw acquisition of POI data.
  
  Each folder for the current city's POI file: file naming: "city - year - month - day.xls" (representing the current number and type of POI summary for this day under this city.eg:Bike_Chicago-2013-01-01.xls).

 - osmid&nbsp;&nbsp;&nbsp;​(OSM Number)
 - name &nbsp;&nbsp;(POI Name)
 - fclass&nbsp;&nbsp;&nbsp;(POI Category)
 - lon  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(Longitude)
 - lat  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(Latitude)
 - FID  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(Order Number)
<img src="https://github.com/Liyue-Chen/HeteContext/blob/main/data/POIS/RAWPOI/Melbourne_POI_img.png" width = "950" height = "600" alt="Chicago_heat" align=center />

The above is the distribution map of POI in the Pedestrian Melbourne dataset. 






   

# Folder:PROCESSED POI is the pre-processed POI data.(The format is pkl files)

When using POI data, for ease of use, we process the Excel file into a pickle file for easy loading. The pickle file stores a two-dimensional numpy array with the dimension of (Numnode, POI_type), where Numnode represents the number of sites and POI_Type is the number of POI types we are considering. The value of a site indicates the number of POIs of that kind within a radius of "R" meters.

"R" represents the site coverage.

Since the size of the area covered by the dataset is different, the value of "R" is also different,The following table shows the values of "R" in different datasets.
"R" as a parameter can be set to different values to generate different pickle files.

|  | Bike_NYC | Bike_DC | Bike_Chicago | Pedestrian_Melbourne |
|:--------:|:--------:|:--------:|:--------:|:--------:|
| R(m)  | 500   | 350   | 350   | 130   |




From the graph, it can be seen that there are many types of POIs, but we only consider some important POIs when using them specifically.For each dataset, we selecte some important POIs. And we take the union of these POIs to get the final important POIs.

## Final Important POIs：


There are two steps to calculating significant POIs for each year.

### Step 1: Calculate the  scores of the POIs and select the POIs with the top 25 values.

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



### Step 2: Remove some strongly correlated POIs based on correlation and prior knowledge.

First, for any two POIs (X,Y),the distribution of them at n sites can be summarized separately by the sequences 
\{
$​​x_1,x_2,x_3.... .x_i$
\}
,
\{
$y_1,y_2,y_3.... .y_i$
\}
.If  X and  Y have strong correlation (Pearson's correlation coefficient is greater than 0.8), we consider that the  correlations of them are strong, and get the combination of POI with strong correlation.

Next, for the POI combination with strong correlation, we further judge by a priori knowledge whether they are similar POI types and need to be removed.There are some examples as follows:
|      | (x1,y1)     | (x2,y2)      | (x3,y3)      |(x4,y4)      |
| :--------: | :--------: | :--------: | :--------: |:--------: |
| X   | bank   | bar   | theatre   | car_sharing|
| Y   | atm   | drinking water   | cinema   |car_rental|


Some combinations like the above, we just need to retain one of them and remove the POI with smaller score.

Through the above two steps of POIs, we get important POIs in a year of the dataset.  Then ,We count a union about the important POIs in each year of the dataset as the dataset's important POIs. Finally, we count a union about the important POIs in all datasets to get the final important POIs.


The final important POIs are listed below





<table>
  <tr>
    <th colspan="7" style="text-align:center;">Important POI List</th>
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

## Processing data from raw POI to pickle files.
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

#Important POI type is  stored in POI_type_file_path.
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


## How to use POI pickle file.
If we use python:
```Python
#Specify the data set and the corresponding time.
Time='2013-01-01'
City='Bike_DC'
R='130'
import pickle

# Specify the file path.
file_path = '{}_{}_R=={}.pkl'.format(City,Time,R)

# Load POI data.
with open(file_path, 'rb') as f:
    data = pickle.load(f)  
```

## Scenedivide Dataset：


In order to judge the traffic prediction within the different regional areas of a city,we divided each city into the following categories: Central Business District (CBD), Central District, and Non-Central District,three types of areas are mutually exclusive, and the prosperity of the area in decreasing order. A total of 4 cities were divided: NYC_Bike, Chicago_Bike, DC_bike, and Melbourne.

We store the polygonal areas of each region in Scenedevide in shape file.

The different areas are divided in detail as follows:

### 1 represents the CBD &nbsp;  &nbsp; 2 represents the central area of the city &nbsp; &nbsp;3 represents the non-central area of the city 

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

### Instructions for using the Scenedivide dataset

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
