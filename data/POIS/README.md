## Document Description：
POIS:The six folders named after cities contain POI data for each of the six cities.


## Data Dictionary：

## Folder:RAWPOI is the raw acquisition of POI data.
  
  Each folder for the current city's POI file: file naming: "city - year - month - day.xls" (representing the current number and type of POI summary for this day under this city.eg:Bike_Chicago-2013-01-01.xls).

 - osmid&nbsp;&nbsp;&nbsp;​(OSM Number)
 - name &nbsp;&nbsp;(POI Name)
 - fclass&nbsp;&nbsp;&nbsp;(POI Category)
 - lon  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(Longitude)
 - lat  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(Latitude)
 - FID  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(Order Number)
<img src="https://github.com/Liyue-Chen/HeteContext/blob/main/data/POIS/RAWPOI/Melbourne_POI_img.png" width = "950" height = "600" alt="Chicago_heat" align=center />

The above is the distribution map of POI in the Melbourne dataset. From the graph, it can be seen that there are many types of POI, but we only consider some important POI categories when using them specifically.






   

## Folder:PROCESSED POI is the pre-processed POI data.(The format is pkl files)

When using POI data, for ease of use, we process the Excel file into a pickle file for easy loading. The pickle file stores a two-dimensional numpy array with the dimension of (Numnode, POI_type), where Numnode represents the number of sites and POI_Type is the number of POI types we are considering. The value of a site indicates the number of POIs of that kind within a radius of N meters.

N represents the site coverage.

Since the size of the area covered by the dataset is different, the value of N is also different.


|  | Bike_NYC | Bike_DC | Bike_Chicago | Pedestrian_Melbourne |
|:--------:|:--------:|:--------:|:--------:|:--------:|
| N(m)  | 500   | 350   | 350   | 130   |


The following are the kinds of POIs considered for different datasets.

## Bike_NYC：



<table>
  <tr>
    <th colspan="6" style="text-align:center;">Bike_NYC POI Type</th>
  </tr>
  <tr>
    <td align="center">cafe</td>
    <td align="center">school</td>
    <td align="center">restaurant</td>
    <td align="center">fast_food</td>
    <td align="center">bicycle_parking</td>
    <td align="center">place_of_worship</td>
  </tr>
  <tr>
    <td align="center">bank</td>
    <td align="center">toilets</td>
    <td align="center">fire_station</td>
    <td align="center">theatre</td>
    <td align="center">hospital</td>
    <td align="center">car_sharing</td>
  </tr>
  <tr>
    <td align="center">library</td>
    <td align="center">pharmacy</td>
    <td align="center">post_office</td>
    <td align="center">parking</td>
    <td align="center">pub</td>
    <td align="center">-</td>
  </tr>
</table>


## Bike_DC：

<table>
  <tr>
    <th colspan="7" style="text-align:center;">Bike_DC POI Type</th>
  </tr>
  <tr>
    <td align="center">cafe</td>
    <td align="center">school</td>
    <td align="center">restaurant</td>
    <td align="center">fast_food</td>
    <td align="center">bar</td>
    <td align="center">place_of_worship</td>
    <td align="center">post_office</td>
  </tr>
  <tr>
    <td align="center">bench</td>
    <td align="center">grave_yard</td>
    <td align="center">post_box</td>
    <td align="center">doctors</td>
    <td align="center">library</td>
    <td align="center">theatre</td>
    <td align="center">nightclub</td>
  </tr>
  <tr>
    <td align="center">parking</td>
    <td align="center">kindergarten</td>
    <td align="center">pharmacy</td>
    <td align="center">embassy</td>
    <td align="center">bank</td>
    <td align="center">-</td>
    <td align="center">-</td>
   
  </tr>
</table>

## Bike_Chicago：


<table>
  <tr>
    <th colspan="6" style="text-align:center;">Bike_DC POI Type</th>
  </tr>
  <tr>
    <td align="center">restaurant</td>
    <td align="center">fast_food</td>
    <td align="center">bank</td>
    <td align="center">hospital</td>
    <td align="center">place_of_worship</td>
    <td align="center">pub</td>
  </tr>
  <tr>
    <td align="center">theatre</td>
    <td align="center">pharmacy</td>
    <td align="center">fountain</td>
    <td align="center">bicycle_parking</td>
    <td align="center">post_box</td>
    <td align="center">clock</td>
  </tr>
  <tr>
    <td align="center">parking</td>
    <td align="center">fire_station</td>
    <td align="center">ferry_terminal</td>
    <td align="center">bicycle_rental</td>
    <td align="center">school</td>
    <td align="center">cafe</td>
  </tr>
</table>


## Pedestrian_Melbourne：
<table>
  <tr>
    <th colspan="6" style="text-align:center;">Bike_DC POI Type</th>
  </tr>
  <tr>
    <td align="center">restaurant</td>
    <td align="center">ice_cream</td>
    <td align="center">telephone</td>
    <td align="center">bench</td>
    <td align="center">vending_machine</td>
    <td align="center">bar</td>
  </tr>
  <tr>
    <td align="center">bank</td>
    <td align="center">toilets</td>
    <td align="center">pharmacy</td>
    <td align="center">post_box</td>
    <td align="center">parking</td>
    <td align="center">cafe</td>
  </tr>
  <tr>
    <td align="center">post_office</td>
    <td align="center">food_court</td>
    <td align="center">theatre</td>
    <td align="center">taxi</td>
    <td align="center">ferry_terminal</td>
    <td align="center">police</td>
  </tr>
</table>

## Methods of POI data use
If we use python:
```Python
#Specify the data set and the corresponding time.
Time='2013-01-01'
Dataset='Bike_DC'
import pickle

# Specify the file path.
file_path = '{}-{}.xls'.format(Dataset,Time)

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
