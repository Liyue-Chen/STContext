## Document Description：
POIS:The six folders named after cities contain POI data for each of the six cities.


## Data Dictionary：

Folder:RAWPOI is the raw acquisition of POI data.
  
  Each folder for the current city's POI file: file naming: "city - year - month - day.xls" (representing the current number and type of POI summary for this day under this city.eg:Bike_Chicago-2013-01-01.xls).


    
 - osmid&nbsp;&nbsp;&nbsp;​(OSM Number)
 - name &nbsp;&nbsp;(POI Name)
 - fclass&nbsp;&nbsp;&nbsp;(POI Category)
 - lon  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(Longitude)
 - lat  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(Latitude)
 - FID  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(Order Number)
<img src="https://github.com/Liyue-Chen/HeteContext/blob/main/data/POIS/RAWPOI/Melbourne_POI_img.png" width = "950" height = "600" alt="Chicago_heat" align=center />

The above is the distribution map of POI in the Melbourne dataset. From the graph, it can be seen that there are many types of POI, and we only consider some important POI categories when using them specifically.






   

Folder:PROCESSED POI is the pre-processed POI data.(The format is  pkl files)

When using POI data, for ease of use, we process the Excel file into a pickle file for easy loading. The pickle file stores a two-dimensional numpy array with the dimension of (Numnode, POI_type), where Numnode represents the number of sites and POI_ Type is the number of POI types we are considering. The value of a site indicates the number of POIs of that kind within a radius of N meters.

Since the size of the area covered by the dataset is different, the value of N is also different.(The value of N represents the site coverage.)


|  | Bike_NYC | Bike_DC | Bike_Chicago | Pedestrian_Melbourne |
|:--------:|:--------:|:--------:|:--------:|:--------:|
| N(m)  | 500   | 350   | 350   | 130   |


The following are the kinds of POIs considered for different datasets.

## Bike_NYC：

| 列1标题 | 列2标题 | 列3标题 | 列4标题 | 列5标题 | 列6标题 |
| ------- | ------- | ------- | ------- | ------- | ------- |
| 内容1   | 内容2   | 内容3   | 内容4   | 内容5   | 内容6   |
| 内容7   | 内容8   | 内容9   | 内容10  | 内容11  | 内容12  |
| 内容13  | 内容14  | 内容15  | 内容16  | 内容17  | 内容18  |
| 内容19  | 内容20  | 内容21  | 内容22  | 内容23  | 内容24  |




  

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
