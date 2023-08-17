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

Folder:PROCESSED POI is the pre-processed POI data.

  

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
