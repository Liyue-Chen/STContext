## Welcome to StackEdit!


## 文件说明：
六个以城市命名的文件夹分别为六个城市的POI数据。


## 数据字典：

_raw为初始获取POI数据

   每个文件夹下为当前城市的POI文件：文件命名方式：“城市-年-月-日.csv”(代表当前这个城市下的这一天的POI数量及种类汇总)

_processed为预处理后数据

   每个文件夹下为当前城市的POI文件：文件命名方式：“城市-年-月-日.csv”(代表当前这个城市下的这一天的POI数量及种类汇总)

注：由于OSM官网上加利福尼亚洲只有2019年以后的数据，因此BAY 和LA 是2019年1月1日的数据。其余城市的数据均为2013年1月1日到1月30日的数据
 - osmid（OSM编号）
 - geometry (地理位置)
 - name （POI名称）
 - fclass (POI种类)
 - lon （经度）
 - lat （维度）

## Scenedivide：


我们针对每个城市按照：城市中心商务区（CBD）、CBD除外的城市中心区，非中心区进行了一个划分，此次总共划分了4个城市：分别是NYC_Bike,Chicago_Bike,DC_bike,Melbourne。
从而为了评判一个城市中不同区域地区内流量预测的情况。我们将每个地区的多边形区域以shp文件格式存储在Scenedevide中。
不同地区的划分如下：

### 1：代表CBD、2：代表城市中心区，3：代表城市非中心区(泛化区，三个区域互斥，且地区繁荣程度依次递减。

- Chicago_Bike:1 2 3 （overlap all nodes）
  Chicago划分为3个区域，Chicago在行政区域上划分了中心商务区，因此这里我们将中心商务区作为1区，将中心商务区周围地区作为2区，将Chicago市其他区域作为3区

  <img src="https://github.com/Liyue-Chen/HeteContext/blob/main/data/POIS/Scene_divide/Chicago_heat.png" width = "600" height = "400" alt="Chicago_heat" align=center />
  
- NYC_Bike:1 2 3（overlap all nodes）
  NYC划分为3个区域，纽约市下属的区域分为曼哈顿区、皇后区、布鲁克林区，根据地区经济繁荣程度，我们将曼哈顿区作为1区，皇后区作为2区，布鲁克林以及纽约市周边bike节点出现地区作为3区

  <img src="https://github.com/Liyue-Chen/HeteContext/blob/main/data/POIS/Scene_divide/NYC_Heat.png" width = "600" height = "400" alt="Chicago_heat" align=center />
  
- DC_bike:1 2 3（overlap all nodes）
  DC划分为3个区域，华盛顿在行政区划上分为4个区，包括东南区、西北区、西南区和东北区，西北地区为华盛顿特区经济最为活跃的地区，因此作为CBD中心1区，其余三个区（东南区、西南区、东北区）作为城市中心区（2区），还有部分bike node不在DC特区内，较为分散，我们将行政特区外的点设置为泛化区（3区）

    <img src="https://github.com/Liyue-Chen/HeteContext/blob/main/data/POIS/Scene_divide/DC_heat.png" width = "600" height = "400" alt="Chicago_heat" align=center />

- Melbourne:1 2（overlap all nodes）
  由于数据集所在地区区域小，因此Melbourne数据集只划分为两个区域：将Melbourne的CBD作为1区将Melbourne非CBD地区作为2区。

  <img src="https://github.com/Liyue-Chen/HeteContext/blob/main/data/POIS/Scene_divide/Melbourne_heat.png" width = "600" height = "400" alt="Chicago_heat" align=center />

