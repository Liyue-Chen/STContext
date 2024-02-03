# STContext

## 目录结构

```
├── README.md
├── analysis
├── data
│   ├── Holidays
│   ├── POIs
│   ├── STData
│   │   ├── Bike_Chicago.zip
│   │   ├── Bike_DC.zip
│   │   ├── Bike_NYC.zip
│   │   ├── METR_LA.zip
│   │   ├── PEMS_BAY.zip
│   │   ├── Pedestrian_Melbourne.pkl.zip
│   │   └── Taxi_Chicago.zip
│   └── Weather
├── experiments
└── processing
```

对目录的安排如下:

- data目录用以存放时空数据和context数据，含有Holidays, POIs, Weather等子文件夹，子文件夹下的文件应使用`城市+数据类型.csv`来命令，例如 `NYC_Weather.csv`
- analysis目录用于存放分析数据所用的代码/notebook文件
- processing目录用以存放数据预处理代码，可以根据需要按照处理的数据类型来命名
- experiments目录用以存放后续进行benchmark实验的代码

## 时空数据集情况

| **Application**  | **Ciity** | **Interval** |     **Time Span**     |
| :--------------: | :-------: | :----------: | :-------------------: |
|   bike-sharing   |    NYC    |    5 mins    | 2013/07/01-2017/09/30 |
|   bike-sharing   |  Chicago  |    5 mins    | 2013/07/01-2017/09/30 |
|   bike-sharing   |    DC     |    5 mins    | 2013/07/01-2017/09/30 |
| pedestrian count | Melbourne |   60 mins    | 2021/01/01-2022/11/01 |
|  vehicle speed   |    LA     |    5 mins    | 2012/03/01-2012/06/28 |
|  vehicle speed   |    BAY    |    5 mins    | 2017/01/01-2017/07/01 |
|   ride-sharing   |  Chicago  |   60 mins    | 2013/01/01-2022/03/31 |

## 天气数据集情况

### 预测天气数据收集情况

- 温度
	- 温度 (Temperature) √
	- 最高温度 (Maximum temperature) ×
	- 最低温度 (Minimum temperature) ×
	- 露点温度 (Dewpoint temperature) √ 
- 湿度
	- 相对湿度 (Relative humidity) √
	- 比湿度 (Specific humidity) √
- 能见度 (Visibility) √
- 风速 (Wind speed (gust)) √
- 风角度
	- 风的 v 分量 (v-component of wind) √
	- 风的 u 分量 (u-component of wind) √
- 空气质量 ×
- 天气状态 ×
- 云层厚度
	- 总云量 (Total cloud cover) √
	- 高云量 (High cloud cover)  ×
	- 中云量 (Medium cloud cover)  ×
	- 低云量 (Low cloud cover)  ×
- 风寒指数 ×


## 节假日数据集情况



## POIs数据集情况
