> 文件与文件夹说明：
> - preprocessed: 预处理后的数据
> - raw: 仅将时间戳调整为不同地区的时间的几乎原始的数据
> - statistics: 统计与分析
> - scene: 选取的场景，其中包含多个csv
> - preprocess: 读取数据的代码
> - ipynb文件为代码文件，目前主要为爬取和数据分析的代码，其中有较多注释


数据字典，目前为中文，以方便查看，后续会翻译成英文
- key: 观测站点的唯一标识符 - (str)
- class: 观测的类别 - (str)
- expire_time_gmt: 观测的过期时间 - (DATE)
- obs_id: 观测的ID - (str)
- obs_name: 观测站点的名称 - (str)
- valid_time_gmt: 观测的有效时间 - (DATE)
- day_ind: 一天中的时间段（白天或晚上） - (str)
- temp: 温度（华氏度） - (float)
- wx_icon: 天气图标的代码 - (int)
- icon_extd: 天气图标的扩展代码 - (int)
- wx_phrase: 天气短语描述 - (str)
- pressure_tend: 气压变化趋势 - (int, 0-稳定, 1-上升, 2-下降)
- pressure_desc: 气压描述 - (str, 其实就是前一字段的描述)
- dewPt: 露点温度（华氏度） - (float)
- heat_index: 酷热指数 - (float)
- rh: 相对湿度 - (float)
- pressure: 气压 - (float)
- vis: 能见度（英里） - (int, 0-10)
- wc: 风寒指数 - (float)
- wdir: 风向（角度） - (int, 以北, 顺时针的角度[10-360], 360代表北) 
- wdir_cardinal: 风向的助记符号 - (str, 描述风向, 比如N代表北风)
- gust: 风速瞬时值（英里/小时） - (float)
- wspd: 风速（英里/小时） - (float)
- max_temp: 最高温度（华氏度） - (float)
- min_temp: 最低温度（华氏度） - (float)
- precip_total: 总降水量（英寸） - (float)
- precip_hrly: 每小时降水量（英寸） - (float)
- snow_hrly: 每小时降雪量（英寸） - (float)
- uv_desc: 紫外线指数描述 - (str)
- feels_like: 体感温度（华氏度） - (float)
- uv_index: 紫外线指数 https://www.epa.gov/sunsafety/calculating-uv-index-0 - (float)

(以下列除了clds，都无数据，只是爬取时有该列，但是并没有数据)
- qualifier: 气象数据的限定条件
- qualifier_svrty: 限定条件的严重程度
- blunt_phrase: 概括的天气短语描述
- terse_phrase: 简短的天气短语描述
- clds: 云量 https://www.eoas.ubc.ca/courses/atsc113/flying/met_concepts/01-met_concepts/01c-cloud_coverage/index.html
- water_temp: 水温
- primary_wave_period: 主波周期
- primary_wave_height: 主波高度
- primary_swell_period: 主涌浪周期
- primary_swell_height: 主涌浪高度
- primary_swell_direction: 主涌浪方向
- secondary_swell_period: 次涌浪周期
- secondary_swell_height: 次涌浪高度
- secondary_swell_direction: 次涌浪方向

> 备注
> - 2023/6/9 Melbourne数据集的时间范围有变化，重新爬取并处理  
值得注意的是，新爬取到的Melbourne数据集，其valid_time_gmt，即时刻的格式，与之前爬取的数据的格式略有不同(用记事本打开就可以看出来），可能是datetime包更新了导致，在”analyze_wxx.ipynb”的“3. 先对其进行处理，使每小时内的所有数据合并”中，在第30行左右的，关于valid_time_gmt的处理代码，用于新爬取到的Melbourne上会报错（但是不是在此处报错，而是在后续运行”4.4 one-hot编码替换原始的wx_phrase”时，由于时间格式的不一致而导致处理函数不能正确处理，从而报错），需要专门为Melbourne调整一下，可以写个if来专门处理
