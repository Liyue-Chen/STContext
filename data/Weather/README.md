> 文件夹说明：
> - preprocessed: 预处理后的数据
> - raw: 仅将时间戳调整为不同地区的时间的几乎原始的数据
> - statistics: 统计与分析


目前为中文，以方便查看，后续会翻译成英文
- key: 观测站点的唯一标识符
- class: 观测的类别
- expire_time_gmt: 观测的过期时间
- obs_id: 观测的ID
- obs_name: 观测站点的名称
- valid_time_gmt: 观测的有效时间
- day_ind: 一天中的时间段（白天或晚上）
- temp: 温度（华氏度）
- wx_icon: 天气图标的代码
- icon_extd: 天气图标的扩展代码
- wx_phrase: 天气短语描述
- pressure_tend: 气压变化趋势 (0-稳定, 1-上升, 2-下降)
- pressure_desc: 气压描述 (其实就是前一字段的描述)
- dewPt: 露点温度（华氏度）
- heat_index: 酷热指数
- rh: 相对湿度
- pressure: 气压
- vis: 能见度（英里）
- wc: 风寒指数
- wdir: 风向（角度） —— —— 以北, 顺时针的角度(取值范围10-360), 360代表北 
- wdir_cardinal: 风向的助记符号
- gust: 风速瞬时值（英里/小时）
- wspd: 风速（英里/小时）
- max_temp: 最高温度（华氏度）
- min_temp: 最低温度（华氏度）
- precip_total: 总降水量（英寸）
- precip_hrly: 每小时降水量（英寸）
- snow_hrly: 每小时降雪量（英寸）
- uv_desc: 紫外线指数描述
- feels_like: 体感温度（华氏度）
- uv_index: 紫外线指数 https://www.epa.gov/sunsafety/calculating-uv-index-0

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
