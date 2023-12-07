import datetime
import calendar
import pandas as pd
start_date = datetime.date(2013, 7, 1)  # 开始日期
end_date = datetime.date(2017, 9, 30)  # 结束日期

date_list = []
for year in range(start_date.year, end_date.year + 1):
    start_month = 1 if year > start_date.year else start_date.month
    end_month = 12 if year < end_date.year else end_date.month
    for month in range(start_month, end_month + 1):
        start_day = 1
        end_day = calendar.monthrange(year, month)[1]
        date_list.append(
            (datetime.date(year, month,
                           start_day), datetime.date(year, month, end_day)))

# 网站API限制，每次只能爬取一个月的数据，故需要每个月的开始时间和结束时间
start_date_ls = []  # 每个月的开始时间
end_date_ls = []  # 每个月的结束时间
# Print the list of start and end dates
for dates in date_list:
    # print(dates[0].strftime('%Y%m%d'), dates[1].strftime('%Y%m%d'))
    start_date_ls.append(str(dates[0].strftime('%Y%m%d')))
    end_date_ls.append(str(dates[1].strftime('%Y%m%d')))

if str(start_date.strftime('%Y%m%d')) < start_date_ls[0]:
    start_date_ls[0] = start_date.strftime('%Y%m%d')
if str(end_date.strftime('%Y%m%d')) < end_date_ls[-1]:
    end_date_ls[-1] = end_date.strftime('%Y%m%d')


import requests


info_crawl = {
	'Melbourne':{
		'url':'https://api.weather.com/v1/location/YMML:9:AU/observations/historical.json',
		'api_key':'e1f10a1e78da46f5b10a1e78da96f525'
	},
	'LA':{
		'url':'https://api.weather.com/v1/location/KBUR:9:US/observations/historical.json',
		'api_key':'e1f10a1e78da46f5b10a1e78da96f525'
	},
	'BAY':{
		'url':'https://api.weather.com/v1/location/KSJC:9:US/observations/historical.json',
		'api_key':'e1f10a1e78da46f5b10a1e78da96f525'
	},
	'Chicago':{
		'url':'https://api.weather.com/v1/location/KMDW:9:US/observations/historical.json',
		'api_key':'e1f10a1e78da46f5b10a1e78da96f525'
	},
	'DC':{
		'url':'https://api.weather.com/v1/location/KDCA:9:US/observations/historical.json',
		'api_key':'e1f10a1e78da46f5b10a1e78da96f525'
	},
	'NYC':{
		'url':'https://api.weather.com/v1/location/KLGA:9:US/observations/historical.json',
		'api_key':'e1f10a1e78da46f5b10a1e78da96f525'
	}
}

for city,value in zip(info_crawl.keys(),info_crawl.values()):
	url = value['url']
	api_key = value['api_key']
	my_list = []

# for循环爬取每个月的数据，加到my_list中，最后再concat到一起
	for i in range(len(start_date_ls)):
	    params = {
	        "apiKey": api_key,
	        "units": "e",
	        "startDate": start_date_ls[i],
	        "endDate": end_date_ls[i]
	    }
	    response = requests.get(url, params=params)
	    if response.status_code == 200:
	        data = response.json()
	        data_list = data['observations']
	        # data_list
	
	        import pandas as pd
	        data_DF_ori = pd.DataFrame(data_list)
	        data_DF_ori
	
	        from datetime import datetime
	        import pytz
	
	        data_DF_1 = data_DF_ori.copy()
	
	        melbourne_timezone = pytz.timezone(
	            'Australia/Melbourne')  # 收集不同城市数据集时，记得更改时区
	        LA_timezone = pytz.timezone('America/Los_Angeles')
	        BAY_timezone = pytz.timezone('America/Los_Angeles')  # 和LA同一个时区
	        Chicago_timezone = pytz.timezone('America/Chicago')
	        DC_timezone = pytz.timezone('US/Eastern')
	        NYC_timezone = pytz.timezone('America/New_York')
	        data_DF_1['valid_time_gmt'] = data_DF_ori['valid_time_gmt'].apply( \
	            lambda x: datetime.fromtimestamp(x, tz=pytz.timezone('utc')).astimezone(NYC_timezone).replace(tzinfo=None))
	        data_DF_1['expire_time_gmt'] = data_DF_ori['expire_time_gmt'].apply( \
	            lambda x: datetime.fromtimestamp(x, tz=pytz.timezone('utc')).astimezone(NYC_timezone).replace(tzinfo=None))
	
	        my_list.append(data_DF_1)
	
	        # print(data)
	    else:
	        print(f"Error happens in iter {i} :", response.status_code,
	              response.text)
	    # data_DF = pd.concat(my_list, ignore_index=True)
	    # print(city,' Finished!')
	    # data_DF.to_csv(city+'_Weather.csv', index=False)