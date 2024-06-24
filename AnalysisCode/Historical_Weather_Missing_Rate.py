import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dateutil.parser import parse
from datetime import timedelta
import pdb
import seaborn as sns
# Read the data from the csv file of city NYC
df = pd.read_csv('data/Weather/raw/NYCWeather_ori_1.csv')

df['vt_dt'] = df['valid_time_gmt'].apply(lambda x: parse(x))
# pdb.set_trace()
start_dt = '2013-06-30 23:30:00'
diff_date_list = []
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14
for i in range(1, len(df)):
    diff_date_list.append((df['vt_dt'].iloc[i] - df['vt_dt'].iloc[i-1])//timedelta(minutes=1))
# calculate time interval in minutes
plt.figure(figsize=(10, 6))
plt.hist(diff_date_list, bins=10,range=[0,100])
plt.xlabel('Time interval in minutes', fontsize=14, fontweight='bold')
plt.ylabel('Number of data points', fontsize=14, fontweight='bold')
plt.title('Histogram of time interval in minutes', fontsize=14, fontweight='bold')
plt.show()
# Thus we use 60 minutes to discretize the time
# temporal alignment check
dt_list = list(map(str,pd.date_range(start='2013-07-01 00:00:00', end='2017-10-01 00:00:00', freq='60min')))
# pdb.set_trace()
origin_dt = parse('2013-07-01 00:00:00')
dt_dict = dict(zip(dt_list,[0 for i in range(len(dt_list))]))
for dt in df['vt_dt']:
    temporal_ind = (dt - parse(start_dt))//timedelta(minutes=60)
    dt_dict[(origin_dt + temporal_ind*timedelta(minutes=60)).strftime('%Y-%m-%d %H:%M:%S')] += 1
count = 0
for k,v in dt_dict.items():
    if v == 0:
        count += 1
        print(k)
# pdb.set_trace()
print(count/len(dt_dict))
# missing data check
print('Variables we want to include:')
included_variables = ['temp', 'rh', 'wspd', 'wdir',  'clds', 'vis', 'precip_hrly',  'snow_hrly','wx_phrase','wc','dewPt']
real_name = ['Temperature','Humidity','Wind Speed', 'Wind Direction','Clouds','Visibility','Precipitation','Snow','Weather State','Wind Chill Index','Dew Point Temperature']
real_name_dict = dict(zip(included_variables,real_name))
for var in included_variables:
    print(var)
    # pdb.set_trace()
    print('Number of missing data points: ', len(df[df[var].isnull()]))
    print('NaN data point value:',df[df[var].isnull()])
    print('Number of data points: ', len(df))
    print('Percentage of missing data points: ', len(df[df[var].isnull()])/len(df))
    print('--------------------------------')
# data distribution check


# 设置 seaborn 风格
sns.set(style="whitegrid", palette="pastel")

# 设置字体和字体大小
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12



for var in included_variables:
    if str(df[var].dtype) == 'object':

        # 设置 seaborn 风格
        sns.set(style="whitegrid", palette="pastel")

        # 设置字体和字体大小
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 12

        # 生成一些示例数据
        categories = df[var].value_counts().index
        values = df[var].value_counts().values

        # 绘制柱形图
        plt.figure(figsize=(10, 6))
        sns.barplot(x=categories, y=values, color='skyblue')

        # 添加数据标签
        for i, value in enumerate(values):
            plt.text(i, value + 1, str(value), ha='center', va='bottom')

        # 添加标签和标题
        plt.xlabel('{}'.format(real_name_dict[var]), fontsize=15, fontweight='bold')
        plt.ylabel('Number of data points', fontsize=15, fontweight='bold')
        plt.title('Bar plot of {}'.format(real_name_dict[var]), fontsize=15, fontweight='bold')

        # 显示图形
        plt.show()
    else:
        # 生成一些示例数据
        data = df[var].dropna()

        # 绘制直方图
        plt.figure(figsize=(10, 6))
        sns.distplot(data, hist=True,kde=True, bins=30, color='skyblue')

        # 绘制概率密度曲线
        # sns.kdeplot(data, color='salmon',bw_adjust=0.5, label='Kernel Density Estimation')

        # 添加图例和标签
        plt.legend()
        plt.title('Distribution of {}'.format(real_name_dict[var]), fontsize=15, fontweight='bold')
        plt.xlabel('{}'.format(real_name_dict[var]), fontsize=15, fontweight='bold')
        plt.ylabel('Frequency/Probability Density', fontsize=15, fontweight='bold')

        # 显示图形
        plt.show()
# pdb.set_trace()
