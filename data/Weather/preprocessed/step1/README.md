在一个小时内，可能有多条数据，为了与时空数据对齐，需要将每小时内的多条数据合并为一条，具体如下：  
（1）对于非数字的列，取第一次出现的  
（2）对于数字的列，若非‘最高气温’和‘最低气温’这两列，则取平均值  
（3）对于最高（最低）气温，取最大（最小）值  
（4）把expire_time替换成修改前的valid_time，再由（1），则处理后的expire_time，即为每小时的第一条数据的具体时间  