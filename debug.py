import numpy as np
import os
import pandas as pd

import torch
import torch.nn as nn

from data import TrainValData, TimeSeriesData

data_folder = "/work/bd/summer2022/insample/datacache"
dailydata = os.path.join(data_folder, 'dailydata') # close, high, low, open, return, tvrvalue, tvrvolume.parquet
labeldata = os.path.join(data_folder, 'labeldata') # 
univdata = os.path.join(data_folder, 'univdata')
xmindata = os.path.join(data_folder, 'xmindata')

df_close = pd.read_parquet(os.path.join(dailydata, 'close.parquet'))
df_high = pd.read_parquet(os.path.join(dailydata, 'high.parquet'))
df_low = pd.read_parquet(os.path.join(dailydata, 'low.parquet'))
df_open = pd.read_parquet(os.path.join(dailydata, 'open.parquet'))
df_tvrvalue = pd.read_parquet(os.path.join(dailydata, 'tvrvalue.parquet'))
df_tvrvolume = pd.read_parquet(os.path.join(dailydata, 'tvrvolume.parquet'))

df_return = pd.read_parquet(os.path.join(dailydata, 'return.parquet'))

df_y0 = pd.read_parquet(os.path.join(labeldata, 'Y_0.parquet'))

features = [df_open, df_high, df_low, df_tvrvalue, df_tvrvolume, df_close]
stock_data_list = []

stocks = df_open.columns.values
for stock in stocks:
    one_stock_features = []
    for feature in features:
        one_stock_features.append(feature[stock].values[:-2].reshape(-1, 1))
    stock_np_features = np.concatenate(one_stock_features, axis=1)
    dates = feature.index.values[:-2]
    labels = df_y0[stock].values[1:]
    # print(stock_np_features.shape, dates.shape, labels.shape)
    stock_data_list.append(TimeSeriesData(dates=dates, data=stock_np_features, labels=labels))
print(len(stocks))

train_val_data = TrainValData(time_series_list=stock_data_list, train_length=800, validate_length=150, history_length=10)
train, val, dates_info = train_val_data.get(20180102, order='by_date')

print('End')