plt.legend(loc='best')
plt.show()

dataset.columns = ['num', 'Date', 'Time', 'Temperature', 'Wind', 'Pressure',
       'Humidity', 'Precipation', 'Heavyrainfall', 'LMH']

dataset.columns

# HOLT'S WINTER MODEL

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

#df = pd.read_csv('rajahmundry_finall.csv',parse_dates=['Month'],index_col='Month')
#df = pd.read_csv('rajahmundry_finall.csv')
df = dataset
#df.index.freq = '3H'
train, test = df.iloc[:7000,[0,1,3]], df.iloc[7001:,[0,1,3]]
#train
model = ExponentialSmoothing(train.Temperature, seasonal='add', seasonal_periods=2880).fit()
#test.Date[8135]
pred = model.predict(start=test.num[7001], end=test.num[8135])
#test.loc[7001,'Date']
plt.figure(figsize=(64,8))
plt.scatter(train.num, train.Temperature, label='Train',s=3)
plt.scatter(test.num, test.Temperature, label='Test',s=3)
plt.scatter(test.num,pred,s=3)
plt.legend(loc='best')
plt.savefig('holt_winter2880scatter.png')
plt.show()

from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(test.Temperature, pred))
print(rms)

# SARIMA MODEL

# Based from the tutorial of Jason Brownlee on Recurrent Neural Networks
%matplotlib inline

from __future__ import print_function

import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"


