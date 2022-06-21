import numpy as np
import datetime as dt
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr

era5 = xr.open_dataset('../../data/smallsampledata-era5.nc')

glofas = xr.open_dataset('../../data/smallsampledata-glofas.nc')

era5

era5['cp']

import sys
sys.path.append("../../")
from python.aux.utils_floodmodel import get_mask_of_basin

danube_catchment = get_mask_of_basin(glofas['dis'].isel(time=0))
dis = glofas['dis'].where(danube_catchment)

maximum = dis.where(dis==dis.max(), drop=True)
lat, lon = float(maximum.latitude), float(maximum.longitude)
lat, lon

poi = dict(latitude=48.35, longitude=13.95)
dis_mean = dis.mean('time')
dis_mean.plot()
plt.gca().plot(lon, lat, color='cyan', marker='o',
               markersize=20, mew=4, markerfacecolor='none')

X = era5[['lsp', 'cp']]

from python.aux.utils_floodmodel import shift_and_aggregate

for var in ['lsp', 'cp']:
    for i in range(1,6):
        newvar = var+'-'+str(i)
        X[newvar] = X[var].shift(time=i)  # previous precip as current day variable

for var in ['lsp', 'cp']:
    for i in range(1,14):
        newvar = var+'+'+str(i)
        X[newvar] = X[var].shift(time=-i) # future precip as current day variable
        
X.data_vars

X['lsp-5-11'] = shift_and_aggregate(X['lsp'], shift=5, aggregate=7)
X['lsp-12-25'] = shift_and_aggregate(X['lsp'], shift=12, aggregate=14)
X['lsp-26-55'] = shift_and_aggregate(X['lsp'], shift=26, aggregate=30)
X['lsp-56-180'] = shift_and_aggregate(X['lsp'], shift=56, aggregate=125)

fig, ax = plt.subplots(figsize=(15,5))

X['lsp'][:60,0,0].plot(ax=ax, label='precipitation current day')
X['lsp-5-11'][:60,0,0].plot(ax=ax, label='precipitation 5-11 days before')
X['lsp-12-25'][:60,0,0].plot(ax=ax, label='precipitation 12-25 days before')
ax.legend()
plt.title('Example timeseries of predictors')

X

from python.aux.utils_floodmodel import reshape_multiday_predictand

y = glofas['dis'].interp(latitude=lat, longitude=lon)

var = y.name
y = y.to_dataset()
for i in range(1,14):
    newvar = var+'+'+str(i)
    y[newvar] = y[var].shift(time=-i) # future precip as current day variable
y = y.to_array(dim='forecast_day')
y.coords['forecast_day'] = range(1,len(y.forecast_day)+1)
y

# aggregate over the space dimension (more complex in the next notebook)
Xagg = X.mean(['latitude', 'longitude'])

Xda, yda = reshape_multiday_predictand(Xagg, y)


