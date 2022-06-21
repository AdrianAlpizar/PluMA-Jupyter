import link_src

import numpy as np
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import dask
from dask.distributed import Client, LocalCluster
cluster = LocalCluster(processes=True) #n_workers=10, threads_per_worker=1,
client = Client(cluster)  # memory_limit='16GB',

import xarray as xr
from dask.diagnostics import ProgressBar

client

from python.aux.ml_flood_config import path_to_data
from python.aux.utils import open_data
from python.aux.utils import shift_time
# define some vars
data_path = f'{path_to_data}danube/'
print(data_path)

# load data
era5 = open_data(data_path, kw='era5')
glofas = open_data(data_path, kw='glofas_ra')
glofas = shift_time(glofas, -dt.timedelta(days=1))

if not 'lsp' in era5:
    lsp = era5['tp']-era5['cp']
    lsp.name = 'lsp'
else:
    lsp = era5['lsp']

reltop = era5['z'].sel(level=500) - era5['z'].sel(level=850)
reltop.name = 'reltop'

q_mean = era5['q'].mean('level')
q_mean.name = 'q_mean'

era5_features = xr.merge([era5['cp'], lsp, reltop, q_mean])
era5_features = era5_features.interp(latitude=glofas.latitude,
                                     longitude=glofas.longitude)
era5_features = era5_features.isel(time=slice(0*365,3*365))
glofas = glofas.isel(time=slice(0*365,3*365))

if len(era5_features.time) < 3000:
    era5_features = era5_features.load()
    glofas = glofas.load()

krems = dict(latitude=48.403, longitude=15.615)

local_region = dict(latitude=slice(krems['latitude']+1.5,
                                   krems['latitude']-1.5),
                   longitude=slice(krems['longitude']-1.5,
                                   krems['longitude']+1.5))

# select area of interest and average over space for all features
dis = glofas.interp(krems)
y = dis.diff('time', 1)
X = era5_features.sel(local_region).mean(['latitude', 'longitude'])
print(X)

def add_shifted_predictors(ds, shifts, variables='all'):
    """Adds additional variables to an array which are shifted in time.

    Parameters
    ----------
    ds : xr.Dataset
    shifts : list of integers
    variables : str or list
    """
    if variables == 'all':
        variables = ds.data_vars

    for var in variables:
        for i in shifts:
            if i == 0: continue  # makes no sense to shift by zero
            newvar = var+'-'+str(i)
            ds[newvar] = ds[var].shift(time=i)
    return ds
shifts = range(1,11)
notshift_vars = ['swvl1', 'swvl2']
shift_vars = [v for v in X.data_vars if not v in notshift_vars]

Xs = add_shifted_predictors(X, shifts, variables=shift_vars)
Xar = Xs.to_array(dim='features')
yar = y.to_array()

# singleton dimension has to have the same name like in X
# so we set it to 'features' too
yar = yar.rename({'variable': 'features'})

# it confuses the concat if one has latitude but the other not
yar = yar.drop(['latitude', 'longitude'])
Xy = xr.concat([Xar, yar], dim='features')

# drop them as we cannot train on nan values
Xyt = Xy.dropna('time', how='any')
predictand = 'dis'
predictors = [v for v in Xyt.coords['features'].values if v != predictand]

Xda = Xyt.loc[predictors]
yda = Xyt.loc[predictand]
time = yda.time
Xda = Xda.chunk(dict(time=-1, features=-1)).data.T
yda = yda.data.squeeze()

import joblib
from sklearn.pipeline import Pipeline
from dask_ml.preprocessing import StandardScaler
from dask_ml.decomposition import PCA

#from dask_ml.xgboost import XGBRegressor
from dask_ml.linear_model import LogisticRegression
from dask_ml.linear_model import LinearRegression

model_kws = dict(n_jobs=-1, max_iter=10000, verbose=True)

pipe = Pipeline([('scaler', StandardScaler()),
                 #('pca', PCA(n_components=6)),
                 ('model', LinearRegression(**model_kws)),],
                verbose=True)

Xda = Xda.persist()

with ProgressBar():
    pipe.fit(Xda, yda)

def add_time(vector, time, name=None):
    """Converts arrays to xarrays with a time coordinate."""
    return xr.DataArray(vector, dims=('time'), coords={'time': time}, name=name)

with ProgressBar():
    ytest = pipe.predict(Xda)

ytest = add_time(ytest, time, name='dis-forecast')
ytest_dis = ytest.cumsum('time')
# initial state + changes = timeseries of forecasted discharge
ytest_dis += dis['dis'][0]

fig, ax = plt.subplots(figsize=(15,5))
dis['dis'].to_pandas().plot(ax=ax, label='dis-reanalysis')
ytest_dis.to_pandas().plot(ax=ax, label='dis-forecast')
plt.legend()


tstart, tend = dt.datetime(1981,6,4), dt.datetime(1981,8,14)
dt_runs = dt.timedelta(days=7)
dt_fxrange = dt.timedelta(days=7)

fig, ax = plt.subplots(figsize=(6,4), dpi=200)
dis['dis'].sel(time=slice(tstart,tend)).to_pandas().plot(ax=ax, label='reanalysis')
plt.legend()
t = tstart
while t < tend:
    t0, t1 = t, t+dt_fxrange
    tm1 = t0-dt.timedelta(days=1)

    fcst = dis['dis'].sel(time=tm1) + ytest.sel(time=slice(t0, t1)).cumsum()  # forecast
    fcst.to_pandas().plot(ax=ax, linestyle='--', color='r')

    t += dt_runs

plt.legend(['reanalysis', 'forecasts'])
ax.set_ylabel('river discharge [m$^3$/s]')

