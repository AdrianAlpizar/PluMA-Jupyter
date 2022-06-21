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

import sys
print(sys.executable)

def shift_time(ds, value):
        ds.coords['time'].values = pd.to_datetime(ds.coords['time'].values) + value
        return ds

static = xr.open_dataset('../data/danube/era5_slt_z_slor_lsm_stationary_field.nc')
static

era5 = xr.open_dataset('../data/danube/era5_danube_pressure_and_single_levels.nc')
era5

glofas = xr.open_dataset('../data/danube/glofas_reanalysis_danube_1981-2002.nc')
glofas = glofas.rename({'lat': 'latitude', 'lon': 'longitude'})  # to have the same name like in era5
glofas = shift_time(glofas, -dt.timedelta(days=1))

glofas_rerun = xr.open_dataset('../data/glofas-freruns/2013051800/glofas2.3_era5wb_reforecast_dis_bigchannels_1000km2_20130518_0.nc')
glofas_rerun = glofas_rerun.rename({'lat': 'latitude', 'lon': 'longitude'})
glofas_rerun = shift_time(glofas_rerun, -dt.timedelta(days=1))

era5.coords

glofas.coords

if not 'lsp' in era5:
        lsp = era5['tp']-era5['cp']
        lsp.name = 'lsp'
else:
                lsp = era5['lsp']

                reltop = era5['z'].sel(level=500) - era5['z'].sel(level=850)
                reltop.name = 'reltop'

                q_mean = era5['q'].mean('level')
                q_mean.name = 'q_mean'

                era5 = xr.merge([era5['cp'], lsp, reltop, q_mean])

era5

era5 = era5.interp(latitude=glofas.latitude, longitude=glofas.longitude)

era5 = era5.isel(time=slice(0*365,3*365))
glofas = glofas.isel(time=slice(0*365,3*365))

if len(era5.time) < 3000:
        era5 = era5.load()
            glofas = glofas.load()

for var in era5:
        fig, ax = plt.subplots(figsize=(25,5))
            era5[var].isel(latitude=5, longitude=33).to_pandas().plot(ax=ax, title=var)

krems = dict(latitude=48.403, longitude=15.615)

surrounding = dict(latitude=slice(krems['latitude']+1, 
                                      krems['latitude']-1),
                                                         longitude=slice(krems['longitude']-1, 
                                                                                               krems['longitude']+1))

def mask_dataframe(df, limit_per_variable: dict):
        for var, (lowlim, uplim) in limit_per_variable.items():
                    df = df[var > lowlim]
                            df = df[var < uplim]
                                return df

                            def normalize(series):
                                    return series #(series-series.mean())/series.std()

                                def normalize_df(df):
                                        dic = []
                                            for var, d in xdf.items():
                                                        dic.append( normalize(d) )
                                                            return #pd.concat(dic)

# select data of interest
dis = glofas.interp(krems)
y = dis #.diff('time', 1)  # forecast time difference of discharge
X = era5.sel(surrounding).mean(['latitude', 'longitude'])
X

#sns.jointplot(x='cp', y='dis', data=merge, kind='hex')

#sns.jointplot(x='lsp', y='dis', data=merge, kind='hex')

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

Xs

Xar = Xs.to_array(dim='features')
yar = y.to_array()

# singleton dimension has to have the same name like in X
# so we set it to 'features' too
yar = yar.rename({'variable': 'features'})

# it confuses the concat if one has latitude but the other not
yar = yar.drop(['latitude', 'longitude'])

Xar

yar

Xy = xr.concat([Xar, yar], dim='features')  

Xyt = Xy.dropna('time', how='any')  # drop them as we cannot train on nan values

Xy.shape
Xyt.shape

Xyt

predictand = 'dis'
predictors = [v for v in Xyt.coords['features'].values if v != predictand]

print(Xyt)
Xda = Xyt.loc[predictors]
yda = Xyt.loc[predictand]

predictors

time = yda.time
Xda = Xda.chunk(dict(time=-1, features=-1)).data.T
yda = yda.data.squeeze()

Xda

yda

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

pipe

Xda = Xda.persist()

with ProgressBar():
        pipe.fit(Xda, yda)

def add_time(vector, time, name=None):
        """Converts arrays to xarrays with a time coordinate."""
            return xr.DataArray(vector, dims=('time'), coords={'time': time}, name=name)

with ProgressBar():
        ytest = pipe.predict(Xda)

ytest = add_time(ytest, time, name='dis-forecast')
ytest += yda[0]  # initial state + changes = timeseries of forecasted discharge

fig, ax = plt.subplots(figsize=(24,5))
Xyt.loc[predictand].to_pandas().plot(ax=ax, label='dis-reanalysis')
ytest.to_pandas().plot(ax=ax, label='dis-forecast')
plt.legend()

def recurrent_predict(pipe, X, n=1):
        """
            
                X : np.array or dask.array
                        dimensions: (n, #features)
                            n : int
                                    number of iterations
                                        """
                                            y = np.zeros(n)
                                                for i in range(n):
                                                            y[i] = pipe.predict(X[np.newaxis, i])
                                                                return y

#t = recurrent_predict(pipe, Xda, 10)

fig, ax = plt.subplots(figsize=(15,5))
t = 0
for t in range(12):
        y = pipe.predict(Xda[t:t+30])
            ydf = pd.Series(index=time.to_pandas().iloc[t:t+30], data=y)
                ydf.plot(ax=ax)

