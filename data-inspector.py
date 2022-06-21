%matplotlib inline
import os
import sys
import numpy as np
import pandas as pd
import seaborn as sb
import xarray as xr
import os.path as path
import matplotlib.pyplot as plt
from aux.ml_flood_config import path_to_data
from aux.utils import calc_stat_moments
from aux.utils import open_data
from aux.plot import Map
import seaborn as sns

from dask.distributed import Client
client = Client(processes=True)
import dask
#dask.config.set(scheduler='processes')
from dask.diagnostics import ProgressBar

# load dask client
client
# define some vars
data_path = f'{path_to_data}danube/'
print(data_path)

era5 = open_data(data_path, kw='era5')
glofas = open_data(data_path, kw='glofas_ra')

sm = calc_stat_moments(era5, dim_aggregator='time', time_constraint=None)
m = Map(figure_kws=dict(figsize=(15,10)))

da_mean = sm['cp'].sel(stat_moments='mean').compute()

#print(da_mean)
#print(da_std)
m.plot(da_mean)
plt.show()

da_std = sm['cp'].sel(stat_moments='std').compute()
m.plot(da_std)

def plot_ts(da, key):
    """Plot a times series for a given xarray dataarray.
    """
    p = sns.lineplot(data=da.to_pandas(), linewidth=2)
    p.set_xlabel('time')
    p.set_ylabel(key)

sm = calc_stat_moments(era5, dim_aggregator='spatial', time_constraint=None)

key = 'lsp'
da = sm[key].sel(stat_moments='mean')
da = da.resample(time="y").mean()

plot_ts(da=da, key='lsp')
plt.show()

key = 'cp'
da = sm[key].sel(stat_moments='mean')
da = da.resample(time="y").mean()

plot_ts(da=da, key='cp')
plt.show()

key = 'z'
da = sm[key].sel(stat_moments='mean')
da = da.resample(time="y").mean()

plot_ts(da=da, key='z')
plt.show()

krems = dict(latitude=48.403, longitude=15.615)
local = dict(latitude=slice(krems['latitude']+1,krems['latitude']-1),
             longitude=slice(krems['longitude']-1, krems['longitude']+1))

#era5 = era5.interp(latitude=glofas.latitude, longitude=glofas.longitude)
start_date = '2013-05-20'
end_date = '2013-06-05'
xds = era5.sel(local).sel(time=slice(start_date, end_date))

glofas = glofas.sel(time=slice(start_date, end_date))
y = glofas.interp(krems)
ydf = y.drop(['latitude', 'longitude']).to_dataframe()
print(xds)
print(ydf)

#print([era5[i] for i in era5.keys()])

ydf_change = ydf.dis.diff(1)
#print(ydf_change)

def corrs(da, y):
    crs = xr.DataArray(np.zeros([da.latitude.values.shape[0], da.longitude.values.shape[0]]),
                       coords=[da.latitude, da.longitude], dims=['latitude', 'longitude'])
    for lat in da.latitude:
        for lon in da.longitude:
            df_iter = da.sel(latitude=lat, longitude=lon).to_series().drop(columns=['latitude', 'longitude'])
            cr = df_iter.corr(y)
            crs.loc[dict(latitude=lat.values, longitude=lon.values)] = cr
    crs.name = 'correlations'
    return crs

def corrs_fast(da, y):
    crs = xr.DataArray(np.zeros([da.latitude.values.shape[0], da.longitude.values.shape[0]]),
                       coords=[da.latitude, da.longitude], dims=['latitude', 'longitude'])
    
    x_anom = da.sel(time=y.time.values)-da.sel(time=y.time.values).mean('time')
    y_anom = y-y.mean('time')
    x_std = x_anom.std()
    y_std = y_anom.std()
    crs = x_anom.dot(y_anom)/(x_std*y_std)
    print(crs.values)
    crs.name = 'correlations'
    return crs

def map3x3(ds, y, var, corr_data=False, plot_kw=False, point=None):
    fig, axes = plt.subplots(3, 3, figsize=(15,15), sharex='col', sharey='row')
    fig.subplots_adjust(hspace=0.35, wspace=0.25)
    
    plt.suptitle(f'correlation gridpoint to river point; variable={var}')
    i = 0
    for axrow in axes:
        axrow[0].set_ylabel('latitude')
        for axcol in axrow:
            axcol.set_title(f't-{i}')
            if not corr_data:
                da = xds[var].shift(time=i)
                crs = corrs(da, y)
                crs.to_netcdf(f'save/crs_{var}_shift_{i}.nc')
                #da = xds[var].shift(time=i)
                #crs = corrs_fast(da, y)
            elif corr_data:
                crs = ds
            if plot_kw:
                if corr_data:
                    crs['correlations'].isel(time_delay=i).plot.imshow(ax=axcol, vmin=-1, vmax=1, cmap='coolwarm_r')
                else:
                    crs.plot.imshow(ax=axcol, vmin=-1, vmax=1, cmap='coolwarm_r')
                axcol.set_title(f't-{i}')
                axcol.plot(point['longitude'], point['latitude'], color='cyan',
                           marker='o', markersize=20, mew=4, markerfacecolor='none')
            i += 1
    return None

#crs_data = xr.open_mfdataset('/home/srvx11/lehre/users/a1303583/ipython/ml_flood/python/save/*crs_lsp*', concat_dim='time_delay')
#print(crs_data)

#print(xds)

map3x3(xds, ydf_change, 'ro', corr_data=False, plot_kw=True, point=krems)
#map3x3(xds, y['dis'].diff('time', 1), 'lsp', corr_data=False, plot_kw=True, point=krems)
#map3x3(crs_data, ydf_change, 'lsp', corr_data=True, plot_kw=True, point=krems)

ydf.plot(y='dis', figsize=(15,5))
fig, ax = plt.subplots()
xdf.plot(y=['lsp', 'cp'], ax=ax, figsize=(15,5))

xdf.cumsum().plot(y=['lsp', 'cp'], figsize=(15,5))

sns.jointplot(x='cp', y='dis', data=merge, kind='hex')

