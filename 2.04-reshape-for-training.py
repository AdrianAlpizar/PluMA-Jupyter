import numpy as np
import datetime as dt
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr

era5 = xr.open_dataset('../../data/smallsampledata-era5.nc')
glofas = xr.open_dataset('../../data/smallsampledata-glofas.nc')

import sys
sys.path.append("../../")
from python.aux.utils_floodmodel import get_mask_of_basin, add_shifted_variables

dis_mean = glofas['dis'].mean('time')
danube_catchment = get_mask_of_basin(dis_mean)
dis = glofas['dis'].where(danube_catchment)

maximum = dis.where(dis==dis.max(), drop=True)
lat, lon = float(maximum.latitude), float(maximum.longitude)

dis.mean('time').plot()
plt.gca().plot(lon, lat, color='cyan', marker='o', 
               markersize=20, mew=4, markerfacecolor='none')

bins = [0, 0.8, 2.4, 10.25, 10000]

dis_mean = dis.mean('time')
cluster = dict()
for i in range(len(bins)-1):
    cluster[str(i)] = (dis_mean >= bins[i]) & (dis_mean < bins[i+1])
    cluster[str(i)].attrs['units'] = None

cluster = xr.Dataset(cluster, coords=dict(clusterId=('clusterId', range(len(bins))),
                                          latitude=('latitude', dis_mean.latitude),
                                          longitude=('longitude', dis_mean.longitude)))
cluster = cluster.to_array('clusterId')

from python.aux.utils_floodmodel import shift_and_aggregate, aggregate_clustersum

shifts = [1, 2, 3]
shift_vars = ['lsp', 'cp']

Xs = add_shifted_variables(era5, shifts, variables=shift_vars)

Xs['lsp-4-10'] = shift_and_aggregate(Xs['lsp'], shift=4, aggregate=7)
Xs['lsp-11-24'] = shift_and_aggregate(Xs['lsp'], shift=14, aggregate=14)
Xs['lsp-25-54'] = shift_and_aggregate(Xs['lsp'], shift=28, aggregate=30)
Xs['lsp-55-180'] = shift_and_aggregate(Xs['lsp'], shift=55, aggregate=126)

sd_diff = Xs['sd'].diff(dim='time')
Xs = Xs.assign({'sd_diff': sd_diff})

ignore_features = ['tcwv', 'rtp_500-850']
Xs = Xs.drop(ignore_features)

X = Xs.interp(latitude=glofas.latitude, longitude=glofas.longitude)

cluster_switch = False
if cluster_switch:
    Xagg = aggregate_clustersum(X, cluster, 'clusterId')
else:
    X_catchment = X.where(danube_catchment, drop=True)
    Xagg = X_catchment.mean(['latitude', 'longitude'])
Xagg.data_vars

y = glofas['dis'].interp(latitude=lat, longitude=lon)
y

Xagg = Xagg.assign({'dis': y})

Xagg.to_netcdf('../../data/features_xy.nc')

def reshape_scalar_predictand(X_dis, y):
    """Reshape, merge predictor/predictand in time, drop nans.

    Parameters
    ----------
        X_dis : xr.Dataset
            variables: time shifted predictors (name irrelevant)
            coords: time, latitude, longitude
        y : xr.DataArray
            coords: time
    """
    if isinstance(X_dis, xr.Dataset):
        X_dis = X_dis.to_array(dim='var_dimension')

    # stack -> seen as one dimension for the model
    stack_dims = [a for a in X_dis.dims if a != 'time']  # all except time
    X_dis = X_dis.stack(features=stack_dims)
    Xar = X_dis.dropna('features', how='all')  # drop features that only contain NaN

    # to be sure that these dims are not in the output
    for coord in ['latitude', 'longitude']:
        if coord in y.coords:
            y = y.drop(coord)

    # merge times
    y.coords['features'] = 'predictand'
    Xy = xr.concat([Xar, y], dim='features')  # maybe merge instead concat?
    Xyt = Xy.dropna('time', how='any')  # drop rows with nan values

    Xda = Xyt[:, :-1]  # last column is predictand
    yda = Xyt[:, -1].drop('features')  # features was only needed in merge
    return Xda, yda

Xda, yda = reshape_scalar_predictand(Xagg, y)

Xda.shape, yda.shape

def reshape_multiday_predictand(X_dis, y):
    """Reshape, merge predictor/predictand in time, drop nans.
    
    Parameters
    ----------
        X_dis : xr.Dataset
            variables: time shifted predictors (name irrelevant)
            coords: time, latitude, longitude
        y : xr.DataArray (multiple variables, multiple timesteps)
            coords: time, forecast_day
    """
    if isinstance(X_dis, xr.Dataset):
        X_dis = X_dis.to_array(dim='var_dimension')

    # stack -> seen as one dimension for the model
    stack_dims = [a for a in X_dis.dims if a != 'time']  # all except time
    X_dis = X_dis.stack(features=stack_dims)
    Xar = X_dis.dropna('features', how='all')  # drop features that only contain NaN

    if not isinstance(y, xr.DataArray):
        raise TypeError('Supply `y` as xr.DataArray.'
                        'with coords (time, forecast_day)!')

    # to be sure that these dims are not in the output
    for coord in ['latitude', 'longitude']:
        if coord in y.coords:
            y = y.drop(coord)

    out_dim = len(y.forecast_day)
    y = y.rename(dict(forecast_day='features'))  # rename temporarily
    Xy = xr.concat([Xar, y], dim='features')  # maybe merge instead concat?
    Xyt = Xy.dropna('time', how='any')  # drop rows with nan values

    Xda = Xyt[:, :-out_dim]  # last column is predictand
    yda = Xyt[:, -out_dim:]  # features was only needed in merge
    yda = yda.rename(dict(features='forecast_day'))  # change renaming back to original
    return Xda, yda

