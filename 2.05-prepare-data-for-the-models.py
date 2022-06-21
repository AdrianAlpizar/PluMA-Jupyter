import numpy as np
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import sys
sys.path.append('../../')
from python.aux.utils_floodmodel import get_mask_of_basin, add_shifted_variables, reshape_scalar_predictand, shift_and_aggregate

# load data
era5 = xr.open_mfdataset('../../data/sampledata-*-era5.nc', combine='by_coords')
glofas = xr.open_mfdataset('../../data/sampledata-*-glofas.nc', combine='by_coords')

danube_catchment = get_mask_of_basin(glofas['dis'].isel(time=0))
dis = glofas['dis'].where(danube_catchment)

era5_basin = era5.interp(latitude=glofas.latitude,
                         longitude=glofas.longitude).where(danube_catchment)

era5_basin['sd'].isel(time=1).plot()

sd_diff = era5_basin['sd'].diff(dim='time')
sd_diff.name = 'sd_diff'
era5_basin = era5_basin.assign({'sd_diff': sd_diff})

era5_basin

# average features in the basin
era5_basin = era5_basin.mean(['latitude', 'longitude'])

era5_basin['lsp-4-11'] = shift_and_aggregate(era5_basin['lsp'], shift=4, aggregate=8)
era5_basin['lsp-12-25'] = shift_and_aggregate(era5_basin['lsp'], shift=12, aggregate=14)
era5_basin['lsp-26-55'] = shift_and_aggregate(era5_basin['lsp'], shift=26, aggregate=30)
era5_basin['lsp-56-180'] = shift_and_aggregate(era5_basin['lsp'], shift=56, aggregate=125)

era5_basin

def prepare_features(x, y, point=None):
    # drop the features which did show no impact in the feature selection
    x = x.drop('tcwv')
    x = x.drop('rtp_500-850')
    #features = ['cp', 'lsp', 'ro', 'swvl1', 'sd', 'sd_diff']
    #x = x[features]
        
    for var in ['lsp', 'cp']:
        for i in range(1,4):
            newvar = var+'-'+str(i)
            x[newvar] = x[var].shift(time=i)  # previous precip as current day variable

    # select the point of interest
    if not point:
        poi = dict(latitude=48.35, longitude=13.95)  # point in upper austria
    elif point == 'krems':
        poi = dict(latitude=48.35, longitude=15.65)  # krems (lower austria), outside the test dataset
    
    # select area of interest and average over space for all features
    y_poi = y.interp(poi)
    y_poi_diff = y_poi.diff('time', 1)  # compare predictors to change in discharge
    y_poi_diff.name = 'dis_diff'

    # merge into one dataset
    sample_data = xr.merge([x, y_poi, y_poi_diff])
    sample_data = sample_data.drop(['latitude', 'longitude'])
    return sample_data
sample_data_1 = prepare_features(era5_basin, dis)

test1 = sample_data_1.to_array(dim='features').T
for f in test1.features:
    plt.figure(figsize=(15,5))
    test1.sel(features=f).plot(ax=plt.gca())

sample_data_1.load()

amax_ind = np.argmax(test1.sel(features='ro').values)-1
print(sample_data_1['ro'][amax_ind])
sample_data_1['ro'][amax_ind] = np.nanmean([sample_data_1['ro'][amax_ind-1], sample_data_1['ro'][amax_ind+1]])
print(sample_data_1['ro'][amax_ind])

sample_data_1.to_netcdf('../../data/features_xy.nc')

sample_data_1['dis'].sel(time=slice('2013-5', '2013-6')).plot()

