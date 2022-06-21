import numpy as np
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

import sys
sys.path.append('../../')

#from python.aux.utils_floodmodel import get_mask_of_basin, add_shifted_variables, reshape_scalar_predictand

poi = dict(latitude=48.35, longitude=13.95)
# load data
era5 = xr.open_dataset('../../data/smallsampledata-era5.nc')
glofas = xr.open_dataset('../../data/smallsampledata-glofas.nc')
era5

sd_diff = era5['sd'].diff(dim='time')
sd_diff.name = 'sd_diff'

era5 = era5.assign({'sd_diff': sd_diff})

dis_mean = glofas['dis'].mean('time')
dis_mean.plot()
plt.title('mean discharge in the dataset domain')

dummy = glofas['dis'].isel(time=0).drop('time')
danube_catchment = get_mask_of_basin(dummy, kw_basins='Danube')
dis_mean.where(danube_catchment).plot()
plt.title('mean discharge in the danube catchment')
plt.gca().plot(poi['longitude'], poi['latitude'], color='cyan', marker='o',
               markersize=20, mew=4, markerfacecolor='none')

def feature_preproc(era5, glofas, timeinit, timeend):

    features = ['cp', 'lsp', 'ro', 'rtp_500-850', 'tcwv',
                'swvl1', 'sd', 'sd_diff']
    era5_features = era5[features]

    # interpolate to glofas grid
    era5_features = era5_features.interp(latitude=glofas.latitude,
                                         longitude=glofas.longitude)
    # time subset
    era5_features = era5_features.sel(time=slice(timeinit, timeend))
    glofas = glofas.sel(time=slice(timeinit, timeend))

    # select the point of interest
    # poi = dict(latitude=48.403, longitude=15.615)  # krems (lower austria), outside the test dataset
    poi = dict(latitude=48.35, longitude=13.95)  # point in upper austria

    dummy = glofas['dis'].isel(time=0)
    danube_catchment = get_mask_of_basin(dummy, kw_basins='Danube')
    X = era5_features.where(danube_catchment).mean(['latitude', 'longitude'])

    # select area of interest and average over space for all features
    dis = glofas.interp(poi)
    y = dis.diff('time', 1)  # compare predictors to change in discharge

    shifts = range(1,3)
    notshift_vars = ['swvl1', 'tcwv', 'rtp_500-850']
    shift_vars = [v for v in X.data_vars if not v in notshift_vars]

    X = add_shifted_variables(X, shifts, variables=shift_vars)

    Xda, yda = reshape_scalar_predictand(X, y)  # reshape into dimensions (time, feature)
    return Xda, yda

import seaborn as sns

def generate_heatmap(X, y, descr='description'):
    df = pd.DataFrame(data=X.values, columns=X.features.values, index=X.time.values)
    df['predictand'] = y
    plt.figure(figsize=(25,25))
    cor = df.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    plt.show()
    cor_predictand = abs(cor['predictand'])
    feature_importance = cor_predictand[cor_predictand > 0.2]
    print(descr)
    print(feature_importance)
    return feature_importance

important_features = []

timeinit, timeend = '1980', '1984'
X, y = feature_preproc(era5=era5, glofas=glofas, timeinit=timeinit, timeend=timeend)

Xtp = X.sel(features=('cp',)) + X.sel(features=('lsp',))
Xprecip = Xtp.where(Xtp>1/1000, drop=True)
X = X.where(Xprecip)
y = y.where(Xprecip)

ft = generate_heatmap(X=X, y=y, descr=f'{timeinit}-{timeend}; |rho|; only days with precip > 1mm')
important_features.append(ft)

timeinit, timeend = '1985', '1989'
X, y = feature_preproc(era5=era5, glofas=glofas, timeinit=timeinit, timeend=timeend)

Xtp = X.sel(features=('cp',)) + X.sel(features=('lsp',))
Xprecip = Xtp.where(Xtp>1/1000, drop=True)
X = X.where(Xprecip)
y = y.where(Xprecip)

ft = generate_heatmap(X=X, y=y, descr=f'{timeinit}-{timeend}; only days with precip > 1mm')
important_features.append(ft)

timeinit, timeend = '1990', '1995'
X, y = feature_preproc(era5=era5, glofas=glofas, timeinit=timeinit, timeend=timeend)
Xtp = X.sel(features=('cp',)) + X.sel(features=('lsp',))

Xprecip = Xtp.where(Xtp>1/1000, drop=True)
X = X.where(Xprecip)
y = y.where(Xprecip)
ft = generate_heatmap(X=X, y=y, descr=f'{timeinit}-{timeend}; only days with precip > 1mm')
important_features.append(ft)

timeinit, timeend = '1989', '1989'
X, y = feature_preproc(era5=era5, glofas=glofas, timeinit=timeinit, timeend=timeend)
Xtp = X.sel(features=('cp',)) + X.sel(features=('lsp',))

#Xprecip = Xtp.where(Xtp>1/1000, drop=True)
#X = X.where(Xprecip)
#y = y.where(Xprecip)
ft = generate_heatmap(X=X, y=y, descr=f'{timeinit}-{timeend}; all days')
important_features.append(ft)

timeinit, timeend = '1989', '1989'
X, y = feature_preproc(era5=era5, glofas=glofas, timeinit=timeinit, timeend=timeend)
Xtp = X.sel(features=('cp',)) + X.sel(features=('lsp',))

Xprecip = Xtp.where(Xtp<1/1000, drop=True)
X = X.where(Xprecip)
y = y.where(Xprecip)
ft = generate_heatmap(X=X, y=y, descr=f'{timeinit}-{timeend}; only days with precip < 1mm')
important_features.append(ft)

timeinit, timeend = '1989', '1989'
X, y = feature_preproc(era5=era5, glofas=glofas, timeinit=timeinit, timeend=timeend)
Xtp = X.sel(features=('cp',)) + X.sel(features=('lsp',))

Xprecip = Xtp.where(Xtp>1/1000, drop=True)
X = X.where(Xprecip)
y = y.where(Xprecip)
ft = generate_heatmap(X=X, y=y, descr=f'{timeinit}-{timeend}; only days with precip > 1mm')
important_features.append(ft)

era5_catchment = era5.interp(latitude=glofas.latitude,
                             longitude=glofas.longitude)
danube_catchment = get_mask_of_basin(dummy, kw_basins='Danube')
era5_catchment = era5_catchment.where(danube_catchment)

sd_mean = era5_catchment['sd'].mean(['latitude', 'longitude'])
sd_diff_mean = era5_catchment['sd_diff'].mean(['latitude', 'longitude'])
dis_mean = glofas['dis'].mean(['latitude', 'longitude'])

dis_mean_slice = dis_mean.sel(time=slice('1989', '1989'))
plt1, = dis_mean_slice.plot(label='dis')
ax2 = plt.gca().twinx()
sd_mean_slice = sd_mean.sel(time=slice('1989', '1989'))
plt2, = sd_mean_slice.plot(label='sd', ax=ax2, color='orange')
rho = np.corrcoef(dis_mean_slice, sd_mean_slice)[1,0]
plt.title(f'corrcoef={rho}')
plt.legend((plt1, plt2), ('dis', 'sd'))

dis_mean_slice = dis_mean.sel(time=slice('1989', '1989'))
plt1, = dis_mean_slice.plot(label='dis')
ax3 = plt.gca().twinx()
sd_diff_mean_slice = sd_diff_mean.sel(time=slice('1989', '1989'))
plt2, = sd_diff_mean_slice.plot(label='sd_diff', ax=ax3, color='orange')
ax3.axhline(0, color='k')
rho = np.corrcoef(dis_mean_slice, sd_diff_mean_slice)[1,0]
plt.title(f'corrcoef={rho}')
plt.legend((plt1, plt2), ('dis', 'sd_diff'))

for entry in important_features:
    print(entry[:-1].sort_values(ascending=False))
    print('#'*30)

timeinit = '2013-05-30'
timeend = '2013-06-15'
X, y = feature_preproc(era5=era5, glofas=glofas, timeinit=timeinit, timeend=timeend)

Xtp = X.sel(features=('cp',)) + X.sel(features=('lsp',))
Xprecip = Xtp.where(Xtp>1/1000, drop=True)
X = X.where(Xprecip)
y = y.where(Xprecip)

ft = generate_heatmap(X=X, y=y, descr=f'{timeinit}-{timeend}; only days with precip > 1mm')

ft[np.abs(ft) > 0.2]

