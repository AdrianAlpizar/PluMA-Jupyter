import link_src

#%matplotlib inline
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from dask.distributed import Client
client = Client(processes=True)
import dask
#dask.config.set(scheduler='processes')
from dask.diagnostics import ProgressBar

from python.aux.ml_flood_config import path_to_data
# load dask client
client
# define some vars
data_path = f'{path_to_data}danube/'
print(data_path)

from python.aux.utils import open_data
# load data
era5 = open_data(data_path, kw='era5')
glofas = open_data(data_path, kw='glofas_ra')

from python.aux.utils import calc_stat_moments
sm = calc_stat_moments(ds=era5, dim_aggregator='time', time_constraint=None)

print(sm)

from python.aux.plot import Map
m = Map(figure_kws=dict(figsize=(15,10)))

da_mean = sm['cp'].sel(stat_moments='mean').compute()
m.plot(da_mean)
plt.title('convective precip averaged over time')
plt.show()

da_mean = sm['cp'].sel(stat_moments='std').compute()
m.plot(da_mean)
plt.title('convective precip standard deviation over time')
plt.show()

da_mean = sm['cp'].sel(stat_moments='vc').compute()

m.plot(da_mean, cbar_kwargs={'fraction': 0.1})
plt.title('convective precip coefficient of variation over time')
plt.show()

sm_time = calc_stat_moments(ds=era5, dim_aggregator='spatial', time_constraint=None)

# resample to yearly values
da_time_mean = sm_time[['cp', 'lsp']].sel(stat_moments='mean').compute().resample(time="y").mean()

da_time_mean['cp'].plot(label='cp')
da_time_mean['lsp'].plot(label='lsp')
plt.legend()
plt.title('convective and large scale precip yearly averages over space')
plt.ylabel('precip')
plt.show()

from python.aux.utils import spatial_cov

cp = era5['cp']
lat = 48.5
lon = 15.5
scov = spatial_cov(cp, lat=lat, lon=lon)

m.plot(scov)
plt.title('spatial covariance for convective precip at the specified location')
m.plot_point(plt.gca(), lat, lon)




