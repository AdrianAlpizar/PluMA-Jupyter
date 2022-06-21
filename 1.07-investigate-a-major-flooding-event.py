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
from python.aux.utils import open_data
from python.aux.plot import Map
from python.aux.utils import calc_stat_moments
from python.aux.utils import spatial_cov

# load dask client
client
# define some vars
data_path = f'{path_to_data}danube/'
print(data_path)

# load data
era5 = open_data(data_path, kw='era5')
glofas = open_data(data_path, kw='glofas_ra')

lat = 48.403
lon = 15.615
krems = dict(latitude=lat, longitude=lon)

start_date = '2013-05-20'
end_date = '2013-07-20'
dis_krems = glofas.sel(time=slice(start_date, end_date)).interp(krems).drop(['latitude', 'longitude'])['dis']
long_term_mean_krems = glofas.interp(krems).drop(['latitude', 'longitude'])['dis'].mean().values
print(dis_krems)

dis_krems.plot(label='discharge')
plt.gca().axhline(long_term_mean_krems, ls='--', color='r', label='long-term average')
plt.legend()
plt.title(f'GLOFAS river discharge at {krems}')

start_date_rise = '2013-05-31'
end_date_rise = '2013-06-06'
dis_krems_rise = glofas.sel(time=slice(start_date_rise, end_date_rise)).interp(krems).drop(['latitude', 'longitude'])['dis']

dis_krems_rise.plot(label='absolute discharge')
plt.legend()
plt.gca().twinx()
dis_krems_rise.diff(dim='time').plot(color='r', label='change per day')
plt.ylabel('change in discharge per day')
plt.title(f'absolute and cahnge per day discharge for {krems}')
plt.legend()

m = Map(figure_kws=dict(figsize=(15,10)))

from python.aux.utils import spatial_cov_2var

dis_krems_rise_diff = dis_krems_rise.diff(dim='time')
cp_rise = era5['cp'].sel(time=slice(start_date_rise, end_date_rise))[1:]

spcov_2 = spatial_cov_2var(dis_krems_rise_diff, cp_rise)
m.plot(spcov_2)
m.plot_point(plt.gca(), lat, lon)
plt.title(f'spatial covariance between discharge at {krems} and convective precip')

lsp_rise = era5['lsp'].sel(time=slice(start_date_rise, end_date_rise))[1:]

spcov_2 = spatial_cov_2var(dis_krems_rise_diff, lsp_rise)
m.plot(spcov_2)
m.plot_point(plt.gca(), lat, lon)
plt.title(f'spatial covariance between discharge at {krems} and large-scale precip')

ro_rise = era5['ro'].sel(time=slice(start_date_rise, end_date_rise))[1:]

spcov_2 = spatial_cov_2var(dis_krems_rise_diff, ro_rise)
m.plot(spcov_2)
m.plot_point(plt.gca(), lat, lon)
plt.title(f'spatial covariance between discharge at {krems} and runoff')


local_region = dict(latitude=slice(krems['latitude']+1.5,
                                   krems['latitude']-1.5),
                    longitude=slice(krems['longitude']-1.5,
                                    krems['longitude']+1.5))

start_date = '2013-05-20'
end_date = '2013-07-20'
xds = era5.sel(local_region).sel(time=slice(start_date, end_date))
yda = dis_krems.copy()
print(xds)
print(yda)

yda_change = yda.diff(dim='time')
yda_change.name = 'dis_change'
print(yda_change)

sm_era5 = calc_stat_moments(xds, dim_aggregator='spatial')
print(sm_era5)

fig, ax = plt.subplots(figsize=(12,7))
for var in sm_era5.variables:
    if not var in ['time', 'level', 'stat_moments']:
        da_iter = sm_era5[var].sel(stat_moments='vc')
        try:
            da_iter.mean(dim='level').plot(ax=ax, label=var)
        except:
            da_iter.plot(ax=ax, label=var)

plt.yscale('log')
plt.grid()
ax2 = ax.twinx()
yda.plot(ax=ax2, ls='--', color='k', label='glofas-discharge')
yda_change.plot(ax=ax2, ls='--', color='grey', label='glofas-discharge change')
fig.legend()

fig, ax = plt.subplots(figsize=(12,7))
for var in sm_era5.variables:
    if not var in ['time', 'level', 'stat_moments']:
        da_iter = sm_era5[var].sel(stat_moments='mean')/sm_era5[var].sel(stat_moments='std')
        try:
            da_iter.mean(dim='level').plot(ax=ax, label=var)
        except:
            da_iter.plot(ax=ax, label=var)

plt.yscale('log')
plt.grid()
ax2 = ax.twinx()
yda.plot(ax=ax2, ls='--', color='k', label='glofas-discharge')
yda_change.plot(ax=ax2, ls='--', color='grey', label='glofas-discharge change')
plt.title('mean/std')
fig.legend()

