import numpy as np
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import dask
dask.config.set(scheduler='synchronous')
#from dask.distributed import Client, LocalCluster
#cluster = LocalCluster(processes=True)
#client = Client(cluster)  # memory_limit='16GB', 

import xarray as xr
from dask.diagnostics import ProgressBar

import sys
print(sys.executable)

def shift_time(ds, value):
    ds.coords['time'].values = pd.to_datetime(ds.coords['time'].values) + value
    return ds

static = xr.open_dataset('../data/danube/era5_slt_z_slor_lsm_stationary_field.nc')
static

glofas = xr.open_dataset('../data/danube/glofas_reanalysis_danube_1981-2002.nc')
glofas = glofas.rename({'lat': 'latitude', 'lon': 'longitude'})  # to have the same name like in era5
glofas = shift_time(glofas, -dt.timedelta(days=1))

z_glofas = static['z'].isel(time=0)/9.81  # converting to m approx.
z_glofas = z_glofas.interp(latitude=glofas.latitude,
                           longitude=glofas.longitude)

z_glofas

dis = glofas['dis']
dis

plt.figure(figsize=(26,6))
i = 48.45
j = 13.75
z_point = z_glofas.sel(latitude=slice(i, i-0.01)).sel(longitude=slice(j, j+0.01))

z_glofas.plot()
dis.mean('time').where(dis.mean('time') > 100).plot(cmap='RdBu')
ax = plt.gca()
p = plt.scatter(j, i, s=1500, marker='o', lw=10, facecolor='none', color='cyan')

plt.figure(figsize=(26,6))
i = 48.45
j = 13.75
z_point = z_glofas.sel(latitude=slice(i, i-0.01)).sel(longitude=slice(j, j+0.01))

z_glofas.where(z_glofas.values >= z_point.values-150).plot()
dis.mean('time').where(dis.mean('time') > 100).plot(cmap='RdBu')
ax = plt.gca()
p = plt.scatter(j, i, s=1500, marker='o', lw=10, facecolor='none', color='cyan')

z_glofas.where(z_glofas.values >= z_point.values)

plt.imshow(dis.mean('time').values)

a = dis.mean('time').values[:,110]
print(np.where(a==a.max()))
# (22, 110) is gridpoint of river and quite large

dis.mean('time').plot()

z_glofas.plot()

meandis = dis.mean('time')
meandis.plot()

plt.figure()
mindis = dis.min('time')
mindis.plot()

river = dis.min('time') > 5
river.plot.pcolormesh()
plt.title('discharge > 5 m^3/s')

di = 10
dj = 10
pct = 0.1  # influencing gridpoint must have mean discharge more than this percentage

#for i in range(di, len(dis.latitude)-dj):
#    for j in range(dj, len(dis.longitude)-dj):
got = False


while not got:
    i = np.random.choice(range(di, len(dis.latitude)-dj))
    j = np.random.choice(range(dj, len(dis.longitude)-dj))
    got = river[i,j]

if river[i,j] == 1:
    print(i,j)
    
    i0, i1 = i-di, i+di
    j0, j1 = j-dj, j+dj
    dis_box = meandis[i0:i1,j0:j1]
    z_box = z_glofas[i0:i1,j0:j1]

    fig, ax = plt.subplots()
    dis_box.plot()
    fig, ax = plt.subplots()
    z_box.where(river==1).plot()

    # select feature gridpoints
    fig, ax = plt.subplots()
    influencer = (dis_box > pct*meandis[i,j])  \
                  &(z_box >= z_glofas[i,j])   \
                  &(river==1)
    influencer.plot()

    # plot center point

#    break
