import os, warnings, sys
import numpy as np
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from dask_ml.preprocessing import StandardScaler
from dask_ml.decomposition import PCA

import dask
dask.config.set(scheduler='threading')

import xarray as xr
from dask.diagnostics import ProgressBar
from joblib import Parallel

sys.path.append('../../')
from python.aux.utils import open_data
from python.aux.ml_flood_config import path_to_data
from python.aux.floodmodels import FlowModel

era5 = open_data(path_to_data+'danube/', kw='era5')

glofas = open_data(path_to_data+'danube/', kw='glofas_ra')

if 'tp' in era5:
    tp = era5['tp']*1000
else:
    tp = (era5['cp']+era5['lsp'])*1000
tp.name = 'total precip [mm]'
tp = tp.interp(latitude=glofas.latitude,
               longitude=glofas.longitude)

from python.aux.utils_floodmodel import add_shifted_variables

shifts = range(1,4)
X = add_shifted_variables(glofas, shifts, variables='all')
X = X.drop('dis')  # current dis is to be predicted, is not a feature

y = glofas['dis']  # just this variable as dataarray

X

N_train = dict(time=slice(None, '1990'))
N_valid = dict(time=slice('1990', '1995'))

# kind, lat, lon will be replaced!
main_dir = '/home/srvx11/lehre/users/a1254888/ipython/ml_flood/'
ff_mod = main_dir+'/models/flowmodel/danube/kind/point_lat_lon_flowmodel.pkl'
ff_hist = main_dir+'/models/flowmodel/danube/kind/point_lat_lon_history.png'
ff_valid = main_dir+'/models/flowmodel/danube/kind/point_lat_lon_validation.png'
ff_upstream = main_dir+'/models/flowmodel/danube/kind/point_lat_lon_upstream.png'

#pipe = Pipeline([('scaler', StandardScaler()),
#                 #('pca', PCA(n_components=6)),
#                 ('model', FlowModel('Ridge', dict(alphas=np.logspace(-3, 2, 6)))),])

model = FlowModel('neural_net', dict(epochs=1000,
                                      ))
pipe = Pipeline([('model', model),])

from python.aux.utils_floodmodel import get_mask_of_basin

danube_gridpoints = get_mask_of_basin(glofas['dis'].isel(time=0), 'Danube')
plt.imshow(danube_gridpoints.astype(int))
plt.show()

def select_riverpoints(dis):
    return (dis > 10)

dis_map_mean = glofas['dis'].mean('time')
is_river = select_riverpoints(dis_map_mean)
mask_river_in_catchment = is_river & danube_gridpoints

plt.imshow(mask_river_in_catchment.astype(int))
plt.title('mask_river_in_catchment')
plt.show()

from python.aux.floodmodels import train_flowmodel

def mkdir(d):
    if not os.path.isdir(d):
        os.makedirs(d)

def replace(string: str, old_new: dict):
    for o, n in old_new.items():
        string = string.replace(o, str(n))
    return string

mkdir(os.path.dirname(ff_mod).replace('kind', model.kind))

grid = mask_river_in_catchment
lats = grid.latitude.values
lons = grid.longitude.values

is_river_in_catchment = grid.values
lons, lats = np.meshgrid(lons, lats)

lats = lats[is_river_in_catchment]
lons = lons[is_river_in_catchment]

task_list = []
#train_flowmodel = dask.delayed(train_flowmodel)  # make it a delayed function

for lat, lon in zip(lats, lons):
    f_mod = replace(ff_mod, dict(lat=lat, lon=lon, kind=model.kind))
    f_hist = replace(ff_hist, dict(lat=lat, lon=lon, kind=model.kind))
    f_valid = replace(ff_valid, dict(lat=lat, lon=lon, kind=model.kind))
    f_upstream = replace(ff_upstream, dict(lat=lat, lon=lon, kind=model.kind))

    #if not os.path.isfile(f_mod):  # if model does not yet exist

    task_list.append(train_flowmodel(X, y, pipe,
                                     lat, lon,
                                     tp, mask_river_in_catchment,
                                     N_train, N_valid,
                                     f_mod, f_hist, f_valid, f_upstream, debug=True))

len(task_list)

with ProgressBar():
    with dask.config.set(scheduler='synchronous'):
        dask.compute(task_list)

Parallel(n_jobs=1, verbose=10)(task_list)

files = os.listdir(os.path.dirname(f_mod))
len(files)

from IPython.display import Image
ddir = "../models/flowmodel/danube/neural_net-1/"

Image(ddir+'point_48.05_10.150000000000034_history.png')

Image(ddir+'point_48.05_10.150000000000034_validation.png')

Image(ddir+'point_49.75_11.75_upstream.png')

Image(ddir+'point_47.75_17.65_upstream.png')

Image(ddir+'point_48.35_18.25_upstream.png')

