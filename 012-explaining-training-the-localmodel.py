import os, warnings
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

import link_src
from python.aux.utils import open_data
from python.aux.ml_flood_config import path_to_data

era5 = open_data(path_to_data+'danube/', kw='era5')

glofas = open_data(path_to_data+'danube/', kw='glofas_ra')

if 'tp' in era5:
    era5['tp'] = era5['tp']*1000
else:
    era5['tp'] = (era5['cp']+era5['lsp'])*1000

# no interpolation necessary

era5['reltop'] = era5['z'].sel(level=500) - era5['z'].sel(level=850)

era5['q_mean'] = era5['q'].mean('level')

listofpredictors = ['reltop', 'q_mean', 'tp', 'ro']
X_local = era5[listofpredictors]

shifts = range(1,4)
#X = add_shifted_predictors(glofas, shifts, variables='all')
#X = X.drop('dis')  # current dis is to be predicted, is not a feature

X_local

from python.aux.utils_flowmodel import add_shifted_predictors

shifts = range(1,4)
X_flow = add_shifted_predictors(glofas, shifts, variables='all')
X_flow = X_flow.drop('dis')  # current dis is to be predicted, is not a feature

N_train = dict(time=slice(None, '1990'))
N_valid = dict(time=slice('1990', '1995'))

# kind, lat, lon will be replaced!
main_dir = '/home/srvx11/lehre/users/a1254888/ipython/ml_flood/'
ff_mod = main_dir+'/models/localmodel/danube/kind/point_lat_lon_localmodel.pkl'
ff_hist = main_dir+'/models/localmodel/danube/kind/point_lat_lon_history.png'
ff_valid = main_dir+'/models/localmodel/danube/kind/point_lat_lon_validation.png'


ff_mod_transport = main_dir+'/models/flowmodel/danube/kind/point_lat_lon_flowmodel.pkl'

from python.aux.floodmodels import LocalModel

#pipe = Pipeline([('scaler', StandardScaler()),
#                 #('pca', PCA(n_components=6)),
#                 ('model', FlowModel('Ridge', dict(alphas=np.logspace(-3, 2, 6)))),])

model = LocalModel('neural_net', dict(epochs=1000,))
pipe = Pipeline([#('pca', PCA(n_components=6)),
                  ('model', model),])

from python.aux.utils_flowmodel import get_mask_of_basin

mask_catchment = get_mask_of_basin(glofas['dis'].isel(time=0), 'Danube')
plt.imshow(mask_catchment.astype(int))
plt.title('Catchment basin of the Danube river')
plt.show()

def select_riverpoints(dis):
    return (dis > 10)

dis_map_mean = glofas['dis'].mean('time')
is_river = select_riverpoints(dis_map_mean)

mask_river_in_catchment = is_river & mask_catchment

plt.imshow(mask_river_in_catchment.astype(int))
plt.title('mask_river_in_catchment')
plt.show()

np.seterr(divide='ignore', invalid='ignore')
from joblib import Parallel, delayed  #  parallel computation
from joblib import dump, load   # saving and loading pipeline objects ("models")
from sklearn.base import clone
from python.aux.utils_flowmodel import select_upstream, preprocess_reshape_flowmodel
@delayed
def train_localmodel(X_local, X_flow, pipe,
                    lat, lon,
                    f_mod, f_hist, f_valid, 
                    f_mod_transport, 
                    debug=False):
    """Train the local model, save it to disk."""

    upstream = select_upstream(mask_river_in_catchment, lat, lon, basin='Danube')
    N_upstream = int(upstream.sum())
    lats, lons = str(lat), str(lon)

    if not os.path.isfile(f_mod_transport) or N_upstream <= 5:
        if debug:
            print(lats, lons, 'is spring.')  # assume constant discharge
            y_flow = glofas['dis'].sel(latitude=lat, longitude=lon).mean('time')
    else:
        dis_point = glofas['dis'].sel(latitude=float(lat), longitude=float(lon))
        tp_box = era5['tp'].sel(latitude=slice(lat+1.5, lat-1.5),
                                longitude=slice(lon-1.5, lon+1.5))
        hasprecip = tp_box.mean(['longitude', 'latitude']) > 0.5
        
        if debug:
            print('predict mean flow using the transport model...')
            print('upstream:', N_upstream)
            
        # prepare the transport model input data
        Xt = X_flow.where(upstream & hasprecip, drop=True)
        if debug:
            plt.imshow(upstream.astype(int))
            plt.title('upstream')
            plt.show()
        
        yt = dis_point
        Xda, yda, time = preprocess_reshape_flowmodel(Xt, yt)
        X_flow = Xda
        if debug:
            print(X_flow.shape)
        
        ppipe = load(f_mod_transport)
        y_flow = ppipe.predict(X_flow)
        # background forecast finished, calculate residual now
        y_res = dis_point - y_flow
        
        Xt = X_local.sel(latitude=slice(lat+1.5, lat-1.5),
                          longitude=slice(lon-1.5, lon+1.5))

        Xt = Xt.where(hasprecip)
        yt = y_res
        Xda, yda, time = preprocess_reshape_flowmodel(Xt, yt)

        X_train = Xda.loc[N_train]
        y_train = yda.loc[N_train]
        X_valid = Xda.loc[N_valid]
        y_valid = yda.loc[N_valid]

        if debug:
            print(X_train.shape, y_train.shape)
            print(X_valid.shape, y_valid.shape)
        ppipe = clone(pipe)
        history = ppipe.fit(X_train.values, y_train.values,
                           model__validation_data=(X_valid.values,
                                                   y_valid.values))

        dump(ppipe, f_mod)

        try:
            h = history.named_steps['model'].m.model.history

            # Plot training & validation loss value
            fig, ax = plt.subplots()
            ax.plot(h.history['loss'], label='loss')
            ax.plot(h.history['val_loss'], label='val_loss')
            plt.title('Model loss')
            ax.set_ylabel('Loss')
            ax.set_xlabel('Epoch')
            plt.legend() #['Train', 'Test'], loc='upper left')
            ax.set_yscale('log')
            fig.savefig(f_hist); plt.close('all')
        except Exception as e:
            warnings.warn(str(e))

        ppipe = load(f_mod)
        y_m = ppipe.predict(X_valid)

        try:
            fig, ax = plt.subplots(figsize=(10,4))
            y_valid.plot(ax=ax, label='reanalysis')
            y_m.plot(ax=ax, label='prediction')
            plt.legend()
            fig.savefig(f_valid); plt.close('all')
        except Exception as e:
            warnings.warn(str(e))

def mkdir(d):
    if not os.path.isdir(d):
        os.makedirs(d)
        
def replace(string: str, old_new: dict):
    for o, n in old_new.items(): 
        string = string.replace(o, str(n))
    return string

mkdir(os.path.dirname(ff_mod).replace('kind', model.kind))
task_list = []

for lon in mask_river_in_catchment.longitude:
    for lat in mask_river_in_catchment.latitude:
        if mask_river_in_catchment.sel(latitude=lat, longitude=lon) == 1:  # valid danube river point
            lat, lon = float(lat), float(lon)
            lat, lon = 48.35, 15.650000000000034
            #print(lat, lon)

            f_mod = replace(ff_mod, dict(lat=lat, lon=lon, kind=model.kind))
            f_hist = replace(ff_hist, dict(lat=lat, lon=lon, kind=model.kind))
            f_valid = replace(ff_valid, dict(lat=lat, lon=lon, kind=model.kind))

            f_mod_transport = replace(ff_mod_transport, dict(lat=lat, lon=lon, kind=model.kind))

            transport_exists = os.path.isfile(f_mod_transport)
            localmodel_exists = os.path.isfile(f_mod)

            if transport_exists and not localmodel_exists:
                task = train_localmodel(X_local, X_flow, pipe,
                                            lat, lon,
                                            f_mod, f_hist, f_valid,
                                            f_mod_transport,
                                            debug=False)
                task_list.append(task)

len(task_list)

Parallel(n_jobs=20, verbose=10)(task_list)

files = os.listdir(os.path.dirname(f_mod))
len(files)

from IPython.display import Image
ddir = "../models/localmodel/danube/neural_net/"

Image(ddir+'point_48.65_16.850000000000023_history.png')

Image(ddir+'point_48.65_16.850000000000023_validation.png')

