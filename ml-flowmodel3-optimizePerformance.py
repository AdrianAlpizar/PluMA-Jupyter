import numpy as np
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import dask
import xarray as xr
from dask.diagnostics import ProgressBar

import joblib
from sklearn.pipeline import Pipeline
from dask_ml.preprocessing import StandardScaler
from dask_ml.decomposition import PCA

#from dask_ml.xgboost import XGBRegressor
#from dask_ml.linear_model import LogisticRegression
from dask_ml.linear_model import LinearRegression

import keras
from keras.layers.core import Dropout

import sys
print(sys.executable)

def shift_time(ds, value):
    ds.coords['time'].values = pd.to_datetime(ds.coords['time'].values) + value
    return ds

static = xr.open_dataset('../data/usa/era5_slt_z_slor_lsm_stationary_field.nc')

era5 = xr.open_dataset('../data/usa/era5_lsp_cp_1981-2017_daysum.nc')

glofas = xr.open_dataset('../data/usa/glofas_reanalysis_usa_1981-2002.nc')
glofas = glofas.rename({'lat': 'latitude', 'lon': 'longitude'})  # to have the same name like in era5
glofas = shift_time(glofas, -dt.timedelta(days=1))
# da.transpose(dims='latitude')  # flip?

glofas = glofas.isel(time=slice(0, 365*3))

box_model = dict(latitude=slice(40, 28), 
                 longitude=slice(-95, -85))

glofas = glofas.sel(box_model)
era5 = era5.sel(box_model)

dis = glofas['dis']

z_glofas = static['z'].isel(time=0)/9.81  # converting to m approx.
z_glofas = z_glofas.interp(latitude=glofas.latitude,
                           longitude=glofas.longitude)
z_glofas = z_glofas.drop('time')  # time is misleading as the topography does not change

tp = (era5['cp']+era5['lsp'])*1000
tp.name = 'total precip [mm]'
tp = tp.interp(latitude=glofas.latitude,
               longitude=glofas.longitude)

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

def correlate(da_3d, da_timeseries, timelag=False):
    a = da_3d - da_3d.mean('time')
    b = da_timeseries - da_timeseries.mean('time')
    N = len(b.coords['time'])
    if timelag:
        b = b.drop('time')
        a = a.drop('time')
    out = b.dot(a)/a.std('time')/b.std()/N
    out.name = 'correlation coefficient'
    return out

def select_river(dis):
    river = dis.min('time') > 5
    river.name = 'river mask [0/1]'
    return river

def select_upstream_river(dis_box, dis_point, z_box, z_point, rivermask, pct):
    lags = [-1, 1]

    timelag_corrs = np.full((len(lags), len(dis_box.latitude), len(dis_box.longitude)), np.nan)
    for t, lag in enumerate(lags):
        if lag > 0:  # dis_box with data from previous timesteps
            cntr = dis_point[lag:]
            dis_box_shift = dis_box[:-lag]
        elif lag < 0:  # dis_box with data from future timesteps
            cntr = dis_point[:lag]
            dis_box_shift = dis_box[-lag:]

        dis_box_relevant = dis_box_shift.where(rivermask==1)
        timelag_corrs[t,:,:] = correlate(dis_box_relevant, cntr, timelag=True)

    lag_influencing = timelag_corrs[1,:,:]>timelag_corrs[0,:,:]
    #plt.imshow(lag_influencing)

    influencer = (dis_box.mean('time') > pct*dis_point.mean('time'))  \
                  &(z_box > z_point)   \
                  &(rivermask==1) & lag_influencing
    influencer.name = 'gridpoints influencing discharge [0/1]'
    #influencer.plot()
    return influencer

shifts = range(1,4)
X_dis = add_shifted_predictors(glofas, shifts, variables='all')
X_dis = X_dis.drop('dis')  # we actually want to predict (t) with (t-1, t-2, t-3)
y_dis = glofas['dis']

i, j = 70, 38
di = 20
dj = 20
pct = 0.1  # influencing gridpoint must have mean discharge more than this percentage

i0, i1 = i-di, i+di
j0, j1 = j-dj, j+dj

tp_box = tp[:, i0:i1, j0:j1]
few_precip = tp_box.mean(['longitude', 'latitude']) < 0.1
print('percentage:',sum(few_precip.astype(int))/few_precip.size) # .plot() #.plot() #'#'
print(few_precip)

fig, ax = plt.subplots(figsize=(25,5))
few_precip.astype(int).to_pandas().plot(ax=ax)

tp_box.where(few_precip).isel(time=0).plot()

dis = dis.load()
disf = dis.where(few_precip)

dis

dis_point = dis[:,i,j]
dis_box = dis[:, i0:i1, j0:j1]
z_point = z_glofas[i,j]
z_box = z_glofas[i0:i1,j0:j1]


rivermask = select_river(dis_box)

upstream = select_upstream_river(dis_box, dis_point, z_box, z_point, rivermask, pct)

upstream.plot()
print(upstream.sum())

def preprocess_reshape(X_dis, y_dis, upstream, i, j):
    X_dis = X_dis.where(upstream)
    X_dis = X_dis.to_array(dim='time_feature')  
    X_dis = X_dis.stack(features=['latitude', 'longitude', 'time_feature'])
    Xar = X_dis.dropna('features', how='all')
    
    yar = y_dis[:,i,j]
    yar = yar.drop(['latitude', 'longitude'])
    yar.coords['features'] = 'dis'
    
    Xy = xr.concat([Xar, yar], dim='features')
    Xyt = Xy.dropna('time', how='any')  # drop them as we cannot train on nan values
    time = Xyt.time
    
    Xda = Xyt[:,:-1]
    yda = Xyt[:,-1]
    return Xda, yda, time

Xda, yda, time = preprocess_reshape(X_dis, y_dis, upstream, i,j)

Xda

for p in range(Xda.shape[1]):
    print(p, float(Xda[:,p].mean()))

X_dis.where(upstream) #.plot()

Xda = Xda.values
yda = yda.values[:, np.newaxis]

print('yda.shape:', Xda.shape, 'yda.shape:',  yda.shape)

class KerasDenseNN(object):
    def __init__(self, **kwargs):
        model = keras.models.Sequential()
        self.cfg = kwargs
        
        model.add(keras.layers.BatchNormalization())
        
        model.add(keras.layers.Dense(8,
                                  kernel_initializer='normal', 
                                  bias_initializer='zeros',
                                  activation='relu')) #('sigmoid'))
        #model.add(Dropout(self.cfg.get('dropout')))
        #model.add(keras.layers.Dense(32))
        #model.add(keras.layers.Activation('sigmoid'))
        #model.add(Dropout(self.cfg.get('dropout')))
        #model.add(keras.layers.Dense(16))
        #model.add(keras.layers.Activation('sigmoid'))
        #model.add(Dropout(self.cfg.get('dropout')))
        #model.add(keras.layers.Dense(8))
        #model.add(keras.layers.Activation('sigmoid'))
        #model.add(Dropout(self.cfg.get('dropout')))
        model.add(keras.layers.Dense(1, activation='linear'))
        #                     bias_initializer=keras.initializers.Constant(value=9000)))
        
        #ha = self.cfg.get('hidden_activation')

        #for N_nodes in self.cfg.get('N_hidden_nodes'):
        #        
        #    model.add(hidden)
        #    model.add(ha.copy())
        #    
        #    if self.cfg.get('dropout'):
        #        model.add(Dropout(self.cfg.get('dropout')))#

        #outputlayer = keras.layers.Dense(1, activation='linear')

        #optimizer_name, options_dict = self.cfg.get('optimizer')
        #optimizer = getattr(keras.optimizers, optimizer_name)(**options_dict)
        #optimizer = keras.optimizers.SGD(lr=0.01)
        rmsprop = keras.optimizers.RMSprop(lr=.1)
        sgd = keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.5, nesterov=True)

        model.compile(loss=self.cfg.get('loss'), 
                      optimizer=rmsprop)
        self.model = model

        self.callbacks = [keras.callbacks.EarlyStopping(monitor='loss',
                            min_delta=1, patience=100, verbose=0, mode='auto',
                            baseline=None, restore_best_weights=True),]

    def predict(self, X):
        return self.model.predict(X).squeeze()

    def fit(self, X, y, **kwargs):
        return self.model.fit(X, y,
                              epochs=self.cfg.get('epochs', None),
                              batch_size=self.cfg.get('batch_size', None),
                              callbacks=self.callbacks,
                              verbose=1,
                              **kwargs)
    

mlp_kws = dict(optimizer=('sgd', dict(lr=1)),
               loss='mean_squared_error',
               #N_hidden_nodes=(4,4),
               #hidden_activation=keras.layers.Activation('sigmoid'), #keras.layers.ReLU(), #-LeakyReLU(alpha=0.3), #'relu',
               #output_activation='linear',
               #bias_initializer='random_uniform',
               batch_size=128,
               dropout=0., #.25,
               epochs=1000,
              )


linear_kws = dict(C=.1, n_jobs=-1, max_iter=10000, verbose=True)


if False:
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('pca', PCA(n_components=4)),
                     ('model', LinearRegression(**linear_kws)),],
                    verbose=True)
if True:
    pipe = Pipeline([#('scaler', StandardScaler()),
                     #('pca', PCA(n_components=2)),
                     ('model', KerasDenseNN(**mlp_kws)),],
                    verbose=False)
    
pipe

type(Xda)

history = pipe.fit(Xda, yda)

keras.utils.print_summary(pipe.named_steps['model'].model)

h = history.named_steps['model'].model.history

# Plot training & validation loss values
plt.plot(h.history['loss'])
#plt.plot(h.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.gca().set_yscale('log')
plt.show()

def add_time(vector, time, name=None):
    """Converts arrays to xarrays with a time coordinate."""
    return xr.DataArray(vector, dims=('time'), coords={'time': time}, name=name)

with ProgressBar():
    ytest = pipe.predict(Xda)

ytest.shape

ytest = add_time(ytest, time, name='dis-forecast')
ytest_dis = ytest #.cumsum('time')
ytest_dis.values
#ytest_dis += y[0]  # initial state + changes = timeseries of forecasted discharge

fig, ax = plt.subplots(figsize=(24,5))
#for p in range(Xda.shape[1]):
    #print(p)
    #print(Xda[:,p])
minpred = add_time(Xda.min(axis=1), time)
maxpred = add_time(Xda.max(axis=1), time)

minpred.plot(ax=ax, linestyle='--', label='predictor-min')
maxpred.plot(ax=ax, linestyle='--', label='predictor-max')


obs = dis[:,i,j].to_pandas()
fcst = ytest_dis.to_pandas()
obs.plot(ax=ax, label='dis-reanalysis')
fcst.plot(ax=ax, label='dis-forecast')
plt.legend()
#plt.gca().set_xlim(dt.datetime(1981,1,1), dt.datetime(1982,1,1)) 

((fcst-obs)/obs*100).plot()

y = add_time(yda.squeeze(), time)
((maxpred-y)/y*100).plot()

s = pipe.named_steps['model'].model.to_json()
import json
with open('flowmodel.json', 'w') as f:
    json.dump(s, f, indent=4)    
